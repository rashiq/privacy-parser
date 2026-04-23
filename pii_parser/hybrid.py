"""Hybrid PII parser: opf model + span-merge + regex fallbacks.

Rationale:
- The opf model gives high-quality context-aware entity detection, but its
  BIOES span decoder sometimes fragments multi-token entities (e.g. a first
  name and last name as two adjacent S-tagged spans).
- In short low-context sentences the model also occasionally misses URLs,
  structured secrets (e.g. `ghp_*`, `sk-*`), and long digit accounts, or
  mislabels them (e.g. account as phone).

The hybrid passes model output through three stages:
  1. merge adjacent same-label spans with a short whitespace/connector gap
  2. run deterministic regex detectors for URL, secret, account_number
  3. resolve overlaps by preferring model output where it exists, falling
     back to regex spans that do not intersect any model span
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .detectors import detect_urls, detect_secrets, detect_accounts
from .model_parser import ModelPIIParser, ParsedSpan
from .postprocess import merge_adjacent_spans


@dataclass(frozen=True)
class HybridParseResult:
    text: str
    spans: tuple[ParsedSpan, ...]
    warning: str | None = None

    def to_dict(self) -> dict:
        payload: dict = {
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
        }
        if self.warning is not None:
            payload["warning"] = self.warning
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _overlaps(a: ParsedSpan, b: ParsedSpan) -> bool:
    return a.start < b.end and b.start < a.end


class HybridPIIParser:
    """Parser = opf model + merger + regex backstop."""

    # Labels where the regex layer is allowed to add new spans (filling gaps
    # the model misses). For private_person/private_phone/private_email/date
    # we trust the model — regex would introduce more noise than signal.
    _REGEX_BACKSTOP_LABELS = ("private_url", "secret", "account_number")

    # Lightly discourage "end then immediately start new span" in favour of
    # continuing the current span. Keeps truly separate adjacent entities
    # (different labels) untouched.
    _DEFAULT_BIASES = {
        "transition_bias_end_to_start": -0.5,
        "transition_bias_inside_to_continue": 0.2,
    }

    def __init__(
        self,
        *,
        model: str | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        enable_merge: bool = True,
        enable_regex_backstop: bool = True,
        enable_viterbi_tuning: bool = True,
    ) -> None:
        biases = self._DEFAULT_BIASES if enable_viterbi_tuning else None
        self._model = ModelPIIParser(
            model=model,
            device=device,
            decode_mode=decode_mode,
            viterbi_biases=biases,
        )
        self._enable_merge = enable_merge
        self._enable_regex_backstop = enable_regex_backstop

    def parse(self, text: str) -> HybridParseResult:
        base = self._model.parse(text)
        spans: list[ParsedSpan] = list(base.spans)

        if self._enable_merge:
            spans = merge_adjacent_spans(spans, text)

        if self._enable_regex_backstop:
            spans = self._apply_regex_backstop(text, spans)

        spans.sort(key=lambda s: (s.start, s.end))
        return HybridParseResult(
            text=base.text,
            spans=tuple(spans),
            warning=base.warning,
        )

    def _apply_regex_backstop(
        self, text: str, model_spans: list[ParsedSpan]
    ) -> list[ParsedSpan]:
        """Add regex spans for URL/secret/account that the model missed or mislabeled.

        Strategy:
        - For each regex candidate, if it overlaps a model span with the same
          label, keep the model span.
        - If it overlaps a model span with a *different* label but the regex
          has very strong evidence (URL starts with http://, secret matches a
          prefix like `sk-`/`ghp_`), prefer the regex label (this corrects the
          common account-number-as-phone mislabel).
        - If it does not overlap any model span, add it.
        """
        candidates: list[ParsedSpan] = []
        for det in (detect_urls, detect_secrets, detect_accounts):
            for s in det(text):
                candidates.append(
                    ParsedSpan(label=s.label, start=s.start, end=s.end, text=s.text)
                )
        if not candidates:
            return model_spans

        out = list(model_spans)
        for cand in candidates:
            overlapping = [i for i, m in enumerate(out) if _overlaps(m, cand)]
            if not overlapping:
                out.append(cand)
                continue
            same_label = [i for i in overlapping if out[i].label == cand.label]
            if same_label:
                # Model already covers it with the right label — defer to model.
                continue
            # Different label overlap. Prefer regex for strong-prefix secrets
            # and schemed URLs; for account_number we also prefer regex if the
            # candidate fully contains the model span (common phone mislabel).
            if cand.label == "private_url" and cand.text.startswith(("http://", "https://")):
                for i in sorted(overlapping, reverse=True):
                    del out[i]
                out.append(cand)
                continue
            if cand.label == "secret" and any(
                cand.text.startswith(p)
                for p in ("sk-", "pk-", "ghp_", "gho_", "xoxb-", "xoxp-", "AKIA", "Bearer ")
            ):
                for i in sorted(overlapping, reverse=True):
                    del out[i]
                out.append(cand)
                continue
            if cand.label == "account_number":
                # Only swap if regex span fully contains the model span AND the
                # model labeled it as a phone — the exact failure mode we saw.
                contained = [
                    i for i in overlapping
                    if out[i].start >= cand.start and out[i].end <= cand.end
                    and out[i].label == "private_phone"
                ]
                if contained:
                    for i in sorted(contained, reverse=True):
                        del out[i]
                    out.append(cand)
        return out


_DEFAULT: HybridPIIParser | None = None


def parse_hybrid(text: str) -> HybridParseResult:
    """Module-level convenience."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = HybridPIIParser(device="cpu")
    return _DEFAULT.parse(text)
