"""Model-backed PII parser.

Thin wrapper around the OpenAI Privacy Filter (opf) checkpoint that returns
*spans* instead of a masked string. Same 1.5B token-classifier, same Viterbi
decoding, same v2 label taxonomy — we just skip the redaction step and hand
back the structured entities.

First call downloads ~3GB of weights into ~/.opf/privacy_filter (or
$OPF_CHECKPOINT).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Literal

from opf._api import OPF, RedactionResult


@dataclass(frozen=True)
class ParsedSpan:
    label: str
    start: int
    end: int
    text: str

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class ModelParseResult:
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


class ModelPIIParser:
    """Parse PII with the opf model. No masking — just structured spans."""

    def __init__(
        self,
        *,
        model: str | None = None,
        device: Literal["cpu", "cuda"] = "cpu",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        output_mode: Literal["typed", "redacted"] = "typed",
        viterbi_biases: dict | None = None,
    ) -> None:
        # output_mode="typed" preserves category labels; "redacted" collapses
        # them all to a single "redacted" label. Keep typed by default because
        # a parser that loses its labels is a regex with extra steps.
        self._opf = OPF(
            model=model,
            device=device,
            output_mode=output_mode,
            decode_mode=decode_mode,
        )
        if viterbi_biases is not None and decode_mode == "viterbi":
            calibration_path = self._write_calibration(viterbi_biases)
            self._opf.set_viterbi_decoder(calibration_path=calibration_path)

    @staticmethod
    def _write_calibration(biases: dict) -> str:
        # opf's calibration loader requires all six keys present.
        required = {
            "transition_bias_background_stay",
            "transition_bias_background_to_start",
            "transition_bias_inside_to_continue",
            "transition_bias_inside_to_end",
            "transition_bias_end_to_background",
            "transition_bias_end_to_start",
        }
        full = {k: float(biases.get(k, 0.0)) for k in required}
        payload = {"operating_points": {"default": {"biases": full}}}
        fd, path = tempfile.mkstemp(prefix="pii_parser_calib_", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        return path

    def parse(self, text: str) -> ModelParseResult:
        result = self._opf.redact(text)
        assert isinstance(result, RedactionResult)
        spans = tuple(
            ParsedSpan(
                label=s.label,
                start=s.start,
                end=s.end,
                text=s.text,
            )
            for s in result.detected_spans
        )
        return ModelParseResult(
            text=result.text,
            spans=spans,
            warning=result.warning,
        )
