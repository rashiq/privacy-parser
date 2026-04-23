"""Public API. Mirrors opf.RedactionResult so consumers can swap backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from .detectors import run_all
from .labels import REDACTED_PLACEHOLDER
from .spans import DetectedSpan, resolve_overlaps

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PIIParseResult:
    schema_version: int
    text: str
    detected_spans: tuple[DetectedSpan, ...]
    redacted_text: str
    summary: dict

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "summary": dict(self.summary),
            "text": self.text,
            "detected_spans": [s.to_dict() for s in self.detected_spans],
            "redacted_text": self.redacted_text,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _apply_redaction(text: str, spans: Iterable[DetectedSpan]) -> str:
    spans = sorted(spans, key=lambda s: s.start)
    out: list[str] = []
    cursor = 0
    for s in spans:
        out.append(text[cursor : s.start])
        out.append(s.placeholder)
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


def _summarize(spans: Iterable[DetectedSpan]) -> dict:
    counts: dict[str, int] = {}
    for s in spans:
        counts[s.label] = counts.get(s.label, 0) + 1
    return {"span_counts": counts, "total_spans": sum(counts.values())}


class PIIParser:
    """Pattern-based PII parser compatible with opf's output schema."""

    def __init__(self, *, placeholder: str = REDACTED_PLACEHOLDER) -> None:
        self._placeholder = placeholder

    def parse(self, text: str) -> PIIParseResult:
        candidates = run_all(text)
        if self._placeholder != REDACTED_PLACEHOLDER:
            candidates = [
                DetectedSpan(s.label, s.start, s.end, s.text, self._placeholder)
                for s in candidates
            ]
        resolved = resolve_overlaps(candidates)
        resolved = [
            DetectedSpan(s.label, s.start, s.end, text[s.start : s.end], s.placeholder)
            for s in resolved
        ]
        redacted = _apply_redaction(text, resolved)
        return PIIParseResult(
            schema_version=SCHEMA_VERSION,
            text=text,
            detected_spans=tuple(resolved),
            redacted_text=redacted,
            summary=_summarize(resolved),
        )

    def redact(self, text: str) -> str:
        return self.parse(text).redacted_text


_DEFAULT = PIIParser()


def parse(text: str) -> PIIParseResult:
    """Module-level convenience. Uses a cached default parser."""
    return _DEFAULT.parse(text)
