"""Span dataclass and overlap resolution."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import PRIORITY, REDACTED_PLACEHOLDER


@dataclass(frozen=True)
class DetectedSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str = REDACTED_PLACEHOLDER

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "placeholder": self.placeholder,
        }


def _overlaps(a: DetectedSpan, b: DetectedSpan) -> bool:
    return a.start < b.end and b.start < a.end


def _beats(a: DetectedSpan, b: DetectedSpan) -> bool:
    """Return True if span ``a`` should be kept over ``b``."""
    len_a = a.end - a.start
    len_b = b.end - b.start
    if len_a != len_b:
        return len_a > len_b
    return PRIORITY.get(a.label, 0) >= PRIORITY.get(b.label, 0)


def resolve_overlaps(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Drop overlapping spans by keeping the longer / higher-priority one.

    Stable across equal cases: earlier-positioned spans win ties after the
    priority check because the input is sorted before pruning.
    """
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[DetectedSpan] = []
    for span in ordered:
        conflict_idx = None
        for i, existing in enumerate(kept):
            if _overlaps(span, existing):
                conflict_idx = i
                break
        if conflict_idx is None:
            kept.append(span)
            continue
        if _beats(span, kept[conflict_idx]):
            kept[conflict_idx] = span
    kept.sort(key=lambda s: s.start)
    return kept
