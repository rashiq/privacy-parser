"""Span postprocessing for the model-backed parser.

The opf model emits BIOES per token, and its span-decoder faithfully converts
those into spans. But this means multi-word entities sometimes arrive as two
*adjacent* same-label spans — e.g. `Quindle` + `Testwick` as two S-tagged
person spans, or `404` + `Nowhere Lane` as an S + B/E pair for the same
address. For a *parser* (entity-level output) we want these glued back.

Rules:
- Two spans with the same label, separated only by whitespace or a small
  set of connectors (`.,-/`, single spaces), are merged.
- For person names we also allow comma + space (for "Last, First" style)
  but cap the gap at 3 chars.
- Non-person spans use a gap cap of 2 chars (space, hyphen, period).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .model_parser import ParsedSpan


# Max gap in characters between two same-label spans that we will still merge.
_MAX_GAP_BY_LABEL: dict[str, int] = {
    "private_person": 3,
    "private_address": 3,
    "private_url": 2,
    "private_email": 0,
    "private_phone": 2,
    "private_date": 3,
    "account_number": 1,
    "secret": 2,
}

# Allowed connector characters between merged spans (stripped before check).
_ALLOWED_CONNECTORS = set(" \t.-,/")


def _gap_is_mergeable(text: str, left_end: int, right_start: int, label: str) -> bool:
    if right_start < left_end:
        return False
    gap = text[left_end:right_start]
    max_gap = _MAX_GAP_BY_LABEL.get(label, 1)
    if len(gap) > max_gap:
        return False
    if not gap:
        return True
    return all(ch in _ALLOWED_CONNECTORS for ch in gap)


def merge_adjacent_spans(spans: Iterable[ParsedSpan], text: str) -> list[ParsedSpan]:
    """Merge same-label spans separated only by a short whitespace/connector gap.

    Input does not need to be sorted; output is sorted by start offset.
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    if not ordered:
        return []
    merged: list[ParsedSpan] = [ordered[0]]
    for current in ordered[1:]:
        last = merged[-1]
        if (
            current.label == last.label
            and _gap_is_mergeable(text, last.end, current.start, current.label)
        ):
            new_end = max(last.end, current.end)
            merged[-1] = replace(
                last,
                start=last.start,
                end=new_end,
                text=text[last.start:new_end],
            )
        else:
            merged.append(current)
    return merged
