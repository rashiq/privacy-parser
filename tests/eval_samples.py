"""Compare parser output against opf's sample ground-truth spans."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pii_parser"))

from pii_parser import PIIParser  # noqa: E402

SAMPLES = ROOT / "privacy-filter" / "examples" / "data" / "sample_eval_five_examples.jsonl"


def load_truth():
    rows = []
    with SAMPLES.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def flatten_truth(spans_field):
    out = []
    for key, ranges in spans_field.items():
        label = key.split(":", 1)[0].strip()
        for start, end in ranges:
            out.append((label, start, end))
    return sorted(out)


def main():
    parser = PIIParser()
    total_truth = 0
    total_pred = 0
    total_tp = 0
    partial_overlap = 0

    for row in load_truth():
        text = row["text"]
        truth = flatten_truth(row["spans"])
        result = parser.parse(text)
        preds = [(s.label, s.start, s.end) for s in result.detected_spans]

        print(f"\n--- {row['info']['id']} ---")
        print(f"TEXT: {text!r}")
        print("TRUTH:")
        for t in truth:
            print(f"  {t}  -> {text[t[1]:t[2]]!r}")
        print("PREDICTED:")
        for p in preds:
            print(f"  {p}  -> {text[p[1]:p[2]]!r}")

        truth_set = set(truth)
        pred_set = set(preds)
        tp = len(truth_set & pred_set)
        total_truth += len(truth_set)
        total_pred += len(pred_set)
        total_tp += tp

        for tl, ts, te in truth:
            for pl, ps, pe in preds:
                if tl == pl and ps < te and ts < pe and (tl, ts, te) != (pl, ps, pe):
                    partial_overlap += 1
                    break

    precision = total_tp / total_pred if total_pred else 0
    recall = total_tp / total_truth if total_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    print("\n=== EXACT-MATCH METRICS ===")
    print(f"truth={total_truth} pred={total_pred} tp={total_tp} "
          f"partial={partial_overlap}")
    print(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


if __name__ == "__main__":
    main()
