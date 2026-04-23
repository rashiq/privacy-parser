"""Exercise HybridPIIParser against the same scenarios as test_model."""

from __future__ import annotations

import json
import time
from pathlib import Path

from pii_parser.hybrid import HybridPIIParser

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "privacy-filter" / "examples" / "data" / "sample_eval_five_examples.jsonl"


def flatten_truth(spans_field):
    out = []
    for key, ranges in spans_field.items():
        label = key.split(":", 1)[0].strip()
        for start, end in ranges:
            out.append((label, start, end))
    return sorted(out)


def exact_match_metrics(parser):
    total_truth = 0
    total_pred = 0
    total_tp = 0
    rows = []
    with SAMPLES.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    for row in rows:
        truth = set(flatten_truth(row["spans"]))
        res = parser.parse(row["text"])
        pred = {(s.label, s.start, s.end) for s in res.spans}
        total_truth += len(truth)
        total_pred += len(pred)
        total_tp += len(truth & pred)
    prec = total_tp / total_pred if total_pred else 0
    rec = total_tp / total_truth if total_truth else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {
        "truth": total_truth,
        "pred": total_pred,
        "tp": total_tp,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def test_case(parser, name, text, expected_labels):
    t0 = time.perf_counter()
    res = parser.parse(text)
    dt = time.perf_counter() - t0
    found = {s.label for s in res.spans}
    missing = set(expected_labels) - found
    extra = found - set(expected_labels)
    status = "PASS" if not missing else "FAIL"
    print(f"[{status}] {name}  ({dt*1000:.0f} ms, {len(res.spans)} spans)")
    print(f"  text:     {text!r}")
    for s in res.spans:
        print(f"    - {s.label}: {s.text!r} [{s.start}:{s.end}]")
    if missing:
        print(f"  MISSING:  {sorted(missing)}")
    if extra:
        print(f"  EXTRA:    {sorted(extra)}")
    print()
    return {"name": name, "status": status, "dt": dt, "missing": missing}


def main():
    print("=" * 70)
    print("Loading hybrid parser (model + merge + regex + viterbi tuning)")
    print("=" * 70)
    t0 = time.perf_counter()
    parser = HybridPIIParser(device="cpu")
    _ = parser.parse("hi")
    print(f"Ready in {time.perf_counter()-t0:.1f}s\n")

    print("=" * 70)
    print("opf sample fixtures (ground-truth)")
    print("=" * 70)
    metrics = exact_match_metrics(parser)
    print(f"  truth={metrics['truth']}  pred={metrics['pred']}  tp={metrics['tp']}")
    print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  f1={metrics['f1']:.3f}\n")

    cases = [
        (
            "basic_contact",
            "Please email alice@example.com or call +1 (415) 555-0198.",
            ["private_email", "private_phone"],
        ),
        (
            "multi_person",
            "Dr. Jane Doe and Mr. John Smith will meet on 2026-01-15.",
            ["private_person", "private_date"],
        ),
        (
            "secrets_and_urls",
            "Reset at https://acme.io/reset?t=qwerty. Rotate ghp_abcdefghijklmnopqrstuvwxyz012345.",
            ["private_url", "secret"],
        ),
        (
            "address_and_account",
            "Send payment to 221B Baker Street, London for account 4532015112830366.",
            ["private_address", "account_number"],
        ),
        (
            "multilingual_mixed",
            "Call Олег Иванов at +7 495 123-45-67 or oleg@example.ru.",
            ["private_person", "private_phone", "private_email"],
        ),
        ("empty", "", []),
        ("no_pii", "The weather is nice today and the clouds look fluffy.", []),
        (
            "dense_paragraph",
            "Hi Quindle Testwick (quindle.testwick@openai.com / +1-415-555-0102), "
            "your package 40702810500001234567 ships to 14 Beautiful Ct, Anytown USA "
            "on 2026-05-17. Password Priv4cy-Filt3r-2026. Portal: https://portal.example.org/reset.",
            [
                "private_person", "private_email", "private_phone",
                "account_number", "private_address", "private_date",
                "secret", "private_url",
            ],
        ),
    ]

    print("=" * 70)
    print("Scenario cases")
    print("=" * 70)
    results = [test_case(parser, *c) for c in cases]

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    avg_ms = sum(r["dt"] for r in results) / max(1, len(results)) * 1000
    print(f"Scenario cases: {passed} passed, {failed} failed, avg {avg_ms:.0f} ms/call")
    print(f"Fixture F1: {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
