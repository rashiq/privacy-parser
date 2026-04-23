# OpenAI Privacy Parser

OpenAI shipped [Privacy Filter](https://github.com/openai/privacy-filter) — a
model that **hides** PII in text. Defenders use it so data doesn't leak.

But every defense has a second side. Attackers need PII too. From logs,
dumps, abandoned S3 buckets, stolen inboxes. Same task: find everything
personal in a pile of text.

**Privacy Filter masks. Privacy Parser extracts.**

Same model, same label taxonomy, same weights. Instead of `<REDACTED>` you
get structured spans — what, where, which type.

Defense: audit your data before it leaks.
Offense: parse someone else's after it did.

The tool doesn't know whose side it's on. It's just a good parser.

---

## Install

```bash
uv venv
uv pip install -e ./privacy-filter
uv pip install -e ./pii_parser
```

First run downloads the opf 1.5B checkpoint (~3 GB) to `~/.opf/privacy_filter/`.

## Use

```python
from pii_parser.hybrid import HybridPIIParser

parser = HybridPIIParser(device="cpu")
result = parser.parse(
    "Hi Quindle Testwick (quindle.testwick@openai.com / +1-415-555-0102), "
    "account 40702810500001234567, 14 Beautiful Ct, Anytown USA, "
    "password Priv4cy-Filt3r-2026."
)
for s in result.spans:
    print(f"{s.label:18}  {s.text}")
```

```
private_person      Quindle Testwick
private_email       quindle.testwick@openai.com
private_phone       +1-415-555-0102
account_number      40702810500001234567
private_address     14 Beautiful Ct, Anytown USA
secret              Priv4cy-Filt3r-2026
```

### CLI

```bash
python -m pii_parser.cli_model "Alice paid 40702810500001234567 on 2026-05-17."
```

## Three backends

| Backend            | Weights | Speed       | F1    |
| ------------------ | ------- | ----------- | ----- |
| `PIIParser`        | none    | µs          | 1.000 |
| `ModelPIIParser`   | 1.5B    | 500 ms CPU  | 0.733 |
| `HybridPIIParser`  | 1.5B    | 600 ms CPU  | 0.929 |

Hybrid = model + span-merge + regex backstop. Ship this one.

## Labels

opf v2 taxonomy — 8 categories:

`private_person` · `private_email` · `private_phone` · `private_address` ·
`private_url` · `private_date` · `account_number` · `secret`

## Architecture

```
text
  ↓
opf 1.5B → BIOES logits → Viterbi (tuned) → char spans
  ↓
span-merge (glues Quindle + Testwick)
  ↓
regex backstop (URL, secret, account — where the model misses)
  ↓
spans[]
```

## Benchmarks

```
python pii_parser/tests/test_hybrid.py
```

```
Fixture F1:  0.929
Scenarios:   8/8 passed
Latency:     ~600 ms CPU
```

## License

Apache-2.0.
