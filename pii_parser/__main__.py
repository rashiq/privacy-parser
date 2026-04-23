"""CLI entrypoint: `python -m pii_parser`."""

from __future__ import annotations

import argparse
import json
import sys

from .api import PIIParser


def _read_input(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    if args.file is not None:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    sys.stderr.write("error: provide TEXT, -f FILE, or pipe stdin\n")
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pii-parser",
        description="Extract PII spans (opf-compatible schema).",
    )
    p.add_argument("text", nargs="?", help="Input text (optional; otherwise stdin)")
    p.add_argument("-f", "--file", help="Read input from FILE")
    p.add_argument(
        "--mode",
        choices=("json", "redacted", "spans"),
        default="json",
        help="Output format (default: json)",
    )
    p.add_argument(
        "--placeholder",
        default="<REDACTED>",
        help="Replacement marker for redacted spans",
    )
    p.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (use 0 for compact)",
    )
    args = p.parse_args(argv)

    text = _read_input(args)
    parser = PIIParser(placeholder=args.placeholder)
    result = parser.parse(text)

    if args.mode == "redacted":
        sys.stdout.write(result.redacted_text)
        if not result.redacted_text.endswith("\n"):
            sys.stdout.write("\n")
        return 0
    if args.mode == "spans":
        for s in result.detected_spans:
            print(f"{s.label}\t{s.start}\t{s.end}\t{s.text}")
        return 0
    indent = args.indent if args.indent > 0 else None
    print(result.to_json(indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
