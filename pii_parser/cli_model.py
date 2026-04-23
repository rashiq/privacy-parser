"""CLI: `python -m pii_parser.cli_model` — model-backed PII parser."""

from __future__ import annotations

import argparse
import sys

from .model_parser import ModelPIIParser


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
        prog="pii-parser-model",
        description="Extract PII spans using the opf model (no masking).",
    )
    p.add_argument("text", nargs="?")
    p.add_argument("-f", "--file")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--model", help="Checkpoint dir (overrides OPF_CHECKPOINT)")
    p.add_argument(
        "--mode",
        choices=("json", "spans"),
        default="json",
    )
    p.add_argument("--indent", type=int, default=2)
    args = p.parse_args(argv)

    text = _read_input(args)
    parser = ModelPIIParser(model=args.model, device=args.device)
    result = parser.parse(text)

    if args.mode == "spans":
        for s in result.spans:
            print(f"{s.label}\t{s.start}\t{s.end}\t{s.text}")
        return 0
    indent = args.indent if args.indent > 0 else None
    print(result.to_json(indent=indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
