"""Command-line entry point for the local MAX importer accuracy gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tournament import run_tournament


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Exit non-zero when any non-reference candidate is rejected.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result = run_tournament(root, args.corpus, args.variants)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if args.require_accepted:
        candidates = [
            value
            for value in result["variants"]
            if value["variant"] != result["reference_variant"]
        ]
        if not candidates or any(value["verdict"] != "accepted" for value in candidates):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
