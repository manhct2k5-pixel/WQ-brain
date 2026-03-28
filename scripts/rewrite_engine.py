#!/usr/bin/env python3
"""Generate rewrite candidates for a diagnosed alpha."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fix_alpha import build_context, build_rewrite_candidates
from scripts.flow_utils import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate raw rewrite candidates for a diagnosed alpha.")
    parser.add_argument("--alpha-id", help="Look up a submitted alpha from the simulation CSV.")
    parser.add_argument("--csv", help="Optional CSV path or repo directory.")
    parser.add_argument("--expression", help="Alpha expression to diagnose directly.")
    parser.add_argument("--errors", nargs="*", default=[], help="Failing checks to diagnose.")
    parser.add_argument("--sharpe", help="Optional Sharpe value to include in the context.")
    parser.add_argument("--fitness", help="Optional fitness value to include in the context.")
    parser.add_argument("--turnover", help="Optional turnover value to include in the context.")
    parser.add_argument("--max-variants-per-family", type=int, default=2)
    parser.add_argument("--max-total", type=int, default=8)
    parser.add_argument("--output", required=True, help="Rewrite candidate JSON output path.")
    args = parser.parse_args()

    context = build_context(args)
    payload = {
        "context": context,
        "candidates": build_rewrite_candidates(
            context,
            max_variants_per_family=args.max_variants_per_family,
            max_total=args.max_total,
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"candidates": len(payload["candidates"]), "output": args.output}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
