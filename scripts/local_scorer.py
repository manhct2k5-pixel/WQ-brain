#!/usr/bin/env python3
"""Score rewrite candidates locally and emit actionable auto-fix candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fix_alpha import (
    build_actionable_auto_fix_candidates,
    build_auto_fix_payload,
    load_auto_fix_store,
    merge_auto_fix_candidates,
)
from scripts.flow_utils import atomic_write_json, load_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Score raw rewrite candidates locally.")
    parser.add_argument("--input", required=True, help="Rewrite candidate JSON from rewrite_engine.py.")
    parser.add_argument("--settings", help="Optional scoring settings override.")
    parser.add_argument("--top-rewrites", type=int, default=5)
    parser.add_argument("--output", required=True, help="Scored auto-fix JSON output path.")
    parser.add_argument("--merge-store", help="Optional canonical actionable store to merge into.")
    args = parser.parse_args()

    rewrite_payload = load_json(Path(args.input), default={})
    context = rewrite_payload.get("context") or {}
    auto_fix_payload = build_auto_fix_payload(
        context,
        csv_path=context.get("resolved_csv"),
        settings=args.settings or context.get("settings"),
        top_rewrites=args.top_rewrites,
    )
    actionable = build_actionable_auto_fix_candidates(context, auto_fix_payload)
    payload = {
        **auto_fix_payload,
        "actionable_candidates": actionable,
    }
    atomic_write_json(args.output, payload)

    if args.merge_store:
        existing = load_auto_fix_store(Path(args.merge_store))
        merged = merge_auto_fix_candidates(existing, actionable)
        atomic_write_json(args.merge_store, merged)

    print(json.dumps({"output": args.output, "actionable_candidates": len(actionable)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
