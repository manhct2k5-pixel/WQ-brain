#!/usr/bin/env python3
"""Seed exactly the submit-ready candidates shown in the daily report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.approve_seeds import merge_into_seed_store
from scripts.daily_best import (
    DEFAULT_TOP_CANDIDATES,
    load_auto_fix_candidates,
    load_json,
    resolve_input_path,
    select_submit_ready_candidates,
)
from scripts.flow_utils import atomic_write_json
from src.program_tokens import validate_token_program
from src.seed_store import SeedStoreCorruptError, load_seed_store, write_seed_store

DEFAULT_INPUT = Path("artifacts/latest/evaluated_candidates.json")
DEFAULT_SEED_PATH = Path("initial-population.pkl")
DEFAULT_MARKDOWN_OUTPUT = Path("artifacts/seed_submit_ready.md")
DEFAULT_JSON_OUTPUT = Path("artifacts/seed_submit_ready.json")


def select_submit_ready_for_seed(
    payload: dict,
    seed_store: dict,
    *,
    top: int = DEFAULT_TOP_CANDIDATES,
    extra_candidates: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    selected = select_submit_ready_candidates(payload, seed_store, top=top, extra_candidates=extra_candidates)
    approved = []
    rejected = []

    for candidate in selected:
        token_program = candidate.get("token_program") or []
        try:
            compiled = validate_token_program(token_program)
        except Exception as exc:
            rejected.append(
                {
                    "expression": candidate.get("expression"),
                    "reason": f"validation failed: {exc}",
                }
            )
            continue

        approved.append(
            {
                **candidate,
                "compiled_expression": str(compiled),
                "seed_source": "submit_ready_report",
            }
        )

    return approved, rejected


def render_markdown(summary: dict) -> str:
    lines = ["# Submit-Ready Seed Approval", ""]
    lines.append(f"- Selected from report: {summary['selected_count']}")
    lines.append(f"- Inserted: {summary['inserted_count']}")
    lines.append(f"- Skipped existing: {summary['skipped_count']}")
    lines.append(f"- Rejected: {summary['rejected_count']}")
    lines.append("")

    if summary["inserted"]:
        lines.append("## Inserted")
        for item in summary["inserted"]:
            lines.append(f"- {item['expression']} (submit_score={item.get('submit_score')})")
        lines.append("")

    if summary["skipped"]:
        lines.append("## Skipped existing")
        for item in summary["skipped"]:
            lines.append(f"- {item['expression']}")
        lines.append("")

    if summary["rejected"]:
        lines.append("## Rejected")
        for item in summary["rejected"]:
            lines.append(f"- {item['expression']}: {item['reason']}")
    return "\n".join(lines)


def resolve_output_path(output: str | None, format_name: str) -> Path:
    if output:
        return Path(output)
    if format_name == "json":
        return DEFAULT_JSON_OUTPUT
    return DEFAULT_MARKDOWN_OUTPUT


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the submit-ready candidates shown in the daily report.")
    parser.add_argument("--input", help="Path to evaluated candidate JSON output.")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_CANDIDATES, help="Maximum number of submit-ready candidates to seed.")
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH), help="Target initial-population pickle path.")
    parser.add_argument("--snapshot-path", help="Optional run-scoped seed snapshot path.")
    parser.add_argument("--auto-fix-input", help="Optional legacy auto-fix candidate JSON to merge into submit-ready seeding.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", help="Optional summary output path. Defaults to artifacts/seed_submit_ready.md or .json.")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    if not input_path.exists():
        print(f"Planner JSON not found: {input_path}", file=sys.stderr)
        return 1

    payload = load_json(input_path)
    seed_path = Path(args.seed_path)
    try:
        seed_store = load_seed_store(seed_path, on_corrupt="raise")
    except SeedStoreCorruptError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    extra_candidates = load_auto_fix_candidates(Path(args.auto_fix_input)) if args.auto_fix_input else []
    approved, rejected = select_submit_ready_for_seed(payload, seed_store, top=args.top, extra_candidates=extra_candidates)
    merged, inserted, skipped = merge_into_seed_store(seed_store, approved)
    write_seed_store(seed_path, merged)
    if args.snapshot_path:
        write_seed_store(Path(args.snapshot_path), merged)

    summary = {
        "selected_count": len(approved),
        "inserted_count": len(inserted),
        "skipped_count": len(skipped),
        "rejected_count": len(rejected),
        "inserted": [
            {
                "expression": item["compiled_expression"],
                "submit_score": item.get("submit_score"),
            }
            for item in approved
            if item["compiled_expression"] in inserted
        ],
        "skipped": [
            {
                "expression": item["compiled_expression"],
            }
            for item in approved
            if item["compiled_expression"] in skipped
        ],
        "rejected": rejected,
    }

    if args.format == "json":
        rendered = json.dumps(summary, indent=2)
    else:
        rendered = render_markdown(summary)

    output_path = resolve_output_path(args.output, args.format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        atomic_write_json(output_path, summary)
    else:
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
