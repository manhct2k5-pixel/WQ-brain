#!/usr/bin/env python3
"""Render a manual-review queue from the latest planning artifacts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.approve_seeds import exceeds_risk_budget, load_seed_store, score_candidate
from scripts.flow_utils import load_json as load_artifact_json


def load_json(path: Path) -> dict:
    return load_artifact_json(
        path,
        default={},
        required_fields=("batch",),
        context=f"manual review input {path}",
        warn=lambda message: print(f"[manual-review] Warning: {message}", file=sys.stderr),
    )


def select_review_candidates(
    payload: dict,
    *,
    top: int,
    existing_seed_store: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    candidates = [item for item in payload.get("batch", {}).get("candidates", []) if isinstance(item, dict)]
    existing_expressions = set(existing_seed_store or {})
    rejected = []

    ranked = sorted(
        (
            {
                **candidate,
                "compiled_expression": candidate.get("compiled_expression") or candidate.get("expression"),
                "selection_score": score_candidate(candidate),
            }
            for candidate in candidates
            if candidate.get("expression") or candidate.get("compiled_expression")
        ),
        key=lambda item: (
            int(bool(item.get("seed_ready"))),
            int(bool(item.get("qualified"))),
            item.get("selection_score", 0.0),
            item.get("candidate_score", 0.0),
            item.get("novelty_score", 0.0),
            item.get("style_alignment_score", 0.0),
        ),
        reverse=True,
    )

    selected = []
    selected_expressions = set()
    thesis_counts = Counter()
    risk_counts = Counter()

    for thesis_cap in (1, 2):
        for candidate in ranked:
            if len(selected) >= top:
                break
            expression = candidate.get("compiled_expression") or candidate.get("expression")
            thesis_id = candidate.get("thesis_id") or "unknown"
            if not expression:
                continue
            if expression in selected_expressions or expression in existing_expressions:
                continue
            if thesis_counts[thesis_id] >= thesis_cap:
                continue
            if exceeds_risk_budget(candidate, risk_counts):
                continue
            selected.append(candidate)
            selected_expressions.add(expression)
            thesis_counts[thesis_id] += 1
            risk_counts.update(candidate.get("risk_tags", []))
        if len(selected) >= top:
            break

    for candidate in ranked:
        expression = candidate.get("compiled_expression") or candidate.get("expression")
        if not expression or expression in selected_expressions:
            continue
        if expression in existing_expressions:
            rejected.append(
                {
                    "expression": candidate.get("expression"),
                    "reason": "already present in seed store",
                }
            )
            continue
        rejected.append(
            {
                "expression": candidate.get("expression"),
                "reason": "held out to preserve thesis/risk diversity",
            }
        )

    return selected, rejected


def render_manual_review(payload: dict, seed_store: dict, top: int) -> str:
    memory = payload.get("memory", {})
    candidates = payload.get("batch", {}).get("candidates", [])
    selected, rejected = select_review_candidates(payload, top=top, existing_seed_store=seed_store)

    lines = ["# Manual Review Queue", ""]
    lines.append("- This file is for human review only.")
    lines.append("- Validate the financial logic yourself before any manual submission in the Brain web UI.")
    lines.append(f"- planner_candidates: {len(candidates)}")
    lines.append(f"- suggested_review_set: {len(selected)}")
    lines.append("")

    style_leaders = memory.get("style_leaders", [])
    if style_leaders:
        lines.append("## Style Leaders")
        for item in style_leaders[:4]:
            lines.append(f"- {item.get('tag')}: learning_score={item.get('learning_score')}")
        lines.append("")

    if selected:
        lines.append("## Review First")
        for index, candidate in enumerate(selected, start=1):
            lines.append(f"- #{index} [{candidate.get('thesis')}] {candidate.get('expression')}")
            lines.append(f"  why: {candidate.get('why')}")
            lines.append(
                f"  selection_score: {candidate.get('selection_score')}, "
                f"novelty: {candidate.get('novelty_score')}, "
                f"style: {candidate.get('style_alignment_score')}"
            )
            lines.append(f"  risk_tags: {', '.join(candidate.get('risk_tags', [])) or 'none'}")
        lines.append("")

    held_out = [item for item in rejected if item.get("reason") == "held out to preserve thesis/risk diversity"]
    if held_out:
        lines.append("## Held Out")
        for item in held_out[:4]:
            lines.append(f"- {item.get('expression')}")
        lines.append("")

    lines.append("## Files")
    lines.append("- artifacts/bao_cao_moi_nhat.md")
    lines.append("- artifacts/lo_tiep_theo.json")
    lines.append("- simulation_results.csv")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a manual review queue from planning artifacts.")
    parser.add_argument("--input", default="artifacts/lo_tiep_theo.json", help="Planner JSON input.")
    parser.add_argument("--seed-path", default="initial-population.pkl", help="Current seed store path.")
    parser.add_argument("--top", type=int, default=4, help="How many candidates to show in the queue.")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    payload_path = Path(args.input)
    if not payload_path.exists():
        print(f"Planner JSON not found: {payload_path}", file=sys.stderr)
        return 1

    payload = load_json(payload_path)
    seed_store = load_seed_store(Path(args.seed_path))
    markdown = render_manual_review(payload, seed_store, args.top)

    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
