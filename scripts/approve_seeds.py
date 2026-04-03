#!/usr/bin/env python3
"""Approve planner candidates into initial-population.pkl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import atomic_write_json, load_json as load_artifact_json
from scripts.lineage_utils import attach_seed_stage, ensure_candidate_lineage
from src.program_tokens import validate_token_program
from src.seed_store import SeedStoreCorruptError, load_seed_store, write_seed_store
from src.submit_gate import submit_gate_fail_reasons

DEFAULT_INPUT = Path("artifacts/lo_tiep_theo.json")
DEFAULT_SEED_PATH = Path("initial-population.pkl")
DEFAULT_MARKDOWN_OUTPUT = Path("artifacts/seed_approval.md")
DEFAULT_JSON_OUTPUT = Path("artifacts/seed_approval.json")


def load_payload(path: str | Path) -> dict:
    return load_artifact_json(
        Path(path),
        default={},
        required_fields=("batch",),
        context=f"seed approval input {path}",
        warn=lambda message: print(f"[approve-seeds] Warning: {message}", file=sys.stderr),
    )

def score_candidate(candidate: dict) -> float:
    risk_tags = set(candidate.get("risk_tags", []))
    risk_penalty = 0.12 * len(risk_tags)
    if {"turnover_risk", "weight_risk"}.issubset(risk_tags):
        risk_penalty += 0.1
    if "unproven_style" in risk_tags:
        risk_penalty += 0.08
    quality_bonus = 0.45 if candidate.get("qualified", True) else -0.1
    return round(
        float(candidate.get("candidate_score", 0.0))
        + 0.35 * float(candidate.get("novelty_score", 0.0))
        + 0.25 * float(candidate.get("style_alignment_score", 0.0))
        + 0.30 * float(candidate.get("confidence_score", 0.0))
        + quality_bonus
        - risk_penalty,
        4,
    )


def exceeds_risk_budget(candidate: dict, risk_counts: Counter) -> bool:
    risk_limits = {
        "turnover_risk": 2,
        "weight_risk": 2,
        "unproven_style": 1,
        "seed_bias_risk": 0,
    }
    for risk_tag in candidate.get("risk_tags", []):
        limit = risk_limits.get(risk_tag)
        if limit is not None and risk_counts[risk_tag] >= limit:
            return True
    return False


def select_candidates(payload: dict, top: int, existing_seed_store: dict | None = None) -> tuple[list[dict], list[dict]]:
    candidates = payload.get("batch", {}).get("candidates", [])
    rejected = []
    validated = []
    existing_expressions = set(existing_seed_store or {})

    ranked = sorted(candidates, key=score_candidate, reverse=True)

    for candidate in ranked:
        if not candidate.get("seed_ready"):
            rejected.append(
                {
                    "expression": candidate.get("expression"),
                    "reason": "seed_ready=false",
                }
            )
            continue

        gate_reasons = submit_gate_fail_reasons(candidate)
        if gate_reasons:
            rejected.append(
                {
                    "expression": candidate.get("expression"),
                    "reason": "submit_gate_failed: " + ", ".join(gate_reasons),
                }
            )
            continue

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

        validated.append(
            {
                **candidate,
                "compiled_expression": str(compiled),
                "selection_score": score_candidate(candidate),
            }
        )

    validated.sort(
        key=lambda item: (item["selection_score"], item.get("candidate_score", 0.0), item.get("novelty_score", 0.0)),
        reverse=True,
    )

    approved = []
    approved_expressions = set()
    thesis_counts = Counter()
    risk_counts = Counter()

    for thesis_cap in (1, 2):
        for candidate in validated:
            if len(approved) >= top:
                break
            expression = candidate["compiled_expression"]
            thesis_id = candidate.get("thesis_id") or "unknown"
            if expression in approved_expressions:
                continue
            if expression in existing_expressions:
                continue
            if thesis_counts[thesis_id] >= thesis_cap:
                continue
            if exceeds_risk_budget(candidate, risk_counts):
                continue
            approved.append(candidate)
            approved_expressions.add(expression)
            thesis_counts[thesis_id] += 1
            risk_counts.update(candidate.get("risk_tags", []))
        if len(approved) >= top:
            break

    for candidate in validated:
        if candidate["compiled_expression"] in approved_expressions:
            continue
        if candidate["compiled_expression"] in existing_expressions:
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

    return approved, rejected


def merge_into_seed_store(seed_store: dict, approved: list[dict]) -> tuple[dict, list[str], list[str]]:
    merged = dict(seed_store)
    inserted = []
    skipped = []

    for candidate in approved:
        expression = candidate["compiled_expression"]
        if expression in merged:
            skipped.append(expression)
            continue
        lineage = ensure_candidate_lineage(
            candidate,
            stage_source=candidate.get("source") or candidate.get("seed_source") or "seed",
            source_detail=candidate.get("seed_source") or candidate.get("source"),
            default_hypothesis_id=candidate.get("thesis_id"),
            default_hypothesis_label=candidate.get("thesis"),
        )
        lineage = attach_seed_stage(
            lineage,
            candidate,
            seed_source=candidate.get("seed_source", "planner"),
            selection_score=candidate.get("selection_score") or candidate.get("submit_score"),
            compiled_expression=expression,
        )
        merged[expression] = {
            "program": candidate["token_program"],
            "fitness": None,
            "result": {
                "status": "PLANNED",
                "source": candidate.get("seed_source", "planner"),
                "thesis_id": candidate.get("thesis_id"),
                "novelty_score": candidate.get("novelty_score"),
                "risk_tags": candidate.get("risk_tags", []),
                "lineage": lineage,
            },
        }
        inserted.append(expression)
    return merged, inserted, skipped


def render_markdown(summary: dict) -> str:
    lines = ["# Seed Approval", ""]
    lines.append(f"- Selected for validation: {summary['selected_count']}")
    lines.append(f"- Inserted: {summary['inserted_count']}")
    lines.append(f"- Skipped existing: {summary['skipped_count']}")
    lines.append(f"- Rejected: {summary['rejected_count']}")
    lines.append("")

    if summary["inserted"]:
        lines.append("## Inserted")
        for item in summary["inserted"]:
            if isinstance(item, dict):
                lines.append(f"- {item['expression']} (selection_score={item.get('selection_score')})")
            else:
                lines.append(f"- {item}")
        lines.append("")

    if summary["skipped"]:
        lines.append("## Skipped existing")
        for expression in summary["skipped"]:
            lines.append(f"- {expression}")
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
    parser = argparse.ArgumentParser(description="Approve planner candidates into initial-population.pkl.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to planner JSON output.")
    parser.add_argument("--top", type=int, default=4, help="Maximum number of candidates to seed.")
    parser.add_argument("--seed-path", default=str(DEFAULT_SEED_PATH), help="Target initial-population pickle path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", help="Optional summary output path. Defaults to artifacts/seed_approval.md or .json.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Planner JSON not found: {input_path}", file=sys.stderr)
        return 1

    payload = load_payload(input_path)
    seed_path = Path(args.seed_path)
    try:
        seed_store = load_seed_store(seed_path, on_corrupt="raise")
    except SeedStoreCorruptError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    approved, rejected = select_candidates(payload, args.top, existing_seed_store=seed_store)
    merged, inserted, skipped = merge_into_seed_store(seed_store, approved)
    write_seed_store(seed_path, merged)

    summary = {
        "selected_count": len(approved),
        "inserted_count": len(inserted),
        "skipped_count": len(skipped),
        "rejected_count": len(rejected),
        "inserted": [
            {
                "expression": item["compiled_expression"],
                "selection_score": item.get("selection_score"),
            }
            for item in approved
            if item["compiled_expression"] in inserted
        ],
        "skipped": skipped,
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
