#!/usr/bin/env python3
"""Build the evaluated candidate pool consumed by daily/feed/seed flows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import atomic_write_json, load_json, read_jsonl
from scripts.lineage_utils import attach_evaluation_stage, ensure_candidate_lineage
from src.submit_gate import candidate_passes_submit_gate, submit_gate_fail_reasons


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _candidate_from_record(record: dict) -> dict:
    planning_qualified = bool(record.get("qualified"))
    planning_quality_label = record.get("quality_label") or ("qualified" if planning_qualified else "watchlist")
    planning_quality_fail_reasons = list(record.get("quality_fail_reasons") or [])
    lineage = ensure_candidate_lineage(
        record,
        stage_source=record.get("source"),
        source_detail=record.get("source"),
        default_hypothesis_id=record.get("thesis_id"),
        default_hypothesis_label=record.get("thesis"),
        default_generation_reason=record.get("why"),
    )
    candidate = {
        "run_id": record.get("run_id"),
        "batch_id": record.get("batch_id"),
        "candidate_id": record.get("candidate_id"),
        "source": record.get("source"),
        "source_stages": record.get("source_stages", [record.get("source")]),
        "thesis": record.get("thesis"),
        "thesis_id": record.get("thesis_id"),
        "why": record.get("why"),
        "expression": record.get("expression") or record.get("regular_code"),
        "compiled_expression": record.get("compiled_expression") or record.get("expression") or record.get("regular_code"),
        "normalized_expression": record.get("normalized_expression"),
        "normalized_compiled_expression": record.get("normalized_compiled_expression"),
        "expression_skeleton": record.get("expression_skeleton"),
        "candidate_signature": record.get("candidate_signature"),
        "structure_signature": record.get("structure_signature"),
        "expression_signature": record.get("expression_signature"),
        "skeleton_signature": record.get("skeleton_signature"),
        "token_program": record.get("token_program") or [],
        "candidate_score": record.get("candidate_score"),
        "confidence_score": record.get("confidence_score"),
        "novelty_score": record.get("novelty_score"),
        "style_alignment_score": record.get("style_alignment_score"),
        "risk_tags": list(record.get("risk_tags") or []),
        "seed_ready": bool(record.get("seed_ready")),
        "qualified": planning_qualified,
        "quality_label": planning_quality_label,
        "quality_fail_reasons": planning_quality_fail_reasons,
        "planning_qualified": planning_qualified,
        "planning_quality_label": planning_quality_label,
        "planning_quality_fail_reasons": planning_quality_fail_reasons,
        "settings": record.get("settings"),
        "local_metrics": record.get("local_metrics") or {},
        "evaluation_backend": record.get("evaluation_backend"),
        "evaluation_status": record.get("evaluation_status"),
        "simulation_result": record.get("simulation_result"),
        "priority_score": record.get("priority_score"),
        "priority_score_before_recent_failures": record.get("priority_score_before_recent_failures"),
        "recent_failure_penalty": record.get("recent_failure_penalty"),
        "recent_failure_reasons": list(record.get("recent_failure_reasons") or []),
        "recent_failure_match_count": record.get("recent_failure_match_count"),
        "dedupe_match_types": list(record.get("dedupe_match_types") or []),
        "duplicate_candidate_count": record.get("duplicate_candidate_count"),
        "merged_candidate_signatures": list(record.get("merged_candidate_signatures") or []),
        "lineage": lineage,
        "evaluated_submit_ready": False,
        "evaluated_fail_reasons": [],
    }
    evaluated_gate_candidate = {
        **candidate,
        "qualified": True,
        "quality_label": "qualified",
    }
    evaluated_fail_reasons = submit_gate_fail_reasons(evaluated_gate_candidate)
    candidate["evaluated_submit_ready"] = not evaluated_fail_reasons
    candidate["evaluated_fail_reasons"] = evaluated_fail_reasons
    candidate["qualified"] = candidate["evaluated_submit_ready"]
    candidate["quality_label"] = "qualified" if candidate["evaluated_submit_ready"] else "watchlist"
    candidate["quality_fail_reasons"] = [] if candidate["evaluated_submit_ready"] else evaluated_fail_reasons
    candidate["lineage"] = attach_evaluation_stage(
        candidate.get("lineage"),
        {**record, **candidate},
        submit_ready=candidate["evaluated_submit_ready"],
        qualified=candidate["qualified"],
        quality_label=candidate["quality_label"],
        fail_reasons=candidate["quality_fail_reasons"],
    )
    return candidate


def build_evaluated_pool(records: list[dict], *, memory: dict | None = None, results_summary: dict | None = None) -> dict:
    candidates = [_candidate_from_record(record) for record in records if isinstance(record, dict)]
    candidates.sort(
        key=lambda item: (
            int(item.get("evaluated_submit_ready", False)),
            to_float((item.get("local_metrics") or {}).get("alpha_score")),
            to_float((item.get("local_metrics") or {}).get("fitness")),
            to_float((item.get("local_metrics") or {}).get("sharpe")),
            to_float(item.get("confidence_score")),
            to_float(item.get("priority_score")),
        ),
        reverse=True,
    )
    return {
        "memory": memory or {},
        "results_summary": results_summary or {},
        "summary": {
            "candidate_count": len(candidates),
            "qualified_count": sum(1 for item in candidates if item.get("qualified")),
            "submit_ready_count": sum(1 for item in candidates if item.get("evaluated_submit_ready")),
            "pending_count": sum(1 for item in candidates if item.get("evaluation_status") == "PENDING"),
        },
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build evaluated candidate JSON from simulation results JSONL.")
    parser.add_argument("--input", required=True, help="Simulation results JSONL input.")
    parser.add_argument("--memory", help="Optional run memory JSON to carry into the evaluated pool.")
    parser.add_argument("--results-summary", help="Optional results summary JSON.")
    parser.add_argument("--output", required=True, help="Evaluated candidate JSON output path.")
    args = parser.parse_args()

    records = read_jsonl(args.input)
    memory = load_json(Path(args.memory), default={}) if args.memory else {}
    results_summary = load_json(Path(args.results_summary), default={}) if args.results_summary else {}
    payload = build_evaluated_pool(records, memory=memory, results_summary=results_summary)
    atomic_write_json(args.output, payload)
    print(json.dumps(payload.get("summary", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
