#!/usr/bin/env python3
"""Evaluate a normalized simulation queue and emit JSONL run results."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from scripts.flow_utils import iso_now, load_json, write_jsonl
from scripts.lineage_utils import attach_evaluation_stage, ensure_candidate_lineage
from src.brain import CHECK_RESULT_COLUMNS, save_alpha_to_csv, simulate
from src.internal_scoring import HistoryIndex, parse_scoring_settings, score_expressions_batch
from src.utils import create_authenticated_session


def _coerce_status(result: dict | None) -> str:
    if not result:
        return "ERROR"
    statuses = {str(result.get(name, "")).upper() for name in CHECK_RESULT_COLUMNS if name in result}
    if "PENDING" in statuses:
        return "PENDING"
    if "NOT FOUND" in statuses:
        return "NOT_FOUND"
    return "COMPLETED"


def _settings_context(settings_value) -> dict:
    parsed = parse_scoring_settings(settings_value)
    return {
        "region": parsed.region,
        "universe": parsed.universe,
        "delay": parsed.delay,
        "decay": parsed.decay,
        "truncation": parsed.truncation,
        "neutralization": parsed.neutralization,
    }


def _history_index(csv_path: str | None) -> HistoryIndex:
    if not csv_path:
        return HistoryIndex()
    try:
        return HistoryIndex.from_csv(csv_path)
    except FileNotFoundError:
        return HistoryIndex()


def _resolve_worldquant_credentials(env_path: Path | None = None) -> tuple[str | None, str | None]:
    resolved_env_path = env_path or (ROOT_DIR / ".env")
    load_dotenv(dotenv_path=resolved_env_path, override=False)
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    return username, password


def _build_record(candidate: dict, local_metrics: dict, *, backend: str, remote_result: dict | None = None) -> dict:
    local = local_metrics or {}
    remote = remote_result or {}
    top = remote if remote_result else local
    evaluation_status = _coerce_status(remote_result) if backend == "worldquant" else "COMPLETED"

    record = {
        "evaluated_at": iso_now(),
        "evaluation_backend": backend,
        "evaluation_status": evaluation_status,
        "run_id": candidate.get("run_id"),
        "batch_id": candidate.get("batch_id"),
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "source_stages": candidate.get("source_stages", [candidate.get("source")]),
        "queue_rank": candidate.get("queue_rank"),
        "priority_score": candidate.get("priority_score"),
        "priority_score_before_recent_failures": candidate.get("priority_score_before_recent_failures"),
        "recent_failure_penalty": candidate.get("recent_failure_penalty"),
        "recent_failure_reasons": candidate.get("recent_failure_reasons", []),
        "recent_failure_match_count": candidate.get("recent_failure_match_count"),
        "thesis": candidate.get("thesis"),
        "thesis_id": candidate.get("thesis_id"),
        "why": candidate.get("why"),
        "expression": candidate.get("expression"),
        "compiled_expression": candidate.get("compiled_expression") or candidate.get("expression"),
        "normalized_expression": candidate.get("normalized_expression"),
        "normalized_compiled_expression": candidate.get("normalized_compiled_expression"),
        "expression_skeleton": candidate.get("expression_skeleton"),
        "candidate_signature": candidate.get("candidate_signature"),
        "structure_signature": candidate.get("structure_signature"),
        "expression_signature": candidate.get("expression_signature"),
        "skeleton_signature": candidate.get("skeleton_signature"),
        "dedupe_match_types": candidate.get("dedupe_match_types", []),
        "duplicate_candidate_count": candidate.get("duplicate_candidate_count"),
        "merged_candidate_signatures": candidate.get("merged_candidate_signatures", []),
        "lineage": ensure_candidate_lineage(
            candidate,
            stage_source=candidate.get("source"),
            source_detail=candidate.get("source"),
            default_hypothesis_id=candidate.get("thesis_id"),
            default_hypothesis_label=candidate.get("thesis"),
            default_generation_reason=candidate.get("why"),
        ),
        "token_program": candidate.get("token_program") or [],
        "seed_ready": candidate.get("seed_ready"),
        "qualified": candidate.get("qualified"),
        "quality_label": candidate.get("quality_label"),
        "quality_fail_reasons": candidate.get("quality_fail_reasons", []),
        "confidence_score": candidate.get("confidence_score"),
        "candidate_score": candidate.get("candidate_score"),
        "novelty_score": candidate.get("novelty_score"),
        "style_alignment_score": candidate.get("style_alignment_score"),
        "settings": candidate.get("settings"),
        "risk_tags": candidate.get("risk_tags", []),
        "local_metrics": local,
        "simulation_result": remote if remote_result else None,
        "alpha_id": top.get("alpha_id") or local.get("alpha_id"),
        "regular_code": top.get("regular_code") or candidate.get("expression"),
        "turnover": top.get("turnover"),
        "returns": top.get("returns"),
        "drawdown": top.get("drawdown"),
        "margin": top.get("margin"),
        "fitness": top.get("fitness"),
        "sharpe": top.get("sharpe"),
    }
    for check_name in CHECK_RESULT_COLUMNS:
        record[check_name] = top.get(check_name, local.get(check_name))
    record["lineage"] = attach_evaluation_stage(record.get("lineage"), record)
    return record


def _local_score_priority_key(item: tuple[int, dict]) -> tuple[float, float]:
    index, candidate = item
    queue_rank = candidate.get("queue_rank")
    try:
        queue_order = -float(queue_rank)
    except (TypeError, ValueError):
        queue_order = float(-index)
    return (
        queue_order,
        float(candidate.get("priority_score") or 0.0),
    )


def _build_skipped_record(candidate: dict, *, backend: str, reason: str) -> dict:
    record = _build_record(candidate, {}, backend=backend)
    record["evaluation_status"] = "SKIPPED_LOCAL_SCORE_LIMIT"
    record["local_scoring_skipped"] = True
    record["local_scoring_skip_reason"] = reason
    record["lineage"] = attach_evaluation_stage(
        record.get("lineage"),
        record,
        submit_ready=False,
        qualified=False,
        quality_label="watchlist",
        fail_reasons=[reason],
    )
    return record


def evaluate_queue(
    queue_payload: dict,
    *,
    backend: str,
    csv_path: str | None = None,
    timeout: int = 300,
    max_local_score_workers: int | None = None,
    min_parallel_local_scoring: int = 4,
    local_score_limit: int | None = None,
) -> list[dict]:
    candidates = [item for item in queue_payload.get("candidates", []) if isinstance(item, dict)]
    history_index = _history_index(csv_path)
    session = None
    evaluate_queue.last_local_scoring_profile = {}

    if backend == "worldquant":
        username, password = _resolve_worldquant_credentials()
        if not username or not password:
            raise RuntimeError(
                "USERNAME or PASSWORD are not available for WorldQuant simulation. "
                "Put them in .env or run a local-only mode such as `bash run_wsl.sh internal` "
                "or `bash run_wsl.sh turbo --scoring internal`."
            )
        session, response = create_authenticated_session(username, password, context="simulate_batch")
        if session is None:
            status_code = getattr(response, "status_code", "n/a")
            detail = ""
            if response is not None:
                try:
                    detail = (response.text or "").strip()
                except Exception:
                    detail = ""
            hint = (
                "Run `bash run_wsl.sh auth` to verify credentials, "
                "or use `bash run_wsl.sh turbo --scoring internal` for local-only runs."
            )
            if status_code == 401:
                raise RuntimeError(
                    "WorldQuant authentication failed in simulate_batch (status=401, INVALID_CREDENTIALS). "
                    "Verify USERNAME/PASSWORD in .env. " + hint
                )
            if status_code == 403:
                raise RuntimeError(
                    "WorldQuant authenticated request was forbidden in simulate_batch (status=403). "
                    "The account may not have access to this endpoint or the session was rejected. "
                    f"{hint} Response={detail or 'n/a'}"
                )
            raise RuntimeError(
                f"WorldQuant authentication failed in simulate_batch (status={status_code}). "
                f"{hint} Response={detail or 'n/a'}"
            )

    indexed_candidates = [(index, candidate) for index, candidate in enumerate(candidates) if candidate.get("expression") or candidate.get("compiled_expression")]
    scored_candidates = list(indexed_candidates)
    skipped_candidate_indices = set()
    limit_applied = False
    if backend == "internal" and local_score_limit is not None and local_score_limit > 0 and len(indexed_candidates) > local_score_limit:
        ranked_candidates = sorted(indexed_candidates, key=_local_score_priority_key, reverse=True)
        scored_candidates = ranked_candidates[:local_score_limit]
        skipped_candidate_indices = {index for index, _ in indexed_candidates} - {index for index, _ in scored_candidates}
        limit_applied = True

    expressions = [(candidate.get("expression") or candidate.get("compiled_expression") or "") for _, candidate in scored_candidates]
    settings_list = [candidate.get("settings") for _, candidate in scored_candidates]
    local_metrics_list, local_scoring_profile = score_expressions_batch(
        expressions,
        history_index=history_index,
        settings_list=settings_list,
        max_workers=max_local_score_workers,
        min_parallel_tasks=min_parallel_local_scoring,
        surrogate_csv_path=csv_path,
        return_profile=True,
    )

    save_started_at = perf_counter()
    local_metrics_by_index = {}
    for (index, _candidate), local_metrics in zip(scored_candidates, local_metrics_list):
        local_metrics_by_index[index] = local_metrics
        if backend == "internal":
            save_alpha_to_csv(local_metrics)
    save_csv_seconds = perf_counter() - save_started_at

    local_scoring_profile = {
        **local_scoring_profile,
        "backend": backend,
        "candidate_count": len(candidates),
        "scored_candidates": len(scored_candidates),
        "skipped_candidates": len(skipped_candidate_indices),
        "max_local_score_workers": max_local_score_workers if max_local_score_workers is not None else 0,
        "min_parallel_local_scoring": min_parallel_local_scoring,
        "local_score_limit": int(local_score_limit) if local_score_limit is not None else 0,
        "limit_applied": limit_applied,
        "save_csv_seconds": round(save_csv_seconds, 4),
    }
    evaluate_queue.last_local_scoring_profile = local_scoring_profile

    records = []
    for index, candidate in indexed_candidates:
        if index in skipped_candidate_indices:
            records.append(_build_skipped_record(candidate, backend=backend, reason="local_score_limit"))
            continue

        local_metrics = local_metrics_by_index.get(index, {})
        expression = candidate.get("expression") or candidate.get("compiled_expression")
        if not expression:
            continue

        if backend == "internal":
            record = _build_record(candidate, local_metrics, backend=backend)
        else:
            remote_result = simulate(
                session,
                expression,
                timeout=timeout,
                settings=_settings_context(candidate.get("settings")),
            )
            record = _build_record(candidate, local_metrics, backend=backend, remote_result=remote_result)
        records.append(record)

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a merged candidate queue.")
    parser.add_argument("--input", required=True, help="Pending simulation queue JSON.")
    parser.add_argument("--output", required=True, help="Simulation results JSONL output path.")
    parser.add_argument("--backend", choices=("internal", "worldquant"), default="internal", help="Evaluation backend.")
    parser.add_argument("--csv", help="Optional CSV history path for local scoring context.")
    parser.add_argument("--timeout", type=int, default=300, help="WorldQuant simulation timeout in seconds.")
    parser.add_argument(
        "--max-local-score-workers",
        "--local-score-workers",
        dest="max_local_score_workers",
        type=int,
        help="Optional worker cap for local scoring. Defaults to auto.",
    )
    parser.add_argument("--min-parallel-local-scoring", type=int, default=4, help="Minimum candidate count before local scoring uses multiprocessing.")
    parser.add_argument("--local-score-limit", type=int, help="Optional cap on how many candidates are locally scored when backend=internal.")
    args = parser.parse_args()

    queue_payload = load_json(Path(args.input), default={})
    records = evaluate_queue(
        queue_payload,
        backend=args.backend,
        csv_path=args.csv,
        timeout=args.timeout,
        max_local_score_workers=args.max_local_score_workers,
        min_parallel_local_scoring=args.min_parallel_local_scoring,
        local_score_limit=args.local_score_limit,
    )
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "records": len(records),
                "output": args.output,
                "backend": args.backend,
                "local_scoring_profile": getattr(evaluate_queue, "last_local_scoring_profile", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
