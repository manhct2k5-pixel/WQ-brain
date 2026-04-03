#!/usr/bin/env python3
"""Run the orchestrated, run-scoped alpha pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.alpha_feed import render_alpha_feed
from scripts.build_evaluated_pool import build_evaluated_pool
from scripts.cleanup_artifacts import cleanup_artifacts
from scripts.daily_best import render_daily_best
from scripts.fix_alpha import (
    build_actionable_auto_fix_candidates,
    build_auto_fix_payload,
    load_auto_fix_store,
    merge_auto_fix_candidates,
)
from scripts.flow_utils import (
    ARTIFACTS_DIR,
    LATEST_DIR,
    STATE_DIR,
    atomic_write_json,
    atomic_write_text,
    copy_file,
    ensure_runtime_layout,
    iso_now,
    latest_metadata_path,
    latest_publish_is_complete,
    load_json,
    make_run_id,
    quarantine_payload,
    read_jsonl,
    sync_to_paths,
)
from scripts.merge_candidate_pool import merge_candidate_pool
from scripts.plan_next_batch import (
    HistoryIndex,
    apply_adaptive_planning_controls,
    build_batch,
    build_memory,
    build_seed_context,
    load_memory,
    load_seed_store,
    merge_memory,
    render_markdown as render_plan_markdown,
)
from scripts.results_digest import (
    build_summary as build_results_summary,
    discover_csv,
    read_result_rows,
    read_rows,
    render_markdown as render_results_markdown,
)
from scripts.simulate_batch import evaluate_queue
from scripts.update_research_memory import update_research_memory
from src.internal_scoring import CHECK_COLUMNS
from src.seed_store import load_seed_store as load_canonical_seed_store
from src.submit_gate import SUBMIT_READY_MIN_FITNESS, SUBMIT_READY_MIN_SHARPE

PROFILE_DEFAULTS = {
    "light": {"count": 8, "queue_limit": 10, "local_score_limit": None, "local_score_workers": 0, "min_parallel_local_scoring": 4},
    "cycle": {"count": 8, "queue_limit": 10, "local_score_limit": None, "local_score_workers": 0, "min_parallel_local_scoring": 4},
    "full": {"count": 12, "queue_limit": 14, "local_score_limit": 12, "local_score_workers": 0, "min_parallel_local_scoring": 4},
    "turbo": {"count": 16, "queue_limit": 18, "local_score_limit": 14, "local_score_workers": 0, "min_parallel_local_scoring": 4},
    "smart": {"count": 10, "queue_limit": 12, "local_score_limit": None, "local_score_workers": 0, "min_parallel_local_scoring": 4},
    "careful": {"count": 6, "queue_limit": 8, "local_score_limit": None, "local_score_workers": 0, "min_parallel_local_scoring": 4},
}

GLOBAL_MEMORY_PATH = STATE_DIR / "global_research_memory.json"
LEGACY_MEMORY_PATH = ARTIFACTS_DIR / "bo_nho_nghien_cuu.json"
LEGACY_AUTO_FIX_PATH = ARTIFACTS_DIR / "auto_fix_candidates.json"
LEGACY_SCOUT_PATH = ARTIFACTS_DIR / "_trinh_sat" / "du_lieu.json"
LATEST_EVALUATED_PATH = LATEST_DIR / "evaluated_candidates.json"
CHECKPOINT_FILE_NAME = "orchestrator_checkpoint.json"
CURRENT_MEMORY_SNAPSHOT_FILE_NAME = "current_memory_snapshot.json"
PREVIOUS_MEMORY_SNAPSHOT_FILE_NAME = "previous_memory_snapshot.json"
STAGE_ORDER = ("planned", "merged", "evaluated", "memory_updated", "published")

DEFAULT_RESOURCE_GUARD_CONFIG = {
    "recent_days": 7,
    "archive_delete_days": 30,
    "important_archive_delete_days": 90,
    "compress_min_bytes": 64 * 1024,
    "max_recent_runs": 24,
    "temp_file_max_age_hours": 12,
}
MANUAL_EXPLORE_DEFAULTS = {
    "exploration_boost": 0.18,
    "exploration_weight_multiplier": 1.6,
    "thesis_limit_bonus": 1,
    "batch_size_bonus": 2,
    "queue_limit_bonus": 2,
}
AUTO_FIX_SYNTHESIS_CONTEXT_LIMIT = 2
AUTO_FIX_SYNTHESIS_TOP_REWRITES = 4


def _profile_config(
    profile: str,
    *,
    count: int | None,
    queue_limit: int | None,
    local_score_limit: int | None = None,
    local_score_workers: int | None = None,
    min_parallel_local_scoring: int | None = None,
) -> dict:
    config = dict(PROFILE_DEFAULTS.get(profile, PROFILE_DEFAULTS["light"]))
    if count is not None:
        config["count"] = count
    if queue_limit is not None:
        config["queue_limit"] = queue_limit
    if local_score_limit is not None:
        config["local_score_limit"] = local_score_limit
    if local_score_workers is not None:
        config["local_score_workers"] = local_score_workers
    if min_parallel_local_scoring is not None:
        config["min_parallel_local_scoring"] = min_parallel_local_scoring
    return config


def _resource_guard_config(args) -> dict:
    config = dict(DEFAULT_RESOURCE_GUARD_CONFIG)
    for key in tuple(config):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    return {
        "recent_days": max(0, int(config["recent_days"])),
        "archive_delete_days": max(0, int(config["archive_delete_days"])),
        "important_archive_delete_days": max(0, int(config["important_archive_delete_days"])),
        "compress_min_bytes": max(0, int(config["compress_min_bytes"])),
        "max_recent_runs": max(0, int(config["max_recent_runs"])),
        "temp_file_max_age_hours": max(0, int(config["temp_file_max_age_hours"])),
    }


def _apply_artifact_resource_guard(summary: dict, args) -> dict:
    base_summary = dict(summary or {})
    try:
        cleanup_summary = cleanup_artifacts(
            artifacts_root=ARTIFACTS_DIR,
            protected_run_ids={str(base_summary.get("run_id") or "")} if base_summary.get("run_id") else None,
            **_resource_guard_config(args),
        )
    except Exception as exc:
        base_summary["artifact_resource_guard"] = {
            "status": "error",
            "error": str(exc),
        }
        return base_summary

    footprint = cleanup_summary.get("artifact_footprint", {}) if isinstance(cleanup_summary, dict) else {}
    base_summary["artifact_resource_guard"] = {
        "status": "ok",
        "status_path": str(ARTIFACTS_DIR / "state" / "artifact_cleanup_status.json"),
        "recent_runs_kept": len(cleanup_summary.get("kept_recent_runs", [])),
        "runs_archived": len(cleanup_summary.get("archived_runs", [])),
        "archives_deleted": len(cleanup_summary.get("deleted_archives", [])),
        "temp_files_removed": len(cleanup_summary.get("temp_files_removed", [])),
        "compressed_files": len(cleanup_summary.get("compressed_files", [])),
        "limits": _resource_guard_config(args),
        "artifact_footprint": footprint,
    }
    return base_summary


def _extract_scout_candidates(payload: dict) -> list[dict]:
    candidates = []
    for key in ("selected", "watchlist", "candidates"):
        items = payload.get(key)
        if isinstance(items, list):
            candidates.extend(item for item in items if isinstance(item, dict))
    return candidates


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _auto_fix_failure_names(candidate: dict) -> list[str]:
    local = candidate.get("local_metrics") or {}
    failures = [
        name
        for name in CHECK_COLUMNS
        if str(local.get(name) or "").strip().upper() == "FAIL"
    ]
    if failures:
        return failures

    risk_tags = set(candidate.get("risk_tags", []))
    sharpe_value = local.get("sharpe")
    fitness_value = local.get("fitness")
    turnover_value = local.get("turnover")

    if sharpe_value not in {None, ""} and _to_float(sharpe_value) < SUBMIT_READY_MIN_SHARPE:
        failures.append("LOW_SHARPE")
    if fitness_value not in {None, ""} and _to_float(fitness_value) < SUBMIT_READY_MIN_FITNESS:
        failures.append("LOW_FITNESS")
    if turnover_value not in {None, ""}:
        turnover = _to_float(turnover_value)
        if turnover > 0.60:
            failures.append("HIGH_TURNOVER")
        elif turnover < 0.08:
            failures.append("LOW_TURNOVER")
    if "weight_risk" in risk_tags:
        failures.append("CONCENTRATED_WEIGHT")
    return _unique_strings(failures)


def _auto_fix_context_sort_key(candidate: dict) -> tuple:
    local = candidate.get("local_metrics") or {}
    verdict_rank = {
        "PASS": 4,
        "LIKELY_PASS": 3,
        "BORDERLINE": 2,
        "FAIL": 1,
    }.get(str(local.get("verdict") or "").upper(), 0)
    return (
        int(bool(candidate.get("seed_ready"))),
        verdict_rank,
        _to_float(candidate.get("confidence_score")),
        _to_float(local.get("alpha_score")),
        _to_float(local.get("fitness")),
        _to_float(local.get("sharpe")),
        -len(candidate.get("quality_fail_reasons") or []),
    )


def _candidate_auto_fix_context(candidate: dict, *, csv_path: str | None) -> dict | None:
    expression = str(candidate.get("compiled_expression") or candidate.get("expression") or "").strip()
    family = str(candidate.get("thesis_id") or candidate.get("family_id") or "").strip()
    local = candidate.get("local_metrics") or {}
    if not expression or not family or candidate.get("qualified") or not local:
        return None

    settings = candidate.get("settings") or (local.get("settings") or {})
    style_tags = local.get("style_tags") or []
    return {
        "alpha_id": candidate.get("candidate_id") or local.get("alpha_id"),
        "expression": expression,
        "failures": _auto_fix_failure_names(candidate),
        "family": family,
        "style_tags": {str(item) for item in style_tags if str(item).strip()},
        "sharpe": _to_float(local.get("sharpe")),
        "fitness": _to_float(local.get("fitness")),
        "turnover": _to_float(local.get("turnover")),
        "settings": settings,
        "resolved_csv": csv_path,
    }


def _synthesize_auto_fix_candidates(
    batch: dict,
    *,
    csv_path: str | None,
    existing_store: dict | None = None,
    context_limit: int = AUTO_FIX_SYNTHESIS_CONTEXT_LIMIT,
) -> tuple[dict, dict]:
    base_store = dict(existing_store) if isinstance(existing_store, dict) else {"generated_at": "", "candidates": []}
    base_candidates = [item for item in (base_store.get("candidates") or []) if isinstance(item, dict)]
    ranked_contexts = []
    seen_expressions = set()

    for raw_candidate in batch.get("candidates", []) if isinstance(batch, dict) else []:
        if not isinstance(raw_candidate, dict):
            continue
        context = _candidate_auto_fix_context(raw_candidate, csv_path=csv_path)
        if context is None:
            continue
        expression = context["expression"]
        if expression in seen_expressions:
            continue
        seen_expressions.add(expression)
        ranked_contexts.append((raw_candidate, context))

    ranked_contexts.sort(key=lambda item: _auto_fix_context_sort_key(item[0]), reverse=True)

    actionable = []
    processed_expressions = []
    for candidate, context in ranked_contexts[: max(0, context_limit)]:
        try:
            auto_fix_payload = build_auto_fix_payload(
                context,
                csv_path=context.get("resolved_csv"),
                settings=context.get("settings"),
                top_rewrites=AUTO_FIX_SYNTHESIS_TOP_REWRITES,
            )
        except Exception as exc:
            _warn_json_issue(
                "Skipping auto-fix synthesis for "
                f"{candidate.get('expression') or candidate.get('compiled_expression')} because {exc}"
            )
            continue
        actionable.extend(build_actionable_auto_fix_candidates(context, auto_fix_payload))
        processed_expressions.append(context["expression"])

    if actionable:
        merged_store = merge_auto_fix_candidates({**base_store, "candidates": base_candidates}, actionable)
    else:
        merged_store = {
            **base_store,
            "generated_at": str(base_store.get("generated_at") or iso_now()),
            "candidates": base_candidates,
        }

    generation_summary = {
        "context_count": len(processed_expressions),
        "generated_candidate_count": len(actionable),
        "available_candidate_count": len(merged_store.get("candidates", [])),
        "context_expressions": processed_expressions,
    }
    return merged_store, generation_summary


def _results_summary_payload(results_path: Path, *, top: int) -> tuple[dict, str]:
    rows = read_result_rows(str(results_path))
    payload = build_results_summary(rows, top)
    markdown = render_results_markdown(payload, str(results_path))
    return payload, markdown


def _warn_json_issue(message: str) -> None:
    print(f"[orchestrator] Warning: {message}", file=sys.stderr)


def _quarantine_candidate_issue(record: dict, *, run_id: str, batch_id: str) -> None:
    quarantine_payload(
        {
            "run_id": run_id,
            "batch_id": batch_id,
            **record,
        },
        reason=str(record.get("reason") or "invalid_candidate"),
        category="candidates",
        label=str(record.get("source") or "candidate"),
        source_path=ARTIFACTS_DIR / "pending_simulation_queue.json",
        root=ARTIFACTS_DIR / "quarantine",
        warn=_warn_json_issue,
    )


def _stable_hash(*parts) -> str:
    joined = "||".join(str(part or "") for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def _stage_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "checkpoint": run_dir / CHECKPOINT_FILE_NAME,
        "current_snapshot": run_dir / CURRENT_MEMORY_SNAPSHOT_FILE_NAME,
        "previous_snapshot": run_dir / PREVIOUS_MEMORY_SNAPSHOT_FILE_NAME,
        "run_memory": run_dir / "run_memory.json",
        "planned_candidates": run_dir / "planned_candidates.json",
        "planned_markdown": run_dir / "planned_candidates.md",
        "auto_fix_candidates": run_dir / "auto_fix_candidates.json",
        "scout_candidates": run_dir / "scout_candidates.json",
        "pending_queue": run_dir / "pending_simulation_queue.json",
        "simulation_results": run_dir / "simulation_results.jsonl",
        "results_summary": run_dir / "results_summary.json",
        "results_markdown": run_dir / "results_summary.md",
        "evaluated_candidates": run_dir / "evaluated_candidates.json",
        "daily_report": run_dir / "alpha_tot_nhat_hom_nay.md",
        "feed_report": run_dir / "bang_tin_alpha.md",
        "orchestrator_summary": run_dir / "orchestrator_summary.json",
    }


def _load_checkpoint(path: Path, *, run_id: str, batch_id: str) -> dict:
    payload = load_json(path, default={})
    if payload.get("run_id") != run_id:
        payload = {}
    checkpoint = {
        "run_id": run_id,
        "batch_id": payload.get("batch_id") or batch_id,
        "updated_at": payload.get("updated_at"),
        "stages": payload.get("stages", {}) if isinstance(payload.get("stages"), dict) else {},
    }
    for stage_name in STAGE_ORDER:
        checkpoint["stages"].setdefault(stage_name, {"state": "pending"})
    return checkpoint


def _write_checkpoint(path: Path, checkpoint: dict) -> None:
    payload = {
        "run_id": checkpoint.get("run_id"),
        "batch_id": checkpoint.get("batch_id"),
        "updated_at": iso_now(),
        "stages": checkpoint.get("stages", {}),
    }
    atomic_write_json(path, payload)


def _mark_stage_done(checkpoint: dict, stage: str, *, details: dict | None = None) -> None:
    details = details or {}
    stage_entry = {
        "state": "done",
        "updated_at": iso_now(),
    }
    if details:
        stage_entry["details"] = details
    checkpoint.setdefault("stages", {})[stage] = stage_entry


def _mark_stages_pending_after(checkpoint: dict, stage: str) -> None:
    try:
        start_index = STAGE_ORDER.index(stage) + 1
    except ValueError:
        return
    for stage_name in STAGE_ORDER[start_index:]:
        checkpoint.setdefault("stages", {})[stage_name] = {"state": "pending"}


def _stage_done(checkpoint: dict, stage: str, *artifacts: Path) -> bool:
    if checkpoint.get("stages", {}).get(stage, {}).get("state") != "done":
        return False
    return all(Path(path).exists() for path in artifacts)


def _batch_id_for_run(run_id: str, profile: str) -> str:
    return f"batch_{_stable_hash(run_id, profile)}"


def _candidate_identity(candidate: dict, *, run_id: str, batch_id: str, default_source: str | None = None) -> dict:
    source = str(candidate.get("source") or default_source or "candidate")
    expression = str(candidate.get("compiled_expression") or candidate.get("expression") or "").strip()
    settings = str(candidate.get("settings") or "").strip()
    thesis_id = str(candidate.get("thesis_id") or candidate.get("thesis") or "").strip()
    candidate_id = candidate.get("candidate_id") or f"cand_{_stable_hash(run_id, batch_id, source, thesis_id, expression, settings)}"
    return {
        **candidate,
        "run_id": candidate.get("run_id") or run_id,
        "batch_id": candidate.get("batch_id") or batch_id,
        "candidate_id": str(candidate_id),
    }


def _stamp_candidates(candidates: list[dict], *, run_id: str, batch_id: str, default_source: str | None = None) -> list[dict]:
    stamped = []
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        stamped.append(_candidate_identity(item, run_id=run_id, batch_id=batch_id, default_source=default_source))
    return stamped


def _summary_with_checkpoint(summary: dict, *, checkpoint: dict, resumed_stages: list[str], executed_stages: list[str]) -> dict:
    return {
        **summary,
        "batch_id": checkpoint.get("batch_id"),
        "checkpoint_path": str(checkpoint.get("checkpoint_path")) if checkpoint.get("checkpoint_path") else None,
        "stage_statuses": {
            stage_name: (checkpoint.get("stages", {}).get(stage_name, {}) or {}).get("state", "pending")
            for stage_name in STAGE_ORDER
        },
        "resumed_stages": list(resumed_stages),
        "executed_stages": list(executed_stages),
        "resumed_from_checkpoint": bool(resumed_stages),
    }


def _payload_matches_run(payload: dict, *, run_id: str, batch_id: str) -> bool:
    if not isinstance(payload, dict):
        return False
    if str(payload.get("run_id") or "") != run_id:
        return False
    payload_batch_id = str(payload.get("batch_id") or "")
    return not payload_batch_id or payload_batch_id == batch_id


def _latest_artifact_paths() -> dict[str, Path]:
    return {
        "daily_report": LATEST_DIR / "alpha_tot_nhat_hom_nay.md",
        "feed_report": LATEST_DIR / "bang_tin_alpha.md",
        "evaluated_candidates": LATEST_DIR / "evaluated_candidates.json",
        "results_summary": LATEST_DIR / "results_summary.json",
        "planned_candidates": LATEST_DIR / "planned_candidates.json",
        "orchestrator_summary": LATEST_DIR / "orchestrator_summary.json",
    }


def _latest_publish_metadata_payload(
    *,
    run_id: str,
    batch_id: str,
    profile: str,
    scoring: str,
    run_dir: Path,
    run_summary_path: Path,
    latest_artifacts: dict[str, Path],
    publish_started_at: str,
    status: str,
    published_at: str | None = None,
) -> dict:
    return {
        "status": status,
        "complete": status == "complete",
        "run_id": run_id,
        "batch_id": batch_id,
        "profile": profile,
        "scoring": scoring,
        "publish_started_at": publish_started_at,
        "published_at": published_at or "",
        "updated_at": iso_now(),
        "source_run_dir": str(run_dir),
        "source_orchestrator_summary": str(run_summary_path),
        "artifacts": {name: str(path) for name, path in latest_artifacts.items()},
    }


def _latest_publish_complete_for_run(*, run_id: str, batch_id: str) -> bool:
    return latest_publish_is_complete(
        latest_dir=LATEST_DIR,
        run_id=run_id,
        batch_id=batch_id,
        required_artifact_names=tuple(_latest_artifact_paths()),
    )


def _unique_strings(values) -> list[str]:
    seen = set()
    ordered = []
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _manual_override_state(args) -> dict:
    payload = getattr(args, "manual_overrides", None)
    raw = dict(payload) if isinstance(payload, dict) else {}

    def _flag(name: str) -> bool:
        if name in raw:
            return bool(raw.get(name))
        return bool(getattr(args, f"manual_{name}", False))

    def _int(name: str, default: int) -> int:
        if name in raw:
            value = raw.get(name)
        else:
            value = getattr(args, f"manual_{name}", default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default

    only_fix = _flag("only_fix")
    disable_scout = _flag("disable_scout") or only_fix
    increase_explore = _flag("increase_explore")
    freeze_memory_update = _flag("freeze_memory_update")
    ignore_block_list = _flag("ignore_block_list")
    allow_exploratory_queue = _flag("allow_exploratory_queue")
    exploratory_queue_limit = _int("exploratory_queue_limit", 2)

    notes = []
    if only_fix:
        notes.append("Queue restricted to auto-fix candidates only.")
    elif disable_scout:
        notes.append("Scout candidates disabled for this run.")
    if increase_explore:
        notes.append("Planner exploration increased for this run.")
    if freeze_memory_update:
        notes.append("Global memory update frozen for this run.")
    if ignore_block_list:
        notes.append("Planner block lists ignored temporarily for this run.")
    if allow_exploratory_queue:
        notes.append(f"Allowing up to {exploratory_queue_limit} exploratory fallback candidate(s) when the strict WorldQuant queue is empty.")

    return {
        "active": any((only_fix, disable_scout, increase_explore, freeze_memory_update, ignore_block_list, allow_exploratory_queue)),
        "only_fix": only_fix,
        "disable_scout": disable_scout,
        "increase_explore": increase_explore,
        "freeze_memory_update": freeze_memory_update,
        "ignore_block_list": ignore_block_list,
        "allow_exploratory_queue": allow_exploratory_queue,
        "exploratory_queue_limit": exploratory_queue_limit,
        "notes": notes,
    }


def _merge_adaptive_controls(base_controls: dict | None, manual_overrides: dict) -> dict:
    merged = dict(base_controls) if isinstance(base_controls, dict) else {}
    if not manual_overrides.get("active"):
        return merged

    warnings = _unique_strings(
        [
            merged.get("warning"),
            "Manual override increased planner exploration." if manual_overrides.get("increase_explore") else "",
            "Manual override ignored planner block lists for this run." if manual_overrides.get("ignore_block_list") else "",
        ]
    )
    if warnings:
        merged["warning"] = " ".join(warnings)

    reason_codes = _unique_strings(list(merged.get("reason_codes", []) or []))
    if manual_overrides.get("increase_explore"):
        reason_codes.append("manual_increase_explore")
        for key, value in MANUAL_EXPLORE_DEFAULTS.items():
            current = merged.get(key)
            if isinstance(value, float):
                try:
                    merged[key] = max(float(current or 0.0), value)
                except (TypeError, ValueError):
                    merged[key] = value
            else:
                try:
                    merged[key] = max(int(current or 0), value)
                except (TypeError, ValueError):
                    merged[key] = value
    if manual_overrides.get("ignore_block_list"):
        reason_codes.append("manual_ignore_block_list")
        merged["ignore_block_list"] = True

    if reason_codes:
        merged["reason_codes"] = _unique_strings(reason_codes)
    if merged:
        merged["active"] = True
        merged["mode"] = str(merged.get("mode") or "manual_override")
    return merged


def _merge_source_bonus_adjustments(base_adjustments: dict | None, manual_overrides: dict) -> dict:
    merged = dict(base_adjustments) if isinstance(base_adjustments, dict) else {}
    if manual_overrides.get("disable_scout"):
        merged.pop("scout", None)
    return merged


def _merge_source_quota_profile(base_profile: dict | None, manual_overrides: dict) -> dict:
    merged = dict(base_profile) if isinstance(base_profile, dict) else {}
    if manual_overrides.get("disable_scout"):
        merged["scout"] = 0.0
    if manual_overrides.get("only_fix"):
        merged["planner"] = 0.0
        merged["auto_fix_rewrite"] = 1.0
        merged["scout"] = 0.0
    return merged


def _exploratory_queue_config(*, scoring: str, manual_overrides: dict, adaptive_controls: dict | None = None) -> dict:
    controls = adaptive_controls if isinstance(adaptive_controls, dict) else {}
    manual_active = bool(manual_overrides.get("allow_exploratory_queue"))
    adaptive_active = bool(controls.get("allow_exploratory_queue"))
    configured_limits = []
    if manual_active:
        configured_limits.append(max(1, int(manual_overrides.get("exploratory_queue_limit") or 2)))
    try:
        adaptive_limit = max(0, int(controls.get("exploratory_queue_limit") or 0))
    except (TypeError, ValueError):
        adaptive_limit = 0
    if adaptive_active and adaptive_limit > 0:
        configured_limits.append(adaptive_limit)
    manual_supported = scoring in {"worldquant", "internal"} and manual_active
    adaptive_supported = scoring == "worldquant" and adaptive_active
    active = manual_supported or adaptive_supported
    return {
        "active": active,
        "limit": max(configured_limits) if configured_limits else max(1, int(manual_overrides.get("exploratory_queue_limit") or 2)),
        "backfill_below_count": max(0, int(controls.get("exploratory_queue_backfill_below_count") or 0)),
        "mode": (
            "manual+adaptive"
            if manual_active and adaptive_active
            else "manual"
            if manual_active
            else "adaptive"
            if adaptive_active
            else "disabled"
        ),
    }


def _memory_stage_done(checkpoint: dict, memory_path: Path) -> bool:
    stage_entry = (checkpoint.get("stages", {}).get("memory_updated", {}) or {})
    if stage_entry.get("state") != "done":
        return False
    details = stage_entry.get("details", {}) if isinstance(stage_entry.get("details"), dict) else {}
    if details.get("frozen"):
        return True
    return memory_path.exists()


def run_pipeline(args) -> dict:
    run_id = args.run_id or make_run_id(args.profile)
    run_dir = ensure_runtime_layout(run_id)
    paths = _stage_paths(run_dir)
    latest_paths = _latest_artifact_paths()
    latest_metadata = latest_metadata_path(LATEST_DIR)
    manual_overrides = _manual_override_state(args)
    retention_tag = str(getattr(args, "retention_tag", "") or "").strip().lower()
    if retention_tag in {"standard", "important", "keep"}:
        atomic_write_json(
            run_dir / "retention_tag.json",
            {
                "tag": retention_tag,
                "set_at": iso_now(),
                "source": "orchestrator_arg",
            },
        )
    checkpoint = _load_checkpoint(paths["checkpoint"], run_id=run_id, batch_id=_batch_id_for_run(run_id, args.profile))
    checkpoint["checkpoint_path"] = paths["checkpoint"]
    resumed_stages: list[str] = []
    executed_stages: list[str] = []

    published_summary = load_json(paths["orchestrator_summary"], default={}) if paths["orchestrator_summary"].exists() else {}
    published_stage_recovered = (
        _payload_matches_run(published_summary, run_id=run_id, batch_id=checkpoint["batch_id"])
        and paths["daily_report"].exists()
        and paths["feed_report"].exists()
        and _latest_publish_complete_for_run(run_id=run_id, batch_id=checkpoint["batch_id"])
    )
    if _stage_done(
        checkpoint,
        "published",
        paths["orchestrator_summary"],
        paths["daily_report"],
        paths["feed_report"],
        latest_metadata,
        *latest_paths.values(),
    ) or published_stage_recovered:
        summary = published_summary
        if published_stage_recovered and checkpoint.get("stages", {}).get("published", {}).get("state") != "done":
            for stage_name in STAGE_ORDER:
                _mark_stage_done(checkpoint, stage_name)
            _write_checkpoint(paths["checkpoint"], checkpoint)
        return _summary_with_checkpoint(
            summary,
            checkpoint=checkpoint,
            resumed_stages=list(STAGE_ORDER),
            executed_stages=[],
        )

    config = _profile_config(
        args.profile,
        count=args.count,
        queue_limit=args.queue_limit,
        local_score_limit=getattr(args, "local_score_limit", None),
        local_score_workers=getattr(args, "local_score_workers", None),
        min_parallel_local_scoring=getattr(args, "min_parallel_local_scoring", None),
    )

    csv_path = discover_csv(args.csv_path)
    rows = read_rows(csv_path)
    if not rows:
        raise RuntimeError("No rows found in simulation history CSV; planner cannot build a new batch.")

    previous_memory_path = Path(args.memory) if args.memory else (GLOBAL_MEMORY_PATH if GLOBAL_MEMORY_PATH.exists() else LEGACY_MEMORY_PATH)
    planning_memory = {}
    current_snapshot = {}
    previous_memory_raw = {}
    batch = {}
    planned_payload = load_json(paths["planned_candidates"], default={}) if paths["planned_candidates"].exists() else {}

    planned_stage_recovered = (
        _payload_matches_run(planned_payload, run_id=run_id, batch_id=checkpoint["batch_id"])
        and paths["run_memory"].exists()
        and paths["planned_markdown"].exists()
        and paths["current_snapshot"].exists()
        and paths["previous_snapshot"].exists()
    )
    if _stage_done(
        checkpoint,
        "planned",
        paths["run_memory"],
        paths["planned_candidates"],
        paths["planned_markdown"],
        paths["current_snapshot"],
        paths["previous_snapshot"],
    ) or planned_stage_recovered:
        planning_memory = load_json(paths["run_memory"], default={})
        batch = dict(planned_payload.get("batch", {})) if isinstance(planned_payload.get("batch"), dict) else {}
        batch["candidates"] = _stamp_candidates(
            batch.get("candidates", []),
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
            default_source="planner",
        )
        current_snapshot = load_json(paths["current_snapshot"], default={})
        previous_memory_raw = load_json(paths["previous_snapshot"], default={})
        if planned_stage_recovered and checkpoint.get("stages", {}).get("planned", {}).get("state") != "done":
            _mark_stage_done(
                checkpoint,
                "planned",
                details={
                    "candidate_count": len(batch.get("candidates", [])),
                    "qualified_count": batch.get("qualified_count", 0),
                },
            )
            _write_checkpoint(paths["checkpoint"], checkpoint)
        resumed_stages.append("planned")
    else:
        previous_memory_raw = load_json(previous_memory_path, default={}) if previous_memory_path.exists() else {}
        previous_memory = load_memory(str(previous_memory_path)) if previous_memory_path.exists() else {}
        atomic_write_json(paths["previous_snapshot"], previous_memory_raw)
        seed_context = build_seed_context(load_seed_store(args.seed_store))
        current_snapshot = build_memory(rows, args.top, history_window=args.history_window, seed_context=seed_context)
        atomic_write_json(paths["current_snapshot"], current_snapshot)
        memory = merge_memory(current_snapshot, previous_memory)
        planning_memory = apply_adaptive_planning_controls(
            memory,
            _merge_adaptive_controls(getattr(args, "adaptive_controls", None), manual_overrides),
        )
        adaptive_controls = planning_memory.get("adaptive_controls", {}) if isinstance(planning_memory.get("adaptive_controls"), dict) else {}
        planner_count = max(1, config["count"] + int(adaptive_controls.get("batch_size_bonus", 0) or 0))
        history_index = HistoryIndex.from_csv(csv_path)
        batch = build_batch(planning_memory, planner_count, include_local_metrics=True, history_index=history_index)
        batch["candidates"] = _stamp_candidates(
            batch.get("candidates", []),
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
            default_source="planner",
        )
        planned_payload = {
            "run_id": run_id,
            "batch_id": checkpoint["batch_id"],
            "memory": planning_memory,
            "batch": batch,
            "source_csv": csv_path,
        }
        atomic_write_json(paths["run_memory"], planning_memory)
        atomic_write_json(paths["planned_candidates"], planned_payload)
        atomic_write_text(paths["planned_markdown"], render_plan_markdown(batch))
        _mark_stages_pending_after(checkpoint, "planned")
        _mark_stage_done(
            checkpoint,
            "planned",
            details={
                "candidate_count": len(batch.get("candidates", [])),
                "qualified_count": batch.get("qualified_count", 0),
            },
        )
        _write_checkpoint(paths["checkpoint"], checkpoint)
        executed_stages.append("planned")

    adaptive_controls = planning_memory.get("adaptive_controls", {}) if isinstance(planning_memory.get("adaptive_controls"), dict) else {}
    queue_limit = max(1, config["queue_limit"] + int(adaptive_controls.get("queue_limit_bonus", 0) or 0))

    queue_payload = {}
    run_auto_fix_payload = load_json(paths["auto_fix_candidates"], default={}) if paths["auto_fix_candidates"].exists() else {}
    auto_fix_generation_summary = (
        dict(run_auto_fix_payload.get("orchestrator_generation", {}))
        if isinstance(run_auto_fix_payload.get("orchestrator_generation"), dict)
        else {
            "context_count": 0,
            "generated_candidate_count": 0,
            "available_candidate_count": len(run_auto_fix_payload.get("candidates", [])) if isinstance(run_auto_fix_payload.get("candidates"), list) else 0,
            "context_expressions": [],
        }
    )
    pending_queue_payload = load_json(paths["pending_queue"], default={}) if paths["pending_queue"].exists() else {}
    merged_stage_recovered = _payload_matches_run(pending_queue_payload, run_id=run_id, batch_id=checkpoint["batch_id"])
    if _stage_done(checkpoint, "merged", paths["pending_queue"]) or merged_stage_recovered:
        queue_payload = pending_queue_payload
        queue_payload["candidates"] = _stamp_candidates(
            queue_payload.get("candidates", []),
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
        )
        if merged_stage_recovered and checkpoint.get("stages", {}).get("merged", {}).get("state") != "done":
            _mark_stage_done(
                checkpoint,
                "merged",
                details={
                    "candidate_count": queue_payload.get("candidate_count", 0),
                    "source_counts": queue_payload.get("source_counts", {}),
                },
            )
            _write_checkpoint(paths["checkpoint"], checkpoint)
        resumed_stages.append("merged")
    else:
        auto_fix_candidates = []
        auto_fix_input = Path(args.auto_fix_input) if args.auto_fix_input else LEGACY_AUTO_FIX_PATH
        auto_fix_payload = load_auto_fix_store(auto_fix_input)
        if not manual_overrides.get("only_fix"):
            auto_fix_payload, auto_fix_generation_summary = _synthesize_auto_fix_candidates(
                batch,
                csv_path=csv_path,
                existing_store=auto_fix_payload,
            )
        else:
            auto_fix_generation_summary = {
                "context_count": 0,
                "generated_candidate_count": 0,
                "available_candidate_count": len(auto_fix_payload.get("candidates", [])),
                "context_expressions": [],
            }
        auto_fix_payload["run_id"] = run_id
        auto_fix_payload["batch_id"] = checkpoint["batch_id"]
        auto_fix_payload["orchestrator_generation"] = auto_fix_generation_summary
        atomic_write_json(paths["auto_fix_candidates"], auto_fix_payload)
        run_auto_fix_payload = auto_fix_payload
        auto_fix_candidates = _stamp_candidates(
            [item for item in auto_fix_payload.get("candidates", []) if isinstance(item, dict)],
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
            default_source="auto_fix_rewrite",
        )

        scout_candidates = []
        scout_input = Path(args.scout_input) if args.scout_input else LEGACY_SCOUT_PATH
        if not manual_overrides.get("disable_scout") and scout_input.exists():
            scout_payload = load_json(
                scout_input,
                default={},
                context=f"scout input {scout_input}",
                warn=_warn_json_issue,
            )
            atomic_write_json(paths["scout_candidates"], scout_payload)
            scout_candidates = _stamp_candidates(
                _extract_scout_candidates(scout_payload),
                run_id=run_id,
                batch_id=checkpoint["batch_id"],
                default_source="scout",
            )

        prior_evaluated_payload = {}
        if latest_publish_is_complete(latest_dir=LATEST_DIR, required_artifact_names=("evaluated_candidates",)):
            prior_evaluated_payload = (
                load_json(
                    LATEST_EVALUATED_PATH,
                    default={},
                    required_fields=("candidates",),
                    context=f"evaluated pool {LATEST_EVALUATED_PATH}",
                    warn=_warn_json_issue,
                )
                if LATEST_EVALUATED_PATH.exists()
                else {}
            )
        elif latest_metadata.exists():
            _warn_json_issue(
                f"Skipping prior evaluated pool from {LATEST_EVALUATED_PATH} because latest publish metadata is incomplete."
            )

        planner_candidates = [] if manual_overrides.get("only_fix") else batch.get("candidates", [])
        queue_payload = merge_candidate_pool(
            planner_candidates=planner_candidates,
            auto_fix_candidates=auto_fix_candidates,
            scout_candidates=scout_candidates,
            prior_evaluated_candidates=prior_evaluated_payload.get("candidates", []),
            limit=queue_limit,
            source_bonus_adjustments=_merge_source_bonus_adjustments(
                getattr(args, "source_bonus_adjustments", None),
                manual_overrides,
            ),
            source_quota_profile=_merge_source_quota_profile(
                getattr(args, "source_quota_profile", None),
                manual_overrides,
            ),
            exploratory_queue=_exploratory_queue_config(
                scoring=args.scoring,
                manual_overrides=manual_overrides,
                adaptive_controls=adaptive_controls,
            ),
            quarantine_callback=lambda record: _quarantine_candidate_issue(
                record,
                run_id=run_id,
                batch_id=checkpoint["batch_id"],
            ),
        )
        queue_payload["run_id"] = run_id
        queue_payload["batch_id"] = checkpoint["batch_id"]
        queue_payload["candidates"] = _stamp_candidates(
            queue_payload.get("candidates", []),
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
        )
        atomic_write_json(paths["pending_queue"], queue_payload)
        _mark_stages_pending_after(checkpoint, "merged")
        _mark_stage_done(
            checkpoint,
            "merged",
            details={
                "candidate_count": queue_payload.get("candidate_count", 0),
                "source_counts": queue_payload.get("source_counts", {}),
            },
        )
        _write_checkpoint(paths["checkpoint"], checkpoint)
        executed_stages.append("merged")

    simulation_records = []
    results_summary_payload = {}
    evaluated_payload = {}
    saved_results_summary = load_json(paths["results_summary"], default={}) if paths["results_summary"].exists() else {}
    saved_evaluated_payload = load_json(paths["evaluated_candidates"], default={}) if paths["evaluated_candidates"].exists() else {}
    evaluated_stage_recovered = (
        _payload_matches_run(saved_results_summary, run_id=run_id, batch_id=checkpoint["batch_id"])
        and _payload_matches_run(saved_evaluated_payload, run_id=run_id, batch_id=checkpoint["batch_id"])
        and paths["simulation_results"].exists()
        and paths["results_markdown"].exists()
    )
    if _stage_done(
        checkpoint,
        "evaluated",
        paths["simulation_results"],
        paths["results_summary"],
        paths["results_markdown"],
        paths["evaluated_candidates"],
    ) or evaluated_stage_recovered:
        simulation_records = read_jsonl(paths["simulation_results"])
        results_summary_payload = saved_results_summary
        evaluated_payload = saved_evaluated_payload
        if isinstance(evaluated_payload.get("candidates"), list):
            evaluated_payload["candidates"] = _stamp_candidates(
                evaluated_payload.get("candidates", []),
                run_id=run_id,
                batch_id=checkpoint["batch_id"],
            )
        if evaluated_stage_recovered and checkpoint.get("stages", {}).get("evaluated", {}).get("state") != "done":
            _mark_stage_done(
                checkpoint,
                "evaluated",
                details={
                    "candidate_count": evaluated_payload.get("summary", {}).get("candidate_count", 0),
                    "submit_ready_count": evaluated_payload.get("summary", {}).get("submit_ready_count", 0),
                },
            )
            _write_checkpoint(paths["checkpoint"], checkpoint)
        resumed_stages.append("evaluated")
    else:
        simulation_records = evaluate_queue(
            queue_payload,
            backend=args.scoring,
            csv_path=csv_path,
            timeout=args.timeout,
            max_local_score_workers=config.get("local_score_workers"),
            min_parallel_local_scoring=config.get("min_parallel_local_scoring", 4),
            local_score_limit=config.get("local_score_limit"),
        )
        local_scoring_profile = getattr(evaluate_queue, "last_local_scoring_profile", {})
        if not isinstance(local_scoring_profile, dict):
            local_scoring_profile = {}
        from scripts.flow_utils import write_jsonl

        write_jsonl(paths["simulation_results"], simulation_records)

        results_summary_payload, results_summary_markdown = _results_summary_payload(paths["simulation_results"], top=args.top)
        results_summary_payload = {
            **results_summary_payload,
            "run_id": run_id,
            "batch_id": checkpoint["batch_id"],
            "local_scoring_profile": local_scoring_profile,
        }
        atomic_write_json(paths["results_summary"], results_summary_payload)
        atomic_write_text(paths["results_markdown"], results_summary_markdown)

        evaluated_payload = build_evaluated_pool(
            read_jsonl(paths["simulation_results"]),
            memory=planning_memory,
            results_summary=results_summary_payload,
        )
        evaluated_payload["run_id"] = run_id
        evaluated_payload["batch_id"] = checkpoint["batch_id"]
        if isinstance(evaluated_payload.get("candidates"), list):
            evaluated_payload["candidates"] = _stamp_candidates(
                evaluated_payload.get("candidates", []),
                run_id=run_id,
                batch_id=checkpoint["batch_id"],
            )
        atomic_write_json(paths["evaluated_candidates"], evaluated_payload)
        _mark_stages_pending_after(checkpoint, "evaluated")
        _mark_stage_done(
            checkpoint,
            "evaluated",
            details={
                "candidate_count": evaluated_payload.get("summary", {}).get("candidate_count", 0),
                "submit_ready_count": evaluated_payload.get("summary", {}).get("submit_ready_count", 0),
            },
        )
        _write_checkpoint(paths["checkpoint"], checkpoint)
        executed_stages.append("evaluated")

    persisted_memory = load_json(GLOBAL_MEMORY_PATH, default={}) if GLOBAL_MEMORY_PATH.exists() else {}
    memory_stage_recovered = str((persisted_memory.get("_meta", {}) or {}).get("last_memory_update_run_id") or "") == run_id
    if _memory_stage_done(checkpoint, GLOBAL_MEMORY_PATH) or memory_stage_recovered:
        if memory_stage_recovered and checkpoint.get("stages", {}).get("memory_updated", {}).get("state") != "done":
            _mark_stage_done(
                checkpoint,
                "memory_updated",
                details={"memory_path": str(GLOBAL_MEMORY_PATH)},
            )
            _write_checkpoint(paths["checkpoint"], checkpoint)
        resumed_stages.append("memory_updated")
    else:
        if manual_overrides.get("freeze_memory_update"):
            _mark_stages_pending_after(checkpoint, "memory_updated")
            _mark_stage_done(
                checkpoint,
                "memory_updated",
                details={
                    "memory_path": str(GLOBAL_MEMORY_PATH),
                    "frozen": True,
                    "manual_override": True,
                },
            )
        else:
            updated_memory = update_research_memory(
                planning_memory,
                current_snapshot=load_json(paths["current_snapshot"], default={}),
                previous_memory=load_json(paths["previous_snapshot"], default={}),
                results_summary=results_summary_payload,
                simulation_records=simulation_records,
            )
            updated_memory.setdefault("_meta", {})
            updated_memory["_meta"]["last_memory_update_run_id"] = run_id
            updated_memory["_meta"]["last_memory_update_batch_id"] = checkpoint["batch_id"]
            atomic_write_json(GLOBAL_MEMORY_PATH, updated_memory)
            _mark_stages_pending_after(checkpoint, "memory_updated")
            _mark_stage_done(
                checkpoint,
                "memory_updated",
                details={"memory_path": str(GLOBAL_MEMORY_PATH)},
            )
        _write_checkpoint(paths["checkpoint"], checkpoint)
        executed_stages.append("memory_updated")

    published_summary = load_json(paths["orchestrator_summary"], default={}) if paths["orchestrator_summary"].exists() else {}
    published_stage_recovered = (
        _payload_matches_run(published_summary, run_id=run_id, batch_id=checkpoint["batch_id"])
        and paths["daily_report"].exists()
        and paths["feed_report"].exists()
        and _latest_publish_complete_for_run(run_id=run_id, batch_id=checkpoint["batch_id"])
    )
    if _stage_done(
        checkpoint,
        "published",
        paths["orchestrator_summary"],
        paths["daily_report"],
        paths["feed_report"],
        latest_metadata,
        *latest_paths.values(),
    ) or published_stage_recovered:
        if isinstance(published_summary, dict) and "global_memory_path" not in published_summary:
            published_summary["global_memory_path"] = str(GLOBAL_MEMORY_PATH)
        summary = _apply_artifact_resource_guard(published_summary, args)
        atomic_write_json(paths["orchestrator_summary"], summary)
        copy_file(paths["orchestrator_summary"], latest_paths["orchestrator_summary"])
        if published_stage_recovered and checkpoint.get("stages", {}).get("published", {}).get("state") != "done":
            _mark_stage_done(
                checkpoint,
                "published",
                details={
                    "daily_report": str(paths["daily_report"]),
                    "feed_report": str(paths["feed_report"]),
                    "latest_metadata": str(latest_metadata),
                },
            )
            _write_checkpoint(paths["checkpoint"], checkpoint)
        resumed_stages.append("published")
        return _summary_with_checkpoint(
            summary,
            checkpoint=checkpoint,
            resumed_stages=resumed_stages,
            executed_stages=executed_stages,
        )

    seed_store = load_canonical_seed_store(Path(args.seed_store))
    auto_fix_report_candidates = [
        item for item in (run_auto_fix_payload.get("candidates", []) if isinstance(run_auto_fix_payload, dict) else []) if isinstance(item, dict)
    ]
    daily_markdown = render_daily_best(
        evaluated_payload,
        seed_store=seed_store,
        top=args.daily_top,
        extra_candidates=auto_fix_report_candidates,
    )
    feed_markdown = render_alpha_feed(
        evaluated_payload,
        limit=args.feed_limit,
        extra_candidates=auto_fix_report_candidates,
    )
    atomic_write_text(paths["daily_report"], daily_markdown)
    atomic_write_text(paths["feed_report"], feed_markdown)

    publish_started_at = iso_now()
    atomic_write_json(
        latest_metadata,
        _latest_publish_metadata_payload(
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
            profile=args.profile,
            scoring=args.scoring,
            run_dir=run_dir,
            run_summary_path=paths["orchestrator_summary"],
            latest_artifacts=latest_paths,
            publish_started_at=publish_started_at,
            status="publishing",
        ),
    )

    copy_file(paths["daily_report"], latest_paths["daily_report"])
    copy_file(paths["feed_report"], latest_paths["feed_report"])
    copy_file(paths["evaluated_candidates"], latest_paths["evaluated_candidates"])
    copy_file(paths["results_summary"], latest_paths["results_summary"])
    copy_file(paths["planned_candidates"], latest_paths["planned_candidates"])

    published_at = iso_now()
    local_scoring_profile = (
        results_summary_payload.get("local_scoring_profile", {})
        if isinstance(results_summary_payload, dict)
        else {}
    )
    summary = {
        "generated_at": published_at,
        "state": "completed",
        "profile": args.profile,
        "scoring": args.scoring,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "batch_id": checkpoint["batch_id"],
        "global_memory_path": str(GLOBAL_MEMORY_PATH),
        "queue_candidates": queue_payload.get("candidate_count", 0),
        "queue_source_counts": queue_payload.get("source_counts", {}),
        "filtered_queue_candidates": queue_payload.get("filtered_counts", {}),
        "exploratory_queue_active": bool(queue_payload.get("exploratory_queue_active")),
        "exploratory_queue_mode": queue_payload.get("exploratory_queue_mode"),
        "exploratory_queue_used": bool(queue_payload.get("exploratory_queue_used")),
        "exploratory_candidate_count": queue_payload.get("exploratory_candidate_count", 0),
        "exploratory_selected_reasons": queue_payload.get("exploratory_selected_reasons", {}),
        "evaluated_candidates": evaluated_payload.get("summary", {}).get("candidate_count", 0),
        "submit_ready_candidates": evaluated_payload.get("summary", {}).get("submit_ready_count", 0),
        "auto_fix_candidates_available": len(auto_fix_report_candidates),
        "auto_fix_candidates_generated": auto_fix_generation_summary.get("generated_candidate_count", 0),
        "auto_fix_contexts_processed": auto_fix_generation_summary.get("context_count", 0),
        "local_scoring_profile": local_scoring_profile,
        "adaptive_controls": adaptive_controls,
        "manual_overrides": manual_overrides,
        "retention_tag": retention_tag or "standard",
        "daily_report": str(latest_paths["daily_report"]),
        "feed_report": str(latest_paths["feed_report"]),
        "latest_orchestrator_summary": str(latest_paths["orchestrator_summary"]),
        "latest_metadata": str(latest_metadata),
        "latest_publish_status": "complete",
        "published_at": published_at,
    }
    summary = _apply_artifact_resource_guard(summary, args)
    atomic_write_json(paths["orchestrator_summary"], summary)
    copy_file(paths["orchestrator_summary"], latest_paths["orchestrator_summary"])
    atomic_write_json(
        latest_metadata,
        _latest_publish_metadata_payload(
            run_id=run_id,
            batch_id=checkpoint["batch_id"],
            profile=args.profile,
            scoring=args.scoring,
            run_dir=run_dir,
            run_summary_path=paths["orchestrator_summary"],
            latest_artifacts=latest_paths,
            publish_started_at=publish_started_at,
            status="complete",
            published_at=published_at,
        ),
    )

    sync_to_paths(paths["daily_report"], ARTIFACTS_DIR / "alpha_tot_nhat_hom_nay.md")
    sync_to_paths(paths["feed_report"], ARTIFACTS_DIR / "bang_tin_alpha.md")
    sync_to_paths(paths["planned_candidates"], ARTIFACTS_DIR / "lo_tiep_theo.json")
    sync_to_paths(paths["planned_markdown"], ARTIFACTS_DIR / "lo_tiep_theo.md")
    sync_to_paths(paths["results_markdown"], ARTIFACTS_DIR / "tom_tat_moi_nhat.md")
    sync_to_paths(paths["auto_fix_candidates"], LEGACY_AUTO_FIX_PATH)
    if GLOBAL_MEMORY_PATH.exists():
        sync_to_paths(GLOBAL_MEMORY_PATH, LEGACY_MEMORY_PATH)
    sync_to_paths(paths["orchestrator_summary"], ARTIFACTS_DIR / "trang_thai_chay.json")
    _mark_stage_done(
        checkpoint,
        "published",
        details={
            "daily_report": str(paths["daily_report"]),
            "feed_report": str(paths["feed_report"]),
            "latest_metadata": str(latest_metadata),
        },
    )
    _write_checkpoint(paths["checkpoint"], checkpoint)
    executed_stages.append("published")
    return _summary_with_checkpoint(
        summary,
        checkpoint=checkpoint,
        resumed_stages=resumed_stages,
        executed_stages=executed_stages,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the orchestrated run-scoped alpha pipeline.")
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="light")
    parser.add_argument("--run-id", help="Optional explicit run id.")
    parser.add_argument("--csv-path", help="Optional simulation history CSV path.")
    parser.add_argument("--memory", help="Optional prior memory JSON path.")
    parser.add_argument("--seed-store", default="initial-population.pkl", help="Seed store path used for planner duplicate context.")
    parser.add_argument("--auto-fix-input", help="Optional auto-fix candidate JSON.")
    parser.add_argument("--scout-input", help="Optional scout candidate JSON.")
    parser.add_argument("--top", type=int, default=10, help="How many top rows to learn from and summarize.")
    parser.add_argument("--count", type=int, help="Override planned candidate count.")
    parser.add_argument("--queue-limit", type=int, help="Override merged queue size.")
    parser.add_argument("--history-window", type=int, default=120, help="Planner history window.")
    parser.add_argument("--scoring", choices=("internal", "worldquant"), default="internal", help="Simulation backend for queue evaluation.")
    parser.add_argument("--timeout", type=int, default=300, help="WorldQuant simulation timeout in seconds.")
    parser.add_argument("--local-score-limit", type=int, help="Optional cap on how many candidates receive local scoring when backend=internal.")
    parser.add_argument("--local-score-workers", type=int, help="Optional worker cap for local scoring. Defaults to auto when omitted or 0.")
    parser.add_argument("--min-parallel-local-scoring", type=int, help="Minimum candidate count before local scoring uses multiprocessing.")
    parser.add_argument("--manual-only-fix", action="store_true", help="Restrict the evaluation queue to auto-fix candidates only for this run.")
    parser.add_argument("--manual-disable-scout", action="store_true", help="Ignore scout candidates for this run.")
    parser.add_argument("--manual-increase-explore", action="store_true", help="Raise planner exploration heuristics for this run.")
    parser.add_argument("--manual-freeze-memory-update", action="store_true", help="Skip writing the global research memory for this run.")
    parser.add_argument("--manual-ignore-block-list", action="store_true", help="Temporarily ignore planner hard/soft block lists for this run.")
    parser.add_argument(
        "--manual-allow-exploratory-queue",
        action="store_true",
        help="When backend=worldquant and the strict queue is empty, allow a very small fallback queue of near-threshold watchlist candidates.",
    )
    parser.add_argument(
        "--manual-exploratory-queue-limit",
        type=int,
        default=2,
        help="Maximum number of fallback exploratory candidates to queue when --manual-allow-exploratory-queue is enabled.",
    )
    parser.add_argument("--daily-top", type=int, default=3, help="Maximum number of daily winners to render.")
    parser.add_argument("--feed-limit", type=int, default=12, help="Maximum number of feed candidates to render.")
    parser.add_argument(
        "--retention-tag",
        choices=("standard", "important", "keep"),
        help="Optional retention tag for this run. Important runs are kept longer in archive; keep runs are never auto-deleted.",
    )
    args = parser.parse_args()

    try:
        summary = run_pipeline(args)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
