#!/usr/bin/env python3
"""Persist a layered research memory snapshot for the next orchestrated run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import atomic_write_json, iso_now, load_json, read_jsonl
from scripts.plan_next_batch import flatten_memory_payload

MEMORY_HALF_LIFE_DAYS = 21.0
BLOCK_HALF_LIFE_DAYS = 14.0
MIN_DECAY_FACTOR = 0.08
WORKING_TOP_ROW_LIMIT = 8
WORKING_SUGGESTION_LIMIT = 4
WORKING_SKELETON_LIMIT = 120
SUMMARY_SKELETON_LIMIT = 320
ARCHIVE_RUN_LIMIT = 90
BLOCK_SCORE_PRUNE_THRESHOLD = 0.45
SOFT_BLOCK_SCORE = 1.35
HARD_BLOCK_SCORE = 3.0
ACTIVE_SOFT_BLOCK_THRESHOLD = 1.0
ACTIVE_HARD_BLOCK_THRESHOLD = 2.5


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _decay_factor(previous_timestamp: str | None, current_timestamp: str, *, half_life_days: float, floor: float = MIN_DECAY_FACTOR) -> float:
    previous_dt = _parse_iso(previous_timestamp)
    current_dt = _parse_iso(current_timestamp)
    if previous_dt is None or current_dt is None:
        return 1.0
    age_days = max(0.0, (current_dt - previous_dt).total_seconds() / 86400.0)
    if age_days <= 0:
        return 1.0
    decay = 0.5 ** (age_days / max(1.0, half_life_days))
    return max(float(floor), float(decay))


def _trim_strings(values: list[str] | None, *, limit: int) -> list[str]:
    items = sorted({str(value) for value in values or [] if value})
    return items[: max(0, limit)]


def _trim_top_rows(rows: list[dict] | None, *, limit: int) -> list[dict]:
    return [item for item in (rows or []) if isinstance(item, dict)][: max(0, limit)]


def _merge_failure_counts(current: dict, previous: dict, *, decay: float, limit: int = 14) -> dict:
    merged = Counter()
    for name, count in (previous or {}).items():
        try:
            scaled = float(count) * decay
        except (TypeError, ValueError):
            continue
        if scaled > 0:
            merged[str(name)] = scaled
    for name, count in (current or {}).items():
        try:
            merged[str(name)] += float(count)
        except (TypeError, ValueError):
            continue
    top_items = sorted(merged.items(), key=lambda item: (-item[1], item[0]))[: max(0, limit)]
    pruned = {}
    for name, value in top_items:
        rounded = int(round(value))
        if rounded > 0:
            pruned[name] = rounded
    return pruned


def _merge_style_leaders(current: list[dict], previous: list[dict], *, decay: float) -> list[dict]:
    merged = {}
    for item in previous or []:
        tag = str(item.get("tag") or "").strip()
        if not tag:
            continue
        merged[tag] = float(item.get("learning_score", 0.0) or 0.0) * decay
    for item in current or []:
        tag = str(item.get("tag") or "").strip()
        if not tag:
            continue
        merged[tag] = max(merged.get(tag, 0.0), float(item.get("learning_score", 0.0) or 0.0))
    ordered = sorted(
        ({"tag": tag, "learning_score": round(score, 4)} for tag, score in merged.items() if score > 0),
        key=lambda item: (item["learning_score"], item["tag"]),
        reverse=True,
    )
    return ordered[:24]


def _merge_family_stats(current: dict, previous: dict, *, decay: float) -> dict:
    merged = {}
    family_ids = sorted(set(current or {}) | set(previous or {}))
    for family_id in family_ids:
        current_stats = dict((current or {}).get(family_id, {}))
        previous_stats = dict((previous or {}).get(family_id, {}))
        attempts = int(current_stats.get("attempts", 0) or 0) + int(round(float(previous_stats.get("attempts", 0) or 0) * decay))
        completed = int(current_stats.get("completed", 0) or 0) + int(round(float(previous_stats.get("completed", 0) or 0) * decay))
        pass_all_count = int(current_stats.get("pass_all_count", 0) or 0) + int(round(float(previous_stats.get("pass_all_count", 0) or 0) * decay))
        serious_failures = int(current_stats.get("serious_failures", 0) or 0) + int(round(float(previous_stats.get("serious_failures", 0) or 0) * decay))
        current_attempts = max(0, int(current_stats.get("attempts", 0) or 0))
        previous_attempts = max(0.0, float(previous_stats.get("attempts", 0) or 0.0) * decay)
        total_attempt_weight = current_attempts + previous_attempts
        avg_research_score = 0.0
        avg_sharpe = 0.0
        avg_fitness = 0.0
        if total_attempt_weight > 0:
            avg_research_score = round(
                (
                    float(current_stats.get("avg_research_score", 0.0) or 0.0) * current_attempts
                    + float(previous_stats.get("avg_research_score", 0.0) or 0.0) * previous_attempts
                )
                / total_attempt_weight,
                4,
            )
            avg_sharpe = round(
                (
                    float(current_stats.get("avg_sharpe", 0.0) or 0.0) * current_attempts
                    + float(previous_stats.get("avg_sharpe", 0.0) or 0.0) * previous_attempts
                )
                / total_attempt_weight,
                4,
            )
            avg_fitness = round(
                (
                    float(current_stats.get("avg_fitness", 0.0) or 0.0) * current_attempts
                    + float(previous_stats.get("avg_fitness", 0.0) or 0.0) * previous_attempts
                )
                / total_attempt_weight,
                4,
            )
        merged[family_id] = {
            "attempts": attempts,
            "completed": completed,
            "pass_all_count": pass_all_count,
            "avg_research_score": avg_research_score,
            "avg_sharpe": avg_sharpe,
            "avg_fitness": avg_fitness,
            "serious_failures": serious_failures,
            "failure_counts": _merge_failure_counts(
                current_stats.get("failure_counts", {}),
                previous_stats.get("failure_counts", {}),
                decay=decay,
                limit=8,
            ),
        }
    return merged


def _merge_block_scores(
    current_items: list[str],
    previous_registry: dict,
    *,
    updated_at: str,
    score_value: float,
    decay_half_life_days: float = BLOCK_HALF_LIFE_DAYS,
) -> dict:
    merged = {}
    for key, payload in (previous_registry or {}).items():
        if not isinstance(payload, dict):
            continue
        decayed = float(payload.get("score", 0.0) or 0.0) * _decay_factor(
            payload.get("updated_at"),
            updated_at,
            half_life_days=decay_half_life_days,
            floor=0.0,
        )
        if decayed >= BLOCK_SCORE_PRUNE_THRESHOLD:
            merged[str(key)] = {
                "score": round(decayed, 4),
                "updated_at": payload.get("updated_at"),
                "level": payload.get("level", "soft"),
            }
    for key in current_items or []:
        item_key = str(key).strip()
        if not item_key:
            continue
        existing = merged.get(item_key, {})
        level = "hard" if score_value >= HARD_BLOCK_SCORE else "soft"
        merged[item_key] = {
            "score": round(max(float(existing.get("score", 0.0) or 0.0), float(score_value)), 4),
            "updated_at": updated_at,
            "level": "hard" if existing.get("level") == "hard" or level == "hard" else "soft",
        }
    return merged


def _registry_to_keys(registry: dict, *, minimum_score: float, require_hard: bool = False) -> list[str]:
    keys = []
    for key, payload in (registry or {}).items():
        if not isinstance(payload, dict):
            continue
        score = float(payload.get("score", 0.0) or 0.0)
        if score < minimum_score:
            continue
        if require_hard and payload.get("level") != "hard":
            continue
        keys.append(str(key))
    return sorted(set(keys))


def _build_working_memory(run_snapshot: dict, *, updated_at: str) -> dict:
    working = flatten_memory_payload(run_snapshot)
    working["_meta"] = {"updated_at": updated_at, "kind": "working"}
    working["top_rows"] = _trim_top_rows(working.get("top_rows", []), limit=WORKING_TOP_ROW_LIMIT)
    working["suggestions"] = [str(item) for item in (working.get("suggestions", []) or [])[:WORKING_SUGGESTION_LIMIT]]
    working["preferred_skeletons"] = _trim_strings(working.get("preferred_skeletons", []), limit=WORKING_SKELETON_LIMIT)
    working["historical_skeletons"] = _trim_strings(working.get("historical_skeletons", []), limit=WORKING_SKELETON_LIMIT)
    return working


def _build_summary_memory(current_snapshot: dict, previous_payload: dict, *, updated_at: str) -> dict:
    previous_summary = previous_payload.get("summary_memory", {}) if isinstance(previous_payload, dict) else {}
    previous_planner = flatten_memory_payload(previous_payload)
    decay = _decay_factor(
        previous_summary.get("_meta", {}).get("updated_at") or previous_payload.get("_meta", {}).get("updated_at"),
        updated_at,
        half_life_days=MEMORY_HALF_LIFE_DAYS,
    )
    current = flatten_memory_payload(current_snapshot)
    previous = flatten_memory_payload(previous_planner)
    summary = {
        "_meta": {"updated_at": updated_at, "kind": "summary", "decay_factor": round(decay, 4)},
        "failure_counts": _merge_failure_counts(current.get("failure_counts", {}), previous.get("failure_counts", {}), decay=decay),
        "family_stats": _merge_family_stats(current.get("family_stats", {}), previous.get("family_stats", {}), decay=decay),
        "style_leaders": _merge_style_leaders(current.get("style_leaders", []), previous.get("style_leaders", []), decay=decay),
        "preferred_skeletons": _trim_strings(
            list(current.get("preferred_skeletons", [])) + list(previous.get("preferred_skeletons", [])),
            limit=WORKING_SKELETON_LIMIT,
        ),
        "historical_skeletons": _trim_strings(
            list(current.get("historical_skeletons", [])) + list(previous.get("historical_skeletons", [])),
            limit=SUMMARY_SKELETON_LIMIT,
        ),
        "seed_context": current.get("seed_context") or previous.get("seed_context", {}),
        "top_rows": _trim_top_rows(current.get("top_rows", []), limit=WORKING_TOP_ROW_LIMIT),
        "suggestions": [str(item) for item in (current.get("suggestions", []) or [])[:WORKING_SUGGESTION_LIMIT]],
    }
    previous_registry = ((previous_summary.get("block_scores") if isinstance(previous_summary, dict) else {}) or {})
    skeleton_registry = {
        "soft": _merge_block_scores(
            current.get("soft_blocked_skeletons", []),
            (previous_registry.get("skeletons", {}) or {}).get("soft", {}),
            updated_at=updated_at,
            score_value=SOFT_BLOCK_SCORE,
        ),
        "hard": _merge_block_scores(
            current.get("hard_blocked_skeletons", current.get("blocked_skeletons", [])),
            (previous_registry.get("skeletons", {}) or {}).get("hard", {}),
            updated_at=updated_at,
            score_value=HARD_BLOCK_SCORE,
        ),
    }
    family_registry = {
        "soft": _merge_block_scores(
            current.get("soft_blocked_families", []),
            (previous_registry.get("families", {}) or {}).get("soft", {}),
            updated_at=updated_at,
            score_value=SOFT_BLOCK_SCORE,
        ),
        "hard": _merge_block_scores(
            current.get("hard_blocked_families", current.get("blocked_families", [])),
            (previous_registry.get("families", {}) or {}).get("hard", {}),
            updated_at=updated_at,
            score_value=HARD_BLOCK_SCORE,
        ),
    }
    summary["block_scores"] = {
        "skeletons": skeleton_registry,
        "families": family_registry,
    }
    summary["soft_blocked_skeletons"] = _registry_to_keys(skeleton_registry["soft"], minimum_score=ACTIVE_SOFT_BLOCK_THRESHOLD)
    summary["soft_blocked_families"] = _registry_to_keys(family_registry["soft"], minimum_score=ACTIVE_SOFT_BLOCK_THRESHOLD)
    summary["hard_blocked_skeletons"] = _registry_to_keys(skeleton_registry["hard"], minimum_score=ACTIVE_HARD_BLOCK_THRESHOLD, require_hard=True)
    summary["hard_blocked_families"] = _registry_to_keys(family_registry["hard"], minimum_score=ACTIVE_HARD_BLOCK_THRESHOLD, require_hard=True)
    summary["blocked_skeletons"] = list(summary["hard_blocked_skeletons"])
    summary["blocked_families"] = list(summary["hard_blocked_families"])
    return summary


def _build_archive_log(
    previous_payload: dict,
    *,
    run_snapshot: dict,
    updated_at: str,
    results_summary: dict | None = None,
    simulation_records: list[dict] | None = None,
) -> dict:
    previous_log = previous_payload.get("archive_log", {}) if isinstance(previous_payload, dict) else {}
    recent_runs = list(previous_log.get("recent_runs", [])) if isinstance(previous_log, dict) else []
    run_entry = {
        "run_at": updated_at,
        "window_rows": int(run_snapshot.get("window_rows", 0) or 0),
        "hard_blocked_families": list(run_snapshot.get("hard_blocked_families", run_snapshot.get("blocked_families", [])) or []),
        "hard_blocked_skeletons": list(run_snapshot.get("hard_blocked_skeletons", run_snapshot.get("blocked_skeletons", [])) or []),
        "soft_blocked_families": list(run_snapshot.get("soft_blocked_families", []) or []),
        "soft_blocked_skeletons": list(run_snapshot.get("soft_blocked_skeletons", []) or []),
        "failure_counts": dict(run_snapshot.get("failure_counts", {}) or {}),
        "top_suggestions": list((run_snapshot.get("suggestions", []) or [])[:3]),
        "results_summary": (results_summary or {}).get("summary") or {},
        "simulation_records": len(simulation_records or []),
    }
    recent_runs.append(run_entry)
    recent_runs = recent_runs[-ARCHIVE_RUN_LIMIT:]
    return {"recent_runs": recent_runs}


def update_research_memory(
    run_memory: dict,
    *,
    current_snapshot: dict | None = None,
    previous_memory: dict | None = None,
    results_summary: dict | None = None,
    simulation_records: list[dict] | None = None,
    updated_at: str | None = None,
) -> dict:
    updated_at = updated_at or iso_now()
    current_snapshot = flatten_memory_payload(current_snapshot or run_memory or {})
    previous_payload = previous_memory if isinstance(previous_memory, dict) else {}
    working_memory = _build_working_memory(current_snapshot, updated_at=updated_at)
    summary_memory = _build_summary_memory(current_snapshot, previous_payload, updated_at=updated_at)
    planner_memory = flatten_memory_payload(
        {
            **summary_memory,
            "top_rows": working_memory.get("top_rows", []),
            "suggestions": working_memory.get("suggestions", []),
            "block_details": {
                "soft": {
                    "skeletons": [{"key": key} for key in summary_memory.get("soft_blocked_skeletons", [])],
                    "families": [{"key": key} for key in summary_memory.get("soft_blocked_families", [])],
                },
                "hard": {
                    "skeletons": [{"key": key} for key in summary_memory.get("hard_blocked_skeletons", [])],
                    "families": [{"key": key} for key in summary_memory.get("hard_blocked_families", [])],
                },
            },
        }
    )
    archive_log = _build_archive_log(
        previous_payload,
        run_snapshot=current_snapshot,
        updated_at=updated_at,
        results_summary=results_summary,
        simulation_records=simulation_records,
    )
    payload = {
        "_meta": {
            "updated_at": updated_at,
            "results_summary": (results_summary or {}).get("summary") or {},
            "simulation_records": len(simulation_records or []),
            "memory_layers": {
                "working_rows": int(working_memory.get("window_rows", 0) or 0),
                "archive_entries": len(archive_log.get("recent_runs", [])),
            },
        },
        "working_memory": working_memory,
        "summary_memory": summary_memory,
        "archive_log": archive_log,
        "planner_memory": planner_memory,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the canonical global research memory from the latest run.")
    parser.add_argument("--run-memory", required=True, help="Run memory JSON produced during planning.")
    parser.add_argument("--current-snapshot", help="Optional unmerged current-run memory JSON used to refresh working memory.")
    parser.add_argument("--previous-memory", help="Optional existing canonical memory JSON used for decay/pruning.")
    parser.add_argument("--results-summary", help="Optional results summary JSON.")
    parser.add_argument("--simulation-results", help="Optional simulation results JSONL.")
    parser.add_argument("--output", required=True, help="Canonical global research memory output path.")
    args = parser.parse_args()

    run_memory = load_json(Path(args.run_memory), default={})
    current_snapshot = load_json(Path(args.current_snapshot), default={}) if args.current_snapshot else run_memory
    previous_memory = load_json(Path(args.previous_memory), default={}) if args.previous_memory else {}
    results_summary = load_json(Path(args.results_summary), default={}) if args.results_summary else {}
    simulation_records = read_jsonl(args.simulation_results) if args.simulation_results else []
    payload = update_research_memory(
        run_memory,
        current_snapshot=current_snapshot,
        previous_memory=previous_memory,
        results_summary=results_summary,
        simulation_records=simulation_records,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": args.output,
                "simulation_records": len(simulation_records),
                "archive_entries": len(payload.get("archive_log", {}).get("recent_runs", [])),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
