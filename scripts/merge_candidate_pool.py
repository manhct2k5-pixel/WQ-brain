#!/usr/bin/env python3
"""Normalize planner, auto-fix, and scout outputs into a single simulation queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import atomic_write_json, iso_now, load_json, quarantine_payload
from scripts.lineage_utils import ensure_candidate_lineage, merge_candidate_lineage
from src.program_tokens import validate_token_program
from src.submit_gate import (
    SUBMIT_READY_MIN_ALPHA_SCORE,
    SUBMIT_READY_MIN_CONFIDENCE,
    SUBMIT_READY_MIN_FITNESS,
    SUBMIT_READY_MIN_SHARPE,
    surrogate_shadow_allows_exploratory_retry,
    summarize_surrogate_shadow_gate,
)

DEFAULT_PRIOR_EVALUATED_INPUT = ROOT_DIR / "artifacts" / "latest" / "evaluated_candidates.json"
DEFAULT_SOURCE_QUOTA_PROFILE = {
    "planner": 0.45,
    "auto_fix_rewrite": 0.30,
    "scout": 0.25,
}
RECENT_FAILURE_PRIORITY_PENALTIES = {
    "recent_failed_exact_match": 18.0,
    "recent_failed_normalized_match": 12.0,
    "recent_failed_skeleton_match": 7.0,
}
NOVELTY_PRIORITY_WEIGHT = 6.0
FAMILY_DIVERSITY_FRESH_BONUS = 3.2
FAMILY_DIVERSITY_REPEAT_PENALTY = 1.45
SOURCE_DIVERSITY_FRESH_BONUS = 1.8
SOURCE_DIVERSITY_REPEAT_PENALTY = 0.95
SOURCE_HISTORY_WEIGHT = 12.0
SOURCE_SOFT_QUOTA_BONUS_WEIGHT = 0.9
SOURCE_SOFT_QUOTA_PENALTY_WEIGHT = 6.0
DEFAULT_EXPLORATORY_QUEUE_LIMIT = 2
EXPLORATORY_ALLOWED_FILTER_REASONS = {"not_seed_ready", "low_confidence_queue", "surrogate_shadow_fail"}
EXPLORATORY_ALPHA_SCORE_BUFFER = 45.0
EXPLORATORY_SHARPE_BUFFER = 0.6
EXPLORATORY_FITNESS_BUFFER = 0.25
EXPLORATORY_CONFIDENCE_BUFFER = 0.20
EXPLORATORY_SURROGATE_CONFIDENCE_FLOOR = 0.40


def normalize_expression(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def skeletonize_expression(expr: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "N", normalize_expression(expr))


def _stable_signature(*parts) -> str:
    joined = "||".join(str(part or "") for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compiled_expression(candidate: dict) -> str:
    compiled = candidate.get("compiled_expression")
    if compiled:
        return str(compiled)
    token_program = candidate.get("token_program") or []
    if token_program:
        try:
            return str(validate_token_program(token_program))
        except Exception:
            pass
    return str(candidate.get("expression") or "")


def _source_bonus(source: str, source_bonus_adjustments: dict | None = None) -> float:
    base = {
        "auto_fix_rewrite": 14.0,
        "planner": 10.0,
        "scout": 6.0,
    }.get(source, 0.0)
    adjustment = to_float((source_bonus_adjustments or {}).get(source))
    return base + adjustment


def _priority_score(candidate: dict, source: str, *, source_bonus_adjustments: dict | None = None) -> float:
    local = candidate.get("local_metrics") or {}
    return round(
        (35.0 if candidate.get("qualified") else 0.0)
        + (12.0 if candidate.get("seed_ready") else 0.0)
        + _source_bonus(source, source_bonus_adjustments)
        + to_float(local.get("alpha_score"))
        + (8.0 * to_float(local.get("fitness")))
        + (6.0 * to_float(local.get("sharpe")))
        + (10.0 * to_float(candidate.get("confidence_score"))),
        4,
    )


def _settings_key(value) -> str:
    return str(value or "").strip()


def _candidate_history_key(expression: str, settings) -> tuple[str, str]:
    return normalize_expression(expression), _settings_key(settings)


def _build_prior_failure_lookup(prior_evaluated_candidates: list[dict] | None) -> set[tuple[str, str]]:
    failed = set()
    for candidate in prior_evaluated_candidates or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("evaluated_submit_ready"):
            continue
        expression = str(candidate.get("compiled_expression") or candidate.get("expression") or "").strip()
        if not expression:
            continue
        failed.add(_candidate_history_key(expression, candidate.get("settings")))
    return failed


def _build_recent_failure_index(prior_evaluated_candidates: list[dict] | None) -> dict[str, dict[str, int]]:
    index = {
        "structure_signatures": {},
        "expression_signatures": {},
        "skeleton_signatures": {},
    }
    for candidate in prior_evaluated_candidates or []:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("evaluated_submit_ready"):
            continue
        expression = str(candidate.get("expression") or candidate.get("compiled_expression") or "").strip()
        compiled_expression = str(candidate.get("compiled_expression") or expression).strip()
        normalized_expression = normalize_expression(expression)
        normalized_compiled_expression = normalize_expression(compiled_expression)
        skeleton = skeletonize_expression(compiled_expression or expression)
        if normalized_compiled_expression:
            signature = _stable_signature("structure", normalized_compiled_expression)
            index["structure_signatures"][signature] = index["structure_signatures"].get(signature, 0) + 1
        if normalized_expression:
            signature = _stable_signature("expression", normalized_expression)
            index["expression_signatures"][signature] = index["expression_signatures"].get(signature, 0) + 1
        if skeleton:
            signature = _stable_signature("skeleton", skeleton)
            index["skeleton_signatures"][signature] = index["skeleton_signatures"].get(signature, 0) + 1
    return index


def _canonical_source_name(candidate: dict) -> str:
    source = str(candidate.get("source") or "").strip()
    if source:
        return source
    source_stages = candidate.get("source_stages") or []
    if isinstance(source_stages, list):
        for item in source_stages:
            text = str(item or "").strip()
            if text:
                return text
    return "unknown"


def _build_source_history_stats(prior_evaluated_candidates: list[dict] | None) -> dict[str, dict[str, float]]:
    stats = {
        "planner": {"attempts": 0, "submit_ready": 0},
        "auto_fix_rewrite": {"attempts": 0, "submit_ready": 0},
        "scout": {"attempts": 0, "submit_ready": 0},
    }
    total_attempts = 0
    total_submit_ready = 0

    for candidate in prior_evaluated_candidates or []:
        if not isinstance(candidate, dict):
            continue
        source = _canonical_source_name(candidate)
        if source not in stats:
            continue
        stats[source]["attempts"] += 1
        total_attempts += 1
        if candidate.get("evaluated_submit_ready"):
            stats[source]["submit_ready"] += 1
            total_submit_ready += 1

    overall_pass_rate = round((total_submit_ready / total_attempts), 4) if total_attempts else 0.0
    for source, record in stats.items():
        attempts = int(record["attempts"])
        submit_ready = int(record["submit_ready"])
        pass_rate = round((submit_ready / attempts), 4) if attempts else 0.0
        record["attempts"] = attempts
        record["submit_ready"] = submit_ready
        record["pass_rate"] = pass_rate
        record["history_bonus"] = round(
            max(-4.0, min(4.0, (pass_rate - overall_pass_rate) * SOURCE_HISTORY_WEIGHT)),
            4,
        ) if attempts else 0.0
    stats["_overall"] = {
        "attempts": total_attempts,
        "submit_ready": total_submit_ready,
        "pass_rate": overall_pass_rate,
        "history_bonus": 0.0,
    }
    return stats


def _normalize_source_quota_profile(source_quota_profile: dict | None, available_counts: dict[str, int]) -> dict[str, float]:
    merged = dict(DEFAULT_SOURCE_QUOTA_PROFILE)
    for source, value in (source_quota_profile or {}).items():
        if source not in merged:
            continue
        merged[source] = max(0.0, to_float(value))

    active_sources = [source for source, count in available_counts.items() if count > 0]
    if not active_sources:
        return {source: 0.0 for source in DEFAULT_SOURCE_QUOTA_PROFILE}

    total = sum(merged.get(source, 0.0) for source in active_sources)
    if total <= 0.0:
        equal_share = round(1.0 / len(active_sources), 4)
        normalized = {source: 0.0 for source in DEFAULT_SOURCE_QUOTA_PROFILE}
        for source in active_sources:
            normalized[source] = equal_share
        return normalized

    normalized = {source: 0.0 for source in DEFAULT_SOURCE_QUOTA_PROFILE}
    for source in active_sources:
        normalized[source] = round(merged.get(source, 0.0) / total, 4)
    return normalized


def _candidate_family_key(candidate: dict) -> str:
    lineage = candidate.get("lineage") if isinstance(candidate.get("lineage"), dict) else {}
    family = str(lineage.get("family") or candidate.get("thesis_id") or candidate.get("thesis") or "").strip()
    return family or "__unknown__"


def _selection_components(
    candidate: dict,
    *,
    selected_source_counts: dict[str, int],
    selected_family_counts: dict[str, int],
    target_count: int,
    source_quota_profile: dict[str, float],
) -> dict[str, float]:
    source = candidate.get("source") or "unknown"
    family_key = _candidate_family_key(candidate)
    source_count = int(selected_source_counts.get(source, 0))
    family_count = int(selected_family_counts.get(family_key, 0))

    novelty_bonus = round(to_float(candidate.get("novelty_score")) * NOVELTY_PRIORITY_WEIGHT, 4)
    diversity_bonus = round(
        (FAMILY_DIVERSITY_FRESH_BONUS if family_count == 0 else -min(4.2, family_count * FAMILY_DIVERSITY_REPEAT_PENALTY))
        + (SOURCE_DIVERSITY_FRESH_BONUS if source_count == 0 else -min(3.4, source_count * SOURCE_DIVERSITY_REPEAT_PENALTY)),
        4,
    )

    quota_share = max(0.0, to_float(source_quota_profile.get(source)))
    soft_quota_target = round(target_count * quota_share, 4) if target_count > 0 else 0.0
    soft_quota_floor = max(1.0, soft_quota_target) if quota_share > 0.0 else soft_quota_target
    soft_quota_bonus = round(
        min(2.6, max(0.0, soft_quota_floor - source_count) * SOURCE_SOFT_QUOTA_BONUS_WEIGHT),
        4,
    ) if target_count > 0 else 0.0
    soft_quota_penalty = round(
        max(0.0, (source_count + 1) - soft_quota_floor) * SOURCE_SOFT_QUOTA_PENALTY_WEIGHT,
        4,
    ) if target_count > 0 else 0.0

    history_bonus = round(to_float(candidate.get("source_history_bonus")), 4)
    selection_priority_score = round(
        to_float(candidate.get("priority_score"))
        + novelty_bonus
        + diversity_bonus
        + history_bonus
        + soft_quota_bonus
        - soft_quota_penalty,
        4,
    )

    return {
        "selection_priority_score": selection_priority_score,
        "novelty_priority_bonus": novelty_bonus,
        "diversity_priority_bonus": diversity_bonus,
        "source_history_bonus": history_bonus,
        "source_soft_quota_target": soft_quota_target,
        "source_soft_quota_bonus": soft_quota_bonus,
        "source_soft_quota_penalty": soft_quota_penalty,
    }


def _rank_candidates_with_soft_quotas(
    candidates: list[dict],
    *,
    limit: int | None,
    source_quota_profile: dict[str, float],
) -> list[dict]:
    remaining = list(candidates)
    ranked = []
    selected_source_counts = {source: 0 for source in DEFAULT_SOURCE_QUOTA_PROFILE}
    selected_family_counts: dict[str, int] = {}
    target_count = min(len(remaining), limit) if limit is not None and limit > 0 else len(remaining)

    while remaining and len(ranked) < target_count:
        best_candidate = None
        best_components = None
        best_key = None
        for candidate in remaining:
            components = _selection_components(
                candidate,
                selected_source_counts=selected_source_counts,
                selected_family_counts=selected_family_counts,
                target_count=target_count,
                source_quota_profile=source_quota_profile,
            )
            sort_key = (
                components["selection_priority_score"],
                to_float(candidate.get("priority_score")),
                to_float((candidate.get("local_metrics") or {}).get("alpha_score")),
                to_float((candidate.get("local_metrics") or {}).get("fitness")),
                to_float((candidate.get("local_metrics") or {}).get("sharpe")),
            )
            if best_key is None or sort_key > best_key:
                best_candidate = candidate
                best_components = components
                best_key = sort_key

        if best_candidate is None or best_components is None:
            break

        best_candidate.update(best_components)
        ranked.append(best_candidate)
        remaining.remove(best_candidate)
        source = best_candidate.get("source") or "unknown"
        selected_source_counts[source] = selected_source_counts.get(source, 0) + 1
        family_key = _candidate_family_key(best_candidate)
        selected_family_counts[family_key] = selected_family_counts.get(family_key, 0) + 1

    return ranked


def _normalize_candidate(
    candidate: dict,
    *,
    source: str,
    source_bonus_adjustments: dict | None = None,
) -> tuple[dict | None, str | None]:
    if not isinstance(candidate, dict):
        return None, "candidate_not_object"
    expression = str(candidate.get("expression") or candidate.get("compiled_expression") or "").strip()
    if not expression:
        return None, "missing_expression"
    token_program = candidate.get("token_program")
    if token_program is not None and token_program != [] and not isinstance(token_program, list):
        return None, "token_program_not_list"
    local_metrics = candidate.get("local_metrics")
    if local_metrics is not None and local_metrics != {} and not isinstance(local_metrics, dict):
        return None, "local_metrics_not_object"
    risk_tags = candidate.get("risk_tags")
    if risk_tags is not None and risk_tags != [] and not isinstance(risk_tags, (list, tuple, set)):
        return None, "risk_tags_not_list"
    fail_reasons = candidate.get("quality_fail_reasons")
    if fail_reasons is not None and fail_reasons != [] and not isinstance(fail_reasons, (list, tuple, set)):
        return None, "quality_fail_reasons_not_list"
    compiled_expression = _compiled_expression(candidate)
    normalized_expression = normalize_expression(expression)
    normalized_compiled_expression = normalize_expression(compiled_expression or expression)
    expression_skeleton = skeletonize_expression(compiled_expression or expression)
    settings_key = _settings_key(candidate.get("settings"))
    lineage = ensure_candidate_lineage(
        candidate,
        stage_source=source,
        source_detail=candidate.get("seed_source") or candidate.get("source_kind") or candidate.get("source") or source,
        default_parent_expression=candidate.get("source_expression"),
        default_parent_alpha_id=candidate.get("source_alpha_id"),
        default_hypothesis_id=candidate.get("thesis_id") or candidate.get("family_id"),
        default_hypothesis_label=candidate.get("thesis") or candidate.get("family"),
        default_family=candidate.get("family_id") or candidate.get("thesis_id"),
        default_family_components=candidate.get("thesis_family_ids") or candidate.get("family_components"),
        default_generation_reason=candidate.get("selection_reason") or candidate.get("repair_status") or candidate.get("why"),
    )

    normalized = {
        "run_id": candidate.get("run_id"),
        "batch_id": candidate.get("batch_id"),
        "candidate_id": candidate.get("candidate_id"),
        "source": source,
        "source_stages": [source],
        "thesis": candidate.get("thesis") or candidate.get("label") or "Untitled candidate",
        "thesis_id": candidate.get("thesis_id"),
        "why": candidate.get("why") or candidate.get("selection_reason") or "",
        "expression": expression,
        "compiled_expression": compiled_expression,
        "normalized_expression": normalized_expression,
        "normalized_compiled_expression": normalized_compiled_expression,
        "expression_skeleton": expression_skeleton,
        "candidate_signature": _stable_signature("candidate", normalized_compiled_expression, settings_key),
        "structure_signature": _stable_signature("structure", normalized_compiled_expression),
        "expression_signature": _stable_signature("expression", normalized_expression),
        "skeleton_signature": _stable_signature("skeleton", expression_skeleton),
        "token_program": token_program or [],
        "candidate_score": candidate.get("candidate_score"),
        "confidence_score": candidate.get("confidence_score"),
        "novelty_score": candidate.get("novelty_score"),
        "style_alignment_score": candidate.get("style_alignment_score"),
        "risk_tags": list(risk_tags or []),
        "seed_ready": bool(candidate.get("seed_ready")),
        "qualified": bool(candidate.get("qualified")),
        "quality_label": candidate.get("quality_label") or ("qualified" if candidate.get("qualified") else "watchlist"),
        "quality_fail_reasons": list(fail_reasons or []),
        "settings": candidate.get("settings"),
        "local_metrics": local_metrics or {},
        "repair_status": candidate.get("repair_status"),
        "priority_score": _priority_score(candidate, source, source_bonus_adjustments=source_bonus_adjustments),
        "priority_score_before_recent_failures": _priority_score(candidate, source, source_bonus_adjustments=source_bonus_adjustments),
        "recent_failure_penalty": 0.0,
        "recent_failure_reasons": [],
        "recent_failure_match_count": 0,
        "dedupe_match_types": [],
        "duplicate_candidate_count": 1,
        "lineage": lineage,
    }
    return normalized, None


def _merge_duplicate_candidate(current: dict, incoming: dict, *, match_type: str) -> dict:
    current_snapshot = dict(current)
    incoming_snapshot = dict(incoming)
    current_sources = set(current.get("source_stages", []))
    current_sources.update(incoming.get("source_stages", []))
    current_risks = set(current.get("risk_tags", [])) | set(incoming.get("risk_tags", []))
    current_fail_reasons = set(current.get("quality_fail_reasons", [])) | set(incoming.get("quality_fail_reasons", []))
    current_match_types = set(current.get("dedupe_match_types", []))
    current_match_types.add(match_type)
    signature_set = set(current.get("merged_candidate_signatures", []))
    signature_set.add(current.get("candidate_signature"))
    signature_set.add(incoming.get("candidate_signature"))
    duplicate_count = max(1, int(current.get("duplicate_candidate_count", 1))) + 1

    better = incoming_snapshot if incoming.get("priority_score", 0.0) >= current.get("priority_score", 0.0) else current_snapshot
    worse = current_snapshot if better is incoming_snapshot else incoming_snapshot
    preserved_queue_rank = current_snapshot.get("queue_rank")
    current.clear()
    current.update(better)
    current["source_stages"] = sorted(item for item in current_sources if item)
    current["risk_tags"] = sorted(item for item in current_risks if item)
    current["quality_fail_reasons"] = sorted(item for item in current_fail_reasons if item)
    current["seed_ready"] = bool(better.get("seed_ready") or worse.get("seed_ready"))
    current["qualified"] = bool(better.get("qualified") or worse.get("qualified"))
    if current["qualified"]:
        current["quality_label"] = "qualified"
    current["dedupe_match_types"] = sorted(current_match_types)
    current["duplicate_candidate_count"] = duplicate_count
    current["merged_candidate_signatures"] = sorted(item for item in signature_set if item)
    current["lineage"] = merge_candidate_lineage(better.get("lineage"), worse.get("lineage"))
    current["lineage"] = ensure_candidate_lineage(
        current,
        stage_source=current.get("source"),
        source_detail=current.get("source"),
        default_hypothesis_id=current.get("thesis_id"),
        default_hypothesis_label=current.get("thesis"),
        default_family=current.get("thesis_id"),
        default_generation_reason=current.get("why"),
    )
    if preserved_queue_rank is not None:
        current["queue_rank"] = preserved_queue_rank
    return current


def _apply_recent_failure_penalty(candidate: dict, recent_failure_index: dict[str, dict[str, int]]) -> None:
    matched_reason = None
    match_count = 0
    if candidate.get("structure_signature") in recent_failure_index.get("structure_signatures", {}):
        matched_reason = "recent_failed_exact_match"
        match_count = int(recent_failure_index["structure_signatures"].get(candidate.get("structure_signature"), 0))
    elif candidate.get("expression_signature") in recent_failure_index.get("expression_signatures", {}):
        matched_reason = "recent_failed_normalized_match"
        match_count = int(recent_failure_index["expression_signatures"].get(candidate.get("expression_signature"), 0))
    elif candidate.get("skeleton_signature") in recent_failure_index.get("skeleton_signatures", {}):
        matched_reason = "recent_failed_skeleton_match"
        match_count = int(recent_failure_index["skeleton_signatures"].get(candidate.get("skeleton_signature"), 0))

    candidate["priority_score_before_recent_failures"] = candidate.get("priority_score", 0.0)
    candidate["recent_failure_match_count"] = match_count
    if not matched_reason:
        candidate["recent_failure_penalty"] = 0.0
        candidate["recent_failure_reasons"] = []
        return

    penalty = RECENT_FAILURE_PRIORITY_PENALTIES[matched_reason]
    candidate["recent_failure_penalty"] = penalty
    candidate["recent_failure_reasons"] = [matched_reason]
    candidate["priority_score"] = round(candidate.get("priority_score", 0.0) - penalty, 4)


def _exploratory_queue_enabled(config: dict | None) -> bool:
    return bool((config or {}).get("active"))


def _exploratory_queue_limit(config: dict | None) -> int:
    try:
        limit = int((config or {}).get("limit") or DEFAULT_EXPLORATORY_QUEUE_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_EXPLORATORY_QUEUE_LIMIT
    return max(1, limit)


def _exploratory_queue_backfill_below_count(config: dict | None) -> int:
    try:
        threshold = int((config or {}).get("backfill_below_count") or 0)
    except (TypeError, ValueError):
        threshold = 0
    return max(0, threshold)


def _exploratory_candidate_eligible(candidate: dict, *, filter_reason: str, exploratory_queue: dict | None) -> bool:
    if not _exploratory_queue_enabled(exploratory_queue):
        return False
    if filter_reason not in EXPLORATORY_ALLOWED_FILTER_REASONS:
        return False

    risk_tags = set(candidate.get("risk_tags") or [])
    if "blocked_family_risk" in risk_tags:
        return False

    confidence_score = to_float(candidate.get("confidence_score"))
    local = candidate.get("local_metrics") or {}
    alpha_score = to_float(local.get("alpha_score"))
    sharpe = to_float(local.get("sharpe"))
    fitness = to_float(local.get("fitness"))

    if filter_reason == "surrogate_shadow_fail":
        return (
            bool(candidate.get("seed_ready"))
            and confidence_score >= EXPLORATORY_SURROGATE_CONFIDENCE_FLOOR
            and surrogate_shadow_allows_exploratory_retry(local)
        )

    confidence_floor = max(0.0, SUBMIT_READY_MIN_CONFIDENCE - EXPLORATORY_CONFIDENCE_BUFFER)
    alpha_floor = max(0.0, SUBMIT_READY_MIN_ALPHA_SCORE - EXPLORATORY_ALPHA_SCORE_BUFFER)
    sharpe_floor = max(0.0, SUBMIT_READY_MIN_SHARPE - EXPLORATORY_SHARPE_BUFFER)
    fitness_floor = max(0.0, SUBMIT_READY_MIN_FITNESS - EXPLORATORY_FITNESS_BUFFER)

    return confidence_score >= confidence_floor and (
        alpha_score >= alpha_floor or sharpe >= sharpe_floor or fitness >= fitness_floor
    )


def _exploratory_queue_gap(candidate: dict, *, filter_reason: str) -> float:
    local = candidate.get("local_metrics") or {}
    confidence_score = to_float(candidate.get("confidence_score"))
    alpha_score = to_float(local.get("alpha_score"))
    sharpe = to_float(local.get("sharpe"))
    fitness = to_float(local.get("fitness"))
    risk_tags = set(candidate.get("risk_tags") or [])

    gap = 0.0
    gap += max(0.0, SUBMIT_READY_MIN_CONFIDENCE - confidence_score) / 0.05
    gap += max(0.0, SUBMIT_READY_MIN_ALPHA_SCORE - alpha_score) / 18.0
    gap += max(0.0, SUBMIT_READY_MIN_SHARPE - sharpe) / 0.35
    gap += max(0.0, SUBMIT_READY_MIN_FITNESS - fitness) / 0.25
    if filter_reason == "surrogate_shadow_fail":
        gap += 0.18
    else:
        gap += 0.45 if filter_reason == "not_seed_ready" else 0.25
    gap += 0.35 if "already_seeded" in risk_tags else 0.0
    gap += 0.30 if "similarity_risk" in risk_tags else 0.0
    gap += 0.20 if "seed_bias_risk" in risk_tags else 0.0
    gap += 0.15 if "soft_blocked_family_risk" in risk_tags else 0.0
    gap += 0.10 if "soft_blocked_skeleton_risk" in risk_tags else 0.0
    return round(gap, 4)


def _mark_exploratory_candidate(candidate: dict, *, filter_reason: str) -> dict:
    candidate["exploratory_queue"] = True
    candidate["exploratory_filter_reason"] = filter_reason
    candidate["exploratory_queue_gap"] = _exploratory_queue_gap(candidate, filter_reason=filter_reason)
    candidate["queue_policy"] = "exploratory_fallback"
    candidate["queue_policy_reason"] = filter_reason
    if filter_reason == "surrogate_shadow_fail":
        candidate["surrogate_verify_first"] = True
        candidate["surrogate_override_mode"] = "exploratory_verify_first"
    return candidate


def _queue_filter_reason(
    candidate: dict,
    *,
    source: str,
    prior_failure_lookup: set[tuple[str, str]],
) -> str | None:
    if not candidate.get("seed_ready"):
        return "not_seed_ready"
    surrogate_gate = summarize_surrogate_shadow_gate(candidate.get("local_metrics"))
    if surrogate_gate["blocked"]:
        return "surrogate_shadow_fail"
    if not candidate.get("qualified") and to_float(candidate.get("confidence_score")) < SUBMIT_READY_MIN_CONFIDENCE:
        return "low_confidence_queue"
    history_key = _candidate_history_key(candidate.get("compiled_expression") or candidate.get("expression") or "", candidate.get("settings"))
    if source == "auto_fix_rewrite" and history_key in prior_failure_lookup:
        return "stale_auto_fix_failed_recently"
    return None


def _dedupe_normalized_candidates(normalized_candidates: list[dict]) -> list[dict]:
    representatives: list[dict] = []
    raw_expression_index: dict[str, dict] = {}
    normalized_expression_index: dict[str, dict] = {}
    structure_signature_index: dict[str, dict] = {}

    for normalized in normalized_candidates:
        match_type = ""
        current = None
        raw_expression = normalized.get("expression") or ""
        if raw_expression:
            current = raw_expression_index.get(raw_expression)
            if current is not None:
                match_type = "exact_expression"
        if current is None and normalized.get("normalized_expression"):
            current = normalized_expression_index.get(normalized["normalized_expression"])
            if current is not None:
                match_type = "normalized_expression"
        if current is None and normalized.get("structure_signature"):
            current = structure_signature_index.get(normalized["structure_signature"])
            if current is not None:
                match_type = "exact_structure"
        if current is None:
            representatives.append(normalized)
            if raw_expression:
                raw_expression_index[raw_expression] = normalized
            if normalized.get("normalized_expression"):
                normalized_expression_index[normalized["normalized_expression"]] = normalized
            if normalized.get("structure_signature"):
                structure_signature_index[normalized["structure_signature"]] = normalized
            continue

        merged_candidate = _merge_duplicate_candidate(current, normalized, match_type=match_type)
        if raw_expression:
            raw_expression_index[raw_expression] = merged_candidate
        if merged_candidate.get("expression"):
            raw_expression_index[merged_candidate["expression"]] = merged_candidate
        if normalized.get("normalized_expression"):
            normalized_expression_index[normalized["normalized_expression"]] = merged_candidate
        if merged_candidate.get("normalized_expression"):
            normalized_expression_index[merged_candidate["normalized_expression"]] = merged_candidate
        if normalized.get("structure_signature"):
            structure_signature_index[normalized["structure_signature"]] = merged_candidate
        if merged_candidate.get("structure_signature"):
            structure_signature_index[merged_candidate["structure_signature"]] = merged_candidate

    return representatives


def _select_ranked_candidates(
    candidates: list[dict],
    *,
    limit: int | None,
    source_quota_profile: dict[str, float],
    filtered_counts: dict[str, int],
) -> list[dict]:
    queue_candidates = []
    skeleton_signature_index: dict[str, dict] = {}
    for candidate in candidates:
        skeleton_signature = candidate.get("skeleton_signature")
        if skeleton_signature and skeleton_signature in skeleton_signature_index:
            current = skeleton_signature_index[skeleton_signature]
            _merge_duplicate_candidate(current, candidate, match_type="skeleton")
            filtered_counts["near_duplicate_skeleton_in_queue"] = filtered_counts.get("near_duplicate_skeleton_in_queue", 0) + 1
            continue
        queue_candidates.append(candidate)
        if skeleton_signature:
            skeleton_signature_index[skeleton_signature] = candidate

    available_source_counts = {
        source: sum(1 for item in queue_candidates if item.get("source") == source)
        for source in DEFAULT_SOURCE_QUOTA_PROFILE
    }
    normalized_source_quota_profile = _normalize_source_quota_profile(source_quota_profile, available_source_counts)
    queue_candidates = _rank_candidates_with_soft_quotas(
        queue_candidates,
        limit=limit,
        source_quota_profile=normalized_source_quota_profile,
    )

    for index, candidate in enumerate(queue_candidates, start=1):
        candidate["queue_rank"] = index
    return queue_candidates


def _select_exploratory_candidates(
    candidates: list[dict],
    *,
    limit: int,
    filtered_counts: dict[str, int],
    existing_candidates: list[dict] | None = None,
    queue_policy_context: str = "strict_queue_empty",
) -> list[dict]:
    queue_candidates = []
    skeleton_signature_index: dict[str, dict] = {}
    existing_skeleton_signatures = {
        item.get("skeleton_signature")
        for item in (existing_candidates or [])
        if item.get("skeleton_signature")
    }
    existing_candidate_signatures = {
        item.get("candidate_signature")
        for item in (existing_candidates or [])
        if item.get("candidate_signature")
    }
    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            to_float(item.get("exploratory_queue_gap")),
            -to_float(item.get("priority_score")),
            -to_float((item.get("local_metrics") or {}).get("alpha_score")),
            -to_float((item.get("local_metrics") or {}).get("fitness")),
            -to_float((item.get("local_metrics") or {}).get("sharpe")),
            -to_float(item.get("confidence_score")),
        ),
    )

    for candidate in ranked_candidates:
        if candidate.get("candidate_signature") in existing_candidate_signatures:
            filtered_counts["duplicate_candidate_in_exploratory_backfill"] = (
                filtered_counts.get("duplicate_candidate_in_exploratory_backfill", 0) + 1
            )
            continue
        skeleton_signature = candidate.get("skeleton_signature")
        if skeleton_signature and skeleton_signature in existing_skeleton_signatures:
            filtered_counts["near_duplicate_skeleton_in_exploratory_backfill"] = (
                filtered_counts.get("near_duplicate_skeleton_in_exploratory_backfill", 0) + 1
            )
            continue
        if skeleton_signature and skeleton_signature in skeleton_signature_index:
            current = skeleton_signature_index[skeleton_signature]
            _merge_duplicate_candidate(current, candidate, match_type="skeleton")
            filtered_counts["near_duplicate_skeleton_in_exploratory_queue"] = (
                filtered_counts.get("near_duplicate_skeleton_in_exploratory_queue", 0) + 1
            )
            continue
        queue_candidates.append(candidate)
        if skeleton_signature:
            skeleton_signature_index[skeleton_signature] = candidate
        if len(queue_candidates) >= limit:
            break

    for index, candidate in enumerate(queue_candidates, start=1):
        candidate["queue_rank"] = index
        candidate["exploratory_queue_rank"] = index
        filter_reason = str(candidate.get("exploratory_filter_reason") or "unknown")
        candidate["queue_policy"] = "exploratory_fallback"
        candidate["queue_policy_reason"] = f"{queue_policy_context}:{filter_reason}"
    return queue_candidates


def _load_planner_candidates(path: Path | None) -> list[dict]:
    payload = load_json(path, default={}) if path else {}
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    batch = payload.get("batch", {})
    return [item for item in batch.get("candidates", []) if isinstance(item, dict)]


def _load_auto_fix_candidates(path: Path | None) -> list[dict]:
    payload = load_json(path, default={}) if path else {}
    return [item for item in payload.get("candidates", []) if isinstance(item, dict)]


def _load_scout_candidates(path: Path | None) -> list[dict]:
    payload = load_json(path, default={}) if path else {}
    candidates = []
    for key in ("selected", "watchlist", "candidates"):
        items = payload.get(key)
        if isinstance(items, list):
            candidates.extend(item for item in items if isinstance(item, dict))
    return candidates


def _load_prior_evaluated_candidates(path: Path | None) -> list[dict]:
    payload = load_json(path, default={}) if path else {}
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return [item for item in candidates if isinstance(item, dict)]
    return []


def merge_candidate_pool(
    *,
    planner_candidates: list[dict],
    auto_fix_candidates: list[dict],
    scout_candidates: list[dict],
    prior_evaluated_candidates: list[dict] | None = None,
    limit: int | None = None,
    source_bonus_adjustments: dict | None = None,
    source_quota_profile: dict | None = None,
    exploratory_queue: dict | None = None,
    quarantine_callback: Callable[[dict], None] | None = None,
) -> dict:
    strict_candidates: list[dict] = []
    exploratory_candidates: list[dict] = []
    source_counts = {
        "planner": 0,
        "auto_fix_rewrite": 0,
        "scout": 0,
    }
    filtered_counts = {}
    quarantined_counts = {}
    prior_failure_lookup = _build_prior_failure_lookup(prior_evaluated_candidates)
    recent_failure_index = _build_recent_failure_index(prior_evaluated_candidates)
    source_history_stats = _build_source_history_stats(prior_evaluated_candidates)

    for source, candidates in (
        ("planner", planner_candidates),
        ("auto_fix_rewrite", auto_fix_candidates),
        ("scout", scout_candidates),
    ):
        for raw_candidate in candidates:
            normalized, invalid_reason = _normalize_candidate(
                raw_candidate,
                source=source,
                source_bonus_adjustments=source_bonus_adjustments,
            )
            if invalid_reason:
                quarantined_counts[invalid_reason] = quarantined_counts.get(invalid_reason, 0) + 1
                if quarantine_callback is not None:
                    quarantine_callback(
                        {
                            "source": source,
                            "reason": invalid_reason,
                            "candidate": raw_candidate,
                        }
                    )
                continue
            filter_reason = _queue_filter_reason(
                normalized,
                source=source,
                prior_failure_lookup=prior_failure_lookup,
            )
            if filter_reason:
                if _exploratory_candidate_eligible(normalized, filter_reason=filter_reason, exploratory_queue=exploratory_queue):
                    exploratory_candidate = _mark_exploratory_candidate(dict(normalized), filter_reason=filter_reason)
                    _apply_recent_failure_penalty(exploratory_candidate, recent_failure_index)
                    history_entry = source_history_stats.get(source, {})
                    exploratory_candidate["source_historical_attempts"] = int(history_entry.get("attempts", 0))
                    exploratory_candidate["source_historical_pass_rate"] = round(to_float(history_entry.get("pass_rate")), 4)
                    exploratory_candidate["source_history_bonus"] = round(to_float(history_entry.get("history_bonus")), 4)
                    exploratory_candidates.append(exploratory_candidate)
                filtered_counts[filter_reason] = filtered_counts.get(filter_reason, 0) + 1
                continue
            source_counts[source] += 1
            strict_candidates.append(normalized)

    representatives = _dedupe_normalized_candidates(strict_candidates)
    for candidate in representatives:
        _apply_recent_failure_penalty(candidate, recent_failure_index)
        source = candidate.get("source") or "unknown"
        history_entry = source_history_stats.get(source, {})
        candidate["source_historical_attempts"] = int(history_entry.get("attempts", 0))
        candidate["source_historical_pass_rate"] = round(to_float(history_entry.get("pass_rate")), 4)
        candidate["source_history_bonus"] = round(to_float(history_entry.get("history_bonus")), 4)

    representatives.sort(
        key=lambda item: (
            item.get("priority_score", 0.0),
            to_float((item.get("local_metrics") or {}).get("alpha_score")),
            to_float((item.get("local_metrics") or {}).get("fitness")),
            to_float((item.get("local_metrics") or {}).get("sharpe")),
        ),
        reverse=True,
    )

    available_source_counts = {
        source: sum(1 for item in representatives if item.get("source") == source)
        for source in DEFAULT_SOURCE_QUOTA_PROFILE
    }
    normalized_source_quota_profile = _normalize_source_quota_profile(source_quota_profile, available_source_counts)
    candidates = _select_ranked_candidates(
        representatives,
        limit=limit,
        source_quota_profile=normalized_source_quota_profile,
        filtered_counts=filtered_counts,
    )

    exploratory_queue_used = False
    exploratory_queue_mode = None
    exploratory_selected_count = 0
    exploratory_selected_reasons: dict[str, int] = {}
    exploratory_queue_active = _exploratory_queue_enabled(exploratory_queue)
    exploratory_representatives = _dedupe_normalized_candidates(exploratory_candidates) if exploratory_queue_active and exploratory_candidates else []
    if not candidates and exploratory_queue_active and exploratory_representatives:
        exploratory_representatives = _dedupe_normalized_candidates(exploratory_candidates)
        exploratory_selected = _select_exploratory_candidates(
            exploratory_representatives,
            limit=min(limit, _exploratory_queue_limit(exploratory_queue)) if limit is not None and limit > 0 else _exploratory_queue_limit(exploratory_queue),
            filtered_counts=filtered_counts,
            queue_policy_context="strict_queue_empty",
        )
        if exploratory_selected:
            candidates = exploratory_selected
            exploratory_queue_used = True
            exploratory_queue_mode = "strict_queue_empty"
            exploratory_selected_count = len(exploratory_selected)
            source_counts = {
                source: sum(1 for item in candidates if item.get("source") == source)
                for source in DEFAULT_SOURCE_QUOTA_PROFILE
            }
            for candidate in exploratory_selected:
                reason = str(candidate.get("exploratory_filter_reason") or "unknown")
                exploratory_selected_reasons[reason] = exploratory_selected_reasons.get(reason, 0) + 1
    elif candidates and exploratory_queue_active:
        backfill_below_count = _exploratory_queue_backfill_below_count(exploratory_queue)
        if backfill_below_count and len(candidates) < backfill_below_count:
            remaining_slots = max(0, limit - len(candidates)) if limit is not None and limit > 0 else _exploratory_queue_limit(exploratory_queue)
            exploratory_backfill_limit = min(
                _exploratory_queue_limit(exploratory_queue),
                max(0, backfill_below_count - len(candidates)),
                remaining_slots,
            )
            if exploratory_backfill_limit > 0 and exploratory_representatives:
                exploratory_selected = _select_exploratory_candidates(
                    exploratory_representatives,
                    limit=exploratory_backfill_limit,
                    filtered_counts=filtered_counts,
                    existing_candidates=candidates,
                    queue_policy_context="strict_queue_sparse",
                )
                if exploratory_selected:
                    candidates.extend(exploratory_selected)
                    for index, candidate in enumerate(candidates, start=1):
                        candidate["queue_rank"] = index
                    exploratory_queue_used = True
                    exploratory_queue_mode = "strict_queue_sparse"
                    exploratory_selected_count = len(exploratory_selected)
                    for candidate in exploratory_selected:
                        reason = str(candidate.get("exploratory_filter_reason") or "unknown")
                        exploratory_selected_reasons[reason] = exploratory_selected_reasons.get(reason, 0) + 1

    selected_source_counts = {
        source: sum(1 for item in candidates if item.get("source") == source)
        for source in DEFAULT_SOURCE_QUOTA_PROFILE
    }

    return {
        "generated_at": iso_now(),
        "source_counts": source_counts,
        "selected_source_counts": selected_source_counts,
        "source_quota_profile": normalized_source_quota_profile,
        "source_historical_pass_rates": {
            source: round(to_float(source_history_stats.get(source, {}).get("pass_rate")), 4)
            for source in DEFAULT_SOURCE_QUOTA_PROFILE
        },
        "filtered_counts": dict(sorted(filtered_counts.items())),
        "quarantined_count": sum(quarantined_counts.values()),
        "quarantined_counts": dict(sorted(quarantined_counts.items())),
        "exploratory_queue_active": exploratory_queue_active,
        "exploratory_queue_mode": exploratory_queue_mode,
        "exploratory_queue_used": exploratory_queue_used,
        "exploratory_candidate_count": exploratory_selected_count if exploratory_queue_used else 0,
        "exploratory_selected_reasons": dict(sorted(exploratory_selected_reasons.items())),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge planner, auto-fix, and scout candidates into one simulation queue.")
    parser.add_argument("--planner-input", help="Planner candidate JSON.")
    parser.add_argument("--auto-fix-input", help="Auto-fix candidate JSON.")
    parser.add_argument("--scout-input", help="Scout candidate JSON.")
    parser.add_argument("--prior-evaluated-input", default=str(DEFAULT_PRIOR_EVALUATED_INPUT), help="Optional evaluated candidate JSON from the previous run used to suppress stale retries.")
    parser.add_argument("--limit", type=int, help="Optional queue size limit.")
    parser.add_argument("--output", required=True, help="Output queue JSON path.")
    args = parser.parse_args()

    output_path = Path(args.output)

    def _quarantine_candidate(record: dict) -> None:
        quarantine_payload(
            record,
            reason=str(record.get("reason") or "invalid_candidate"),
            category="candidates",
            label=str(record.get("source") or "candidate"),
            source_path=output_path,
        )

    payload = merge_candidate_pool(
        planner_candidates=_load_planner_candidates(Path(args.planner_input)) if args.planner_input else [],
        auto_fix_candidates=_load_auto_fix_candidates(Path(args.auto_fix_input)) if args.auto_fix_input else [],
        scout_candidates=_load_scout_candidates(Path(args.scout_input)) if args.scout_input else [],
        prior_evaluated_candidates=_load_prior_evaluated_candidates(Path(args.prior_evaluated_input)) if args.prior_evaluated_input else [],
        limit=args.limit,
        quarantine_callback=_quarantine_candidate,
    )
    atomic_write_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
