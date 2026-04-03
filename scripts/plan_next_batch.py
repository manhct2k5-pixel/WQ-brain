#!/usr/bin/env python3
"""Build the next alpha batch from prior results and structured thesis templates."""

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
from scripts.lineage_utils import ensure_candidate_lineage

try:
    from .expression_lint import build_summary as build_lint_summary
    from .results_digest import (
        build_summary as build_digest_summary,
        classify_expression_family,
        coerce_float,
        compute_research_score,
        discover_csv,
        failing_checks,
        normalize_expression,
        passes_all_checks,
        read_rows,
        row_has_pending_checks,
        skeletonize,
    )
except ImportError:
    from expression_lint import build_summary as build_lint_summary
    from results_digest import (
        build_summary as build_digest_summary,
        classify_expression_family,
        coerce_float,
        compute_research_score,
        discover_csv,
        failing_checks,
        normalize_expression,
        passes_all_checks,
        read_rows,
        row_has_pending_checks,
        skeletonize,
    )

from src.internal_scoring import CHECK_COLUMNS, HistoryIndex, score_expression
from src.program_tokens import render_token_program
from src.seed_store import load_seed_store as load_shared_seed_store
from src.submit_gate import (
    SUBMIT_READY_MIN_ALPHA_SCORE,
    SUBMIT_READY_MIN_FITNESS,
    SUBMIT_READY_MIN_SHARPE,
    local_metrics_pass_submit_gate,
)

IMMEDIATE_BLOCK_FAILURES = {"MATCHES_COMPETITION"}
REPEATED_BLOCK_FAILURES = {"SELF_CORRELATION", "CONCENTRATED_WEIGHT", "LOW_SHARPE_AND_LOW_FITNESS"}
TURNOVER_FAILURES = {"HIGH_TURNOVER", "LOW_TURNOVER"}
FAMILY_BLOCK_MIN_COMPLETED = 3
SKELETON_BLOCK_MIN_COMPLETED = 2
PLANNER_FAMILY_SETTINGS = {
    "technical_indicator": ("USA", "TOP3000", 1, 6, 0.05, "Industry"),
    "reversal_conditioned": ("USA", "TOP1000", 1, 3, 0.03, "Industry"),
    "simple_price_patterns": ("USA", "TOP3000", 1, 5, 0.04, "Industry"),
    "vwap_dislocation": ("USA", "TOP200", 1, 3, 0.02, "Industry"),
    "pv_divergence": ("USA", "TOP1000", 1, 5, 0.04, "Industry"),
    "shock_response": ("USA", "TOP1000", 1, 6, 0.04, "Industry"),
    "residual_beta": ("USA", "TOP3000", 1, 7, 0.06, "Industry"),
}

THESIS_LIBRARY = [
    {
        "id": "pv_divergence",
        "label": "Price-volume divergence",
        "why": "Targets disagreement between price action and liquidity flow.",
        "variants": [
            {
                "variant_id": "pv_rank_corr_10",
                "token_program": ["VOLUME", "TSR_10", "CLOSE", "TSR_10", "CORR_10", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["volume", "correlation", "rank", "normalization"],
            },
            {
                "variant_id": "pv_rank_corr_21",
                "token_program": ["VOLUME", "TSR_21", "CLOSE", "TSR_21", "CORR_21", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["volume", "correlation", "rank", "normalization"],
            },
            {
                "variant_id": "pv_delta_z_21",
                "token_program": ["PV_DIV2", "TSZ_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["volume", "rank", "normalization"],
            },
            {
                "variant_id": "pv_rank_spread_21",
                "token_program": ["VOLUME", "TSR_21", "CLOSE", "TSR_21", "SUB", "RANK"],
                "risk_tags": [],
                "style_tags": ["volume", "rank"],
            },
        ],
    },
    {
        "id": "vwap_dislocation",
        "label": "VWAP dislocation",
        "why": "Looks for price dislocation versus intraday anchor prices.",
        "variants": [
            {
                "variant_id": "vwap_abs_z_21",
                "token_program": ["FACTOR_3", "TSZ_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["vwap", "rank", "normalization"],
            },
            {
                "variant_id": "vwap_reversion_band",
                "token_program": ["FACTOR_3", "TSZ_21", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["vwap", "rank", "normalization", "winsorize"],
            },
            {
                "variant_id": "vwap_normalized_spread",
                "token_program": ["CLOSE", "VWAP", "SUB", "RET", "STD_21", "DIV", "RANK", "ZSCORE"],
                "risk_tags": [],
                "style_tags": ["vwap", "rank", "normalization", "volatility"],
            },
            {
                "variant_id": "vwap_midprice_spread",
                "token_program": ["FACTOR_2", "WINSORIZE", "RANK"],
                "risk_tags": ["weight_risk"],
                "style_tags": ["vwap", "rank", "normalization", "winsorize"],
            },
        ],
    },
    {
        "id": "reversal_conditioned",
        "label": "Conditioned reversal",
        "why": "Keeps reversal structure while controlling noise and turnover.",
        "variants": [
            {
                "variant_id": "rev_inverse_vol",
                "token_program": ["REVERSAL", "RET", "STD_21", "INV", "MUL", "RANK"],
                "risk_tags": [],
                "style_tags": ["reversal", "rank", "volatility"],
            },
            {
                "variant_id": "rev_volume_surprise",
                "token_program": ["RET", "VOLUME", "ADV", "DIV", "MUL", "TSZ_21", "RANK"],
                "risk_tags": ["turnover_risk"],
                "style_tags": ["reversal", "volume", "rank", "normalization"],
            },
            {
                "variant_id": "rev_vwap_abs",
                "token_program": ["REVERSAL", "FACTOR_3", "MUL", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["reversal", "vwap", "rank", "winsorize"],
            },
        ],
    },
    {
        "id": "simple_price_patterns",
        "label": "Simple price hypothesis",
        "why": "Starts from a clean price-based hypothesis, then makes it ratio-like, ranked, and robust.",
        "variants": [
            {
                "variant_id": "simple_inv_price_rank",
                "token_program": ["PRICE_INV", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "rank", "winsorize", "book_alpha_design"],
            },
            {
                "variant_id": "simple_price_delay_rank",
                "token_program": ["PRICE_DELAY3", "RANK", "TSZ_21"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "reversal", "rank", "normalization", "book_alpha_design"],
            },
            {
                "variant_id": "simple_price_delay_corr",
                "token_program": ["PRICE_DELAY_CORR", "TSZ_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "correlation", "rank", "normalization", "trend", "book_alpha_design"],
            },
            {
                "variant_id": "simple_trend_volume_confirm",
                "token_program": ["TREND_VOLUME_3", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "trend", "volume", "rank", "winsorize", "book_alpha_design"],
            },
            {
                "variant_id": "simple_ranked_reversal_ladder",
                "token_program": ["REVERSAL", "SURPRISE_21", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "reversal", "rank", "winsorize", "normalization", "book_alpha_design"],
            },
            {
                "variant_id": "simple_inv_corr_div_ladder",
                "token_program": ["PRICE_INV", "CORR_DIV", "MUL", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "correlation", "rank", "winsorize", "low_correlation", "book_alpha_design"],
            },
            {
                "variant_id": "simple_delay_corr_div_rank",
                "token_program": ["PRICE_DELAY3", "CORR_DIV", "MUL", "TSZ_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "correlation", "rank", "normalization", "trend", "low_correlation", "book_alpha_design"],
            },
            {
                "variant_id": "simple_inv_delay_blend",
                "token_program": ["PRICE_INV", "PRICE_DELAY3", "ADD", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "reversal", "rank", "winsorize", "book_alpha_design"],
            },
            {
                "variant_id": "simple_delay_volume_ratio",
                "token_program": ["PRICE_DELAY3", "VOLUME", "ADV", "DIV", "MUL", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "volume", "rank", "winsorize", "trend", "book_alpha_design"],
            },
            {
                "variant_id": "simple_price_corr_div_regime",
                "token_program": ["PRICE_DELAY_CORR", "CORR_DIV", "MUL", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["simple", "ratio_like", "correlation", "rank", "winsorize", "trend", "low_correlation", "book_alpha_design"],
            },
        ],
    },
    {
        "id": "residual_beta",
        "label": "Residual or de-beta structure",
        "why": "Aims to reduce market-direction overlap and self-correlation.",
        "variants": [
            {
                "variant_id": "residual_beta_reg",
                "token_program": ["CLOSE", "BETA", "REG_RESD_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "rank", "beta"],
            },
            {
                "variant_id": "residual_volume_beta_corr",
                "token_program": ["VOLUME", "TSZ_63", "VOLUME", "BETA", "REG_RESD_63", "CORR_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "volume", "correlation", "rank", "normalization", "beta"],
            },
            {
                "variant_id": "residual_volume_inverse_corr_coef",
                "token_program": ["VOLUME", "VOLUME", "TSZ_63", "PRICE_INV", "CORR_21", "REG_COEF_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "volume", "correlation", "rank", "normalization", "beta", "ratio_like"],
            },
            {
                "variant_id": "residual_beta_volume_chain",
                "token_program": ["BETA", "VOLUME", "TSZ_63", "ADV", "REG_RESD_63", "CORR_10", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "volume", "correlation", "rank", "normalization", "beta"],
            },
            {
                "variant_id": "residual_adv_price_ranked_coef",
                "token_program": ["ADV", "TSZ_63", "BETA", "TSR_21", "PRICE_INV", "TSR_21", "CORR_21", "REG_COEF_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "volume", "correlation", "rank", "normalization", "beta", "ratio_like"],
            },
            {
                "variant_id": "residual_corr_coef",
                "token_program": ["RET", "CORR_SPY", "REG_COEF_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "correlation", "rank", "beta"],
            },
            {
                "variant_id": "residual_risk_spread",
                "token_program": ["IDIO_RISK", "SYS_RISK", "SUB", "RANK", "ZSCORE"],
                "risk_tags": [],
                "style_tags": ["residual", "rank", "normalization", "volatility"],
            },
            {
                "variant_id": "residual_risk_ratio_rank",
                "token_program": ["IDIO_RISK", "SYS_RISK", "DIV", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["residual", "beta", "ratio_like", "rank", "winsorize", "low_correlation"],
            },
            {
                "variant_id": "residual_volume_adv_beta_resid",
                "token_program": ["VOLUME", "ADV", "DIV", "BETA", "REG_RESD_63", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "beta", "ratio_like", "volume", "rank", "normalization", "low_correlation"],
            },
            {
                "variant_id": "residual_volume_adv_beta_coef",
                "token_program": ["VOLUME", "ADV", "DIV", "BETA", "REG_COEF_63", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["residual", "beta", "ratio_like", "volume", "rank", "winsorize", "low_correlation"],
            },
            {
                "variant_id": "residual_corr_div_beta_coef",
                "token_program": ["CORR_DIV", "BETA", "REG_COEF_63", "RANK", "TSZ_21"],
                "risk_tags": [],
                "style_tags": ["residual", "beta", "correlation", "rank", "normalization", "low_correlation"],
            },
            {
                "variant_id": "residual_risk_ratio_beta_corr",
                "token_program": ["IDIO_RISK", "SYS_RISK", "DIV", "BETA", "CORR_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["residual", "beta", "ratio_like", "correlation", "rank", "low_correlation"],
            },
        ],
    },
    {
        "id": "shock_response",
        "label": "Volatility or liquidity shock response",
        "why": "Looks for reversion or continuation after unusual activity.",
        "variants": [
            {
                "variant_id": "shock_volume_close",
                "token_program": ["CLOSE", "TSZ_21", "VOLUME", "ADV", "DIV", "MUL", "RANK"],
                "risk_tags": ["turnover_risk"],
                "style_tags": ["volume", "rank", "normalization"],
            },
            {
                "variant_id": "shock_price_volume_ranked",
                "token_program": ["CLOSE", "TSZ_21", "VOLUME", "ADV", "DIV", "MUL", "WINSORIZE", "RANK"],
                "risk_tags": [],
                "style_tags": ["volume", "rank", "normalization", "winsorize"],
            },
            {
                "variant_id": "shock_vol_over_adv",
                "token_program": ["RET", "STD_21", "ADV", "DIV", "TSZ_21", "RANK"],
                "risk_tags": ["turnover_risk"],
                "style_tags": ["volume", "rank", "normalization", "volatility"],
            },
            {
                "variant_id": "shock_vwap_sigma",
                "token_program": ["FACTOR_3", "RET", "STD_10", "MUL", "RANK", "WINSORIZE"],
                "risk_tags": ["turnover_risk"],
                "style_tags": ["vwap", "rank", "volatility", "winsorize"],
            },
        ],
    },
    {
        "id": "technical_indicator",
        "label": "Technical indicator blend",
        "why": "Uses oscillator, momentum, and band-style signals inspired by classic factor libraries.",
        "variants": [
            {
                "variant_id": "tech_open_volume_corr",
                "token_program": ["ZERO", "OPEN", "VOLUME", "CORR_10", "SUB", "RANK", "WINSORIZE"],
                "risk_tags": [],
                "style_tags": ["technical", "volume", "correlation", "rank", "winsorize", "cross_sectional"],
            },
            {
                "variant_id": "tech_sum_spread",
                "token_program": ["CLOSE", "TSS_10", "CLOSE", "TSS_21", "SUB", "CLOSE", "TSS_10", "DIV", "RANK"],
                "risk_tags": [],
                "style_tags": ["technical", "trend", "momentum", "rank", "normalization"],
            },
            {
                "variant_id": "tech_boll_volume",
                "token_program": ["BOLL", "VOLUME", "SURPRISE_21", "MUL", "RANK"],
                "risk_tags": [],
                "style_tags": ["technical", "momentum", "band", "volume", "rank", "normalization"],
            },
            {
                "variant_id": "tech_momentum_band",
                "token_program": ["MOMENTUM", "BOLL", "MUL", "RANK"],
                "risk_tags": [],
                "style_tags": ["technical", "momentum", "band", "rank", "trend"],
            },
            {
                "variant_id": "tech_williams_z",
                "token_program": ["WILLIAMS", "TSZ_21", "RANK"],
                "risk_tags": [],
                "style_tags": ["technical", "oscillator", "momentum", "rank", "normalization"],
            },
            {
                "variant_id": "tech_momentum_volume",
                "token_program": ["MOMENTUM", "VOLUME", "SURPRISE_63", "MUL", "WINSORIZE", "RANK"],
                "risk_tags": ["turnover_risk"],
                "style_tags": ["technical", "momentum", "volume", "rank", "winsorize"],
            },
        ],
    },
]

THESIS_INDEX = {item["id"]: item for item in THESIS_LIBRARY}
HARD_RISK_TAGS = {"similarity_risk", "seed_bias_risk", "already_seeded", "blocked_family_risk"}
SOFT_RISK_TAGS = {"turnover_risk", "weight_risk", "unproven_style", "soft_blocked_family_risk", "soft_blocked_skeleton_risk"}
MERGED_MEMORY_PREVIOUS_WEIGHT = 0.2
MERGED_MEMORY_COLD_START_WEIGHT = 0.5
PLANNER_MEMORY_STYLE_LIMIT = 24
PLANNER_MEMORY_PREFERRED_LIMIT = 96
PLANNER_MEMORY_HISTORY_LIMIT = 320
PLANNER_MEMORY_TOP_ROW_LIMIT = 12
PLANNER_MEMORY_SUGGESTION_LIMIT = 6
PLANNER_MEMORY_FAILURE_LIMIT = 14
PLANNER_MEMORY_FAMILY_FAILURE_LIMIT = 8
ADAPTIVE_MAX_EXPLORATION_BOOST = 0.35
ADAPTIVE_MIN_RISK_PENALTY_MULTIPLIER = 0.5
ADAPTIVE_MAX_SOFT_BLOCK_REOPEN_FAMILIES = 4
ADAPTIVE_MAX_SOFT_BLOCK_REOPEN_SKELETONS = 12


def serious_failures(row: dict) -> set[str]:
    failures = set()
    raw_failures = set(failing_checks(row))
    if "MATCHES_COMPETITION" in raw_failures:
        failures.add("MATCHES_COMPETITION")
    if "SELF_CORRELATION" in raw_failures:
        failures.add("SELF_CORRELATION")
    if "CONCENTRATED_WEIGHT" in raw_failures:
        failures.add("CONCENTRATED_WEIGHT")
    if {"LOW_SHARPE", "LOW_FITNESS"}.issubset(raw_failures):
        failures.add("LOW_SHARPE_AND_LOW_FITNESS")
    return failures


def load_memory(path: str | None) -> dict:
    if not path:
        return {}
    memory_path = Path(path)
    if not memory_path.exists():
        return {}
    payload = load_artifact_json(
        memory_path,
        default={},
        context=f"planner memory {memory_path}",
        warn=lambda message: print(f"[plan-next-batch] Warning: {message}", file=sys.stderr),
    )
    return flatten_memory_payload(payload)


def _limited_sorted_unique(values, limit: int) -> list[str]:
    items = sorted({str(value) for value in values if value})
    if limit <= 0:
        return []
    return items[:limit]


def _prune_failure_counts(payload: dict, *, limit: int) -> dict:
    counter = Counter()
    for name, count in (payload or {}).items():
        try:
            numeric = float(count)
        except (TypeError, ValueError):
            continue
        if numeric <= 0:
            continue
        counter[str(name)] = numeric
    top_items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[: max(0, limit)]
    pruned = {}
    for name, value in top_items:
        rounded = int(round(value))
        if rounded > 0:
            pruned[name] = rounded
    return pruned


def _normalize_style_leaders(style_leaders: list[dict] | None) -> list[dict]:
    normalized = {}
    for item in style_leaders or []:
        tag = str(item.get("tag") or "").strip()
        if not tag:
            continue
        try:
            learning_score = float(item.get("learning_score", 0.0) or 0.0)
        except (TypeError, ValueError):
            learning_score = 0.0
        normalized[tag] = {
            "tag": tag,
            "learning_score": round(learning_score, 4),
        }
    ordered = sorted(normalized.values(), key=lambda item: (item["learning_score"], item["tag"]), reverse=True)
    return ordered[:PLANNER_MEMORY_STYLE_LIMIT]


def _normalize_block_details(memory: dict) -> dict:
    raw = memory.get("block_details", {}) if isinstance(memory, dict) else {}
    normalized = {
        "soft": {"skeletons": [], "families": []},
        "hard": {"skeletons": [], "families": []},
    }
    for level in ("soft", "hard"):
        for group, fallback_field in (
            ("skeletons", f"{level}_blocked_skeletons"),
            ("families", f"{level}_blocked_families"),
        ):
            seen = set()
            entries = []
            for item in ((raw.get(level, {}) if isinstance(raw, dict) else {}).get(group, []) or []):
                key = str(item.get("key") if isinstance(item, dict) else item or "").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                if isinstance(item, dict):
                    entries.append({"key": key, **{k: v for k, v in item.items() if k != "key"}})
                else:
                    entries.append({"key": key})
            for key in memory.get(fallback_field, []) or []:
                key = str(key).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                entries.append({"key": key})
            normalized[level][group] = entries
    return normalized


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _adaptive_controls(memory: dict) -> dict:
    controls = memory.get("adaptive_controls", {})
    return controls if isinstance(controls, dict) else {}


def _soft_block_reopen_candidates(memory: dict, *, group: str) -> list[str]:
    field = f"soft_blocked_{group}"
    registry = ((((memory.get("block_scores") or {}).get(group, {})) or {}).get("soft", {})) or {}
    ranked = []
    for key in memory.get(field, []) or []:
        key = str(key).strip()
        if not key:
            continue
        payload = registry.get(key, {}) if isinstance(registry, dict) else {}
        score = _coerce_float(payload.get("score"), 999.0) if isinstance(payload, dict) else 999.0
        updated_at = str(payload.get("updated_at") or "") if isinstance(payload, dict) else ""
        ranked.append((score, updated_at, key))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[2] for item in ranked]


def apply_adaptive_planning_controls(memory: dict, controls: dict | None) -> dict:
    adjusted = flatten_memory_payload(memory)
    if not isinstance(controls, dict) or not controls:
        adjusted["adaptive_controls"] = {}
        return adjusted

    ignore_block_list = bool(controls.get("ignore_block_list"))
    ignored_blocked_families = list(adjusted.get("blocked_families", []))
    ignored_blocked_skeletons = list(adjusted.get("blocked_skeletons", []))
    ignored_soft_blocked_families = list(adjusted.get("soft_blocked_families", []))
    ignored_soft_blocked_skeletons = list(adjusted.get("soft_blocked_skeletons", []))

    if ignore_block_list:
        reopened_families = []
        reopened_skeletons = []
        adjusted["blocked_families"] = []
        adjusted["blocked_skeletons"] = []
        adjusted["hard_blocked_families"] = []
        adjusted["hard_blocked_skeletons"] = []
        adjusted["soft_blocked_families"] = []
        adjusted["soft_blocked_skeletons"] = []
    else:
        reopened_family_limit = min(
            ADAPTIVE_MAX_SOFT_BLOCK_REOPEN_FAMILIES,
            max(0, _coerce_int(controls.get("reopen_soft_blocked_families_count"), 0)),
        )
        reopened_skeleton_limit = min(
            ADAPTIVE_MAX_SOFT_BLOCK_REOPEN_SKELETONS,
            max(0, _coerce_int(controls.get("reopen_soft_blocked_skeletons_count"), 0)),
        )
        reopened_families = _soft_block_reopen_candidates(adjusted, group="families")[:reopened_family_limit]
        reopened_skeletons = _soft_block_reopen_candidates(adjusted, group="skeletons")[:reopened_skeleton_limit]

        adjusted["soft_blocked_families"] = [item for item in adjusted.get("soft_blocked_families", []) if item not in set(reopened_families)]
        adjusted["soft_blocked_skeletons"] = [item for item in adjusted.get("soft_blocked_skeletons", []) if item not in set(reopened_skeletons)]

    warning = str(controls.get("warning") or "").strip()
    suggestions = [str(item) for item in adjusted.get("suggestions", []) or [] if str(item).strip()]
    if warning:
        suggestions = [warning] + [item for item in suggestions if item != warning]
    adjusted["suggestions"] = suggestions[:PLANNER_MEMORY_SUGGESTION_LIMIT]
    adjusted["adaptive_controls"] = {
        "active": True,
        "mode": str(controls.get("mode") or "stagnation_recovery"),
        "reason_codes": [str(item) for item in (controls.get("reason_codes", []) or []) if item],
        "warning": warning,
        "exploration_boost": round(
            _clip(_coerce_float(controls.get("exploration_boost"), 0.0), 0.0, ADAPTIVE_MAX_EXPLORATION_BOOST),
            4,
        ),
        "exploration_weight_multiplier": round(max(1.0, _coerce_float(controls.get("exploration_weight_multiplier"), 1.0)), 4),
        "soft_block_penalty_multiplier": round(
            _clip(_coerce_float(controls.get("soft_block_penalty_multiplier"), 1.0), 0.0, 1.0),
            4,
        ),
        "candidate_risk_penalty_multiplier": round(
            _clip(
                _coerce_float(controls.get("candidate_risk_penalty_multiplier"), 1.0),
                ADAPTIVE_MIN_RISK_PENALTY_MULTIPLIER,
                1.0,
            ),
            4,
        ),
        "thesis_limit_bonus": max(0, _coerce_int(controls.get("thesis_limit_bonus"), 0)),
        "batch_size_bonus": max(0, _coerce_int(controls.get("batch_size_bonus"), 0)),
        "queue_limit_bonus": max(0, _coerce_int(controls.get("queue_limit_bonus"), 0)),
        "reopened_soft_blocked_families": reopened_families,
        "reopened_soft_blocked_skeletons": reopened_skeletons,
        "ignore_block_list": ignore_block_list,
        "ignored_blocked_families": ignored_blocked_families if ignore_block_list else [],
        "ignored_blocked_skeletons": ignored_blocked_skeletons if ignore_block_list else [],
        "ignored_soft_blocked_families": ignored_soft_blocked_families if ignore_block_list else [],
        "ignored_soft_blocked_skeletons": ignored_soft_blocked_skeletons if ignore_block_list else [],
    }
    adjusted["block_details"] = _normalize_block_details(adjusted)
    return flatten_memory_payload(adjusted)


def flatten_memory_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    memory = payload.get("planner_memory") if isinstance(payload.get("planner_memory"), dict) else payload
    normalized = dict(memory)
    normalized["failure_counts"] = _prune_failure_counts(normalized.get("failure_counts", {}), limit=PLANNER_MEMORY_FAILURE_LIMIT)
    normalized["blocked_skeletons"] = _limited_sorted_unique(
        normalized.get("blocked_skeletons") or normalized.get("hard_blocked_skeletons", []),
        PLANNER_MEMORY_PREFERRED_LIMIT,
    )
    normalized["blocked_families"] = _limited_sorted_unique(
        normalized.get("blocked_families") or normalized.get("hard_blocked_families", []),
        len(THESIS_LIBRARY),
    )
    normalized["hard_blocked_skeletons"] = list(normalized["blocked_skeletons"])
    normalized["hard_blocked_families"] = list(normalized["blocked_families"])
    normalized["soft_blocked_skeletons"] = _limited_sorted_unique(
        normalized.get("soft_blocked_skeletons", []),
        PLANNER_MEMORY_PREFERRED_LIMIT,
    )
    normalized["soft_blocked_families"] = _limited_sorted_unique(
        normalized.get("soft_blocked_families", []),
        len(THESIS_LIBRARY),
    )
    normalized["preferred_skeletons"] = _limited_sorted_unique(
        normalized.get("preferred_skeletons", []),
        PLANNER_MEMORY_PREFERRED_LIMIT,
    )
    normalized["historical_skeletons"] = _limited_sorted_unique(
        normalized.get("historical_skeletons", []),
        PLANNER_MEMORY_HISTORY_LIMIT,
    )
    normalized["suggestions"] = [str(item) for item in (normalized.get("suggestions", []) or [])[:PLANNER_MEMORY_SUGGESTION_LIMIT]]
    normalized["top_rows"] = list((normalized.get("top_rows", []) or [])[:PLANNER_MEMORY_TOP_ROW_LIMIT])
    normalized["style_leaders"] = _normalize_style_leaders(normalized.get("style_leaders", []))
    family_stats = {}
    for family_id, stats in (normalized.get("family_stats", {}) or {}).items():
        entry = default_family_stats()
        entry.update(stats if isinstance(stats, dict) else {})
        entry["failure_counts"] = _prune_failure_counts(entry.get("failure_counts", {}), limit=PLANNER_MEMORY_FAMILY_FAILURE_LIMIT)
        family_stats[family_id] = entry
    normalized["family_stats"] = family_stats
    normalized["block_details"] = _normalize_block_details(normalized)
    return normalized
def load_seed_store(path: str | Path | None = None) -> dict:
    seed_path = Path(path) if path else ROOT_DIR / "initial-population.pkl"
    return load_shared_seed_store(seed_path)


def build_seed_context(seed_store: dict) -> dict:
    family_counts = Counter()
    planned_family_counts = Counter()
    seeded_skeletons = set()
    planned_skeletons = set()

    for expression, entry in seed_store.items():
        if not expression:
            continue
        family_id = classify_expression_family(expression)
        skeleton = skeletonize(expression)
        family_counts.update([family_id])
        seeded_skeletons.add(skeleton)

        result = entry.get("result", {}) if isinstance(entry, dict) else {}
        status = str(result.get("status", "")).upper()
        source = str(result.get("source", "")).lower()
        if status == "PLANNED" or source == "planner":
            planned_family_counts.update([family_id])
            planned_skeletons.add(skeleton)

    return {
        "family_counts": dict(sorted(family_counts.items())),
        "planned_family_counts": dict(sorted(planned_family_counts.items())),
        "seeded_skeletons": sorted(seeded_skeletons),
        "planned_skeletons": sorted(planned_skeletons),
    }


def default_family_stats() -> dict:
    return {
        "attempts": 0,
        "completed": 0,
        "pass_all_count": 0,
        "real_attempts": 0,
        "real_completed": 0,
        "real_pass_all_count": 0,
        "real_fail_count": 0,
        "real_fail_streak": 0,
        "avg_research_score": 0.0,
        "avg_sharpe": 0.0,
        "avg_fitness": 0.0,
        "serious_failures": 0,
        "failure_counts": {},
    }


def default_style_stats() -> dict:
    return {
        "attempts": 0,
        "completed": 0,
        "pass_all_count": 0,
        "avg_research_score": 0.0,
        "top_hits": 0,
    }


def expression_style_tags(expr: str) -> set[str]:
    normalized = normalize_expression(expr).lower()
    tags = set()
    if "ts_corr" in normalized:
        tags.add("correlation")
    if "volume" in normalized:
        tags.add("volume")
    if "vwap" in normalized:
        tags.add("vwap")
    if "rank(" in normalized or "ts_rank" in normalized:
        tags.add("rank")
    if "zscore" in normalized or "ts_zscore" in normalized or "winsorize" in normalized:
        tags.add("normalization")
    if "winsorize" in normalized:
        tags.add("winsorize")
    if "beta_last_" in normalized or "ts_regression" in normalized or "systematic_risk" in normalized or "unsystematic_risk" in normalized:
        tags.update({"residual", "beta"})
    if "close/ts_delay(close,180)-1" in normalized:
        tags.update({"momentum", "technical", "trend"})
    if "-ts_zscore(close,21)" in normalized:
        tags.update({"band", "technical"})
    if "ts_mean(high,10)-close" in normalized:
        tags.update({"oscillator", "technical"})
    if "ts_corr(open,volume,10)" in normalized:
        tags.update({"technical", "volume", "correlation", "cross_sectional"})
    if "ts_sum(close,10)" in normalized or "ts_sum(close,21)" in normalized:
        tags.update({"technical", "trend", "momentum"})
    if "1-close/ts_delay" in normalized or "ts_delay(close" in normalized:
        tags.add("reversal")
    if "ts_std_dev" in normalized or "systematic_risk" in normalized or "unsystematic_risk" in normalized:
        tags.add("volatility")
    return tags


def update_running_average(current_average: float, current_count: int, value: float | None) -> float:
    if value is None:
        return current_average
    return ((current_average * current_count) + value) / (current_count + 1)


def is_real_brain_result(row: dict) -> bool:
    alpha_id = str((row or {}).get("alpha_id") or "").strip().upper()
    if not alpha_id:
        return False
    return not alpha_id.startswith("LOCAL-")


def merge_real_fail_streak(current_stats: dict, previous_stats: dict) -> int:
    current_real_attempts = int(current_stats.get("real_attempts", 0) or 0)
    current_real_completed = int(current_stats.get("real_completed", 0) or 0)
    if current_real_attempts <= 0 or current_real_completed <= 0:
        return int(previous_stats.get("real_fail_streak", 0) or 0)

    current_streak = int(current_stats.get("real_fail_streak", 0) or 0)
    if current_streak <= 0:
        return 0

    # A real pass inside the current window already reset the streak locally,
    # so only carry the previous tail when the window is fail-only.
    if int(current_stats.get("real_pass_all_count", 0) or 0) > 0:
        return current_streak
    return current_streak + int(previous_stats.get("real_fail_streak", 0) or 0)


def merge_family_stats(current: dict, previous: dict) -> dict:
    merged = {}
    for family_id in sorted(set(current) | set(previous)):
        stats = default_family_stats()
        current_stats = current.get(family_id, {})
        previous_stats = previous.get(family_id, {})
        current_attempts = current_stats.get("attempts", 0)
        previous_weight = MERGED_MEMORY_PREVIOUS_WEIGHT if current_attempts else MERGED_MEMORY_COLD_START_WEIGHT
        weighted_previous_attempts = previous_stats.get("attempts", 0) * previous_weight
        weighted_previous_completed = previous_stats.get("completed", 0) * previous_weight
        weighted_previous_pass_all = previous_stats.get("pass_all_count", 0) * previous_weight
        weighted_previous_real_attempts = previous_stats.get("real_attempts", 0) * previous_weight
        weighted_previous_real_completed = previous_stats.get("real_completed", 0) * previous_weight
        weighted_previous_real_pass_all = previous_stats.get("real_pass_all_count", 0) * previous_weight
        weighted_previous_real_fail_count = previous_stats.get("real_fail_count", 0) * previous_weight
        weighted_previous_serious = previous_stats.get("serious_failures", 0) * previous_weight

        stats["attempts"] = int(round(current_attempts + weighted_previous_attempts))
        stats["completed"] = int(round(current_stats.get("completed", 0) + weighted_previous_completed))
        stats["pass_all_count"] = int(round(current_stats.get("pass_all_count", 0) + weighted_previous_pass_all))
        stats["real_attempts"] = int(round(current_stats.get("real_attempts", 0) + weighted_previous_real_attempts))
        stats["real_completed"] = int(round(current_stats.get("real_completed", 0) + weighted_previous_real_completed))
        stats["real_pass_all_count"] = int(round(current_stats.get("real_pass_all_count", 0) + weighted_previous_real_pass_all))
        stats["real_fail_count"] = int(round(current_stats.get("real_fail_count", 0) + weighted_previous_real_fail_count))
        stats["serious_failures"] = int(round(current_stats.get("serious_failures", 0) + weighted_previous_serious))
        stats["real_fail_streak"] = merge_real_fail_streak(current_stats, previous_stats)

        previous_attempts = weighted_previous_attempts
        total_attempts = current_attempts + previous_attempts
        if total_attempts:
            stats["avg_research_score"] = round(
                (
                    current_stats.get("avg_research_score", 0.0) * current_attempts
                    + previous_stats.get("avg_research_score", 0.0) * previous_attempts
                )
                / total_attempts,
                4,
            )
            stats["avg_sharpe"] = round(
                (
                    current_stats.get("avg_sharpe", 0.0) * current_attempts
                    + previous_stats.get("avg_sharpe", 0.0) * previous_attempts
                )
                / total_attempts,
                4,
            )
            stats["avg_fitness"] = round(
                (
                    current_stats.get("avg_fitness", 0.0) * current_attempts
                    + previous_stats.get("avg_fitness", 0.0) * previous_attempts
                )
                / total_attempts,
                4,
            )

        merged_failures = Counter()
        for name, count in previous_stats.get("failure_counts", {}).items():
            weighted_count = int(round(float(count) * previous_weight))
            if weighted_count:
                merged_failures[name] = weighted_count
        merged_failures.update(current_stats.get("failure_counts", {}))
        stats["failure_counts"] = dict(sorted(merged_failures.items()))
        merged[family_id] = stats
    return merged


def derive_family_blocklists(family_stats: dict) -> tuple[set[str], set[str]]:
    blocked_families = set()
    soft_blocked_families = set()
    for family_id, stats in family_stats.items():
        completed = int(stats.get("completed", 0) or 0)
        serious_count = int(stats.get("serious_failures", 0) or 0)
        pass_all_count = int(stats.get("pass_all_count", 0) or 0)
        fail_rate = (serious_count / completed) if completed else 0.0
        real_fail_streak = int(stats.get("real_fail_streak", 0) or 0)
        if real_fail_streak >= 3:
            blocked_families.add(family_id)
            continue
        if completed >= FAMILY_BLOCK_MIN_COMPLETED and pass_all_count == 0 and serious_count >= 3 and fail_rate >= 0.75:
            blocked_families.add(family_id)
            continue
        if real_fail_streak >= 2:
            soft_blocked_families.add(family_id)
            continue
        if completed >= 2 and serious_count >= 2 and (fail_rate >= 0.5 or pass_all_count == 0):
            soft_blocked_families.add(family_id)
    return blocked_families, soft_blocked_families - blocked_families


def summarize_style_leaders(style_stats: dict) -> list[dict]:
    max_top_hits = max((stats.get("top_hits", 0) for stats in style_stats.values()), default=0)
    leaders = []
    for tag, stats in style_stats.items():
        completed = stats.get("completed", 0)
        avg_score = max(0.0, stats.get("avg_research_score", 0.0))
        pass_rate = (stats.get("pass_all_count", 0) / completed) if completed else 0.0
        avg_component = min(1.5, avg_score) / 1.5
        top_component = (stats.get("top_hits", 0) / max_top_hits) if max_top_hits else 0.0
        learning_score = round(0.45 * avg_component + 0.35 * pass_rate + 0.20 * top_component, 4)
        leaders.append(
            {
                "tag": tag,
                "attempts": stats.get("attempts", 0),
                "completed": completed,
                "pass_all_count": stats.get("pass_all_count", 0),
                "avg_research_score": round(stats.get("avg_research_score", 0.0), 4),
                "top_hits": stats.get("top_hits", 0),
                "learning_score": learning_score,
            }
        )
    leaders.sort(key=lambda item: (item["learning_score"], item["top_hits"], item["tag"]), reverse=True)
    return leaders


def score_style_alignment(style_tags: list[str] | set[str], memory: dict) -> float:
    if not style_tags:
        return 0.2
    leader_map = {item["tag"]: item["learning_score"] for item in memory.get("style_leaders", [])}
    scores = []
    for tag in style_tags:
        if tag in leader_map:
            scores.append(leader_map[tag])
        else:
            scores.append(0.18)
    return round(sum(scores) / len(scores), 4)


def normalize_candidate_score(value: float) -> float:
    return max(0.0, min(1.0, (value + 0.2) / 1.2))


def planner_settings_label(family_id: str) -> str:
    region, universe, delay, decay, truncation, neutralization = PLANNER_FAMILY_SETTINGS.get(
        family_id,
        ("USA", "TOP3000", 1, 5, 0.05, "Market"),
    )
    return (
        f"{region}, {universe}, Decay {decay}, Delay {delay}, "
        f"Truncation {truncation:.2f}, Neutralization {neutralization}"
    )


def summarize_local_metrics(result: dict) -> dict:
    return {
        "alpha_id": result.get("alpha_id"),
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "alpha_score": result.get("alpha_score"),
        "sharpe": result.get("sharpe"),
        "fitness": result.get("fitness"),
        "turnover": result.get("turnover"),
        "returns": result.get("returns"),
        "drawdown": result.get("drawdown"),
        "margin": result.get("margin"),
        "uniqueness_proxy": result.get("uniqueness_proxy"),
        "style_tags": result.get("style_tags", []),
        "settings": result.get("settings", {}),
        "OUT_OF_SAMPLE_ALIGNMENT": result.get("OUT_OF_SAMPLE_ALIGNMENT"),
        "surrogate_shadow": result.get("surrogate_shadow", {}),
        "surrogate_shadow_status": result.get("surrogate_shadow_status"),
        "surrogate_shadow_preview_verdict": result.get("surrogate_shadow_preview_verdict"),
        "surrogate_shadow_alignment": result.get("surrogate_shadow_alignment"),
        "surrogate_shadow_penalty": result.get("surrogate_shadow_penalty"),
        "surrogate_shadow_reasons": result.get("surrogate_shadow_reasons", []),
        "surrogate_shadow_hard_signal": result.get("surrogate_shadow_hard_signal"),
        **{name: result.get(name) for name in CHECK_COLUMNS},
    }


def score_local_confidence(local_metrics: dict | None) -> float | None:
    if not local_metrics:
        return None

    verdict_score = {
        "PASS": 1.0,
        "LIKELY_PASS": 0.8,
        "BORDERLINE": 0.45,
        "FAIL": 0.08,
    }.get(str(local_metrics.get("verdict") or "").upper(), 0.18)

    alpha_score = coerce_float(local_metrics.get("alpha_score")) or 0.0
    sharpe = coerce_float(local_metrics.get("sharpe")) or 0.0
    fitness = coerce_float(local_metrics.get("fitness")) or 0.0
    turnover = coerce_float(local_metrics.get("turnover"))

    alpha_component = max(0.0, min(1.0, alpha_score / (SUBMIT_READY_MIN_ALPHA_SCORE + 15.0)))
    sharpe_component = max(0.0, min(1.0, sharpe / (SUBMIT_READY_MIN_SHARPE + 0.7)))
    fitness_component = max(0.0, min(1.0, fitness / (SUBMIT_READY_MIN_FITNESS + 0.8)))

    turnover_component = 0.65
    if turnover is not None:
        if 0.08 <= turnover <= 0.60:
            turnover_component = 1.0
        elif 0.05 <= turnover <= 0.80:
            turnover_component = 0.78
        else:
            turnover_component = 0.35

    return round(
        (0.34 * verdict_score)
        + (0.24 * alpha_component)
        + (0.18 * sharpe_component)
        + (0.16 * fitness_component)
        + (0.08 * turnover_component),
        4,
    )


def annotate_candidate_with_local_metrics(candidate: dict, history_index: HistoryIndex | None = None) -> dict:
    annotated = dict(candidate)
    settings_label = planner_settings_label(str(candidate.get("thesis_id") or ""))
    result = score_expression(
        annotated.get("expression", ""),
        history_index=history_index or HistoryIndex(),
        settings=settings_label,
    )
    resolved_settings = result.get("settings", {})
    annotated["settings"] = resolved_settings.get("label", settings_label)
    annotated["local_metrics"] = summarize_local_metrics(result)
    return annotated


def compute_family_confidence(family_id: str, memory: dict) -> float:
    stats = memory.get("family_stats", {}).get(family_id, default_family_stats())
    attempts = stats.get("attempts", 0)
    completed = stats.get("completed", 0)
    pass_all = stats.get("pass_all_count", 0)
    avg_score = max(0.0, stats.get("avg_research_score", 0.0))
    serious_failures = stats.get("serious_failures", 0)
    real_fail_streak = int(stats.get("real_fail_streak", 0) or 0)

    if attempts == 0:
        return 0.35

    coverage = min(1.0, attempts / 8)
    pass_rate = (pass_all / completed) if completed else 0.0
    fail_penalty = min(1.0, serious_failures / max(1, attempts))
    confidence = (
        0.40 * (min(1.5, avg_score) / 1.5)
        + 0.20 * pass_rate
        + 0.20 * coverage
        + 0.20 * max(0.0, 1.0 - fail_penalty)
    )
    confidence -= min(0.35, real_fail_streak * 0.12)
    return round(max(0.0, min(1.0, confidence)), 4)


def evaluate_quality_gate(candidate: dict, family_id: str, memory: dict) -> dict:
    family_confidence = compute_family_confidence(family_id, memory)
    style_alignment = float(candidate.get("style_alignment_score", 0.0))
    novelty_score = float(candidate.get("novelty_score", 0.0))
    candidate_score = float(candidate.get("candidate_score", 0.0))
    risk_tags = set(candidate.get("risk_tags", []))
    local_metrics = candidate.get("local_metrics") or {}
    local_alpha_score = coerce_float(local_metrics.get("alpha_score")) or 0.0
    local_sharpe = coerce_float(local_metrics.get("sharpe")) or 0.0
    local_fitness = coerce_float(local_metrics.get("fitness")) or 0.0
    local_verdict = str(local_metrics.get("verdict") or "").upper()

    hard_risks = sorted(risk_tags & HARD_RISK_TAGS)
    soft_risks = sorted(risk_tags & SOFT_RISK_TAGS)
    strong_local_support = (
        bool(local_metrics)
        and local_verdict in {"PASS", "LIKELY_PASS"}
        and local_alpha_score >= max(72.0, SUBMIT_READY_MIN_ALPHA_SCORE + 7.0)
        and local_sharpe >= (SUBMIT_READY_MIN_SHARPE + 0.15)
        and local_fitness >= (SUBMIT_READY_MIN_FITNESS + 0.35)
    )

    stages = {
        "structural": bool(candidate.get("seed_ready")) and not hard_risks,
        "statistical": (
            candidate_score >= 0.18 and novelty_score >= 0.72
        ) or (
            strong_local_support and novelty_score >= 0.60
        ),
        "history_fit": (
            family_confidence >= 0.24 and style_alignment >= 0.085
        ) or (
            strong_local_support and style_alignment >= 0.07
        ),
        "risk": len(soft_risks) <= 3,
        "local": True if not local_metrics else local_metrics_pass_submit_gate(local_metrics),
    }
    base_confidence = round(
        0.34 * normalize_candidate_score(candidate_score)
        + 0.24 * min(1.0, novelty_score)
        + 0.20 * min(1.0, style_alignment * 4.0)
        + 0.22 * family_confidence
        - 0.05 * len(soft_risks),
        4,
    )
    local_confidence = score_local_confidence(local_metrics)
    confidence_score = base_confidence
    if local_confidence is not None:
        confidence_score = round((0.65 * base_confidence) + (0.35 * local_confidence), 4)
        if not stages["local"]:
            confidence_score = round(min(confidence_score, 0.44), 4)
    qualified = all(stages.values()) and confidence_score >= 0.45

    quality_fail_reasons = []
    if not stages["structural"]:
        quality_fail_reasons.append("structural")
    if not stages["statistical"]:
        quality_fail_reasons.append("statistical")
    if not stages["history_fit"]:
        quality_fail_reasons.append("history_fit")
    if not stages["risk"]:
        quality_fail_reasons.append("risk")
    if not stages["local"]:
        quality_fail_reasons.append("local")
    if confidence_score < 0.45:
        quality_fail_reasons.append("confidence")

    return {
        "family_confidence_score": family_confidence,
        "confidence_score": confidence_score,
        "quality_stages": stages,
        "hard_risk_tags": hard_risks,
        "soft_risk_tags": soft_risks,
        "qualified": qualified,
        "quality_fail_reasons": quality_fail_reasons,
        "quality_label": "qualified" if qualified else "watchlist",
    }


def build_memory(rows: list[dict], top_n: int, history_window: int = 120, seed_context: dict | None = None) -> dict:
    window_rows = rows[-history_window:]
    digest = build_digest_summary(window_rows, top_n)
    blocked_skeletons = set()
    blocked_families = set()
    soft_blocked_skeletons = set()
    soft_blocked_families = set()
    historical_skeletons = set()
    preferred_skeletons = set()
    skeleton_stats = {}
    family_stats = {}
    style_stats = {}

    for row in window_rows:
        expr = row.get("regular_code", "")
        if not expr:
            continue

        family = classify_expression_family(expr)
        skeleton = skeletonize(expr)
        historical_skeletons.add(skeleton)
        pending_checks = row_has_pending_checks(row)
        row_failures = failing_checks(row)
        serious = serious_failures(row)
        research_score = compute_research_score(row)
        passed_all = passes_all_checks(row)
        style_tags = expression_style_tags(expr)
        real_brain_row = is_real_brain_result(row)

        stats = skeleton_stats.setdefault(
            skeleton,
            {
                "completed": 0,
                "serious_failures": 0,
                "immediate_block": False,
            },
        )
        if not pending_checks:
            stats["completed"] += 1
        stats["serious_failures"] += len(serious)
        if "MATCHES_COMPETITION" in serious:
            stats["immediate_block"] = True

        family_entry = family_stats.setdefault(family, default_family_stats())
        family_entry["attempts"] += 1
        family_entry["avg_research_score"] = update_running_average(
            family_entry["avg_research_score"],
            family_entry["attempts"] - 1,
            research_score,
        )
        family_entry["avg_sharpe"] = update_running_average(
            family_entry["avg_sharpe"],
            family_entry["attempts"] - 1,
            coerce_float(row.get("sharpe")),
        )
        family_entry["avg_fitness"] = update_running_average(
            family_entry["avg_fitness"],
            family_entry["attempts"] - 1,
            coerce_float(row.get("fitness")),
        )
        if not pending_checks:
            family_entry["completed"] += 1
        if passed_all:
            family_entry["pass_all_count"] += 1
        family_entry["serious_failures"] += len(serious)
        failure_counts = Counter(family_entry["failure_counts"])
        failure_counts.update(row_failures)
        family_entry["failure_counts"] = dict(sorted(failure_counts.items()))
        if real_brain_row:
            family_entry["real_attempts"] += 1
            if not pending_checks:
                family_entry["real_completed"] += 1
                if passed_all:
                    family_entry["real_pass_all_count"] += 1
                    family_entry["real_fail_streak"] = 0
                else:
                    family_entry["real_fail_count"] += 1
                    family_entry["real_fail_streak"] += 1

        for tag in style_tags:
            style_entry = style_stats.setdefault(tag, default_style_stats())
            style_entry["attempts"] += 1
            style_entry["avg_research_score"] = update_running_average(
                style_entry["avg_research_score"],
                style_entry["attempts"] - 1,
                research_score,
            )
            if not pending_checks:
                style_entry["completed"] += 1
            if passed_all:
                style_entry["pass_all_count"] += 1

    for skeleton, stats in skeleton_stats.items():
        completed = stats["completed"]
        serious_count = stats["serious_failures"]
        fail_rate = (serious_count / completed) if completed else 0.0
        if stats["immediate_block"]:
            blocked_skeletons.add(skeleton)
            continue
        if completed >= SKELETON_BLOCK_MIN_COMPLETED and serious_count >= 2 and fail_rate >= 0.6:
            blocked_skeletons.add(skeleton)
            continue
        if serious_count >= 1 and completed >= 1 and (serious_count >= 2 or fail_rate >= 0.5):
            soft_blocked_skeletons.add(skeleton)

    blocked_families, soft_blocked_families = derive_family_blocklists(family_stats)

    for top_row in digest["top_rows"][: max(1, min(5, len(digest["top_rows"])))]:
        expr = top_row.get("regular_code", "")
        if expr:
            preferred_skeletons.add(skeletonize(expr))
            for tag in expression_style_tags(expr):
                style_entry = style_stats.setdefault(tag, default_style_stats())
                style_entry["top_hits"] += 1

    style_leaders = summarize_style_leaders(style_stats)

    memory = {
        "history_window": history_window,
        "source_rows": len(rows),
        "window_rows": len(window_rows),
        "failure_counts": digest["summary"].get("failure_counts", {}),
        "blocked_skeletons": sorted(blocked_skeletons),
        "blocked_families": sorted(blocked_families),
        "hard_blocked_skeletons": sorted(blocked_skeletons),
        "hard_blocked_families": sorted(blocked_families),
        "soft_blocked_skeletons": sorted(soft_blocked_skeletons - blocked_skeletons),
        "soft_blocked_families": sorted(soft_blocked_families - blocked_families),
        "preferred_skeletons": sorted(preferred_skeletons),
        "historical_skeletons": sorted(historical_skeletons),
        "family_stats": family_stats,
        "style_leaders": style_leaders,
        "seed_context": seed_context or {
            "family_counts": {},
            "planned_family_counts": {},
            "seeded_skeletons": [],
            "planned_skeletons": [],
        },
        "top_rows": digest["top_rows"],
        "suggestions": digest["suggestions"],
    }
    memory["block_details"] = _normalize_block_details(memory)
    return flatten_memory_payload(memory)


def merge_memory(current: dict, previous: dict) -> dict:
    if not previous:
        return current

    previous_explicit_block_details = isinstance(previous.get("block_details"), dict) if isinstance(previous, dict) else False
    current = flatten_memory_payload(current)
    previous = flatten_memory_payload(previous)
    merged_failures = Counter(previous.get("failure_counts", {}))
    merged_failures.update(current.get("failure_counts", {}))
    current["failure_counts"] = dict(sorted(merged_failures.items()))
    current["preferred_skeletons"] = sorted(set(current.get("preferred_skeletons", [])) | set(previous.get("preferred_skeletons", [])))
    current["historical_skeletons"] = sorted(set(current.get("historical_skeletons", [])) | set(previous.get("historical_skeletons", [])))
    current["family_stats"] = merge_family_stats(current.get("family_stats", {}), previous.get("family_stats", {}))
    derived_blocked_families, derived_soft_blocked_families = derive_family_blocklists(current["family_stats"])
    merged_style_scores = {}
    for item in previous.get("style_leaders", []):
        merged_style_scores[item["tag"]] = float(item.get("learning_score", 0.0))
    for item in current.get("style_leaders", []):
        tag = item["tag"]
        previous_score = merged_style_scores.get(tag)
        current_score = float(item.get("learning_score", 0.0))
        merged_style_scores[tag] = round((previous_score + current_score) / 2, 4) if previous_score is not None else current_score
    current["style_leaders"] = [
        {"tag": tag, "learning_score": score}
        for tag, score in sorted(merged_style_scores.items(), key=lambda pair: (pair[1], pair[0]), reverse=True)
    ]
    previous_block_details = previous.get("block_details", {})
    if previous_explicit_block_details and previous_block_details:
        previous_hard_skeletons = {item.get("key") for item in previous_block_details.get("hard", {}).get("skeletons", []) if item.get("key")}
        previous_hard_families = {item.get("key") for item in previous_block_details.get("hard", {}).get("families", []) if item.get("key")}
        current["blocked_skeletons"] = sorted(set(current.get("blocked_skeletons", [])) | previous_hard_skeletons)
        current["blocked_families"] = sorted(set(current.get("blocked_families", [])) | previous_hard_families | derived_blocked_families)
        current["hard_blocked_skeletons"] = list(current["blocked_skeletons"])
        current["hard_blocked_families"] = list(current["blocked_families"])
        previous_soft_skeletons = {item.get("key") for item in previous_block_details.get("soft", {}).get("skeletons", []) if item.get("key")}
        previous_soft_families = {item.get("key") for item in previous_block_details.get("soft", {}).get("families", []) if item.get("key")}
        current["soft_blocked_skeletons"] = sorted(set(current.get("soft_blocked_skeletons", [])) | previous_soft_skeletons)
        current["soft_blocked_families"] = sorted(
            (set(current.get("soft_blocked_families", [])) | previous_soft_families | derived_soft_blocked_families)
            - set(current["blocked_families"])
        )
    else:
        current["blocked_skeletons"] = list(current.get("blocked_skeletons", []))
        current["blocked_families"] = sorted(set(current.get("blocked_families", [])) | derived_blocked_families)
        current["hard_blocked_skeletons"] = list(current.get("hard_blocked_skeletons", current.get("blocked_skeletons", [])))
        current["hard_blocked_families"] = list(current["blocked_families"])
        current["soft_blocked_skeletons"] = list(current.get("soft_blocked_skeletons", []))
        current["soft_blocked_families"] = sorted(
            (set(current.get("soft_blocked_families", [])) | derived_soft_blocked_families)
            - set(current["blocked_families"])
        )
    current["seed_context"] = current.get("seed_context") or previous.get("seed_context", {})
    current["block_details"] = _normalize_block_details(current)
    return flatten_memory_payload(current)


def rank_theses(memory: dict) -> list[dict]:
    failure_counts = memory.get("failure_counts", {})
    blocked_families = set(memory.get("blocked_families", []))
    soft_blocked_families = set(memory.get("soft_blocked_families", []))
    adaptive_controls = _adaptive_controls(memory)
    exploration_weight_multiplier = max(1.0, _coerce_float(adaptive_controls.get("exploration_weight_multiplier"), 1.0))
    soft_block_penalty_multiplier = _clip(
        _coerce_float(adaptive_controls.get("soft_block_penalty_multiplier"), 1.0),
        0.0,
        1.0,
    )
    priority_map = {
        "SELF_CORRELATION": {"residual_beta": 1.0, "simple_price_patterns": 0.95, "shock_response": 0.6},
        "MATCHES_COMPETITION": {"simple_price_patterns": 1.0, "residual_beta": 0.9, "shock_response": 0.4},
        "CONCENTRATED_WEIGHT": {"vwap_dislocation": 0.7, "simple_price_patterns": 0.45, "shock_response": 0.5},
        "LOW_SHARPE": {"pv_divergence": 0.5, "vwap_dislocation": 0.5, "reversal_conditioned": 0.5, "technical_indicator": 0.6, "simple_price_patterns": 0.55},
        "LOW_FITNESS": {"pv_divergence": 0.4, "reversal_conditioned": 0.5, "technical_indicator": 0.45, "simple_price_patterns": 0.5, "residual_beta": 0.35},
        "HIGH_TURNOVER": {"shock_response": 0.8, "vwap_dislocation": 0.4, "simple_price_patterns": 0.3, "residual_beta": 0.25},
        "LOW_TURNOVER": {"reversal_conditioned": 0.5, "pv_divergence": 0.4, "technical_indicator": 0.2, "simple_price_patterns": 0.25},
    }

    max_failure = max(failure_counts.values(), default=1)
    ranked = []
    for thesis in THESIS_LIBRARY:
        family_id = thesis["id"]
        family_stat = memory.get("family_stats", {}).get(family_id, default_family_stats())
        seed_context = memory.get("seed_context", {})
        thesis_style_tags = sorted({tag for variant in thesis["variants"] for tag in variant.get("style_tags", [])})
        family_seed_count = seed_context.get("family_counts", {}).get(family_id, 0)
        planned_seed_count = seed_context.get("planned_family_counts", {}).get(family_id, 0)
        attempts = family_stat.get("attempts", 0)
        completed = family_stat.get("completed", 0)
        pass_all_count = family_stat.get("pass_all_count", 0)
        serious_failures_count = family_stat.get("serious_failures", 0)
        real_attempts = int(family_stat.get("real_attempts", 0) or 0)
        real_completed = int(family_stat.get("real_completed", 0) or 0)
        real_pass_all_count = int(family_stat.get("real_pass_all_count", 0) or 0)
        real_fail_count = int(family_stat.get("real_fail_count", 0) or 0)
        real_fail_streak = int(family_stat.get("real_fail_streak", 0) or 0)
        style_alignment = score_style_alignment(thesis_style_tags, memory)

        failure_alignment = 0.0
        for failure_name, count in failure_counts.items():
            failure_alignment += (count / max_failure) * priority_map.get(failure_name, {}).get(family_id, 0.0)

        success_rate = (pass_all_count / completed) if completed else 0.0
        serious_fail_rate = min(1.0, (serious_failures_count / completed) if completed else 0.0)
        real_pass_rate = (real_pass_all_count / real_completed) if real_completed else 0.0
        real_fail_rate = min(1.0, (real_fail_count / real_completed) if real_completed else 0.0)
        avg_score = max(0.0, family_stat.get("avg_research_score", 0.0))
        exploration_bonus = 1.0 / (1.0 + attempts)
        fresh_family_bonus = 0.15 if family_seed_count == 0 else 0.0
        fresh_real_family_bonus = 0.08 if real_attempts == 0 else 0.0
        seed_bias_penalty = min(1.0, planned_seed_count * 0.45 + family_seed_count * 0.12)
        blocked_penalty = 1.0 if family_id in blocked_families else 0.0
        soft_block_penalty = 1.0 if family_id in soft_blocked_families else 0.0
        real_fail_streak_penalty = min(1.0, real_fail_streak / 3.0)

        thesis_score = round(
            0.28 * failure_alignment
            + 0.25 * avg_score
            + 0.18 * success_rate
            + 0.10 * real_pass_rate
            + 0.12 * exploration_bonus * exploration_weight_multiplier
            + 0.17 * style_alignment
            + fresh_family_bonus
            + fresh_real_family_bonus
            - 0.35 * serious_fail_rate
            - 0.30 * real_fail_rate
            - 0.38 * real_fail_streak_penalty
            - 0.30 * seed_bias_penalty
            - 0.22 * soft_block_penalty * soft_block_penalty_multiplier
            - 0.80 * blocked_penalty,
            4,
        )
        ranked.append(
            {
                **thesis,
                "thesis_score": thesis_score,
                "style_alignment_score": style_alignment,
                "family_stats": family_stat,
                "blocked": family_id in blocked_families,
                "soft_blocked": family_id in soft_blocked_families,
            }
        )

    ranked.sort(key=lambda item: (item["thesis_score"], item["id"]), reverse=True)
    return ranked


def build_candidate(
    thesis: dict,
    variant: dict,
    memory: dict,
    *,
    history_index: HistoryIndex | None = None,
) -> dict | None:
    token_program = variant["token_program"]
    if thesis.get("blocked"):
        return None
    try:
        expression = render_token_program(token_program)
        seed_ready = True
    except Exception:
        return None

    skeleton = skeletonize(expression)
    if skeleton in set(memory.get("blocked_skeletons", [])):
        return None
    soft_blocked_skeletons = set(memory.get("soft_blocked_skeletons", []))
    adaptive_controls = _adaptive_controls(memory)
    exploration_boost = _clip(
        _coerce_float(adaptive_controls.get("exploration_boost"), 0.0),
        0.0,
        ADAPTIVE_MAX_EXPLORATION_BOOST,
    )
    candidate_risk_penalty_multiplier = _clip(
        _coerce_float(adaptive_controls.get("candidate_risk_penalty_multiplier"), 1.0),
        ADAPTIVE_MIN_RISK_PENALTY_MULTIPLIER,
        1.0,
    )

    family_stats = memory.get("family_stats", {}).get(thesis["id"], default_family_stats())
    seed_context = memory.get("seed_context", {})
    historical_skeletons = set(memory.get("historical_skeletons", []))
    preferred_skeletons = set(memory.get("preferred_skeletons", []))
    seeded_skeletons = set(seed_context.get("seeded_skeletons", []))
    planned_skeletons = set(seed_context.get("planned_skeletons", []))
    style_alignment = score_style_alignment(variant.get("style_tags", []), memory)
    historical_seen = skeleton in historical_skeletons
    attempts = max(1, family_stats.get("attempts", 0))
    family_saturation = min(1.0, attempts / max(1, memory.get("window_rows", 1)))
    novelty_score = round(
        (0.65 if not historical_seen else 0.25)
        + (0.35 * (1.0 - family_saturation))
        + (0.1 if skeleton in preferred_skeletons else 0.0)
        - (0.25 if skeleton in seeded_skeletons else 0.0),
        4,
    )
    novelty_score = round(novelty_score + exploration_boost, 4)
    novelty_score = max(0.0, min(1.2, novelty_score))

    risk_tags = list(variant.get("risk_tags", []))
    failure_counts = family_stats.get("failure_counts", {})
    if failure_counts.get("HIGH_TURNOVER", 0) >= 2 and "turnover_risk" not in risk_tags:
        risk_tags.append("turnover_risk")
    if failure_counts.get("CONCENTRATED_WEIGHT", 0) >= 2 and "weight_risk" not in risk_tags:
        risk_tags.append("weight_risk")
    if historical_seen:
        risk_tags.append("similarity_risk")
    if skeleton in seeded_skeletons:
        risk_tags.append("seed_bias_risk")
    if skeleton in planned_skeletons:
        risk_tags.append("already_seeded")
    if thesis.get("blocked"):
        risk_tags.append("blocked_family_risk")
    if thesis.get("soft_blocked"):
        risk_tags.append("soft_blocked_family_risk")
    if skeleton in soft_blocked_skeletons:
        risk_tags.append("soft_blocked_skeleton_risk")
    if style_alignment < 0.22:
        risk_tags.append("unproven_style")

    seed_ready = (
        seed_ready
        and novelty_score >= 0.55
        and "similarity_risk" not in risk_tags
        and "blocked_family_risk" not in risk_tags
        and "already_seeded" not in risk_tags
    )
    candidate_score = round(
        thesis["thesis_score"] + novelty_score + (0.45 * style_alignment) - ((0.18 * candidate_risk_penalty_multiplier) * len(risk_tags)),
        4,
    )
    candidate = {
        "variant_id": variant["variant_id"],
        "thesis_id": thesis["id"],
        "thesis": thesis["label"],
        "why": thesis["why"],
        "expression": expression,
        "token_program": token_program,
        "novelty_score": novelty_score,
        "style_alignment_score": style_alignment,
        "risk_tags": sorted(set(risk_tags)),
        "seed_ready": seed_ready,
        "candidate_score": candidate_score,
        "skeleton": skeleton,
    }
    if history_index is not None:
        candidate = annotate_candidate_with_local_metrics(candidate, history_index=history_index)
    candidate.update(evaluate_quality_gate(candidate, thesis["id"], memory))
    candidate["lineage"] = ensure_candidate_lineage(
        candidate,
        stage_source="planner",
        default_hypothesis_id=thesis["id"],
        default_hypothesis_label=thesis["label"],
        default_family=thesis["id"],
        default_family_components=[thesis["id"]],
        default_generation_reason=thesis["why"],
    )
    return candidate


def _batch_candidate_sort_key(entry: dict) -> tuple:
    candidate = entry["candidate"]
    return (
        int(candidate.get("qualified", False)),
        int(candidate.get("seed_ready", False)),
        float(candidate.get("confidence_score", 0.0) or 0.0),
        float(candidate.get("candidate_score", 0.0) or 0.0),
        float(candidate.get("novelty_score", 0.0) or 0.0),
        float(candidate.get("style_alignment_score", 0.0) or 0.0),
        -int(entry.get("thesis_rank", 0) or 0),
    )


def _extend_batch_selection(
    selected: list[dict],
    seen_skeletons: set[str],
    pool: list[dict],
    *,
    max_candidates: int,
) -> int:
    added = 0
    for entry in sorted(pool, key=_batch_candidate_sort_key, reverse=True):
        if len(selected) >= max_candidates:
            break
        candidate = entry["candidate"]
        skeleton = str(candidate.get("skeleton") or "")
        if skeleton and skeleton in seen_skeletons:
            continue
        selected.append(candidate)
        if skeleton:
            seen_skeletons.add(skeleton)
        added += 1
    return added


def build_batch(
    memory: dict,
    max_candidates: int,
    *,
    include_local_metrics: bool = False,
    history_index: HistoryIndex | None = None,
) -> dict:
    adaptive_controls = _adaptive_controls(memory)
    ranked_theses = rank_theses(memory)
    selected = []
    notes = []
    seen_skeletons = set()
    primary_pool = []
    overflow_pool = []

    for thesis_rank, thesis in enumerate(ranked_theses):
        if thesis.get("blocked"):
            continue

        thesis_candidates = []
        thesis_seen_skeletons = set()
        for variant in thesis["variants"]:
            candidate = build_candidate(
                thesis,
                variant,
                memory,
                history_index=history_index if include_local_metrics else None,
            )
            if candidate is None:
                continue
            if candidate["skeleton"] in thesis_seen_skeletons:
                continue
            thesis_seen_skeletons.add(candidate["skeleton"])
            thesis_candidates.append(candidate)

        thesis_candidates.sort(
            key=lambda item: (
                int(item.get("qualified", False)),
                int(item.get("seed_ready", False)),
                item.get("confidence_score", 0.0),
                item["candidate_score"],
                item["novelty_score"],
                item.get("style_alignment_score", 0.0),
            ),
            reverse=True,
        )
        planned_family_count = memory.get("seed_context", {}).get("planned_family_counts", {}).get(thesis["id"], 0)
        total_seed_count = memory.get("seed_context", {}).get("family_counts", {}).get(thesis["id"], 0)
        thesis_limit = 1 if planned_family_count >= 1 or total_seed_count >= 3 else 2
        thesis_limit += max(0, _coerce_int(adaptive_controls.get("thesis_limit_bonus"), 0))
        for candidate in thesis_candidates[:thesis_limit]:
            primary_pool.append({"candidate": candidate, "thesis_rank": thesis_rank})
        for candidate in thesis_candidates[thesis_limit:]:
            overflow_pool.append({"candidate": candidate, "thesis_rank": thesis_rank})

    _extend_batch_selection(selected, seen_skeletons, primary_pool, max_candidates=max_candidates)
    overflow_added = 0
    if len(selected) < max_candidates and overflow_pool:
        overflow_added = _extend_batch_selection(selected, seen_skeletons, overflow_pool, max_candidates=max_candidates)
        if overflow_added:
            notes.append(
                f"Primary thesis caps underfilled the batch; added {overflow_added} overflow exploration candidate(s) from lower-priority slots."
            )
    selected.sort(key=lambda item: _batch_candidate_sort_key({"candidate": item}), reverse=True)

    if not selected:
        notes.append("No supported candidate survived the current blocklist. Relax memory filters or expand the template library.")

    lint = build_lint_summary([item["expression"] for item in selected]) if selected else {"summary": {}, "expressions": []}
    if lint["summary"].get("near_duplicate_groups"):
        notes.append("Batch still contains near-duplicate skeletons. Trim before heavy runs.")

    if memory.get("blocked_families"):
        notes.append(
            "Hard-blocked families this round: " + ", ".join(memory["blocked_families"])
        )
    if memory.get("soft_blocked_families"):
        notes.append("Soft-blocked families under watch: " + ", ".join(memory["soft_blocked_families"]))
    if adaptive_controls.get("active"):
        warning = str(adaptive_controls.get("warning") or "").strip()
        if warning:
            notes.insert(0, warning)
        if adaptive_controls.get("ignore_block_list"):
            notes.append("Manual override ignored planner block lists for this round.")
        reopened_families = adaptive_controls.get("reopened_soft_blocked_families", []) or []
        reopened_skeletons = adaptive_controls.get("reopened_soft_blocked_skeletons", []) or []
        if reopened_families:
            notes.append("Adaptive reopen for soft-blocked families: " + ", ".join(reopened_families))
        if reopened_skeletons:
            notes.append("Adaptive reopen for soft-blocked skeletons: " + ", ".join(reopened_skeletons[:4]))

    if memory.get("suggestions"):
        notes.extend(memory["suggestions"][:2])

    deduped_notes = []
    seen_notes = set()
    for note in notes:
        note = str(note).strip()
        if not note or note in seen_notes:
            continue
        seen_notes.add(note)
        deduped_notes.append(note)
    notes = deduped_notes

    qualified_count = sum(1 for item in selected if item.get("qualified"))
    watchlist_count = len(selected) - qualified_count
    if qualified_count == 0:
        notes.append("No candidate passed the strict quality gate. Review the watchlist carefully before promoting anything.")
    else:
        notes.append(f"Strict quality gate passed: {qualified_count} candidate(s). Watchlist only: {watchlist_count}.")

    if include_local_metrics:
        selected = [
            item if item.get("local_metrics") else annotate_candidate_with_local_metrics(item, history_index=history_index)
            for item in selected
        ]

    return {
        "candidates": selected,
        "qualified_count": qualified_count,
        "watchlist_count": watchlist_count,
        "lint_summary": lint.get("summary", {}),
        "notes": notes,
    }


def render_markdown(batch: dict) -> str:
    lines = ["# Next Alpha Batch", ""]
    if batch["notes"]:
        lines.append("## Notes")
        for note in batch["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    qualified_candidates = [item for item in batch["candidates"] if item.get("qualified")]
    watchlist_candidates = [item for item in batch["candidates"] if not item.get("qualified")]

    def render_candidate_section(title: str, items: list[dict]) -> None:
        if not items:
            return
        lines.append(title)
        for index, item in enumerate(items, start=1):
            lines.append(f"- #{index} [{item['thesis']}] {item['expression']}")
            lines.append(f"  why: {item['why']}")
            lines.append(f"  quality_label: {item['quality_label']}")
            lines.append(f"  confidence_score: {item['confidence_score']}")
            lines.append(f"  novelty_score: {item['novelty_score']}")
            lines.append(f"  style_alignment_score: {item['style_alignment_score']}")
            lines.append(f"  seed_ready: {item['seed_ready']}")
            if item.get("settings"):
                lines.append(f"  settings: {item['settings']}")
            local = item.get("local_metrics", {})
            if local:
                lines.append(f"  verdict: {local.get('verdict')} ({local.get('confidence')})")
                lines.append(f"  alpha_score: {local.get('alpha_score')}")
                lines.append(f"  sharpe: {local.get('sharpe')}")
                lines.append(f"  fitness: {local.get('fitness')}")
                lines.append(f"  turnover: {local.get('turnover')}")
            if item["risk_tags"]:
                lines.append(f"  risk_tags: {', '.join(item['risk_tags'])}")
        lines.append("")

    render_candidate_section("## Strictly Qualified Candidates", qualified_candidates)
    render_candidate_section("## Watchlist Candidates (Reference Only)", watchlist_candidates)

    lines.append("## Lint summary")
    summary = batch["lint_summary"]
    lines.append(f"- total_expressions: {summary.get('total_expressions', 0)}")
    lines.append(f"- exact_duplicate_groups: {len(summary.get('exact_duplicate_groups', []))}")
    lines.append(f"- near_duplicate_groups: {len(summary.get('near_duplicate_groups', []))}")
    family_counts = summary.get("family_counts", {})
    if family_counts:
        lines.append(f"- family_counts: {json.dumps(family_counts, ensure_ascii=True)}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan the next WorldQuant alpha batch.")
    parser.add_argument("csv_path", nargs="?", help="CSV path or directory with simulation results.")
    parser.add_argument("--memory", help="Optional prior research memory JSON.")
    parser.add_argument("--top", type=int, default=10, help="How many top rows to learn from.")
    parser.add_argument("--count", type=int, default=8, help="How many candidates to produce.")
    parser.add_argument("--history-window", type=int, default=120, help="How many recent rows to consider.")
    parser.add_argument("--seed-store", help="Optional seed store path. Defaults to initial-population.pkl.")
    parser.add_argument("--write-memory", help="Optional path to write refreshed memory JSON.")
    parser.add_argument("--write-batch", help="Optional path to write candidate expressions.")
    parser.add_argument("--write-plan", help="Optional path to write the full next-batch JSON payload.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    try:
        csv_path = discover_csv(args.csv_path)
        rows = read_rows(csv_path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        return 1

    seed_context = build_seed_context(load_seed_store(args.seed_store))
    memory = build_memory(rows, args.top, history_window=args.history_window, seed_context=seed_context)
    memory = merge_memory(memory, load_memory(args.memory))
    history_index = HistoryIndex.from_csv(csv_path)

    if args.write_memory:
        atomic_write_json(Path(args.write_memory), memory)

    batch = build_batch(memory, args.count, include_local_metrics=True, history_index=history_index)

    if args.write_batch:
        output = Path(args.write_batch)
        approved_expressions = [item["expression"] for item in batch["candidates"] if item.get("qualified")]
        output.write_text(
            "\n".join(approved_expressions) + ("\n" if approved_expressions else ""),
            encoding="utf-8",
        )

    payload = {
        "memory": memory,
        "batch": batch,
        "source_csv": csv_path,
    }

    if args.write_plan:
        atomic_write_json(Path(args.write_plan), payload)

    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(render_markdown(batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
