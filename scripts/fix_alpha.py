#!/usr/bin/env python3
"""Diagnose a failing alpha and suggest concrete repair directions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from .plan_next_batch import THESIS_INDEX, expression_style_tags
    from .results_digest import (
        CHECK_COLUMNS,
        classify_expression_family,
        coerce_float,
        discover_csv,
        failing_checks,
        normalize_expression,
        read_rows,
    )
except ImportError:
    from plan_next_batch import THESIS_INDEX, expression_style_tags
    from results_digest import (
        CHECK_COLUMNS,
        classify_expression_family,
        coerce_float,
        discover_csv,
        failing_checks,
        normalize_expression,
        read_rows,
    )

from scripts.flow_utils import atomic_write_json, load_json as load_artifact_json
from scripts.lineage_utils import ensure_candidate_lineage
from src.program_tokens import render_token_program
from src.internal_scoring import HistoryIndex, format_scoring_settings, parse_scoring_settings, score_expression, score_expressions_batch
from src.program_tokens import validate_token_program
from src.submit_gate import (
    SUBMIT_READY_MIN_ALPHA_SCORE,
    SUBMIT_READY_MIN_FITNESS,
    SUBMIT_READY_MIN_SHARPE,
    SUBMIT_READY_VERDICTS,
    local_metrics_pass_submit_gate,
)

PROMISING_ALPHA_BUFFER = 7.0
PROMISING_SHARPE_BUFFER = 0.2
PROMISING_FITNESS_BUFFER = 0.15
DEFAULT_AUTO_FIX_CANDIDATES = Path("artifacts/auto_fix_candidates.json")


def find_row_by_alpha_id(alpha_id: str, csv_path: str | None = None) -> dict | None:
    path = discover_csv(csv_path)
    rows = read_rows(path)
    for row in reversed(rows):
        if str(row.get("alpha_id", "")).strip() == alpha_id:
            return row
    return None


def _extract_context_settings(row: dict | None) -> dict:
    if not row:
        return {}
    settings = {}
    for field in ("region", "universe", "delay", "decay", "neutralization", "truncation"):
        value = row.get(field)
        if value not in {None, ""}:
            settings[field] = value
    return settings


def _resolve_history_csv(csv_path: str | None = None) -> str | None:
    try:
        return str(discover_csv(csv_path))
    except FileNotFoundError:
        return csv_path or None


def choose_target_families(family: str, failures: list[str]) -> list[str]:
    ordered = []

    def add(name: str) -> None:
        if name in THESIS_INDEX and name not in ordered:
            ordered.append(name)

    add(family)

    failure_set = set(failures)
    if "MATCHES_COMPETITION" in failure_set:
        add("technical_indicator")
        add("shock_response")
        add("residual_beta")
    if "SELF_CORRELATION" in failure_set:
        add("residual_beta")
        add("technical_indicator")
    if "CONCENTRATED_WEIGHT" in failure_set:
        add("vwap_dislocation")
        add("technical_indicator")
    if "HIGH_TURNOVER" in failure_set:
        add("residual_beta")
        add("vwap_dislocation")
    if "LOW_TURNOVER" in failure_set:
        add("pv_divergence")
        add("technical_indicator")
    if {"LOW_SHARPE", "LOW_FITNESS"}.issubset(failure_set):
        add("technical_indicator")
        add("residual_beta")
    elif "LOW_SHARPE" in failure_set:
        add("technical_indicator")
    elif "LOW_FITNESS" in failure_set:
        add("vwap_dislocation")

    if not ordered:
        add("technical_indicator")
    return ordered[:3]


def build_issue_notes(expression: str, failures: list[str], family: str) -> tuple[list[str], list[str]]:
    normalized = normalize_expression(expression).lower()
    issue_notes = []
    repair_steps = []

    if "LOW_SHARPE" in failures:
        issue_notes.append("Sharpe thấp: alpha đang có tín hiệu yếu hoặc quá nhiễu.")
        repair_steps.append("Thêm normalization mạnh hơn như `rank`, `zscore`, `winsorize`, hoặc chuyển sang thesis khác thay vì chỉ chỉnh window.")
        repair_steps.append("Thử blend motif ít tương quan hơn, ví dụ momentum + reversal hoặc residual + volume, thay vì chỉ đẩy cùng một style mạnh hơn.")

    if "LOW_FITNESS" in failures:
        issue_notes.append("Fitness thấp: payoff chưa đủ sạch hoặc phân bố tín hiệu chưa ổn.")
        repair_steps.append("Giảm cấu trúc quá gắt, tránh mẫu số mong manh, và ưu tiên công thức ổn định hơn với `std`, `adv`, hoặc rank-based spread.")
        repair_steps.append("So sánh trực tiếp với một biến thể đơn giản hơn; nếu bản đơn giản ổn định hơn thì rollback thay vì chồng thêm operator.")

    if "HIGH_TURNOVER" in failures:
        issue_notes.append("Turnover cao: alpha đang phản ứng quá nhanh.")
        repair_steps.append("Làm chậm tín hiệu bằng window dài hơn như 21/63, thêm smoothing, hoặc dùng biến nền ổn định hơn thay vì delta ngắn.")
        repair_steps.append("Ưu tiên `ts_mean`, `ts_rank`, `ts_zscore`, rồi mới rank ngoài cùng để giữ structure nhưng giảm nhảy tín hiệu.")

    if "LOW_TURNOVER" in failures:
        issue_notes.append("Turnover thấp: alpha quá ì.")
        repair_steps.append("Rút ngắn horizon, thêm thành phần `ts_delta`, `ts_rank`, hoặc trigger phản ứng nhanh hơn.")

    if "CONCENTRATED_WEIGHT" in failures:
        issue_notes.append("Trọng số tập trung: alpha có cực trị quá mạnh hoặc denominator thiếu ổn định.")
        repair_steps.append("Bọc tín hiệu bằng `rank`, `zscore`, `winsorize`, và tránh `inverse(...)` trên đại lượng có thể rất nhỏ.")

    if "SELF_CORRELATION" in failures:
        issue_notes.append("Self-correlation cao: alpha đang quá giống những gì bạn đã thử trước đó.")
        repair_steps.append("Đổi family hoặc thêm phần residual/de-beta như `ts_regression`, `beta_last_60_days_spy`, hoặc risk spread.")
        repair_steps.append("Đổi skeleton thật sự hoặc phối với family khác thay vì chỉ đổi tham số lookback.")

    if "MATCHES_COMPETITION" in failures:
        issue_notes.append("Match competition: cần đổi skeleton, không nên chỉ tinh chỉnh tham số.")
        repair_steps.append("Rời khỏi family hiện tại và viết lại thesis theo một cấu trúc khác hẳn.")

    if "LOW_SUB_UNIVERSE_SHARPE" in failures:
        issue_notes.append("Sub-universe Sharpe thấp: tín hiệu không bền ở phân đoạn khó hơn.")
        repair_steps.append("Ưu tiên liquidity/volatility conditioning và tránh expression quá phụ thuộc vài cổ phiếu cực trị.")

    if "inverse(" in normalized and "CONCENTRATED_WEIGHT" not in failures:
        repair_steps.append("Alpha có `inverse(...)`, nên kiểm tra denominator có thể gây weight concentration không.")

    if "ts_delta(" in normalized or "returns" in normalized:
        repair_steps.append("Alpha dùng tín hiệu ngắn hạn; nếu kết quả nhiễu, thử thay bằng `ts_rank(...,21)` hoặc `ts_zscore(...,21)`.")

    if family == "technical_indicator":
        repair_steps.append("Family kỹ thuật thường hợp hơn khi có normalization ngoài cùng như `rank(ts_zscore(...))` thay vì raw oscillator.")
    elif family == "residual_beta":
        repair_steps.append("Family residual nên giữ mục tiêu khử beta/risk rõ ràng, tránh trộn thêm quá nhiều reversal ngắn hạn.")
    elif family == "pv_divergence":
        repair_steps.append("Family price-volume divergence thường khỏe hơn khi dùng `rank` hoặc `winsorize` để giảm outlier.")

    return issue_notes, list(dict.fromkeys(repair_steps))


def build_experiment_plan(context: dict) -> list[str]:
    failures = set(context["failures"])
    style_tags = set(context["style_tags"])
    plan = [
        "Giữ một baseline cố định, chỉ sửa một điểm mỗi lần rồi rescore để tránh overfitting và khó truy nguyên tác động.",
        "Chạy cùng một thesis qua các horizon 5/10/21/63 để xem edge nằm ở nhịp nào thay vì khóa sớm vào một window.",
        "So sánh raw signal với các lớp bọc `rank`, `zscore`, `winsorize`, và `rank(ts_zscore(...))` để tăng IC sạch hơn.",
    ]

    if "HIGH_TURNOVER" in failures:
        plan.append("Tạo thêm một nhánh chậm hơn bằng `ts_mean`, `ts_rank`, hoặc `ts_zscore` trước khi cân nhắc operator mới.")
    if "LOW_TURNOVER" in failures:
        plan.append("Tạo thêm một nhánh nhanh hơn với lookback ngắn hơn hoặc trigger kiểu `ts_delta` để kiểm tra alpha có đang quá ì không.")
    if {"SELF_CORRELATION", "MATCHES_COMPETITION"} & failures:
        plan.append("Viết lại một bản khác family hoặc khác skeleton hoàn toàn; chỉ tune tham số sẽ khó cải thiện uniqueness.")
    if {"momentum", "reversal"} & style_tags:
        plan.append("Thử blend 0.6 alpha hiện tại với 0.4 residual/volume để giảm phụ thuộc vào một motif duy nhất.")
    elif {"technical", "vwap"} & style_tags:
        plan.append("Thử blend alpha hiện tại với một nhánh residual hoặc volume để tăng độ độc nhất mà không phá thesis gốc.")
    if "normalization" not in style_tags:
        plan.append("Thêm normalization ngoài cùng nếu chưa có, vì nhiều alpha fail do phân bố weight quá gắt chứ không hẳn do thesis sai.")

    plan.append("Giữ kết quả out-of-sample riêng. Nếu bản đơn giản hơn thắng trên giai đoạn holdout thì giữ bản đơn giản.")
    return list(dict.fromkeys(plan))[:7]


def choose_variant_priority(variant: dict, failures: list[str]) -> tuple[float, str]:
    style_tags = set(variant.get("style_tags", []))
    risk_tags = set(variant.get("risk_tags", []))
    failure_set = set(failures)
    score = 0.0

    if "LOW_SHARPE" in failure_set:
        if {"technical", "momentum"} & style_tags:
            score += 1.0
        if "winsorize" in style_tags or "normalization" in style_tags:
            score += 0.5
    if "LOW_FITNESS" in failure_set:
        if "trend" in style_tags or "cross_sectional" in style_tags:
            score += 0.8
    if "CONCENTRATED_WEIGHT" in failure_set:
        if "winsorize" in style_tags or "rank" in style_tags:
            score += 1.0
    if "HIGH_TURNOVER" in failure_set and "turnover_risk" not in risk_tags:
        score += 0.8
    if "SELF_CORRELATION" in failure_set and ("residual" in style_tags or "beta" in style_tags):
        score += 0.8
    if "MATCHES_COMPETITION" in failure_set and "cross_sectional" in style_tags:
        score += 0.6
    return (score, variant.get("variant_id", ""))


def render_rewrite_examples(target_families: list[str], current_expression: str, failures: list[str]) -> list[dict]:
    current_norm = normalize_expression(current_expression)
    examples = []
    for family_id in target_families:
        thesis = THESIS_INDEX.get(family_id)
        if not thesis:
            continue
        ranked_variants = sorted(
            thesis.get("variants", []),
            key=lambda variant: choose_variant_priority(variant, failures),
            reverse=True,
        )
        for variant in ranked_variants:
            try:
                expression = render_token_program(variant["token_program"])
            except Exception:
                continue
            if normalize_expression(expression) == current_norm:
                continue
            examples.append(
                {
                    "family": thesis["label"],
                    "why": thesis["why"],
                    "expression": expression,
                }
            )
            break
    return examples[:3]


def build_rewrite_candidates(
    context: dict,
    *,
    max_variants_per_family: int = 2,
    max_total: int = 8,
) -> list[dict]:
    current_expression = context["expression"]
    failures = context["failures"]
    current_norm = normalize_expression(current_expression)
    candidates = []
    seen = {current_norm}

    for family_id in choose_target_families(context["family"], failures):
        thesis = THESIS_INDEX.get(family_id)
        if not thesis:
            continue
        ranked_variants = sorted(
            thesis.get("variants", []),
            key=lambda variant: choose_variant_priority(variant, failures),
            reverse=True,
        )
        added_for_family = 0
        for variant in ranked_variants:
            try:
                expression = render_token_program(variant["token_program"])
            except Exception:
                continue
            normalized = normalize_expression(expression)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                {
                    "family_id": family_id,
                    "family": thesis["label"],
                    "why": thesis["why"],
                    "variant_id": variant.get("variant_id", ""),
                    "expression": expression,
                    "style_tags": list(variant.get("style_tags", [])),
                    "token_program": list(variant.get("token_program", [])),
                }
            )
            added_for_family += 1
            if added_for_family >= max(1, max_variants_per_family):
                break
            if len(candidates) >= max_total:
                return candidates[:max_total]
    return candidates[:max_total]


def _build_history_index(csv_path: str | None = None) -> HistoryIndex:
    if not csv_path:
        return HistoryIndex()
    try:
        return HistoryIndex.from_csv(csv_path)
    except FileNotFoundError:
        return HistoryIndex()


def _submit_gap(result: dict) -> float:
    verdict = str(result.get("verdict") or "").upper()
    alpha_score = coerce_float(result.get("alpha_score"))
    sharpe = coerce_float(result.get("sharpe"))
    fitness = coerce_float(result.get("fitness"))
    gap = 0.0
    if verdict not in SUBMIT_READY_VERDICTS:
        gap += 1.0
    gap += max(0.0, SUBMIT_READY_MIN_ALPHA_SCORE - alpha_score) / 7.0
    gap += max(0.0, SUBMIT_READY_MIN_SHARPE - sharpe) / 0.2
    gap += max(0.0, SUBMIT_READY_MIN_FITNESS - fitness) / 0.15
    return round(gap, 4)


def _rewrite_status(result: dict) -> str:
    if local_metrics_pass_submit_gate(result):
        return "submit_ready"

    verdict = str(result.get("verdict") or "").upper()
    alpha_score = coerce_float(result.get("alpha_score"))
    sharpe = coerce_float(result.get("sharpe"))
    fitness = coerce_float(result.get("fitness"))
    if (
        verdict in SUBMIT_READY_VERDICTS
        and alpha_score >= SUBMIT_READY_MIN_ALPHA_SCORE - PROMISING_ALPHA_BUFFER
        and sharpe >= SUBMIT_READY_MIN_SHARPE - PROMISING_SHARPE_BUFFER
        and fitness >= SUBMIT_READY_MIN_FITNESS - PROMISING_FITNESS_BUFFER
    ):
        return "promising"
    return "weak"


def _planner_confidence_score(result: dict, repair_status: str) -> float:
    confidence_label = str(result.get("confidence") or "").upper()
    label_score = {
        "HIGH": 0.78,
        "MEDIUM": 0.62,
        "LOW": 0.38,
    }.get(confidence_label, 0.32)
    quality_proxy = coerce_float(result.get("quality_proxy"))
    stability_proxy = coerce_float(result.get("stability_proxy"))
    blended = max(label_score * 0.55 + quality_proxy * 0.3 + stability_proxy * 0.15, 0.0)
    if repair_status == "submit_ready":
        blended = max(blended, 0.45)
    return round(min(0.95, blended), 4)


def _risk_tags_from_result(result: dict, repair_status: str) -> list[str]:
    risk_tags = []
    if str(result.get("HIGH_TURNOVER") or "").upper() == "FAIL":
        risk_tags.append("turnover_risk")
    if str(result.get("CONCENTRATED_WEIGHT") or "").upper() == "FAIL":
        risk_tags.append("weight_risk")
    if repair_status == "promising":
        risk_tags.append("unproven_style")
    return risk_tags


def _normalize_auto_fix_candidate(candidate: dict) -> dict:
    compiled_expression = candidate.get("compiled_expression") or candidate.get("expression")
    return {
        **candidate,
        "compiled_expression": compiled_expression,
    }


def build_actionable_auto_fix_candidates(context: dict, auto_fix_payload: dict) -> list[dict]:
    actionable = []
    source_label = context.get("alpha_id") or context.get("expression")
    for item in auto_fix_payload.get("candidates", []):
        repair_status = item.get("repair_status")
        if repair_status not in {"submit_ready", "promising"}:
            continue
        result = item["result"]
        token_program = item.get("token_program") or []
        compiled_expression = item.get("expression")
        if token_program:
            try:
                compiled_expression = str(validate_token_program(token_program))
            except Exception:
                compiled_expression = item.get("expression")
        candidate = {
            "thesis": f"Auto-fix rewrite [{item['family']}]",
            "thesis_id": item.get("family_id"),
            "why": (
                f"Auto-fixed from {source_label}; {item['why']} "
                f"(repair_status={repair_status}, submit_gap={item.get('submit_gap')})."
            ),
            "expression": item["expression"],
            "compiled_expression": compiled_expression,
            "candidate_score": round(coerce_float(result.get("alpha_score")) / 100.0, 4),
            "confidence_score": _planner_confidence_score(result, repair_status),
            "novelty_score": round(coerce_float(result.get("uniqueness_proxy")), 4),
            "style_alignment_score": round(coerce_float(result.get("stability_proxy")), 4),
            "risk_tags": _risk_tags_from_result(result, repair_status),
            "seed_ready": repair_status in {"submit_ready", "promising"},
            "qualified": repair_status == "submit_ready",
            "quality_label": "qualified" if repair_status == "submit_ready" else "watchlist",
            "settings": result.get("settings", {}).get("label"),
            "local_metrics": result,
            "token_program": token_program,
            "repair_status": repair_status,
            "submit_gap": item.get("submit_gap"),
            "source": "auto_fix_rewrite",
            "source_alpha_id": context.get("alpha_id"),
            "source_expression": context.get("expression"),
            "improvement": item.get("improvement", {}),
        }
        candidate["lineage"] = ensure_candidate_lineage(
            candidate,
            stage_source="auto_fix_rewrite",
            default_parent_expression=context.get("expression"),
            default_parent_alpha_id=context.get("alpha_id"),
            default_parent_source="prior_candidate",
            default_hypothesis_id=item.get("family_id"),
            default_hypothesis_label=item.get("family"),
            default_family=item.get("family_id"),
            default_family_components=[item.get("family_id")] if item.get("family_id") else [],
            default_generation_reason=f"repair_status={repair_status}",
        )
        actionable.append(_normalize_auto_fix_candidate(candidate))
    return actionable


def load_auto_fix_store(path: str | Path | None) -> dict:
    if not path:
        return {"generated_at": "", "candidates": []}
    file_path = Path(path)
    if not file_path.exists():
        return {"generated_at": "", "candidates": []}
    payload = load_artifact_json(
        file_path,
        default={"generated_at": "", "candidates": []},
        context=f"auto-fix candidate store {file_path}",
        warn=lambda message: print(f"[fix-alpha] Warning: {message}", file=sys.stderr),
    )
    if not isinstance(payload, dict):
        return {"generated_at": "", "candidates": []}
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        payload["candidates"] = []
    return payload


def merge_auto_fix_candidates(existing: dict, new_candidates: list[dict]) -> dict:
    merged = {}
    for candidate in existing.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        expression = candidate.get("compiled_expression") or candidate.get("expression")
        if not expression:
            continue
        merged[expression] = candidate

    for candidate in new_candidates:
        expression = candidate.get("compiled_expression") or candidate.get("expression")
        if not expression:
            continue
        current = merged.get(expression)
        if current is None:
            merged[expression] = candidate
            continue
        current_score = coerce_float((current.get("local_metrics") or {}).get("alpha_score"))
        new_score = coerce_float((candidate.get("local_metrics") or {}).get("alpha_score"))
        if new_score >= current_score:
            merged[expression] = candidate

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidates": sorted(
            merged.values(),
            key=lambda item: (
                2 if item.get("repair_status") == "submit_ready" else 1 if item.get("repair_status") == "promising" else 0,
                coerce_float((item.get("local_metrics") or {}).get("alpha_score")),
                coerce_float((item.get("local_metrics") or {}).get("sharpe")),
                coerce_float((item.get("local_metrics") or {}).get("fitness")),
            ),
            reverse=True,
        ),
    }


def build_auto_fix_payload(
    context: dict,
    *,
    csv_path: str | None = None,
    settings: str | dict | None = None,
    top_rewrites: int = 5,
    max_variants_per_family: int = 2,
) -> dict:
    resolved_settings = parse_scoring_settings(settings or context.get("settings"))
    history_index = _build_history_index(csv_path or context.get("resolved_csv"))
    baseline = score_expression(
        context["expression"],
        history_index=history_index,
        settings=resolved_settings,
    )
    rewrites = build_rewrite_candidates(
        context,
        max_variants_per_family=max_variants_per_family,
        max_total=max(top_rewrites * 2, 6),
    )

    baseline_alpha = coerce_float(baseline.get("alpha_score"))
    baseline_sharpe = coerce_float(baseline.get("sharpe"))
    baseline_fitness = coerce_float(baseline.get("fitness"))
    ranked_candidates = []

    scored_rewrites: list[tuple[dict, dict]] = []
    if rewrites:
        try:
            rewrite_results = score_expressions_batch(
                [str(rewrite.get("expression") or "") for rewrite in rewrites],
                history_index=history_index,
                settings_list=[resolved_settings] * len(rewrites),
            )
            scored_rewrites = list(zip(rewrites, rewrite_results))
        except Exception:
            for rewrite in rewrites:
                try:
                    result = score_expression(
                        rewrite["expression"],
                        history_index=history_index,
                        settings=resolved_settings,
                    )
                except Exception:
                    continue
                scored_rewrites.append((rewrite, result))

    for rewrite, result in scored_rewrites:

        improvement = {
            "alpha_score_delta": round(coerce_float(result.get("alpha_score")) - baseline_alpha, 1),
            "sharpe_delta": round(coerce_float(result.get("sharpe")) - baseline_sharpe, 2),
            "fitness_delta": round(coerce_float(result.get("fitness")) - baseline_fitness, 2),
        }
        status = _rewrite_status(result)
        ranked_candidates.append(
            {
                **rewrite,
                "result": result,
                "repair_status": status,
                "submit_gap": _submit_gap(result),
                "improvement": improvement,
            }
        )

    ranked_candidates.sort(
        key=lambda item: (
            2 if item["repair_status"] == "submit_ready" else 1 if item["repair_status"] == "promising" else 0,
            coerce_float(item["result"].get("alpha_score")),
            coerce_float(item["result"].get("sharpe")),
            coerce_float(item["result"].get("fitness")),
            item["improvement"]["alpha_score_delta"],
            item["improvement"]["sharpe_delta"],
            item["improvement"]["fitness_delta"],
        ),
        reverse=True,
    )

    return {
        "settings_label": format_scoring_settings(resolved_settings),
        "baseline": baseline,
        "candidates": ranked_candidates[: max(1, top_rewrites)],
    }


def render_fix_report(context: dict, auto_fix_payload: dict | None = None) -> str:
    expression = context["expression"]
    failures = context["failures"]
    family = context["family"]
    style_tags = sorted(context["style_tags"])
    issue_notes, repair_steps = build_issue_notes(expression, failures, family)
    target_families = choose_target_families(family, failures)
    rewrite_examples = render_rewrite_examples(target_families, expression, failures)

    lines = ["# Alpha Fix Guide", ""]
    lines.append("## Input")
    if context.get("alpha_id"):
        lines.append(f"- alpha_id: {context['alpha_id']}")
    lines.append(f"- family: {family}")
    lines.append(f"- expression: {expression}")
    lines.append(f"- failures: {', '.join(failures) if failures else 'none provided'}")
    lines.append(f"- style_tags: {', '.join(style_tags) if style_tags else 'none'}")
    if context.get("sharpe") is not None:
        lines.append(f"- sharpe: {context['sharpe']}")
    if context.get("fitness") is not None:
        lines.append(f"- fitness: {context['fitness']}")
    if context.get("turnover") is not None:
        lines.append(f"- turnover: {context['turnover']}")
    lines.append("")

    if issue_notes:
        lines.append("## Diagnosis")
        for item in issue_notes:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## How To Fix")
    if repair_steps:
        for item in repair_steps[:6]:
            lines.append(f"- {item}")
    else:
        lines.append("- Chưa có lỗi cụ thể. Hãy truyền các check fail hoặc alpha_id từ CSV để chẩn đoán chính xác hơn.")
    lines.append("")

    if rewrite_examples:
        lines.append("## Rewrite Directions")
        for item in rewrite_examples:
            lines.append(f"- [{item['family']}] {item['expression']}")
            lines.append(f"  why: {item['why']}")
        lines.append("")

    if auto_fix_payload is not None:
        baseline = auto_fix_payload["baseline"]
        baseline_label = (
            f"{baseline.get('verdict')} | alpha_score={baseline.get('alpha_score')} | "
            f"sharpe={baseline.get('sharpe')} | fitness={baseline.get('fitness')}"
        )
        lines.append("## Auto Rewrite Scoreboard")
        lines.append(f"- scoring_settings: {auto_fix_payload.get('settings_label')}")
        lines.append(f"- baseline: {baseline_label}")
        candidates = auto_fix_payload.get("candidates", [])
        actionable = [item for item in candidates if item.get("repair_status") in {"submit_ready", "promising"}]
        if not candidates:
            lines.append("- Không tạo được rewrite candidate hợp lệ nào để score local.")
            lines.append("")
        else:
            lines.append(f"- actionable_rewrites: {len(actionable)}")
            if auto_fix_payload.get("candidate_store_path"):
                lines.append(f"- candidate_store: {auto_fix_payload.get('candidate_store_path')}")
            lines.append("")
            for index, item in enumerate(candidates, start=1):
                result = item["result"]
                improvement = item["improvement"]
                lines.append(f"### Candidate {index}: {item['repair_status']}")
                lines.append(f"- family: {item['family']}")
                lines.append(f"- variant_id: {item.get('variant_id') or 'n/a'}")
                lines.append(f"- expression: {item['expression']}")
                lines.append(
                    f"- alpha_score: {result.get('alpha_score')} ({improvement['alpha_score_delta']:+.1f} vs baseline)"
                )
                lines.append(
                    f"- sharpe: {result.get('sharpe')} ({improvement['sharpe_delta']:+.2f} vs baseline)"
                )
                lines.append(
                    f"- fitness: {result.get('fitness')} ({improvement['fitness_delta']:+.2f} vs baseline)"
                )
                lines.append(f"- submit_gap: {item.get('submit_gap')}")
                lines.append(f"- why: {item['why']}")
                lines.append("")

    experiment_plan = build_experiment_plan(context)
    if experiment_plan:
        lines.append("## Optimization Loop")
        for item in experiment_plan:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Usage")
    lines.append("- `bash run_wsl.sh fix --alpha-id YOUR_ALPHA_ID`")
    lines.append("- `bash run_wsl.sh fix --expression \"rank(...)\" --errors LOW_SHARPE LOW_FITNESS`")
    lines.append("- `bash run_wsl.sh fix --alpha-id YOUR_ALPHA_ID --auto-rewrite --top-rewrites 5`")
    return "\n".join(lines)


def build_context(args) -> dict:
    row = None
    if args.alpha_id:
        row = find_row_by_alpha_id(args.alpha_id, args.csv)
        if row is None:
            raise ValueError(f"Could not find alpha_id '{args.alpha_id}' in the simulation CSV.")

    expression = args.expression or (row.get("regular_code") if row else None)
    if not expression:
        raise ValueError("You must provide --expression or --alpha-id.")

    failures = [item.upper() for item in args.errors] if args.errors else []
    if not failures and row:
        failures = failing_checks(row)

    return {
        "alpha_id": args.alpha_id or (row.get("alpha_id") if row else None),
        "expression": expression,
        "failures": failures,
        "family": classify_expression_family(expression),
        "style_tags": expression_style_tags(expression),
        "sharpe": coerce_float(args.sharpe) if args.sharpe is not None else coerce_float(row.get("sharpe") if row else None),
        "fitness": coerce_float(args.fitness) if args.fitness is not None else coerce_float(row.get("fitness") if row else None),
        "turnover": coerce_float(args.turnover) if args.turnover is not None else coerce_float(row.get("turnover") if row else None),
        "settings": _extract_context_settings(row),
        "resolved_csv": _resolve_history_csv(args.csv),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose a failing alpha and suggest repair directions.")
    parser.add_argument("--alpha-id", help="Look up a submitted alpha from the simulation CSV.")
    parser.add_argument("--csv", help="Optional CSV path or repo directory.")
    parser.add_argument("--expression", help="Alpha expression to diagnose directly.")
    parser.add_argument("--errors", nargs="*", default=[], choices=CHECK_COLUMNS, help="Failing checks to diagnose.")
    parser.add_argument("--sharpe", help="Optional Sharpe value to include in the report.")
    parser.add_argument("--fitness", help="Optional fitness value to include in the report.")
    parser.add_argument("--turnover", help="Optional turnover value to include in the report.")
    parser.add_argument("--settings", help="Optional scoring settings string for auto-rewrite rescoring.")
    parser.add_argument("--auto-rewrite", action="store_true", help="Generate and locally rescore rewrite candidates for the diagnosed alpha.")
    parser.add_argument("--top-rewrites", type=int, default=5, help="Maximum number of scored rewrite candidates to show when --auto-rewrite is enabled.")
    parser.add_argument("--write-candidates", default=str(DEFAULT_AUTO_FIX_CANDIDATES), help="JSON artifact used to store actionable auto-fix candidates for daily/feed/seed flows.")
    parser.add_argument("--output", help="Optional output markdown path.")
    args = parser.parse_args()

    try:
        context = build_context(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    auto_fix_payload = None
    if args.auto_rewrite:
        auto_fix_payload = build_auto_fix_payload(
            context,
            csv_path=context.get("resolved_csv"),
            settings=args.settings or context.get("settings"),
            top_rewrites=args.top_rewrites,
        )
        actionable_candidates = build_actionable_auto_fix_candidates(context, auto_fix_payload)
        auto_fix_store_path = Path(args.write_candidates)
        existing_store = load_auto_fix_store(auto_fix_store_path)
        merged_store = merge_auto_fix_candidates(existing_store, actionable_candidates)
        auto_fix_store_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(auto_fix_store_path, merged_store)
        auto_fix_payload["actionable_count"] = len(actionable_candidates)
        auto_fix_payload["candidate_store_path"] = str(auto_fix_store_path)

    markdown = render_fix_report(context, auto_fix_payload=auto_fix_payload)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
