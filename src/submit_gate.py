"""Shared submit-ready gates for planner, reports, and seed approval."""

from __future__ import annotations

SUBMIT_READY_VERDICTS = {"PASS", "LIKELY_PASS"}
SUBMIT_READY_MIN_ALPHA_SCORE = 65.0
SUBMIT_READY_MIN_SHARPE = 1.4
SUBMIT_READY_MIN_FITNESS = 1.0
SUBMIT_READY_MIN_CONFIDENCE = 0.45


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def local_metrics_pass_submit_gate(local_metrics: dict | None) -> bool:
    local = local_metrics or {}
    if str(local.get("verdict") or "").upper() not in SUBMIT_READY_VERDICTS:
        return False
    if to_float(local.get("alpha_score")) < SUBMIT_READY_MIN_ALPHA_SCORE:
        return False
    if to_float(local.get("sharpe")) < SUBMIT_READY_MIN_SHARPE:
        return False
    if to_float(local.get("fitness")) < SUBMIT_READY_MIN_FITNESS:
        return False
    return True


def submit_gate_fail_reasons(candidate: dict) -> list[str]:
    reasons = []
    if not candidate.get("seed_ready"):
        reasons.append("seed_ready=false")
    if not candidate.get("qualified"):
        reasons.append("qualified=false")
    if candidate.get("quality_label") != "qualified":
        reasons.append("quality_label!=qualified")
    if to_float(candidate.get("confidence_score")) < SUBMIT_READY_MIN_CONFIDENCE:
        reasons.append(f"confidence<{SUBMIT_READY_MIN_CONFIDENCE}")

    local = candidate.get("local_metrics") or {}
    if not local:
        reasons.append("missing_local_metrics")
        return reasons

    verdict = str(local.get("verdict") or "").upper()
    if verdict not in SUBMIT_READY_VERDICTS:
        reasons.append(f"verdict={verdict or 'n/a'}")
    if to_float(local.get("alpha_score")) < SUBMIT_READY_MIN_ALPHA_SCORE:
        reasons.append(f"alpha_score<{SUBMIT_READY_MIN_ALPHA_SCORE}")
    if to_float(local.get("sharpe")) < SUBMIT_READY_MIN_SHARPE:
        reasons.append(f"sharpe<{SUBMIT_READY_MIN_SHARPE}")
    if to_float(local.get("fitness")) < SUBMIT_READY_MIN_FITNESS:
        reasons.append(f"fitness<{SUBMIT_READY_MIN_FITNESS}")
    return reasons


def candidate_passes_submit_gate(candidate: dict) -> bool:
    return not submit_gate_fail_reasons(candidate)
