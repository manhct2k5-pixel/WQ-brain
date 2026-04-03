"""Shared submit-ready gates for planner, reports, and seed approval."""

from __future__ import annotations

SUBMIT_READY_VERDICTS = {"PASS", "LIKELY_PASS"}
SUBMIT_READY_MIN_ALPHA_SCORE = 65.0
SUBMIT_READY_MIN_SHARPE = 1.4
SUBMIT_READY_MIN_FITNESS = 1.0
SUBMIT_READY_MIN_CONFIDENCE = 0.45
SURROGATE_VERIFY_FIRST_MIN_ALPHA_SCORE = 35.0
SURROGATE_VERIFY_FIRST_MIN_SHARPE = 1.15
SURROGATE_VERIFY_FIRST_MIN_FITNESS = 0.9
SURROGATE_VERIFY_FIRST_MAX_PENALTY = 20.0


def to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_surrogate_shadow_gate(local_metrics: dict | None) -> dict:
    local = local_metrics or {}
    shadow = local.get("surrogate_shadow") or {}
    status = str(local.get("surrogate_shadow_status") or shadow.get("status") or "unavailable").lower()
    preview_verdict = str(
        local.get("surrogate_shadow_preview_verdict") or shadow.get("preview_verdict") or "UNAVAILABLE"
    ).upper()
    alignment = str(local.get("surrogate_shadow_alignment") or shadow.get("alignment") or "unknown")
    hard_signal = str(local.get("surrogate_shadow_hard_signal") or shadow.get("hard_signal") or "none")
    reasons = []
    blocked = False

    if status == "ready" and preview_verdict == "FAIL":
        blocked = True
        reasons.append("surrogate_preview_fail")
    if status == "ready" and hard_signal == "severe_mismatch":
        blocked = True
        reasons.append("surrogate_severe_mismatch")

    exploratory_eligible = (
        status == "ready"
        and preview_verdict == "FAIL"
        and to_float(local.get("alpha_score")) >= SURROGATE_VERIFY_FIRST_MIN_ALPHA_SCORE
        and to_float(local.get("sharpe")) >= SURROGATE_VERIFY_FIRST_MIN_SHARPE
        and to_float(local.get("fitness")) >= SURROGATE_VERIFY_FIRST_MIN_FITNESS
        and to_float(local.get("surrogate_shadow_penalty")) <= SURROGATE_VERIFY_FIRST_MAX_PENALTY
    )

    return {
        "status": status,
        "preview_verdict": preview_verdict,
        "alignment": alignment,
        "hard_signal": hard_signal,
        "blocked": blocked,
        "reasons": reasons,
        "exploratory_eligible": exploratory_eligible,
    }


def surrogate_shadow_allows_exploratory_retry(local_metrics: dict | None) -> bool:
    return bool(summarize_surrogate_shadow_gate(local_metrics).get("exploratory_eligible"))


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
    surrogate_gate = summarize_surrogate_shadow_gate(local)
    if surrogate_gate["blocked"]:
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
    surrogate_gate = summarize_surrogate_shadow_gate(local)
    if surrogate_gate["blocked"]:
        reasons.extend(surrogate_gate["reasons"])
    return reasons


def candidate_passes_submit_gate(candidate: dict) -> bool:
    return not submit_gate_fail_reasons(candidate)
