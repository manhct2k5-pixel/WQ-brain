#!/usr/bin/env python3
"""Runtime health checks and alert severity helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SEVERITY_ORDER = {
    "ok": 0,
    "warning": 1,
    "error": 2,
    "critical": 3,
}

API_ERROR_TOKENS = (
    "worldquant",
    "invalid_credentials",
    "authentication failed",
    "simulation_limit_exceeded",
    "status=401",
    "status=403",
    "status code: 429",
    "forbidden in simulate_batch",
)
MEMORY_REQUIRED_FIELDS = (
    "working_memory",
    "summary_memory",
    "archive_log",
    "planner_memory",
)


def _severity_rank(level: str) -> int:
    return SEVERITY_ORDER.get(str(level or "ok").lower(), 0)


def _max_severity(levels: list[str]) -> str:
    resolved = "ok"
    for level in levels:
        if _severity_rank(level) >= _severity_rank(resolved):
            resolved = str(level or "ok").lower()
    return resolved


def classify_api_failure(message: str | Exception | None) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in API_ERROR_TOKENS)


def _threshold_severity(value: float, *, warning: float, error: float, critical: float) -> str:
    if critical > 0 and value >= critical:
        return "critical"
    if error > 0 and value >= error:
        return "error"
    if warning > 0 and value >= warning:
        return "warning"
    return "ok"


def _check_shared_json_file(path_value: str | None) -> dict:
    if not path_value:
        return {
            "name": "shared_file_readability",
            "severity": "ok",
            "status": "skipped",
            "message": "No shared file path was provided for this round.",
            "path": None,
        }

    path = Path(path_value)
    if not path.exists():
        return {
            "name": "shared_file_readability",
            "severity": "warning",
            "status": "missing",
            "message": f"Shared file is missing: {path}.",
            "path": str(path),
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "name": "shared_file_readability",
            "severity": "error",
            "status": "unreadable",
            "message": f"Shared file cannot be read as JSON: {path} ({exc}).",
            "path": str(path),
        }

    if not isinstance(payload, dict):
        return {
            "name": "shared_file_readability",
            "severity": "error",
            "status": "invalid_payload",
            "message": f"Shared file does not contain a JSON object: {path}.",
            "path": str(path),
        }

    if path.name == "latest_metadata.json" and payload.get("complete") is False:
        return {
            "name": "shared_file_readability",
            "severity": "warning",
            "status": "incomplete",
            "message": f"Shared file is readable but publish metadata is incomplete: {path}.",
            "path": str(path),
        }

    return {
        "name": "shared_file_readability",
        "severity": "ok",
        "status": "ok",
        "message": f"Shared file is readable: {path}.",
        "path": str(path),
    }


def _check_memory_file_size(
    path_value: str | None,
    *,
    warning_mb: float,
    error_mb: float,
    critical_mb: float,
) -> dict:
    if not path_value:
        return {
            "name": "memory_file_size",
            "severity": "ok",
            "status": "skipped",
            "message": "No memory file path was provided for this round.",
            "path": None,
        }

    path = Path(path_value)
    if not path.exists():
        return {
            "name": "memory_file_size",
            "severity": "warning",
            "status": "missing",
            "message": f"Memory file is missing: {path}.",
            "path": str(path),
        }

    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return {
            "name": "memory_file_size",
            "severity": "error",
            "status": "stat_failed",
            "message": f"Memory file size could not be read: {path} ({exc}).",
            "path": str(path),
        }

    size_mb = round(size_bytes / (1024 * 1024), 3)
    severity = _threshold_severity(
        size_mb,
        warning=warning_mb,
        error=error_mb,
        critical=critical_mb,
    )
    status = "oversized" if severity != "ok" else "ok"
    message = (
        f"Memory file size is {size_mb} MiB at {path}."
        if severity == "ok"
        else f"Memory file grew to {size_mb} MiB at {path}."
    )
    return {
        "name": "memory_file_size",
        "severity": severity,
        "status": status,
        "message": message,
        "path": str(path),
        "size_bytes": size_bytes,
        "size_mb": size_mb,
    }


def _check_memory_payload_shape(path_value: str | None) -> dict:
    if not path_value:
        return {
            "name": "memory_payload_shape",
            "severity": "ok",
            "status": "skipped",
            "message": "No memory file path was provided for this round.",
            "path": None,
        }

    path = Path(path_value)
    if not path.exists():
        return {
            "name": "memory_payload_shape",
            "severity": "warning",
            "status": "missing",
            "message": f"Memory file is missing: {path}.",
            "path": str(path),
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "name": "memory_payload_shape",
            "severity": "error",
            "status": "unreadable",
            "message": f"Memory file cannot be read as JSON: {path} ({exc}).",
            "path": str(path),
        }

    if not isinstance(payload, dict):
        return {
            "name": "memory_payload_shape",
            "severity": "error",
            "status": "invalid_payload",
            "message": f"Memory file does not contain a JSON object: {path}.",
            "path": str(path),
        }

    missing_fields = [field for field in MEMORY_REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        return {
            "name": "memory_payload_shape",
            "severity": "warning",
            "status": "missing_fields",
            "message": f"Memory file is readable but missing fields: {', '.join(missing_fields)}.",
            "path": str(path),
            "missing_fields": missing_fields,
        }

    return {
        "name": "memory_payload_shape",
        "severity": "ok",
        "status": "ok",
        "message": f"Memory file contains the expected top-level sections: {path}.",
        "path": str(path),
    }


def _check_api_error_streak(
    consecutive_api_failures: int,
    *,
    enabled: bool,
    warning_streak: int,
    error_streak: int,
    critical_streak: int,
) -> dict:
    if not enabled:
        return {
            "name": "api_error_streak",
            "severity": "ok",
            "status": "disabled",
            "message": "API streak check is disabled for non-remote scoring mode.",
            "consecutive_failures": 0,
        }

    streak = max(0, int(consecutive_api_failures or 0))
    severity = _threshold_severity(
        streak,
        warning=warning_streak,
        error=error_streak,
        critical=critical_streak,
    )
    status = "elevated" if severity != "ok" else "ok"
    message = (
        f"Consecutive API failures: {streak}."
        if severity != "ok"
        else f"Consecutive API failures are under control: {streak}."
    )
    return {
        "name": "api_error_streak",
        "severity": severity,
        "status": status,
        "message": message,
        "consecutive_failures": streak,
    }


def _check_submit_ready_freshness(
    *,
    round_index: int,
    last_submit_ready_round: int,
    last_submit_ready_at: str | None,
    warning_rounds: int,
    error_rounds: int,
    critical_rounds: int,
) -> dict:
    if round_index <= 0:
        return {
            "name": "submit_ready_freshness",
            "severity": "ok",
            "status": "skipped",
            "message": "No rounds have completed yet, so submit-ready freshness is not available.",
            "rounds_since_submit_ready": 0,
            "last_submit_ready_at": last_submit_ready_at,
        }

    if last_submit_ready_round > 0:
        rounds_since = max(0, round_index - last_submit_ready_round)
    else:
        rounds_since = round_index

    severity = _threshold_severity(
        rounds_since,
        warning=warning_rounds,
        error=error_rounds,
        critical=critical_rounds,
    )
    status = "stale" if severity != "ok" else "ok"
    since_text = last_submit_ready_at or "none yet"
    if severity == "ok":
        message = (
            f"Submit-ready alpha freshness is healthy; rounds since last pass={rounds_since}, last_submit_ready_at={since_text}."
        )
    else:
        message = (
            f"It has been {rounds_since} round(s) since the last submit-ready alpha; last_submit_ready_at={since_text}."
        )
    return {
        "name": "submit_ready_freshness",
        "severity": severity,
        "status": status,
        "message": message,
        "rounds_since_submit_ready": rounds_since,
        "last_submit_ready_at": last_submit_ready_at,
    }


def evaluate_orchestrator_loop_health(
    *,
    summary: dict | None,
    round_index: int,
    scoring_backend: str,
    consecutive_api_failures: int,
    last_submit_ready_round: int,
    last_submit_ready_at: str | None,
    memory_warning_mb: float = 4.0,
    memory_error_mb: float = 8.0,
    memory_critical_mb: float = 16.0,
    api_warning_streak: int = 2,
    api_error_streak: int = 4,
    api_critical_streak: int = 6,
    no_pass_warning_rounds: int = 3,
    no_pass_error_rounds: int = 6,
    no_pass_critical_rounds: int = 12,
) -> dict:
    summary = summary or {}
    checks = {
        "shared_file_readability": _check_shared_json_file(summary.get("latest_metadata")),
        "memory_file_size": _check_memory_file_size(
            summary.get("global_memory_path") or summary.get("memory_path"),
            warning_mb=memory_warning_mb,
            error_mb=memory_error_mb,
            critical_mb=memory_critical_mb,
        ),
        "memory_payload_shape": _check_memory_payload_shape(
            summary.get("global_memory_path") or summary.get("memory_path"),
        ),
        "api_error_streak": _check_api_error_streak(
            consecutive_api_failures,
            enabled=str(scoring_backend or "").strip().lower() == "worldquant",
            warning_streak=api_warning_streak,
            error_streak=api_error_streak,
            critical_streak=api_critical_streak,
        ),
        "submit_ready_freshness": _check_submit_ready_freshness(
            round_index=round_index,
            last_submit_ready_round=last_submit_ready_round,
            last_submit_ready_at=last_submit_ready_at,
            warning_rounds=no_pass_warning_rounds,
            error_rounds=no_pass_error_rounds,
            critical_rounds=no_pass_critical_rounds,
        ),
    }

    active_alerts = [value for value in checks.values() if value.get("severity") != "ok"]
    active_alerts.sort(key=lambda item: _severity_rank(item.get("severity")), reverse=True)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "highest_severity": _max_severity([item.get("severity", "ok") for item in checks.values()]),
        "active_alert_count": len(active_alerts),
        "checks": checks,
        "active_alerts": active_alerts,
    }


def health_alert_lines(health: dict | None) -> list[tuple[str, str]]:
    payload = health or {}
    alerts = list(payload.get("active_alerts") or [])
    lines: list[tuple[str, str]] = []
    for alert in alerts:
        severity = str(alert.get("severity") or "warning").lower()
        name = str(alert.get("name") or "health_check")
        message = str(alert.get("message") or "").strip() or "Health check raised an alert."
        lines.append((severity, f"[health][{severity}] {name}: {message}"))
    return lines
