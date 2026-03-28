#!/usr/bin/env python3
"""Run the orchestrated alpha pipeline repeatedly until stopped."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import STATE_DIR, atomic_write_json, iso_now, make_run_id
from scripts.orchestrator import PROFILE_DEFAULTS, run_pipeline
from scripts.runtime_control import clear_stale_lock_files, clear_stale_stop_file, remove_runtime_file
from scripts.system_health import classify_api_failure, evaluate_orchestrator_loop_health, health_alert_lines
from scripts.system_logging import configure_runtime_logging, redirect_standard_streams

DEFAULT_STOP_FILE = STATE_DIR / "DUNG_LOOP"
DEFAULT_STATUS_FILE = STATE_DIR / "orchestrator_loop_status.json"
ROUND_HISTORY_LIMIT = 24


def _sleep_until_next_round(seconds: int, stop_file: Path) -> bool:
    remaining = max(0, int(seconds))
    while remaining > 0:
        if stop_file.exists():
            return False
        chunk = min(5, remaining)
        time.sleep(chunk)
        remaining -= chunk
    return True


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _make_loop_run_id(prefix: str | None, round_index: int) -> str:
    return f"{make_run_id(prefix)}_r{round_index:03d}"


def _status_payload(
    *,
    state: str,
    round_index: int,
    started_at: str | None = None,
    finished_at: str | None = None,
    next_run_at: str | None = None,
    reason: str | None = None,
    stop_file: Path | None = None,
    status_file: Path | None = None,
    run_id: str | None = None,
    summary: dict | None = None,
    consecutive_failures: int = 0,
    error: str | None = None,
    stagnation: dict | None = None,
    health: dict | None = None,
    manual_overrides: dict | None = None,
) -> dict:
    payload = {
        "pid": os.getpid(),
        "state": state,
        "round_index": round_index,
        "started_at": started_at,
        "finished_at": finished_at,
        "next_run_at": next_run_at,
        "reason": reason,
        "run_id": run_id,
        "consecutive_failures": consecutive_failures,
        "summary": summary or {},
        "error": error,
        "stagnation": stagnation or {},
        "health": health or {},
        "manual_overrides": manual_overrides or {},
    }
    if stop_file is not None:
        payload["stop_file"] = str(stop_file)
    if status_file is not None:
        payload["status_file"] = str(status_file)
        log_dir = Path(status_file).parent
        payload["system_log"] = str(log_dir / "system.log")
        payload["error_log"] = str(log_dir / "error.log")
    return payload


def _health_config(args) -> dict:
    return {
        "memory_warning_mb": float(getattr(args, "health_memory_warning_mb", 4.0)),
        "memory_error_mb": float(getattr(args, "health_memory_error_mb", 8.0)),
        "memory_critical_mb": float(getattr(args, "health_memory_critical_mb", 16.0)),
        "api_warning_streak": int(getattr(args, "health_api_warning_streak", 2)),
        "api_error_streak": int(getattr(args, "health_api_error_streak", 4)),
        "api_critical_streak": int(getattr(args, "health_api_critical_streak", 6)),
        "no_pass_warning_rounds": int(getattr(args, "health_no_pass_warning_rounds", 3)),
        "no_pass_error_rounds": int(getattr(args, "health_no_pass_error_rounds", 6)),
        "no_pass_critical_rounds": int(getattr(args, "health_no_pass_critical_rounds", 12)),
    }


def _build_health(
    *,
    args,
    round_index: int,
    summary: dict | None,
    consecutive_api_failures: int,
    last_submit_ready_round: int,
    last_submit_ready_at: str | None,
) -> dict:
    return evaluate_orchestrator_loop_health(
        summary=summary,
        round_index=round_index,
        scoring_backend=args.scoring,
        consecutive_api_failures=consecutive_api_failures,
        last_submit_ready_round=last_submit_ready_round,
        last_submit_ready_at=last_submit_ready_at,
        **_health_config(args),
    )


def _health_signature(health: dict | None) -> tuple:
    payload = health or {}
    alerts = tuple(
        (
            str(item.get("name") or ""),
            str(item.get("severity") or ""),
            str(item.get("status") or ""),
        )
        for item in (payload.get("active_alerts") or [])
    )
    return (str(payload.get("highest_severity") or "ok"), alerts)


def _emit_health_alerts(logger, health: dict | None, *, previous_health: dict | None = None) -> None:
    current_signature = _health_signature(health)
    previous_signature = _health_signature(previous_health)
    current_severity = str((health or {}).get("highest_severity") or "ok").lower()
    previous_severity = str((previous_health or {}).get("highest_severity") or "ok").lower()

    if current_signature == previous_signature:
        return

    if current_severity == "ok":
        if previous_severity != "ok":
            message = "[health][ok] Runtime health recovered."
            logger.info(message)
            print(message, file=sys.stderr)
        return

    for severity, line in health_alert_lines(health):
        if severity == "critical":
            logger.critical(line)
        elif severity == "error":
            logger.error(line)
        elif severity == "warning":
            logger.warning(line)
        else:
            logger.info(line)
        print(line, file=sys.stderr)


def _build_run_args(
    args,
    *,
    run_id: str,
    adaptive_controls: dict | None = None,
    source_bonus_adjustments: dict | None = None,
    source_quota_profile: dict | None = None,
    manual_overrides: dict | None = None,
):
    return SimpleNamespace(
        profile=args.profile,
        run_id=run_id,
        csv_path=args.csv_path,
        memory=args.memory,
        seed_store=args.seed_store,
        auto_fix_input=args.auto_fix_input,
        scout_input=args.scout_input,
        top=args.top,
        count=args.count,
        queue_limit=args.queue_limit,
        history_window=args.history_window,
        scoring=args.scoring,
        timeout=args.timeout,
        local_score_limit=args.local_score_limit,
        local_score_workers=args.local_score_workers,
        min_parallel_local_scoring=args.min_parallel_local_scoring,
        daily_top=args.daily_top,
        feed_limit=args.feed_limit,
        adaptive_controls=adaptive_controls or {},
        source_bonus_adjustments=source_bonus_adjustments or {},
        source_quota_profile=source_quota_profile or {},
        manual_overrides=manual_overrides or {},
    )


def _coerce_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _manual_override_payload(args) -> dict:
    only_fix = bool(getattr(args, "manual_only_fix", False))
    disable_scout = bool(getattr(args, "manual_disable_scout", False)) or only_fix
    payload = {
        "active": any(
            (
                only_fix,
                disable_scout,
                bool(getattr(args, "manual_increase_explore", False)),
                bool(getattr(args, "manual_freeze_memory_update", False)),
                bool(getattr(args, "manual_ignore_block_list", False)),
            )
        ),
        "only_fix": only_fix,
        "disable_scout": disable_scout,
        "increase_explore": bool(getattr(args, "manual_increase_explore", False)),
        "freeze_memory_update": bool(getattr(args, "manual_freeze_memory_update", False)),
        "ignore_block_list": bool(getattr(args, "manual_ignore_block_list", False)),
    }
    return payload


def _round_metrics(summary: dict) -> dict:
    evaluated = max(0, _coerce_int((summary or {}).get("evaluated_candidates")))
    submit_ready = max(0, _coerce_int((summary or {}).get("submit_ready_candidates")))
    pass_rate = round((submit_ready / evaluated), 4) if evaluated else 0.0
    return {
        "run_id": (summary or {}).get("run_id"),
        "evaluated_candidates": evaluated,
        "submit_ready_candidates": submit_ready,
        "pass_rate": pass_rate,
    }


def _stagnation_state(round_history: list[dict], args) -> dict:
    dry_threshold = max(0, _coerce_int(getattr(args, "stagnation_no_submit_ready_rounds", 0)))
    window = max(0, _coerce_int(getattr(args, "stagnation_window", 0)))
    min_pass_rate = max(0.0, _coerce_float(getattr(args, "stagnation_min_pass_rate", 0.0)))

    consecutive_no_submit_ready = 0
    for item in reversed(round_history):
        if _coerce_int(item.get("submit_ready_candidates")) > 0:
            break
        consecutive_no_submit_ready += 1

    recent_window = round_history[-window:] if window else []
    evaluated_total = sum(max(0, _coerce_int(item.get("evaluated_candidates"))) for item in recent_window)
    submit_ready_total = sum(max(0, _coerce_int(item.get("submit_ready_candidates"))) for item in recent_window)
    rolling_pass_rate = round((submit_ready_total / evaluated_total), 4) if evaluated_total else 0.0

    reason_codes = []
    if dry_threshold and consecutive_no_submit_ready >= dry_threshold:
        reason_codes.append("consecutive_no_submit_ready")
    if window and len(recent_window) >= window and rolling_pass_rate < min_pass_rate:
        reason_codes.append("low_pass_rate")

    active = bool(reason_codes)
    warning = ""
    adaptive_controls = {}
    source_bonus_adjustments = {}
    source_quota_profile = {}
    if active:
        warning = (
            f"Stagnation detected: {consecutive_no_submit_ready} dry round(s), "
            f"rolling submit-ready rate {rolling_pass_rate:.1%} over {len(recent_window)} round(s). "
            "Recovery mode will raise exploration, reopen soft blocks, and prioritize scout candidates."
        )
        adaptive_controls = {
            "mode": "stagnation_recovery",
            "reason_codes": reason_codes,
            "warning": warning,
            "exploration_boost": max(0.0, _coerce_float(getattr(args, "adaptive_exploration_boost", 0.0))),
            "exploration_weight_multiplier": max(1.0, _coerce_float(getattr(args, "adaptive_exploration_weight_multiplier", 1.0))),
            "soft_block_penalty_multiplier": min(1.0, max(0.0, _coerce_float(getattr(args, "adaptive_soft_block_penalty_multiplier", 1.0)))),
            "candidate_risk_penalty_multiplier": min(
                1.0,
                max(0.5, _coerce_float(getattr(args, "adaptive_candidate_risk_penalty_multiplier", 1.0))),
            ),
            "reopen_soft_blocked_families_count": max(0, _coerce_int(getattr(args, "adaptive_reopen_soft_families", 0))),
            "reopen_soft_blocked_skeletons_count": max(0, _coerce_int(getattr(args, "adaptive_reopen_soft_skeletons", 0))),
            "thesis_limit_bonus": max(0, _coerce_int(getattr(args, "adaptive_thesis_limit_bonus", 0))),
            "batch_size_bonus": max(0, _coerce_int(getattr(args, "adaptive_batch_size_bonus", 0))),
            "queue_limit_bonus": max(0, _coerce_int(getattr(args, "adaptive_queue_limit_bonus", 0))),
        }
        scout_boost = max(0.0, _coerce_float(getattr(args, "adaptive_scout_priority_boost", 0.0)))
        if scout_boost:
            source_bonus_adjustments["scout"] = scout_boost
        source_quota_profile = {
            "planner": max(0.0, _coerce_float(getattr(args, "adaptive_planner_soft_quota", 0.0))),
            "auto_fix_rewrite": max(0.0, _coerce_float(getattr(args, "adaptive_fix_soft_quota", 0.0))),
            "scout": max(0.0, _coerce_float(getattr(args, "adaptive_scout_soft_quota", 0.0))),
        }

    return {
        "active": active,
        "reason_codes": reason_codes,
        "consecutive_no_submit_ready": consecutive_no_submit_ready,
        "window_size": window,
        "window_evaluated_candidates": evaluated_total,
        "window_submit_ready_candidates": submit_ready_total,
        "rolling_pass_rate": rolling_pass_rate,
        "warning": warning,
        "adaptive_controls": adaptive_controls,
        "source_bonus_adjustments": source_bonus_adjustments,
        "source_quota_profile": source_quota_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep running orchestrator rounds on a fixed interval until the user stops the loop.",
    )
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="light")
    parser.add_argument("--run-id-prefix", help="Optional prefix for each generated run_id.")
    parser.add_argument("--csv-path", help="Optional simulation history CSV path.")
    parser.add_argument("--memory", help="Optional prior memory JSON path.")
    parser.add_argument("--seed-store", default="initial-population.pkl", help="Seed store path used for planner duplicate context.")
    parser.add_argument("--auto-fix-input", help="Optional auto-fix candidate JSON.")
    parser.add_argument("--scout-input", help="Optional scout candidate JSON.")
    parser.add_argument("--top", type=int, default=10, help="How many top rows to learn from and summarize.")
    parser.add_argument("--count", type=int, help="Override planned candidate count.")
    parser.add_argument("--queue-limit", type=int, help="Override merged queue size.")
    parser.add_argument("--history-window", type=int, default=120, help="Planner history window.")
    parser.add_argument("--scoring", choices=("internal", "worldquant"), default="worldquant", help="Simulation backend for queue evaluation.")
    parser.add_argument("--timeout", type=int, default=300, help="WorldQuant simulation timeout in seconds.")
    parser.add_argument("--local-score-limit", type=int, help="Optional cap on how many candidates receive local scoring when backend=internal.")
    parser.add_argument("--local-score-workers", type=int, help="Optional worker cap for local scoring. Defaults to auto when omitted or 0.")
    parser.add_argument("--min-parallel-local-scoring", type=int, default=4, help="Minimum candidate count before local scoring uses multiprocessing.")
    parser.add_argument("--manual-only-fix", action="store_true", help="Restrict each round to auto-fix candidates only.")
    parser.add_argument("--manual-disable-scout", action="store_true", help="Disable scout candidates for each round.")
    parser.add_argument("--manual-increase-explore", action="store_true", help="Increase planner exploration for each round.")
    parser.add_argument("--manual-freeze-memory-update", action="store_true", help="Freeze global memory updates for each round.")
    parser.add_argument("--manual-ignore-block-list", action="store_true", help="Temporarily ignore planner hard/soft block lists for each round.")
    parser.add_argument("--daily-top", type=int, default=3, help="Maximum number of daily winners to render.")
    parser.add_argument("--feed-limit", type=int, default=12, help="Maximum number of feed candidates to render.")
    parser.add_argument("--interval-minutes", type=int, default=30, help="Minutes to wait between rounds.")
    parser.add_argument("--max-rounds", type=int, default=0, help="Optional safety cap. 0 means unlimited.")
    parser.add_argument("--max-consecutive-failures", type=int, default=3, help="Stop after this many consecutive failed rounds. 0 means unlimited.")
    parser.add_argument("--stagnation-no-submit-ready-rounds", type=int, default=3, help="Enable recovery mode after this many consecutive rounds without submit-ready output. 0 disables the check.")
    parser.add_argument("--stagnation-window", type=int, default=4, help="Rolling round window used to detect low submit-ready pass rate. 0 disables the check.")
    parser.add_argument("--stagnation-min-pass-rate", type=float, default=0.08, help="Minimum submit-ready rate across the stagnation window before recovery mode is triggered.")
    parser.add_argument("--adaptive-exploration-boost", type=float, default=0.16, help="Temporary novelty boost applied when recovery mode is active.")
    parser.add_argument("--adaptive-exploration-weight-multiplier", type=float, default=1.45, help="Temporary multiplier for planner exploration bias when recovery mode is active.")
    parser.add_argument("--adaptive-soft-block-penalty-multiplier", type=float, default=0.35, help="Temporary multiplier applied to soft-block penalties during recovery mode.")
    parser.add_argument("--adaptive-candidate-risk-penalty-multiplier", type=float, default=0.85, help="Temporary multiplier applied to candidate risk penalties during recovery mode.")
    parser.add_argument("--adaptive-reopen-soft-families", type=int, default=2, help="How many soft-blocked families to reopen temporarily during recovery mode.")
    parser.add_argument("--adaptive-reopen-soft-skeletons", type=int, default=4, help="How many soft-blocked skeletons to reopen temporarily during recovery mode.")
    parser.add_argument("--adaptive-thesis-limit-bonus", type=int, default=1, help="Extra variants allowed per thesis during recovery mode.")
    parser.add_argument("--adaptive-batch-size-bonus", type=int, default=2, help="Extra planner candidates requested during recovery mode.")
    parser.add_argument("--adaptive-queue-limit-bonus", type=int, default=3, help="Extra queue slots allowed during recovery mode.")
    parser.add_argument("--adaptive-scout-priority-boost", type=float, default=8.0, help="Additional queue priority bonus applied to scout candidates during recovery mode.")
    parser.add_argument("--adaptive-planner-soft-quota", type=float, default=0.30, help="Planner soft share target during recovery mode.")
    parser.add_argument("--adaptive-fix-soft-quota", type=float, default=0.30, help="Auto-fix soft share target during recovery mode.")
    parser.add_argument("--adaptive-scout-soft-quota", type=float, default=0.40, help="Scout soft share target during recovery mode.")
    parser.add_argument("--health-memory-warning-mb", type=float, default=4.0, help="Raise a warning when the shared memory artifact exceeds this size in MiB.")
    parser.add_argument("--health-memory-error-mb", type=float, default=8.0, help="Raise an error when the shared memory artifact exceeds this size in MiB.")
    parser.add_argument("--health-memory-critical-mb", type=float, default=16.0, help="Raise a critical alert when the shared memory artifact exceeds this size in MiB.")
    parser.add_argument("--health-api-warning-streak", type=int, default=2, help="Raise a warning after this many consecutive API-related failures.")
    parser.add_argument("--health-api-error-streak", type=int, default=4, help="Raise an error after this many consecutive API-related failures.")
    parser.add_argument("--health-api-critical-streak", type=int, default=6, help="Raise a critical alert after this many consecutive API-related failures.")
    parser.add_argument("--health-no-pass-warning-rounds", type=int, default=3, help="Raise a warning after this many rounds without a submit-ready alpha.")
    parser.add_argument("--health-no-pass-error-rounds", type=int, default=6, help="Raise an error after this many rounds without a submit-ready alpha.")
    parser.add_argument("--health-no-pass-critical-rounds", type=int, default=12, help="Raise a critical alert after this many rounds without a submit-ready alpha.")
    parser.add_argument("--stop-file", default=str(DEFAULT_STOP_FILE), help="If this file appears, the loop exits gracefully.")
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS_FILE), help="JSON status file updated after every round.")
    parser.add_argument("--clear-stop-file", action="store_true", help="Remove the stop file before starting, if it exists.")
    parser.add_argument(
        "--clear-stale-locks",
        action="store_true",
        help="Remove stale *.lock files under the loop runtime folders before starting.",
    )
    args = parser.parse_args()

    stop_file = Path(args.stop_file)
    status_file = Path(args.status_file)
    current_run_id = {"value": None}
    log_bundle = configure_runtime_logging(
        "orchestrator_loop",
        log_dir=status_file.parent,
        run_id_getter=lambda: current_run_id["value"],
    )
    logger = log_bundle.logger

    with redirect_standard_streams(log_bundle):
        logger.info(
            "Loop startup stop_file=%s status_file=%s system_log=%s error_log=%s",
            stop_file,
            status_file,
            log_bundle.system_log_path,
            log_bundle.error_log_path,
        )
        clear_stale_stop_file(stop_file, warn=lambda message: print(f"[loop] Warning: {message}", file=sys.stderr))
        if args.clear_stop_file:
            remove_runtime_file(stop_file)
        if args.clear_stale_locks:
            clear_stale_lock_files(
                {
                    stop_file.parent,
                    status_file.parent,
                    STATE_DIR,
                },
                warn=lambda message: print(f"[loop] Warning: {message}", file=sys.stderr),
            )

        round_index = 0
        consecutive_failures = 0
        consecutive_api_failures = 0
        last_summary = {}
        last_run_id = None
        last_submit_ready_round = 0
        last_submit_ready_at = None
        round_history = []
        manual_overrides = _manual_override_payload(args)
        last_stagnation = _stagnation_state(round_history, args)
        last_health = _build_health(
            args=args,
            round_index=round_index,
            summary=last_summary,
            consecutive_api_failures=consecutive_api_failures,
            last_submit_ready_round=last_submit_ready_round,
            last_submit_ready_at=last_submit_ready_at,
        )

        try:
            while True:
                stagnation = _stagnation_state(round_history, args)
                if stagnation.get("active") and not last_stagnation.get("active"):
                    print(f"[loop] Warning: {stagnation['warning']}", file=sys.stderr)
                elif not stagnation.get("active") and last_stagnation.get("active"):
                    print("[loop] Recovery mode cleared after submit-ready output improved.", file=sys.stderr)
                last_stagnation = stagnation

                if stop_file.exists():
                    health = _build_health(
                        args=args,
                        round_index=round_index,
                        summary=last_summary,
                        consecutive_api_failures=consecutive_api_failures,
                        last_submit_ready_round=last_submit_ready_round,
                        last_submit_ready_at=last_submit_ready_at,
                    )
                    _emit_health_alerts(logger, health, previous_health=last_health)
                    last_health = health
                    payload = _status_payload(
                        state="stopped",
                        round_index=round_index,
                        finished_at=iso_now(),
                        reason="stop_file_detected",
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=last_run_id,
                        summary=last_summary,
                        consecutive_failures=consecutive_failures,
                        stagnation=stagnation,
                        health=health,
                        manual_overrides=manual_overrides,
                    )
                    _write_status(status_file, payload)
                    print(f"[loop] Stop file detected before round start: {stop_file}")
                    return 0

                round_index += 1
                started_at = iso_now()
                run_id = _make_loop_run_id(args.run_id_prefix or args.profile, round_index)
                current_run_id["value"] = run_id
                payload = _status_payload(
                    state="running",
                    round_index=round_index,
                    started_at=started_at,
                    run_id=run_id,
                    stop_file=stop_file,
                    status_file=status_file,
                    consecutive_failures=consecutive_failures,
                    stagnation=stagnation,
                    health=last_health,
                    manual_overrides=manual_overrides,
                )
                _write_status(status_file, payload)
                logger.info("Round %s started with run_id=%s", round_index, run_id)
                print(f"[loop] Round {round_index} started at {started_at} (run_id={run_id})")

                try:
                    summary = run_pipeline(
                        _build_run_args(
                            args,
                            run_id=run_id,
                            adaptive_controls=stagnation.get("adaptive_controls"),
                            source_bonus_adjustments=stagnation.get("source_bonus_adjustments"),
                            source_quota_profile=stagnation.get("source_quota_profile"),
                            manual_overrides=manual_overrides,
                        )
                    )
                except Exception as exc:  # pragma: no cover - handled in tests via RuntimeError
                    logger.exception("Round %s crashed", round_index)
                    consecutive_failures += 1
                    if classify_api_failure(exc):
                        consecutive_api_failures += 1
                    else:
                        consecutive_api_failures = 0
                    finished_at = iso_now()
                    health = _build_health(
                        args=args,
                        round_index=round_index,
                        summary=last_summary,
                        consecutive_api_failures=consecutive_api_failures,
                        last_submit_ready_round=last_submit_ready_round,
                        last_submit_ready_at=last_submit_ready_at,
                    )
                    _emit_health_alerts(logger, health, previous_health=last_health)
                    last_health = health
                    payload = _status_payload(
                        state="error",
                        round_index=round_index,
                        started_at=started_at,
                        finished_at=finished_at,
                        reason="round_failed",
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=run_id,
                        summary=last_summary,
                        consecutive_failures=consecutive_failures,
                        error=f"{type(exc).__name__}: {exc}",
                        stagnation=stagnation,
                        health=health,
                        manual_overrides=manual_overrides,
                    )
                    _write_status(status_file, payload)
                    print(f"[loop] Round {round_index} failed: {exc}", file=sys.stderr)
                    if args.max_consecutive_failures and consecutive_failures >= args.max_consecutive_failures:
                        payload["state"] = "stopped"
                        payload["reason"] = "max_consecutive_failures"
                        _write_status(status_file, payload)
                        print(
                            f"[loop] Reached max consecutive failures ({args.max_consecutive_failures}). Stopping.",
                            file=sys.stderr,
                        )
                        return 1
                else:
                    consecutive_failures = 0
                    consecutive_api_failures = 0
                    last_summary = summary
                    last_run_id = run_id
                    finished_at = iso_now()
                    if int(summary.get("submit_ready_candidates", 0) or 0) > 0:
                        last_submit_ready_round = round_index
                        last_submit_ready_at = finished_at
                    round_history.append(_round_metrics(summary))
                    round_history = round_history[-ROUND_HISTORY_LIMIT:]
                    stagnation = _stagnation_state(round_history, args)
                    last_stagnation = stagnation
                    health = _build_health(
                        args=args,
                        round_index=round_index,
                        summary=summary,
                        consecutive_api_failures=consecutive_api_failures,
                        last_submit_ready_round=last_submit_ready_round,
                        last_submit_ready_at=last_submit_ready_at,
                    )
                    _emit_health_alerts(logger, health, previous_health=last_health)
                    last_health = health
                    next_run_at = None
                    if not (args.max_rounds and round_index >= args.max_rounds):
                        next_run_at = (
                            datetime.now() + timedelta(minutes=max(0, args.interval_minutes))
                        ).isoformat(timespec="seconds")
                    payload = _status_payload(
                        state="sleeping",
                        round_index=round_index,
                        started_at=started_at,
                        finished_at=finished_at,
                        next_run_at=next_run_at,
                        reason="round_completed",
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=run_id,
                        summary=summary,
                        consecutive_failures=0,
                        stagnation=stagnation,
                        health=health,
                        manual_overrides=manual_overrides,
                    )
                    _write_status(status_file, payload)
                    logger.info(
                        "Round %s completed queue_candidates=%s submit_ready_candidates=%s",
                        round_index,
                        summary.get("queue_candidates", 0),
                        summary.get("submit_ready_candidates", 0),
                    )
                    print(
                        json.dumps(
                            {
                                "round_index": round_index,
                                "run_id": run_id,
                                "queue_candidates": summary.get("queue_candidates", 0),
                                "submit_ready_candidates": summary.get("submit_ready_candidates", 0),
                                "adaptive_recovery": bool(stagnation.get("active")),
                                "health": health.get("highest_severity", "ok"),
                                "daily_report": summary.get("daily_report"),
                                "feed_report": summary.get("feed_report"),
                            },
                            indent=2,
                        )
                    )

                if args.max_rounds and round_index >= args.max_rounds:
                    health = _build_health(
                        args=args,
                        round_index=round_index,
                        summary=last_summary,
                        consecutive_api_failures=consecutive_api_failures,
                        last_submit_ready_round=last_submit_ready_round,
                        last_submit_ready_at=last_submit_ready_at,
                    )
                    _emit_health_alerts(logger, health, previous_health=last_health)
                    last_health = health
                    payload = _status_payload(
                        state="stopped",
                        round_index=round_index,
                        finished_at=iso_now(),
                        reason="max_rounds",
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=last_run_id,
                        summary=last_summary,
                        consecutive_failures=consecutive_failures,
                        stagnation=last_stagnation,
                        health=health,
                        manual_overrides=manual_overrides,
                    )
                    _write_status(status_file, payload)
                    logger.info("Loop stopped after reaching max_rounds=%s", args.max_rounds)
                    print(f"[loop] Reached max rounds ({args.max_rounds}). Stopping.")
                    return 0

                wait_seconds = max(0, args.interval_minutes) * 60
                next_run_note = f"{args.interval_minutes} minute(s)"
                if wait_seconds == 0:
                    next_run_note = "0 minute(s)"
                print(f"[loop] Sleeping {next_run_note}. Create {stop_file} or press Ctrl+C to stop.")
                if not _sleep_until_next_round(wait_seconds, stop_file):
                    health = _build_health(
                        args=args,
                        round_index=round_index,
                        summary=last_summary,
                        consecutive_api_failures=consecutive_api_failures,
                        last_submit_ready_round=last_submit_ready_round,
                        last_submit_ready_at=last_submit_ready_at,
                    )
                    _emit_health_alerts(logger, health, previous_health=last_health)
                    last_health = health
                    payload = _status_payload(
                        state="stopped",
                        round_index=round_index,
                        finished_at=iso_now(),
                        reason="stop_file_detected",
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=last_run_id,
                        summary=last_summary,
                        consecutive_failures=consecutive_failures,
                        stagnation=last_stagnation,
                        health=health,
                        manual_overrides=manual_overrides,
                    )
                    _write_status(status_file, payload)
                    logger.info("Loop stopped because stop file was detected")
                    print(f"[loop] Stop file detected: {stop_file}")
                    return 0
        except KeyboardInterrupt:
            health = _build_health(
                args=args,
                round_index=round_index,
                summary=last_summary,
                consecutive_api_failures=consecutive_api_failures,
                last_submit_ready_round=last_submit_ready_round,
                last_submit_ready_at=last_submit_ready_at,
            )
            _emit_health_alerts(logger, health, previous_health=last_health)
            payload = _status_payload(
                state="stopped",
                round_index=round_index,
                finished_at=iso_now(),
                reason="keyboard_interrupt",
                stop_file=stop_file,
                status_file=status_file,
                run_id=last_run_id,
                summary=last_summary,
                consecutive_failures=consecutive_failures,
                stagnation=last_stagnation,
                health=health,
                manual_overrides=manual_overrides,
            )
            _write_status(status_file, payload)
            logger.info("Loop interrupted by keyboard")
            print("\n[loop] Stopped by user.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
