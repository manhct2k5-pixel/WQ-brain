#!/usr/bin/env python3
"""Run scout repeatedly until a reportable ready-to-submit alpha appears or the user stops the loop."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from scripts.flow_utils import atomic_write_json, iso_now, load_json, make_run_id
from scripts.runtime_control import clear_stale_lock_files, clear_stale_stop_file, remove_runtime_file
from scripts.system_logging import configure_runtime_logging, redirect_standard_streams

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLAN = ROOT_DIR / "artifacts/_trinh_sat/du_lieu.json"
DEFAULT_STOP_FILE = ROOT_DIR / "artifacts/_trinh_sat/DUNG"
DEFAULT_STATUS_FILE = ROOT_DIR / "artifacts/_trinh_sat/scout_loop_status.json"


def _warn(message: str) -> None:
    print(f"[scout-loop] Warning: {message}", file=sys.stderr)


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _make_loop_run_id(round_index: int) -> str:
    return f"{make_run_id('scout_loop')}_r{round_index:03d}"


def _status_payload(
    *,
    state: str,
    round_index: int,
    stop_file: Path,
    status_file: Path,
    run_id: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    reason: str | None = None,
    error: str | None = None,
    plan_path: str | None = None,
) -> dict:
    payload = {
        "pid": os.getpid(),
        "state": state,
        "round_index": round_index,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "reason": reason,
        "error": error,
        "stop_file": str(stop_file),
        "status_file": str(status_file),
    }
    log_dir = Path(status_file).parent
    payload["system_log"] = str(log_dir / "system.log")
    payload["error_log"] = str(log_dir / "error.log")
    if plan_path is not None:
        payload["plan_path"] = str(plan_path)
    return payload


def _validate_payload(payload: dict) -> str | None:
    if not isinstance(payload.get("batch"), dict):
        return "field 'batch' must be an object"
    if not isinstance(payload.get("selected"), list):
        return "field 'selected' must be a list"
    reportable_selected = payload.get("reportable_selected")
    if reportable_selected is not None and not isinstance(reportable_selected, list):
        return "field 'reportable_selected' must be a list when present"
    qualified_count = payload.get("batch", {}).get("qualified_count")
    if qualified_count is not None:
        try:
            int(qualified_count)
        except (TypeError, ValueError):
            return "field 'batch.qualified_count' must be numeric when present"
    return None


def _load_payload(path: str | Path) -> dict:
    return load_json(
        path,
        default={},
        required_fields=("batch", "selected"),
        validator=_validate_payload,
        context=f"scout payload {Path(path)}",
        warn=_warn,
    )


def _strict_pick_count(payload: dict) -> int:
    return sum(1 for item in payload.get("selected", []) if item.get("selection_mode") == "strict")


def _reportable_pick_count(payload: dict) -> int:
    reportable = payload.get("reportable_selected")
    if isinstance(reportable, list):
        return len(reportable)
    return _strict_pick_count(payload)


def _sleep_until_next_round(seconds: int, stop_file: Path) -> bool:
    remaining = max(0, seconds)
    while remaining > 0:
        if stop_file.exists():
            return False
        chunk = min(5, remaining)
        time.sleep(chunk)
        remaining -= chunk
    return True


def _emit_subprocess_output(result) -> None:
    stdout_text = str(getattr(result, "stdout", "") or "")
    stderr_text = str(getattr(result, "stderr", "") or "")
    if stdout_text:
        sys.stdout.write(stdout_text)
        if not stdout_text.endswith("\n"):
            sys.stdout.write("\n")
    if stderr_text:
        sys.stderr.write(stderr_text)
        if not stderr_text.endswith("\n"):
            sys.stderr.write("\n")
    sys.stdout.flush()
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep running scout until at least one reportable pick appears, or until the user stops the loop.",
    )
    parser.add_argument("--poll-seconds", type=int, default=900, help="Seconds to wait between scout runs when no strict pick is found.")
    parser.add_argument(
        "--until-reportable-count",
        "--until-strict-count",
        dest="until_reportable_count",
        type=int,
        default=1,
        help="Stop once this many reportable picks are present.",
    )
    parser.add_argument("--max-rounds", type=int, default=0, help="Optional safety cap. 0 means unlimited.")
    parser.add_argument("--plan-path", default=str(DEFAULT_PLAN), help="Path to the scout payload JSON used to inspect strict picks.")
    parser.add_argument("--stop-file", default=str(DEFAULT_STOP_FILE), help="If this file appears, the loop exits gracefully.")
    parser.add_argument("--status-file", default=str(DEFAULT_STATUS_FILE), help="JSON status file updated after every scout round.")
    parser.add_argument("--subprocess-timeout", type=int, default=1800, help="Timeout in seconds for each scout subprocess invocation.")
    parser.add_argument("--clear-stop-file", action="store_true", help="Remove the stop file before starting, if it exists.")
    parser.add_argument(
        "--clear-stale-locks",
        action="store_true",
        help="Remove stale *.lock files under the scout runtime folders before starting.",
    )
    parser.add_argument(
        "--manual-stop-only",
        action="store_true",
                help="Ignore reportable-pick stop conditions and keep running until the user stops the loop manually.",
    )
    parser.add_argument(
        "--allow-degraded-feedback",
        action="store_true",
        help="Do not force --require-feedback-healthy on scout runs. Use only if you intentionally want degraded feedback mode.",
    )
    args, scout_args = parser.parse_known_args()

    stop_file = Path(args.stop_file)
    status_file = Path(args.status_file)
    current_run_id = {"value": None}
    log_bundle = configure_runtime_logging(
        "scout_loop",
        log_dir=status_file.parent,
        run_id_getter=lambda: current_run_id["value"],
    )
    logger = log_bundle.logger

    with redirect_standard_streams(log_bundle):
        logger.info(
            "Scout loop startup stop_file=%s status_file=%s system_log=%s error_log=%s",
            stop_file,
            status_file,
            log_bundle.system_log_path,
            log_bundle.error_log_path,
        )
        clear_stale_stop_file(stop_file, warn=_warn)
        if args.clear_stop_file:
            remove_runtime_file(stop_file)
        if args.clear_stale_locks:
            clear_stale_lock_files(
                {
                    stop_file.parent,
                    status_file.parent,
                    Path(args.plan_path).parent,
                },
                warn=_warn,
            )

        round_index = 0
        last_run_id = None
        try:
            while True:
                if stop_file.exists():
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=last_run_id,
                            finished_at=iso_now(),
                            reason="stop_file_detected",
                            plan_path=args.plan_path,
                        ),
                    )
                    logger.info("Scout loop stopped because stop file already existed")
                    print(f"Stop file detected before round start: {stop_file}")
                    return 0

                round_index += 1
                started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                run_id = _make_loop_run_id(round_index)
                current_run_id["value"] = run_id
                last_run_id = run_id
                _write_status(
                    status_file,
                    _status_payload(
                        state="running",
                        round_index=round_index,
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=run_id,
                        started_at=iso_now(),
                        reason="round_started",
                        plan_path=args.plan_path,
                    ),
                )
                logger.info("Scout round %s started with run_id=%s", round_index, run_id)
                print(f"[scout-loop] Round {round_index} started at {started_at}")
                command = [sys.executable, str(ROOT_DIR / "scripts/scout_ideas.py")]
                if not args.allow_degraded_feedback and "--require-feedback-healthy" not in scout_args:
                    command.append("--require-feedback-healthy")
                command.extend(scout_args)
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT_DIR,
                        capture_output=True,
                        text=True,
                        timeout=max(1, int(args.subprocess_timeout)),
                    )
                except subprocess.TimeoutExpired as exc:
                    _emit_subprocess_output(exc)
                    logger.error("Scout subprocess timed out after %s seconds", args.subprocess_timeout)
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=run_id,
                            started_at=iso_now(),
                            finished_at=iso_now(),
                            reason="scout_timeout",
                            error=f"timeout={max(1, int(args.subprocess_timeout))}",
                            plan_path=args.plan_path,
                        ),
                    )
                    print(f"[scout-loop] Scout timed out after {max(1, int(args.subprocess_timeout))} seconds. Stopping loop.")
                    return 124
                _emit_subprocess_output(result)
                if result.returncode != 0:
                    logger.error("Scout subprocess exited with code %s", result.returncode)
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=run_id,
                            started_at=iso_now(),
                            finished_at=iso_now(),
                            reason="scout_failed",
                            error=f"returncode={result.returncode}",
                            plan_path=args.plan_path,
                        ),
                    )
                    if result.returncode == 2:
                        print("[scout-loop] Scout halted on feedback health check. Stopping loop until feedback is fixed.")
                    else:
                        print(f"[scout-loop] Scout exited with code {result.returncode}. Stopping loop.")
                    return result.returncode

                payload = _load_payload(args.plan_path)
                if not payload:
                    _warn("Plan payload is unavailable or invalid. Skipping stop-condition checks for this round.")
                strict_count = _strict_pick_count(payload)
                reportable_count = _reportable_pick_count(payload)
                qualified_count = int(payload.get("batch", {}).get("qualified_count", 0) or 0)
                logger.info(
                    "Scout round %s completed reportable_picks=%s strict_picks=%s qualified_count=%s",
                    round_index,
                    reportable_count,
                    strict_count,
                    qualified_count,
                )
                print(
                    f"[scout-loop] Round {round_index} finished. reportable_picks={reportable_count}, "
                    f"strict_picks={strict_count}, qualified_count={qualified_count}"
                )

                if not args.manual_stop_only and reportable_count >= max(1, args.until_reportable_count):
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=run_id,
                            started_at=started_at,
                            finished_at=iso_now(),
                            reason="reportable_target_reached",
                            plan_path=args.plan_path,
                        ),
                    )
                    logger.info("Scout loop stopped because reportable target was reached")
                    print("[scout-loop] Found enough reportable picks. Stopping.")
                    return 0
                if args.manual_stop_only and reportable_count >= max(1, args.until_reportable_count):
                    print("[scout-loop] Reportable picks found, but manual-stop-only is enabled. Continuing.")

                if args.max_rounds and round_index >= args.max_rounds:
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=run_id,
                            started_at=started_at,
                            finished_at=iso_now(),
                            reason="max_rounds",
                            plan_path=args.plan_path,
                        ),
                    )
                    logger.info("Scout loop stopped after reaching max_rounds=%s", args.max_rounds)
                    print(f"[scout-loop] Reached max rounds ({args.max_rounds}). Stopping.")
                    return 0

                _write_status(
                    status_file,
                    _status_payload(
                        state="sleeping",
                        round_index=round_index,
                        stop_file=stop_file,
                        status_file=status_file,
                        run_id=run_id,
                        started_at=started_at,
                        finished_at=iso_now(),
                        reason="round_completed",
                        plan_path=args.plan_path,
                    ),
                )
                print(
                    f"[scout-loop] No reportable pick yet. Sleeping {args.poll_seconds} seconds. "
                    f"Create {stop_file} or press Ctrl+C to stop."
                )
                if not _sleep_until_next_round(args.poll_seconds, stop_file):
                    _write_status(
                        status_file,
                        _status_payload(
                            state="stopped",
                            round_index=round_index,
                            stop_file=stop_file,
                            status_file=status_file,
                            run_id=run_id,
                            finished_at=iso_now(),
                            reason="stop_file_detected",
                            plan_path=args.plan_path,
                        ),
                    )
                    logger.info("Scout loop stopped because stop file was detected")
                    print(f"[scout-loop] Stop file detected: {stop_file}")
                    return 0
        except KeyboardInterrupt:
            _write_status(
                status_file,
                _status_payload(
                    state="stopped",
                    round_index=round_index,
                    stop_file=stop_file,
                    status_file=status_file,
                    run_id=last_run_id,
                    finished_at=iso_now(),
                    reason="keyboard_interrupt",
                    plan_path=args.plan_path,
                ),
            )
            logger.info("Scout loop interrupted by keyboard")
            print("\n[scout-loop] Stopped by user.")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
