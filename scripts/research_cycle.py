#!/usr/bin/env python3
"""Run guarded research cycles around the main alpha engine."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from scripts.render_cycle_report import build_report
from scripts.results_digest import build_summary as build_digest_summary
from scripts.results_digest import discover_csv, read_rows
from scripts.flow_utils import atomic_write_json
from src.run_profiles import ensure_artifacts_dir
from src.utils import create_authenticated_session

REPO_ROOT = ROOT_DIR
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RUN_STATUS_PATH = ARTIFACTS_DIR / "trang_thai_chay.json"


def read_csv_rows(csv_path: Path | None = None) -> list[dict]:
    candidates = [csv_path] if csv_path else [REPO_ROOT / "simulation_results.csv", REPO_ROOT / "simulations.csv"]
    for candidate in candidates:
        if candidate and candidate.exists():
            return read_rows(str(candidate))
    return []


def summarize_rows(rows: list[dict], top_n: int = 10) -> dict:
    if not rows:
        return {
            "rows": 0,
            "pending_count": 0,
            "best_research_score": None,
            "pass_all_count": 0,
        }
    summary = build_digest_summary(rows, top_n)["summary"]
    return {
        "rows": summary.get("rows", 0),
        "pending_count": summary.get("pending_count", 0),
        "best_research_score": summary.get("best_research_score"),
        "pass_all_count": summary.get("pass_all_count", 0),
    }


def inspect_run_output(output_text: str) -> dict:
    text = output_text.upper()
    rate_limit_events = text.count("STATUS CODE: 429") + text.count("SIMULATION_LIMIT_EXCEEDED")
    auth_failed = "STATUS CODE: 401" in text or "INVALID_CREDENTIALS" in text
    return {
        "auth_failed": auth_failed,
        "rate_limit_events": rate_limit_events,
    }


def choose_next_action(stop_reason: str) -> str:
    actions = {
        "completed": "Review artifacts/lo_tiep_theo.json and run seed only for candidates you want to promote.",
        "no_new_rows": "Inspect the latest logs, then retry with test or light after the API settles.",
        "pending_backlog": "Wait for pending checks to settle and use digest/plan before starting another cycle.",
        "no_improvement": "Review lo_tiep_theo.json and adjust or seed selectively before another run.",
        "rate_limited": "Pause for a few minutes before the next cycle to protect quota.",
        "auth_failed": "Run auth again and verify the Brain credentials in .env.",
        "run_failed": "Inspect logs/brain-learn for the failing round before retrying.",
    }
    return actions.get(stop_reason, "Review artifacts and logs before starting another cycle.")


def write_status(status: dict, output_path: Path = RUN_STATUS_PATH) -> None:
    ensure_artifacts_dir(output_path.parent)
    atomic_write_json(output_path, status)


def write_cycle_report(repo_root: Path) -> None:
    artifacts_dir = repo_root / "artifacts"
    ensure_artifacts_dir(artifacts_dir)
    (artifacts_dir / "bao_cao_moi_nhat.md").write_text(build_report(repo_root), encoding="utf-8")


def run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=max(1, int(timeout)),
            check=False,
        )
        output = str(result.stdout or "")
        if output:
            print(output, end="", flush=True)
        return subprocess.CompletedProcess(
            command,
            result.returncode,
            stdout=output,
            stderr="",
        )
    except subprocess.TimeoutExpired as exc:
        partial_output = str(exc.output or "")
        if partial_output:
            print(partial_output, end="", flush=True)
        timeout_message = (
            f"[research_cycle] Subprocess timed out after {max(1, int(timeout))} seconds: {' '.join(command)}\n"
        )
        print(timeout_message, end="", file=sys.stderr, flush=True)
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=partial_output + timeout_message,
            stderr="timeout",
        )


def refresh_artifacts(repo_root: Path, *, subprocess_timeout: int) -> int:
    ensure_artifacts_dir(repo_root / "artifacts")
    rows = read_csv_rows()
    if not rows:
        return 0

    digest_result = run_subprocess(
        [sys.executable, "scripts/results_digest.py", "--format", "markdown"],
        cwd=repo_root,
        timeout=subprocess_timeout,
    )
    (repo_root / "artifacts/tom_tat_moi_nhat.md").write_text(digest_result.stdout, encoding="utf-8")
    if digest_result.returncode != 0:
        return digest_result.returncode

    plan_result = run_subprocess(
        [
            sys.executable,
            "scripts/plan_next_batch.py",
            "--format",
            "markdown",
            "--memory",
            "artifacts/bo_nho_nghien_cuu.json",
            "--write-memory",
            "artifacts/bo_nho_nghien_cuu.json",
            "--write-batch",
            "artifacts/bieu_thuc_ung_vien.txt",
            "--write-plan",
            "artifacts/lo_tiep_theo.json",
        ],
        cwd=repo_root,
        timeout=subprocess_timeout,
    )
    (repo_root / "artifacts/lo_tiep_theo.md").write_text(plan_result.stdout, encoding="utf-8")
    return plan_result.returncode


def authentication_check() -> tuple[bool, int | None]:
    load_dotenv(REPO_ROOT / ".env")
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    if not username or not password:
        return False, None

    session, response = create_authenticated_session(username, password, context="Research cycle auth check")
    if session is None:
        status_code = response.status_code if response is not None else None
        return False, status_code
    return True, 201


def should_stop_after_round(
    *,
    before_summary: dict,
    after_summary: dict,
    round_events: dict,
    max_pending_ratio: float,
) -> str | None:
    if round_events["auth_failed"]:
        return "auth_failed"
    if round_events["rate_limit_events"]:
        return "rate_limited"
    if after_summary["rows"] <= before_summary["rows"]:
        return "no_new_rows"

    new_rows = after_summary["rows"] - before_summary["rows"]
    pending_delta = max(0, after_summary["pending_count"] - before_summary["pending_count"])
    pending_ratio = (pending_delta / new_rows) if new_rows else 0.0
    if pending_ratio > max_pending_ratio:
        return "pending_backlog"

    previous_best = before_summary.get("best_research_score")
    current_best = after_summary.get("best_research_score")
    if previous_best is not None and current_best is not None and current_best <= previous_best:
        return "no_improvement"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run guarded WorldQuant research cycles.")
    parser.add_argument("--profile", choices=("careful", "smart", "light", "full"), default="light", help="Run profile for each round.")
    parser.add_argument("--rounds", type=int, default=1, help="Maximum number of rounds to run.")
    parser.add_argument("--cooldown-seconds", type=int, default=180, help="Cooldown between rounds.")
    parser.add_argument("--max-pending-ratio", type=float, default=0.35, help="Pending-ratio stop threshold.")
    parser.add_argument("--subprocess-timeout", type=int, default=1800, help="Timeout in seconds for each child process launched by the research cycle.")
    args = parser.parse_args()

    auth_ok, auth_status = authentication_check()
    if not auth_ok:
        status = {
            "profile": args.profile,
            "rounds_requested": args.rounds,
            "rounds_completed": 0,
            "new_rows": 0,
            "pending_count": 0,
            "best_research_score": None,
            "score_delta_vs_previous_cycle": None,
            "rate_limit_events": 0,
            "stop_reason": "auth_failed",
            "next_action": choose_next_action("auth_failed"),
            "auth_status": auth_status,
        }
        write_status(status)
        write_cycle_report(REPO_ROOT)
        print(status["next_action"], file=sys.stderr)
        return 1

    before_rows = read_csv_rows()
    baseline_summary = summarize_rows(before_rows)
    previous_summary = dict(baseline_summary)
    total_rate_limit_events = 0
    rounds_completed = 0
    stop_reason = "completed"

    for round_index in range(args.rounds):
        result = run_subprocess(
            [sys.executable, "main.py", "--mode", args.profile],
            cwd=REPO_ROOT,
            timeout=args.subprocess_timeout,
        )
        round_events = inspect_run_output((result.stdout or "") + "\n" + (result.stderr or ""))
        total_rate_limit_events += round_events["rate_limit_events"]

        if result.returncode != 0:
            stop_reason = "auth_failed" if round_events["auth_failed"] else "run_failed"
            break

        rounds_completed += 1
        refresh_code = refresh_artifacts(REPO_ROOT, subprocess_timeout=args.subprocess_timeout)
        if refresh_code != 0:
            stop_reason = "run_failed"
            break

        current_rows = read_csv_rows()
        current_summary = summarize_rows(current_rows)
        stop_candidate = should_stop_after_round(
            before_summary=previous_summary,
            after_summary=current_summary,
            round_events=round_events,
            max_pending_ratio=args.max_pending_ratio,
        )
        previous_summary = current_summary
        if stop_candidate:
            stop_reason = stop_candidate
            break
        if round_index + 1 < args.rounds:
            time.sleep(args.cooldown_seconds)

    after_rows = read_csv_rows()
    final_summary = summarize_rows(after_rows)
    new_rows = max(0, final_summary["rows"] - baseline_summary["rows"])
    previous_best = baseline_summary.get("best_research_score")
    final_best = final_summary.get("best_research_score")
    if previous_best is None or final_best is None:
        score_delta = None
    else:
        score_delta = round(final_best - previous_best, 4)

    status = {
        "profile": args.profile,
        "rounds_requested": args.rounds,
        "rounds_completed": rounds_completed,
        "new_rows": new_rows,
        "pending_count": final_summary["pending_count"],
        "best_research_score": final_best,
        "score_delta_vs_previous_cycle": score_delta,
        "rate_limit_events": total_rate_limit_events,
        "stop_reason": stop_reason,
        "next_action": choose_next_action(stop_reason),
    }
    write_status(status)
    write_cycle_report(REPO_ROOT)

    if stop_reason in {"auth_failed", "run_failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
