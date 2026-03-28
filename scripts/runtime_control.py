#!/usr/bin/env python3
"""Helpers and CLI for runtime stop files and stale lock cleanup."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import atomic_write_json, iso_now, load_json


def pid_is_alive(pid: int | str | None) -> bool:
    try:
        pid_value = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_value <= 0:
        return False
    try:
        os.kill(pid_value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def load_runtime_metadata(path: str | Path) -> dict:
    return load_json(path, default={}, use_lock=False)


def write_signal_file(
    path: str | Path,
    *,
    kind: str,
    reason: str,
    run_id: str | None = None,
    pid: int | None = None,
    target_pid: int | None = None,
    source_status_file: str | Path | None = None,
    extra: dict | None = None,
) -> dict:
    payload = {
        "kind": kind,
        "pid": int(pid or os.getpid()),
        "target_pid": int(target_pid) if target_pid is not None else None,
        "timestamp": iso_now(),
        "run_id": run_id,
        "reason": reason,
    }
    if source_status_file is not None:
        payload["source_status_file"] = str(source_status_file)
    if isinstance(extra, dict) and extra:
        payload.update(extra)
    atomic_write_json(path, payload, use_lock=False)
    return payload


def remove_runtime_file(path: str | Path) -> bool:
    file_path = Path(path)
    if not file_path.exists():
        return False
    file_path.unlink()
    return True


def _warning(warn: Callable[[str], None] | None, message: str) -> None:
    if warn is not None:
        warn(message)


def clear_stale_stop_file(path: str | Path, *, warn: Callable[[str], None] | None = None) -> bool:
    stop_path = Path(path)
    if not stop_path.exists():
        return False
    payload = load_runtime_metadata(stop_path)
    if not payload:
        return False
    candidate_pid = payload.get("target_pid") or payload.get("pid")
    if not pid_is_alive(candidate_pid):
        remove_runtime_file(stop_path)
        _warning(
            warn,
            f"Removed stale stop file {stop_path} (pid {candidate_pid} is no longer alive).",
        )
        return True
    return False


def _lock_cleanup_reason(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    payload = load_runtime_metadata(path)
    if payload:
        pid = payload.get("pid")
        if not pid_is_alive(pid):
            return f"dead pid {pid}"
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size == 0:
        return "legacy empty lock"
    return "invalid lock metadata"


def clear_stale_lock_files(
    roots: Iterable[str | Path],
    *,
    warn: Callable[[str], None] | None = None,
) -> list[str]:
    removed: list[str] = []
    seen: set[Path] = set()
    for root in roots:
        root_path = Path(root)
        if root_path in seen:
            continue
        seen.add(root_path)
        if not root_path.exists():
            continue
        if root_path.is_file():
            candidates = [root_path] if root_path.name.endswith(".lock") else []
        else:
            candidates = sorted(path for path in root_path.rglob("*.lock") if path.is_file())
        for lock_path in candidates:
            reason = _lock_cleanup_reason(lock_path)
            if not reason:
                continue
            remove_runtime_file(lock_path)
            removed.append(str(lock_path))
            _warning(warn, f"Removed stale lock file {lock_path} ({reason}).")
    return removed


def request_stop_file(
    stop_file: str | Path,
    *,
    status_file: str | Path | None = None,
    reason: str = "user_requested_stop",
    run_id: str | None = None,
) -> dict:
    status_payload = load_runtime_metadata(status_file) if status_file else {}
    target_pid = status_payload.get("pid") if isinstance(status_payload, dict) else None
    resolved_run_id = run_id
    if not resolved_run_id and isinstance(status_payload, dict):
        resolved_run_id = str(status_payload.get("run_id") or "") or None
        if not resolved_run_id:
            resolved_run_id = str(((status_payload.get("summary") or {}).get("run_id")) or "") or None
    return write_signal_file(
        stop_file,
        kind="stop_request",
        reason=reason,
        run_id=resolved_run_id,
        target_pid=target_pid if pid_is_alive(target_pid) else None,
        source_status_file=status_file,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write runtime stop files with metadata or clear stale lock files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stop_parser = subparsers.add_parser("request-stop", help="Create a stop file with metadata from the latest status file.")
    stop_parser.add_argument("--stop-file", required=True, help="Target stop file path.")
    stop_parser.add_argument("--status-file", help="Optional status file used to capture target pid and run id.")
    stop_parser.add_argument("--reason", default="user_requested_stop", help="Reason recorded in the stop file.")
    stop_parser.add_argument("--run-id", help="Optional explicit run id to record.")

    cleanup_parser = subparsers.add_parser("clear-stale-locks", help="Remove stale *.lock files under the given roots.")
    cleanup_parser.add_argument("roots", nargs="+", help="Folders or lock files to inspect.")

    args = parser.parse_args()
    if args.command == "request-stop":
        payload = request_stop_file(
            args.stop_file,
            status_file=args.status_file,
            reason=args.reason,
            run_id=args.run_id,
        )
        print(f"Wrote stop file {args.stop_file} for run_id={payload.get('run_id') or '-'} target_pid={payload.get('target_pid') or '-'}")
        return 0
    removed = clear_stale_lock_files(args.roots, warn=lambda message: print(message, file=sys.stderr))
    print(f"Removed {len(removed)} stale lock file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
