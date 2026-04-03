#!/usr/bin/env python3
"""Apply retention, archival, and compression policies to run artifacts."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.flow_utils import (
    ARCHIVE_DIR,
    ARTIFACTS_DIR,
    LEGACY_RUNS_DIR,
    RECENT_RUNS_DIR,
    STATE_DIR,
    atomic_write_json,
    iso_now,
    load_json,
)

COMPRESSIBLE_SUFFIXES = {".jsonl", ".csv"}
TEMP_FILE_SUFFIXES = {".tmp", ".partial"}
RETENTION_TAGS = {"standard", "important", "keep"}
RETENTION_MARKERS = {
    ".important": "important",
    ".keep": "keep",
}
ARCHIVE_METADATA_FILE = "archive_metadata.json"
RETENTION_TAG_FILE = "retention_tag.json"
STATUS_PATH = STATE_DIR / "artifact_cleanup_status.json"


def _iter_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _read_json_file_loose(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def _parse_timestamp(value: str | None) -> float | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _latest_file_mtime(run_dir: Path) -> float:
    latest = run_dir.stat().st_mtime
    for path in run_dir.rglob("*"):
        try:
            stat = path.stat()
        except OSError:
            continue
        latest = max(latest, stat.st_mtime)
    return latest


def _reference_timestamp(run_dir: Path) -> float:
    archive_metadata = _read_json_file_loose(run_dir / ARCHIVE_METADATA_FILE)
    original_ts = archive_metadata.get("original_last_updated_ts")
    if isinstance(original_ts, (int, float)):
        return float(original_ts)

    summary_payload = _read_json_file_loose(run_dir / "orchestrator_summary.json")
    summary_ts = _parse_timestamp(summary_payload.get("generated_at"))
    if summary_ts is not None:
        return summary_ts

    retention_payload = _read_json_file_loose(run_dir / RETENTION_TAG_FILE)
    retention_ts = _parse_timestamp(retention_payload.get("set_at"))
    if retention_ts is not None:
        return retention_ts

    return _latest_file_mtime(run_dir)


def _age_days(run_dir: Path, *, now_ts: float) -> float:
    return max(0.0, (now_ts - _reference_timestamp(run_dir)) / 86400.0)


def _retention_tag(run_dir: Path) -> tuple[str, str]:
    for marker_name, tag in RETENTION_MARKERS.items():
        if (run_dir / marker_name).exists():
            return tag, marker_name

    tag_payload = _read_json_file_loose(run_dir / RETENTION_TAG_FILE)
    tag = str(tag_payload.get("tag") or "").strip().lower()
    if tag in RETENTION_TAGS:
        return tag, RETENTION_TAG_FILE

    summary_payload = _read_json_file_loose(run_dir / "orchestrator_summary.json")
    summary_tag = str(summary_payload.get("retention_tag") or "").strip().lower()
    if summary_tag in RETENTION_TAGS:
        return summary_tag, "orchestrator_summary.json"
    if int(summary_payload.get("submit_ready_candidates") or 0) > 0:
        return "important", "submit_ready_candidates"

    return "standard", "default"


def _delete_threshold_days(tag: str, *, archive_delete_days: int, important_archive_delete_days: int) -> int | None:
    if tag == "keep":
        return None
    if tag == "important":
        return important_archive_delete_days
    return archive_delete_days


def _compress_file(path: Path) -> Path | None:
    if path.suffix not in COMPRESSIBLE_SUFFIXES:
        return None
    compressed_path = path.with_name(f"{path.name}.gz")
    if compressed_path.exists():
        return None

    source_stat = path.stat()
    with path.open("rb") as source_handle, gzip.open(compressed_path, "wb", compresslevel=6) as target_handle:
        shutil.copyfileobj(source_handle, target_handle)
    os.utime(compressed_path, (source_stat.st_atime, source_stat.st_mtime))
    path.unlink()
    return compressed_path


def _compress_run_files(run_dir: Path, *, min_bytes: int) -> list[str]:
    compressed: list[str] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in COMPRESSIBLE_SUFFIXES:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < min_bytes:
            continue
        compressed_path = _compress_file(path)
        if compressed_path is not None:
            compressed.append(str(compressed_path.relative_to(run_dir)))
    return compressed


def _is_temp_artifact_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.endswith(".lock"):
        return False
    if path.suffix.lower() in TEMP_FILE_SUFFIXES:
        return True
    return False


def _cleanup_temp_files(artifacts_root: Path, *, max_age_seconds: int) -> list[str]:
    removed: list[str] = []
    now_ts = datetime.now().timestamp()
    for path in sorted(artifacts_root.rglob("*")):
        if not _is_temp_artifact_file(path):
            continue
        try:
            age_seconds = max(0.0, now_ts - path.stat().st_mtime)
        except OSError:
            continue
        if max_age_seconds > 0 and age_seconds < max_age_seconds:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(str(path.relative_to(artifacts_root)))
    return removed


def _directory_footprint(path: Path) -> dict:
    total_bytes = 0
    file_count = 0
    log_bytes = 0
    log_file_count = 0
    largest_files: list[tuple[int, str]] = []

    if not path.exists():
        return {
            "bytes": 0,
            "megabytes": 0.0,
            "file_count": 0,
            "log_bytes": 0,
            "log_file_count": 0,
            "largest_files": [],
        }

    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        try:
            size = candidate.stat().st_size
        except OSError:
            continue
        total_bytes += size
        file_count += 1
        if candidate.suffix.lower().startswith(".log") or ".log." in candidate.name:
            log_bytes += size
            log_file_count += 1
        largest_files.append((size, str(candidate.relative_to(path))))

    largest_files.sort(reverse=True)
    return {
        "bytes": total_bytes,
        "megabytes": round(total_bytes / (1024 * 1024), 2),
        "file_count": file_count,
        "log_bytes": log_bytes,
        "log_file_count": log_file_count,
        "largest_files": [
            {"path": relative_path, "bytes": size}
            for size, relative_path in largest_files[:5]
        ],
    }


def describe_artifact_footprint(artifacts_root: Path = ARTIFACTS_DIR) -> dict:
    buckets = {
        "recent_runs": artifacts_root / RECENT_RUNS_DIR.name,
        "archive": artifacts_root / ARCHIVE_DIR.name,
        "latest": artifacts_root / "latest",
        "state": artifacts_root / "state",
        "quarantine": artifacts_root / "quarantine",
    }
    footprint = {
        "root": str(artifacts_root),
        "total": _directory_footprint(artifacts_root),
        "buckets": {},
        "recent_run_count": len(_iter_run_dirs(buckets["recent_runs"])),
        "archive_run_count": len(_iter_run_dirs(buckets["archive"])),
    }
    for name, bucket_path in buckets.items():
        footprint["buckets"][name] = _directory_footprint(bucket_path)
    return footprint


def _move_run(source_dir: Path, target_root: Path) -> Path:
    target_root.mkdir(parents=True, exist_ok=True)
    target_dir = target_root / source_dir.name
    if target_dir.exists():
        target_dir = target_root / f"{source_dir.name}__archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.move(str(source_dir), str(target_dir))
    return target_dir


def _write_archive_metadata(
    run_dir: Path,
    *,
    source_dir: Path,
    original_last_updated_ts: float,
    retention_tag: str,
    retention_tag_source: str,
    compressed_files: list[str],
) -> None:
    existing = load_json(run_dir / ARCHIVE_METADATA_FILE, default={}, use_lock=False)
    payload = {
        **existing,
        "archived_at": iso_now(),
        "source_run_dir": str(source_dir),
        "original_last_updated_ts": original_last_updated_ts,
        "retention_tag": retention_tag,
        "retention_tag_source": retention_tag_source,
        "compressed_files": compressed_files,
    }
    atomic_write_json(run_dir / ARCHIVE_METADATA_FILE, payload, use_lock=False)


def cleanup_artifacts(
    *,
    artifacts_root: Path = ARTIFACTS_DIR,
    recent_days: int = 7,
    archive_delete_days: int = 30,
    important_archive_delete_days: int = 90,
    compress_min_bytes: int = 64 * 1024,
    max_recent_runs: int = 24,
    temp_file_max_age_hours: int = 12,
    protected_run_ids: set[str] | None = None,
) -> dict:
    recent_runs_dir = artifacts_root / RECENT_RUNS_DIR.name
    archive_dir = artifacts_root / ARCHIVE_DIR.name
    legacy_runs_dir = artifacts_root / LEGACY_RUNS_DIR.name
    state_dir = artifacts_root / STATE_DIR.name
    recent_runs_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    now_ts = datetime.now().timestamp()
    protected = {str(item).strip() for item in (protected_run_ids or set()) if str(item).strip()}
    summary = {
        "generated_at": iso_now(),
        "artifacts_root": str(artifacts_root),
        "recent_runs_dir": str(recent_runs_dir),
        "archive_dir": str(archive_dir),
        "legacy_runs_dir": str(legacy_runs_dir),
        "recent_days": recent_days,
        "archive_delete_days": archive_delete_days,
        "important_archive_delete_days": important_archive_delete_days,
        "compress_min_bytes": compress_min_bytes,
        "max_recent_runs": max_recent_runs,
        "temp_file_max_age_hours": temp_file_max_age_hours,
        "protected_run_ids": sorted(protected),
        "migrated_legacy_runs": [],
        "archived_runs": [],
        "deleted_archives": [],
        "kept_recent_runs": [],
        "kept_archives": [],
        "compressed_files": [],
        "temp_files_removed": [],
    }

    summary["temp_files_removed"] = _cleanup_temp_files(
        artifacts_root,
        max_age_seconds=max(0, int(temp_file_max_age_hours * 3600)),
    )

    for legacy_run in _iter_run_dirs(legacy_runs_dir):
        if legacy_run.name in protected:
            continue
        moved_path = _move_run(legacy_run, recent_runs_dir)
        summary["migrated_legacy_runs"].append(
            {
                "run_id": legacy_run.name,
                "target": str(moved_path),
            }
        )
    try:
        if legacy_runs_dir.exists() and not any(legacy_runs_dir.iterdir()):
            legacy_runs_dir.rmdir()
    except OSError:
        pass

    recent_run_dirs = sorted(_iter_run_dirs(recent_runs_dir), key=_reference_timestamp, reverse=True)
    kept_recent_count = 0
    for run_dir in recent_run_dirs:
        if run_dir.name in protected:
            kept_recent_count += 1
            summary["kept_recent_runs"].append(
                {
                    "run_id": run_dir.name,
                    "age_days": round(_age_days(run_dir, now_ts=now_ts), 2),
                    "retention_tag": _retention_tag(run_dir)[0],
                    "reason": "protected_run_id",
                }
            )
            continue
        tag, tag_source = _retention_tag(run_dir)
        age_days = _age_days(run_dir, now_ts=now_ts)
        keep_due_to_age = age_days <= recent_days
        keep_due_to_count = max_recent_runs <= 0 or kept_recent_count < max_recent_runs
        if keep_due_to_age and keep_due_to_count:
            kept_recent_count += 1
            summary["kept_recent_runs"].append(
                {
                    "run_id": run_dir.name,
                    "age_days": round(age_days, 2),
                    "retention_tag": tag,
                }
            )
            continue
        compressed_files = _compress_run_files(run_dir, min_bytes=compress_min_bytes)
        original_last_updated_ts = _reference_timestamp(run_dir)
        archived_dir = _move_run(run_dir, archive_dir)
        _write_archive_metadata(
            archived_dir,
            source_dir=run_dir,
            original_last_updated_ts=original_last_updated_ts,
            retention_tag=tag,
            retention_tag_source=tag_source,
            compressed_files=compressed_files,
        )
        summary["compressed_files"].extend(f"{archived_dir.name}/{path}" for path in compressed_files)
        summary["archived_runs"].append(
            {
                "run_id": archived_dir.name,
                "age_days": round(age_days, 2),
                "retention_tag": tag,
                "retention_tag_source": tag_source,
                "archive_path": str(archived_dir),
                "archive_reason": "max_recent_runs" if keep_due_to_age and not keep_due_to_count else "recent_days",
            }
        )

    for run_dir in _iter_run_dirs(archive_dir):
        if run_dir.name in protected:
            summary["kept_archives"].append(
                {
                    "run_id": run_dir.name,
                    "age_days": round(_age_days(run_dir, now_ts=now_ts), 2),
                    "retention_tag": _retention_tag(run_dir)[0],
                    "delete_after_days": None,
                    "reason": "protected_run_id",
                }
            )
            continue
        tag, _tag_source = _retention_tag(run_dir)
        age_days = _age_days(run_dir, now_ts=now_ts)
        delete_threshold = _delete_threshold_days(
            tag,
            archive_delete_days=archive_delete_days,
            important_archive_delete_days=important_archive_delete_days,
        )
        if delete_threshold is not None and age_days > delete_threshold:
            shutil.rmtree(run_dir)
            summary["deleted_archives"].append(
                {
                    "run_id": run_dir.name,
                    "age_days": round(age_days, 2),
                    "retention_tag": tag,
                }
            )
            continue
        summary["kept_archives"].append(
            {
                "run_id": run_dir.name,
                "age_days": round(age_days, 2),
                "retention_tag": tag,
                "delete_after_days": delete_threshold,
            }
        )

    summary["artifact_footprint"] = describe_artifact_footprint(artifacts_root)

    status_path = artifacts_root / "state" / STATUS_PATH.name
    atomic_write_json(status_path, summary, use_lock=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Archive and clean old run artifacts.")
    parser.add_argument("--artifacts-root", default=str(ARTIFACTS_DIR), help="Artifact root folder.")
    parser.add_argument("--recent-days", type=int, default=7, help="Keep runs in recent_runs for this many days before archiving.")
    parser.add_argument("--archive-delete-days", type=int, default=30, help="Delete standard archived runs after this many days.")
    parser.add_argument(
        "--important-archive-delete-days",
        type=int,
        default=90,
        help="Delete important archived runs after this many days. 'keep' tagged runs are never auto-deleted.",
    )
    parser.add_argument(
        "--compress-min-bytes",
        type=int,
        default=64 * 1024,
        help="Only gzip .jsonl/.csv files at or above this size when archiving.",
    )
    parser.add_argument(
        "--max-recent-runs",
        type=int,
        default=24,
        help="Maximum number of newest recent_runs folders to keep unarchived. 0 disables the count cap.",
    )
    parser.add_argument(
        "--temp-file-max-age-hours",
        type=int,
        default=12,
        help="Delete leftover *.tmp/*.partial artifact files at or above this age. 0 removes all such temp files immediately.",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root)
    summary = cleanup_artifacts(
        artifacts_root=artifacts_root,
        recent_days=max(0, args.recent_days),
        archive_delete_days=max(0, args.archive_delete_days),
        important_archive_delete_days=max(0, args.important_archive_delete_days),
        compress_min_bytes=max(0, args.compress_min_bytes),
        max_recent_runs=max(0, args.max_recent_runs),
        temp_file_max_age_hours=max(0, args.temp_file_max_age_hours),
    )
    print(
        f"[cleanup_artifacts] archived={len(summary['archived_runs'])} "
        f"deleted={len(summary['deleted_archives'])} "
        f"migrated={len(summary['migrated_legacy_runs'])} "
        f"compressed={len(summary['compressed_files'])} "
        f"temp_removed={len(summary['temp_files_removed'])} "
        f"artifact_mb={summary.get('artifact_footprint', {}).get('total', {}).get('megabytes', 0.0)}"
    )
    print(str(artifacts_root / "state" / STATUS_PATH.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
