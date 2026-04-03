#!/usr/bin/env python3
"""Helpers for run-scoped artifact orchestration."""

from __future__ import annotations

import os
import json
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
QUARANTINE_DIR = ARTIFACTS_DIR / "quarantine"
LEGACY_RUNS_DIR = ARTIFACTS_DIR / "runs"
RECENT_RUNS_DIR = ARTIFACTS_DIR / "recent_runs"
ARCHIVE_DIR = ARTIFACTS_DIR / "archive"
RUNS_DIR = RECENT_RUNS_DIR
LATEST_DIR = ARTIFACTS_DIR / "latest"
STATE_DIR = ARTIFACTS_DIR / "state"
LATEST_METADATA_FILE_NAME = "latest_metadata.json"
JSON_ARTIFACT_SCHEMA_VERSION = 1

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


@dataclass(frozen=True)
class JsonArtifactSchema:
    version: int = JSON_ARTIFACT_SCHEMA_VERSION
    required_fields: tuple[str, ...] = ()
    validator: Callable[[dict[str, Any]], str | None] | None = None
    migrate: Callable[[dict[str, Any], int | None], dict[str, Any]] | None = None


def _validate_candidate_payload(payload: dict[str, Any]) -> str | None:
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        return None
    batch = payload.get("batch")
    if isinstance(batch, dict) and isinstance(batch.get("candidates"), list):
        return None
    return "expected candidates or batch.candidates to be a list"


def _validate_scout_payload(payload: dict[str, Any]) -> str | None:
    if isinstance(payload.get("selected"), list):
        return None
    return _validate_candidate_payload(payload)


GENERIC_ARTIFACT_JSON_SCHEMA = JsonArtifactSchema()
ARTIFACT_JSON_SCHEMAS: dict[str, JsonArtifactSchema] = {
    "latest_run.json": JsonArtifactSchema(
        required_fields=("generated_at", "profile", "history", "hall_of_fame"),
    ),
    "trang_thai_chay.json": JsonArtifactSchema(),
    "run_memory.json": JsonArtifactSchema(),
    "current_memory_snapshot.json": JsonArtifactSchema(),
    "previous_memory_snapshot.json": JsonArtifactSchema(),
    "bo_nho_nghien_cuu.json": JsonArtifactSchema(),
    "global_research_memory.json": JsonArtifactSchema(),
    "bo_nho.json": JsonArtifactSchema(),
    "phan_hoi_brain.json": JsonArtifactSchema(),
    "kien_thuc_tu_hoc.json": JsonArtifactSchema(),
    "cache_tim_kiem.json": JsonArtifactSchema(),
    "trang_thai_bao_cao.json": JsonArtifactSchema(),
    "retention_tag.json": JsonArtifactSchema(),
    "archive_metadata.json": JsonArtifactSchema(),
    "artifact_cleanup_status.json": JsonArtifactSchema(),
    "scout_loop_status.json": JsonArtifactSchema(),
    "orchestrator_loop_status.json": JsonArtifactSchema(),
    "planned_candidates.json": JsonArtifactSchema(
        required_fields=("batch",),
        validator=_validate_candidate_payload,
    ),
    "lo_tiep_theo.json": JsonArtifactSchema(
        required_fields=("batch",),
        validator=_validate_candidate_payload,
    ),
    "pending_simulation_queue.json": JsonArtifactSchema(
        required_fields=("candidates", "candidate_count"),
    ),
    "results_summary.json": JsonArtifactSchema(
        required_fields=("summary",),
    ),
    "evaluated_candidates.json": JsonArtifactSchema(
        required_fields=("candidates", "summary"),
    ),
    "auto_fix_candidates.json": JsonArtifactSchema(
        required_fields=("candidates",),
    ),
    "du_lieu.json": JsonArtifactSchema(
        required_fields=("batch", "selected", "watchlist"),
        validator=_validate_scout_payload,
    ),
    "scout_candidates.json": JsonArtifactSchema(
        required_fields=("batch",),
        validator=_validate_scout_payload,
    ),
    "latest_metadata.json": JsonArtifactSchema(
        required_fields=("status", "complete", "artifacts"),
    ),
    "orchestrator_checkpoint.json": JsonArtifactSchema(
        required_fields=("run_id", "batch_id", "stages"),
    ),
    "orchestrator_summary.json": JsonArtifactSchema(
        required_fields=("run_id", "profile"),
    ),
    "seed_approval.json": JsonArtifactSchema(
        required_fields=("selected_count", "inserted_count", "rejected_count"),
    ),
    "seed_submit_ready.json": JsonArtifactSchema(
        required_fields=("selected_count", "inserted_count", "rejected_count"),
    ),
}


def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_run_id(prefix: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        safe_prefix = "".join(ch for ch in str(prefix) if ch.isalnum() or ch in {"-", "_"}).strip("_-")
        if safe_prefix:
            return f"{safe_prefix}_{stamp}"
    return stamp


def ensure_runtime_layout(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    RECENT_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_metadata_path(latest_dir: str | Path | None = None) -> Path:
    resolved_latest_dir = Path(latest_dir) if latest_dir is not None else LATEST_DIR
    return resolved_latest_dir / LATEST_METADATA_FILE_NAME


def _default_payload(default):
    return {} if default is None else default


def _emit_warning(message: str, *, warn: Callable[[str], None] | None = None) -> None:
    if warn is not None:
        warn(message)


def _path_looks_like_artifact_json(path: Path) -> bool:
    if path.suffix.lower() != ".json":
        return False
    if path.name in ARTIFACT_JSON_SCHEMAS:
        return True
    return "artifacts" in path.parts


def _schema_for_path(path: str | Path | None) -> JsonArtifactSchema | None:
    if path is None:
        return None
    file_path = Path(path)
    if file_path.suffix.lower() != ".json":
        return None
    if file_path.name in ARTIFACT_JSON_SCHEMAS:
        return ARTIFACT_JSON_SCHEMAS[file_path.name]
    if _path_looks_like_artifact_json(file_path):
        return GENERIC_ARTIFACT_JSON_SCHEMA
    return None


def _coerce_schema_version(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _default_migrate_artifact_payload(payload: dict[str, Any], _from_version: int | None) -> dict[str, Any]:
    return dict(payload)


def _migrate_artifact_payload(
    payload: dict[str, Any],
    *,
    schema: JsonArtifactSchema,
    label: str,
    warn: Callable[[str], None] | None = None,
) -> dict[str, Any] | None:
    raw_version = payload.get("schema_version")
    version = _coerce_schema_version(raw_version)

    if raw_version is None:
        migrated = dict(payload)
        migrated["schema_version"] = schema.version
        _emit_warning(
            f"{label}: missing schema_version; treating as legacy JSON and migrating to schema_version={schema.version}.",
            warn=warn,
        )
        return migrated

    if version is None:
        _emit_warning(f"{label}: invalid schema_version value {raw_version!r}.", warn=warn)
        return None

    if version > schema.version:
        _emit_warning(
            f"{label}: unsupported schema_version={version}; current reader supports up to {schema.version}.",
            warn=warn,
        )
        return None

    migrated = dict(payload)
    if version < schema.version:
        migrate = schema.migrate or _default_migrate_artifact_payload
        try:
            migrated = migrate(dict(payload), version)
        except Exception as exc:
            _emit_warning(
                f"{label}: failed to migrate schema_version {version} -> {schema.version} ({exc}).",
                warn=warn,
            )
            return None
        if not isinstance(migrated, dict):
            _emit_warning(f"{label}: schema migrator must return a JSON object.", warn=warn)
            return None
        _emit_warning(
            f"{label}: migrated schema_version {version} -> {schema.version}.",
            warn=warn,
        )

    migrated["schema_version"] = schema.version
    return migrated


def _validate_artifact_payload(payload: dict[str, Any], *, schema: JsonArtifactSchema) -> str | None:
    missing_fields = [field for field in schema.required_fields if field not in payload]
    if missing_fields:
        return f"missing required fields: {', '.join(missing_fields)}"
    if schema.validator is not None:
        return schema.validator(payload)
    return None


def _prepare_json_payload_for_write(path: str | Path, payload):
    schema = _schema_for_path(path)
    if schema is None or not isinstance(payload, dict):
        return payload
    prepared = dict(payload)
    prepared["schema_version"] = schema.version
    return prepared


def _safe_name_fragment(value: str | None, *, default: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:80] or default


def _artifact_root_for_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).resolve(strict=False)
    parts = resolved.parts
    if "artifacts" not in parts:
        return None
    index = parts.index("artifacts")
    return Path(*parts[: index + 1])


def _default_quarantine_root(path: str | Path | None = None, *, root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    artifact_root = _artifact_root_for_path(path)
    if artifact_root is not None:
        return artifact_root / "quarantine"
    if path is not None:
        return Path(path).resolve(strict=False).parent / "quarantine"
    return QUARANTINE_DIR


def _quarantine_target(
    category_dir: Path,
    *,
    label: str | None = None,
    source_path: str | Path | None = None,
    suffix: str = ".json",
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source_fragment = _safe_name_fragment(Path(source_path).stem if source_path else None, default="artifact")
    label_fragment = _safe_name_fragment(label, default="quarantine")
    token_source = f"{source_path or ''}|{label or ''}|{stamp}"
    token = str(abs(hash(token_source)))[:10]
    safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return category_dir / f"{stamp}__{label_fragment}__{source_fragment}__{token}{safe_suffix}"


def quarantine_payload(
    payload,
    *,
    reason: str,
    category: str = "records",
    label: str | None = None,
    source_path: str | Path | None = None,
    details: dict[str, Any] | None = None,
    root: str | Path | None = None,
    warn: Callable[[str], None] | None = None,
) -> Path:
    quarantine_root = _default_quarantine_root(source_path, root=root)
    category_dir = quarantine_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    target_path = _quarantine_target(category_dir, label=label, source_path=source_path)
    wrapped = {
        "kind": category.rstrip("s") or "record",
        "quarantined_at": iso_now(),
        "reason": reason,
        "source_path": str(source_path) if source_path is not None else "",
        "details": details or {},
        "payload": payload,
    }
    atomic_write_json(target_path, wrapped, use_lock=False)
    _emit_warning(f"Quarantined {wrapped['kind']} to {target_path} ({reason}).", warn=warn)
    return target_path


def quarantine_file(
    path: str | Path,
    *,
    reason: str,
    details: dict[str, Any] | None = None,
    raw_content: str | None = None,
    root: str | Path | None = None,
    warn: Callable[[str], None] | None = None,
) -> Path | None:
    file_path = Path(path)
    if not file_path.exists() and raw_content is None:
        return None

    quarantine_root = _default_quarantine_root(file_path, root=root)
    files_dir = quarantine_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    suffix = file_path.suffix or ".txt"
    target_path = _quarantine_target(files_dir, label="file", source_path=file_path, suffix=suffix)
    action = "snapshot"

    if file_path.exists():
        artifact_root = _artifact_root_for_path(file_path)
        if artifact_root is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(file_path, target_path)
            action = "moved"
        else:
            shutil.copy2(file_path, target_path)
            action = "copied"
    else:
        atomic_write_text(target_path, raw_content or "", use_lock=False)

    metadata_path = target_path.parent / f"{target_path.name}.meta.json"
    atomic_write_json(
        metadata_path,
        {
            "kind": "file",
            "quarantined_at": iso_now(),
            "reason": reason,
            "source_path": str(file_path),
            "quarantine_path": str(target_path),
            "action": action,
            "details": details or {},
        },
        use_lock=False,
    )
    _emit_warning(f"Quarantined file to {target_path} ({reason}).", warn=warn)
    return target_path


def _lock_path(path: str | Path) -> Path:
    file_path = Path(path)
    return file_path.parent / f".{file_path.name}.lock"


def _write_lock_metadata(handle, lock_path: Path, *, shared: bool) -> None:
    if shared:
        return
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "kind": "file_lock",
                    "pid": os.getpid(),
                    "timestamp": iso_now(),
                    "mode": "shared" if shared else "exclusive",
                    "lock_path": str(lock_path),
                },
                ensure_ascii=True,
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
    except OSError:
        pass


@contextmanager
def file_lock(path: str | Path, *, shared: bool = False, enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if fcntl is not None:
            mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(handle.fileno(), mode)
        _write_lock_metadata(handle, lock_path, shared=shared)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_RDONLY"):
        return
    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def load_json(
    path: str | Path | None,
    default=None,
    *,
    required_fields: Iterable[str] | None = None,
    validator: Callable[[dict[str, Any]], str | None] | None = None,
    context: str | None = None,
    warn: Callable[[str], None] | None = None,
    use_lock: bool = True,
):
    fallback = _default_payload(default)
    if path is None:
        return fallback
    file_path = Path(path)
    if not file_path.exists():
        return fallback
    quarantine_root = _default_quarantine_root(file_path)

    def _quarantine(reason: str, *, details: dict[str, Any] | None = None) -> None:
        if _schema_for_path(file_path) is None:
            return
        quarantine_file(
            file_path,
            reason=reason,
            details=details,
            root=quarantine_root,
            warn=warn,
        )

    try:
        with file_lock(file_path, shared=True, enabled=use_lock):
            payload = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        label = context or str(file_path)
        _emit_warning(f"{label}: cannot read JSON ({exc}).", warn=warn)
        return fallback
    except json.JSONDecodeError as exc:
        label = context or str(file_path)
        _emit_warning(f"{label}: invalid JSON ({exc}).", warn=warn)
        _quarantine("invalid_json", details={"context": label, "error": str(exc)})
        return fallback
    if not isinstance(payload, dict):
        label = context or str(file_path)
        _emit_warning(f"{label}: expected a JSON object at the top level.", warn=warn)
        _quarantine("top_level_not_object", details={"context": label, "type": type(payload).__name__})
        return fallback
    label = context or str(file_path)
    schema = _schema_for_path(file_path)
    if schema is not None:
        migrated_payload = _migrate_artifact_payload(payload, schema=schema, label=label, warn=warn)
        if migrated_payload is None:
            _quarantine("schema_version_error", details={"context": label})
            return fallback
        payload = migrated_payload
        validation_error = _validate_artifact_payload(payload, schema=schema)
        if validation_error:
            _emit_warning(f"{label}: {validation_error}.", warn=warn)
            _quarantine("schema_validation_failed", details={"context": label, "error": validation_error})
            return fallback
    if required_fields:
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            joined = ", ".join(missing_fields)
            _emit_warning(f"{label}: missing required fields: {joined}.", warn=warn)
            _quarantine("missing_required_fields", details={"context": label, "fields": missing_fields})
            return fallback
    if validator is not None:
        validation_error = validator(payload)
        if validation_error:
            _emit_warning(f"{label}: {validation_error}.", warn=warn)
            _quarantine("validator_failed", details={"context": label, "error": validation_error})
            return fallback
    return payload


def atomic_write_text(path: str | Path, content: str, *, use_lock: bool = True) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with file_lock(file_path, enabled=use_lock):
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temp_path = Path(handle.name)
            os.replace(temp_path, file_path)
            _fsync_directory(file_path.parent)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    return file_path


def atomic_write_json(path: str | Path, payload, *, use_lock: bool = True) -> Path:
    prepared_payload = _prepare_json_payload_for_write(path, payload)
    return atomic_write_text(path, json.dumps(prepared_payload, indent=2, ensure_ascii=True), use_lock=use_lock)


def copy_file(source: str | Path, target: str | Path, *, use_lock: bool = True) -> Path:
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    with file_lock(target_path, enabled=use_lock):
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                dir=target_path.parent,
                prefix=f".{target_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
            shutil.copy2(source_path, temp_path)
            with temp_path.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_path, target_path)
            _fsync_directory(target_path.parent)
        except Exception:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
    return target_path


def write_jsonl(path: str | Path, records: list[dict]) -> Path:
    lines = [json.dumps(record, ensure_ascii=True) for record in records]
    content = "\n".join(lines)
    if lines:
        content += "\n"
    return atomic_write_text(path, content)


def read_jsonl(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    records = []
    quarantine_root = _default_quarantine_root(file_path)
    for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            quarantine_payload(
                {"line_number": line_number, "raw_line": raw_line},
                reason="invalid_jsonl_line",
                category="jsonl_lines",
                label=file_path.stem,
                source_path=file_path,
                details={},
                root=quarantine_root,
            )
            continue
        if isinstance(payload, dict):
            records.append(payload)
            continue
        quarantine_payload(
            {"line_number": line_number, "raw_line": raw_line, "parsed_type": type(payload).__name__},
            reason="jsonl_top_level_not_object",
            category="jsonl_lines",
            label=file_path.stem,
            source_path=file_path,
            details={},
            root=quarantine_root,
        )
    return records


def sync_to_paths(source: str | Path, *targets: str | Path) -> None:
    for target in targets:
        copy_file(source, target)


def load_latest_publish_metadata(latest_dir: str | Path | None = None, *, warn: Callable[[str], None] | None = None) -> dict:
    metadata_path = latest_metadata_path(latest_dir)
    return load_json(
        metadata_path,
        default={},
        context=f"latest publish metadata {metadata_path}",
        warn=warn,
    )


def latest_publish_is_complete(
    *,
    latest_dir: str | Path | None = None,
    run_id: str | None = None,
    batch_id: str | None = None,
    required_artifact_names: Iterable[str] | None = None,
    warn: Callable[[str], None] | None = None,
) -> bool:
    metadata = load_latest_publish_metadata(latest_dir, warn=warn)
    if not metadata:
        return False
    if metadata.get("status") != "complete" or not bool(metadata.get("complete")):
        return False
    if run_id is not None and str(metadata.get("run_id") or "") != str(run_id):
        return False
    if batch_id is not None and str(metadata.get("batch_id") or "") != str(batch_id):
        return False
    artifact_map = metadata.get("artifacts")
    if required_artifact_names:
        if not isinstance(artifact_map, dict):
            return False
        for artifact_name in required_artifact_names:
            artifact_path = artifact_map.get(artifact_name)
            if not artifact_path or not Path(artifact_path).exists():
                return False
    return True
