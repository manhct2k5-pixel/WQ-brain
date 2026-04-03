from __future__ import annotations

import os
import pickle
import shutil
import tempfile
from datetime import datetime
from pathlib import Path


class SeedStoreCorruptError(RuntimeError):
    """Raised when the seed store cannot be safely read for an in-place update."""

    def __init__(self, path: Path, reason: str):
        self.path = Path(path)
        self.reason = reason
        super().__init__(
            f"Seed store is corrupted or unreadable: {self.path} ({reason}). "
            "Refusing to overwrite it automatically."
        )


READ_ERRORS = (EOFError, OSError, pickle.UnpicklingError, AttributeError, ImportError, ModuleNotFoundError, ValueError)


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


def backup_seed_store_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    return resolved_path.with_name(f"{resolved_path.name}.bak")


def dated_backup_seed_store_path(path: str | Path, *, timestamp: datetime | None = None) -> Path:
    resolved_path = Path(path)
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d")
    return resolved_path.with_name(f"{resolved_path.name}.{stamp}.bak")


def _seed_store_read_candidates(path: Path) -> list[Path]:
    candidates = [path]
    backup_path = backup_seed_store_path(path)
    if backup_path.exists():
        candidates.append(backup_path)
    dated_backups = sorted(
        path.parent.glob(f"{path.name}.*.bak"),
        key=lambda item: item.name,
        reverse=True,
    )
    candidates.extend(item for item in dated_backups if item not in candidates)
    return candidates


def _format_corruption_reason(errors: list[tuple[Path, str]]) -> str:
    return "; ".join(f"{path.name}: {reason}" for path, reason in errors)


def _allowed_type_label(allowed_types: type | tuple[type, ...] | None) -> str:
    if allowed_types is None:
        return "payload"
    if isinstance(allowed_types, tuple):
        return ", ".join(item.__name__ for item in allowed_types)
    return allowed_types.__name__


def _load_seed_payload(
    path: str | Path,
    *,
    on_corrupt: str = "empty",
    allowed_types: type | tuple[type, ...] | None = None,
    with_info: bool = False,
):
    resolved_path = Path(path)
    info = {
        "requested_path": str(resolved_path),
        "source_path": None,
        "fallback_used": False,
        "errors": [],
    }
    errors: list[tuple[Path, str]] = []

    for candidate_path in _seed_store_read_candidates(resolved_path):
        if not candidate_path.exists():
            continue
        try:
            with candidate_path.open("rb") as handle:
                payload = pickle.load(handle)
        except READ_ERRORS as exc:
            errors.append((candidate_path, str(exc)))
            info["errors"].append({"path": str(candidate_path), "reason": str(exc)})
            continue
        if allowed_types is not None and not isinstance(payload, allowed_types):
            reason = f"expected {_allowed_type_label(allowed_types)}, got {type(payload).__name__}"
            errors.append((candidate_path, reason))
            info["errors"].append({"path": str(candidate_path), "reason": reason})
            continue
        info["source_path"] = str(candidate_path)
        info["fallback_used"] = candidate_path != resolved_path
        return (payload, info) if with_info else payload

    if on_corrupt == "raise" and errors:
        raise SeedStoreCorruptError(resolved_path, _format_corruption_reason(errors))
    payload = {}
    return (payload, info) if with_info else payload


def load_seed_payload(
    path: str | Path,
    *,
    on_corrupt: str = "empty",
    allowed_types: type | tuple[type, ...] | None = None,
    with_info: bool = False,
):
    return _load_seed_payload(
        path,
        on_corrupt=on_corrupt,
        allowed_types=allowed_types,
        with_info=with_info,
    )


def load_seed_store(path: str | Path, *, on_corrupt: str = "empty") -> dict:
    payload = _load_seed_payload(path, on_corrupt=on_corrupt, allowed_types=dict)
    return payload if isinstance(payload, dict) else {}


def _atomic_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with source.open("rb") as source_handle, os.fdopen(fd, "wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(tmp_path, target)
        _fsync_directory(target.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def write_seed_store(path: str | Path, payload: dict) -> None:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(resolved_path.parent),
        prefix=f".{resolved_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, resolved_path)
        _fsync_directory(resolved_path.parent)
        _atomic_copy_file(resolved_path, backup_seed_store_path(resolved_path))
        _atomic_copy_file(resolved_path, dated_backup_seed_store_path(resolved_path))
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
