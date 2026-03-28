#!/usr/bin/env python3
"""Shared runtime logging helpers for long-running loops."""

from __future__ import annotations

import io
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Iterator

DEFAULT_LOG_MAX_BYTES = 2 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 4


class RunContextFilter(logging.Filter):
    def __init__(self, run_id_getter: Callable[[], str | None]):
        super().__init__()
        self._run_id_getter = run_id_getter

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            run_id = self._run_id_getter()
        except Exception:
            run_id = None
        record.run_id = str(run_id or "-")
        return True


class LoggingTee(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int, original_stream):
        self._logger = logger
        self._level = level
        self._original_stream = original_stream
        self._buffer = ""

    def write(self, text: str) -> int:
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return 0
        self._original_stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        self._original_stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._original_stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._original_stream.fileno()

    def _emit(self, line: str) -> None:
        message = line.rstrip()
        if not message:
            return
        self._logger.log(self._level, message)


@dataclass
class RuntimeLoggingBundle:
    logger: logging.Logger
    stdout_logger: logging.Logger
    stderr_logger: logging.Logger
    system_log_path: Path
    error_log_path: Path

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()


def configure_runtime_logging(
    module_name: str,
    *,
    log_dir: str | Path,
    run_id_getter: Callable[[], str | None],
    log_max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    log_backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> RuntimeLoggingBundle:
    resolved_log_dir = Path(log_dir)
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    system_log_path = resolved_log_dir / "system.log"
    error_log_path = resolved_log_dir / "error.log"

    logger = logging.getLogger(f"brain_learn.runtime.{module_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    context_filter = RunContextFilter(run_id_getter)
    formatter = logging.Formatter(
        "%(asctime)s|%(levelname)s|module=%(name)s|run_id=%(run_id)s|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler_kwargs = {
        "filename": system_log_path,
        "mode": "a",
        "encoding": "utf-8",
    }
    if log_max_bytes and log_max_bytes > 0 and log_backup_count and log_backup_count > 0:
        system_handler = RotatingFileHandler(
            maxBytes=max(1, int(log_max_bytes)),
            backupCount=max(1, int(log_backup_count)),
            **handler_kwargs,
        )
    else:
        system_handler = logging.FileHandler(**handler_kwargs)
    system_handler.setLevel(logging.INFO)
    system_handler.setFormatter(formatter)
    system_handler.addFilter(context_filter)

    handler_kwargs = {
        "filename": error_log_path,
        "mode": "a",
        "encoding": "utf-8",
    }
    if log_max_bytes and log_max_bytes > 0 and log_backup_count and log_backup_count > 0:
        error_handler = RotatingFileHandler(
            maxBytes=max(1, int(log_max_bytes)),
            backupCount=max(1, int(log_backup_count)),
            **handler_kwargs,
        )
    else:
        error_handler = logging.FileHandler(**handler_kwargs)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    error_handler.addFilter(context_filter)

    logger.addHandler(system_handler)
    logger.addHandler(error_handler)

    stdout_logger = logger.getChild("stdout")
    stderr_logger = logger.getChild("stderr")
    return RuntimeLoggingBundle(
        logger=logger,
        stdout_logger=stdout_logger,
        stderr_logger=stderr_logger,
        system_log_path=system_log_path,
        error_log_path=error_log_path,
    )


@contextmanager
def redirect_standard_streams(bundle: RuntimeLoggingBundle) -> Iterator[None]:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    stdout_tee = LoggingTee(bundle.stdout_logger, logging.INFO, original_stdout)
    stderr_tee = LoggingTee(bundle.stderr_logger, logging.ERROR, original_stderr)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    try:
        yield
    finally:
        try:
            stdout_tee.flush()
            stderr_tee.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            bundle.close()
