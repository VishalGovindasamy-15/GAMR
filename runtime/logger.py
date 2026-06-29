"""
Structured logging setup for GAMR.

Creates three log files per run:
  runtime.log   — main runtime events
  scheduler.log — scheduler decisions
  monitor.log   — hardware metric collection

All loggers also write to stdout.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s"
_DATE_FMT = "%Y-%m-%dT%H:%M:%S"


def setup_logging(run_dir: Path, log_level: str = "INFO") -> None:
    """
    Initialise logging for the current run.

    Must be called after run_dir is created and before any loggers are used.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # ── Console (stdout) ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(level)

    # ── Root logger ───────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(_file_handler(run_dir / "runtime.log", formatter, level))

    # ── Dedicated scheduler log ───────────────────────────────────────
    _add_file_handler(
        "gamr.scheduler", run_dir / "scheduler.log", formatter, level
    )

    # ── Dedicated monitor log ─────────────────────────────────────────
    _add_file_handler(
        "gamr.monitor", run_dir / "monitor.log", formatter, level
    )


def _file_handler(
    path: Path, formatter: logging.Formatter, level: int
) -> logging.FileHandler:
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(level)
    return handler


def _add_file_handler(
    logger_name: str, path: Path, formatter: logging.Formatter, level: int
) -> None:
    logger = logging.getLogger(logger_name)
    logger.addHandler(_file_handler(path, formatter, level))


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced GAMR logger, e.g. get_logger('runtime') → 'gamr.runtime'."""
    return logging.getLogger(f"gamr.{name}")
