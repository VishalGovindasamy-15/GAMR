"""
Recorder — black-box flight recorder for GAMR.

Phase 4 implementation. Writes every event timestamped to:
  events.json, gpu.csv, ram.csv, vram.csv, latency.csv

Stub registered here so imports don't break in Phase 2/3.
Full implementation in Phase 4 (Week 5).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("gamr.recorder")


class Recorder:
    """
    Stub recorder. Accepts events but discards them (Phase 2/3).
    Phase 4 replaces the body with real file writers.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        logger.debug("Recorder initialised (stub — Phase 4 fills this in).")

    def record(self, event_type: str, object_id: str = "", payload: dict = None) -> None:
        """Record a single event. No-op until Phase 4."""
        pass

    def flush(self) -> None:
        """Flush all pending writes. No-op until Phase 4."""
        pass
