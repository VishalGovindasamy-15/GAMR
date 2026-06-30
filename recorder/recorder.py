"""
Recorder — black-box flight recorder for GAMR (Phase 4 full implementation).

Writes every event timestamped to:
  events.json  — NDJSON, one event per line, fully replay-able
  gpu.csv      — periodic GPU util % + VRAM snapshots (from MONITOR_METRICS)
  ram.csv      — periodic RAM used/free/util snapshots
  vram.csv     — periodic VRAM used/free snapshots
  latency.csv  — per-layer load and compute latencies

Thread-safe: inference runs in asyncio.to_thread(); monitor runs in the event
loop. Both write to the same files through a threading.Lock.
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import orjson

from runtime.event_bus import Event, EventBus, EventType

logger = logging.getLogger("gamr.recorder")

# ── CSV column schemas ─────────────────────────────────────────────────────
_GPU_COLS     = ["ts", "gpu_util_pct", "vram_used_gb", "vram_free_gb"]
_RAM_COLS     = ["ts", "ram_used_gb", "ram_free_gb", "ram_util_pct"]
_VRAM_COLS    = ["ts", "vram_used_gb", "vram_free_gb"]
_LATENCY_COLS = ["ts", "layer_index", "layer_name", "event_type", "latency_ms", "vram_used_gb"]


class Recorder:
    """
    Black-box flight recorder.

    Usage (from runtime):
        recorder = Recorder(run_dir, event_bus)
        # ... run inference, monitor publishes events ...
        recorder.close()          # flush + close files
    """

    def __init__(self, run_dir: Path, event_bus: EventBus) -> None:
        self._run_dir = run_dir
        self._bus     = event_bus
        self._lock    = threading.Lock()
        self._closed  = False

        # Open all output files
        self._events_fh  = (run_dir / "events.json").open("w", encoding="utf-8")
        self._gpu_fh     = (run_dir / "gpu.csv").open("w", newline="", encoding="utf-8")
        self._ram_fh     = (run_dir / "ram.csv").open("w", newline="", encoding="utf-8")
        self._vram_fh    = (run_dir / "vram.csv").open("w", newline="", encoding="utf-8")
        self._latency_fh = (run_dir / "latency.csv").open("w", newline="", encoding="utf-8")

        # CSV writers
        self._gpu_csv     = csv.DictWriter(self._gpu_fh,     fieldnames=_GPU_COLS)
        self._ram_csv     = csv.DictWriter(self._ram_fh,     fieldnames=_RAM_COLS)
        self._vram_csv    = csv.DictWriter(self._vram_fh,    fieldnames=_VRAM_COLS)
        self._latency_csv = csv.DictWriter(self._latency_fh, fieldnames=_LATENCY_COLS)
        for w in (self._gpu_csv, self._ram_csv, self._vram_csv, self._latency_csv):
            w.writeheader()

        # Subscribe to all event bus events
        self._bus.subscribe(self._on_bus_event)
        logger.info(f"Recorder initialised → {run_dir}")

    # ── Event Bus subscription ─────────────────────────────────────────

    async def _on_bus_event(self, event: Event) -> None:
        """Called by the EventBus for every published event."""
        payload = event.payload or {}
        self._write_event(
            ts=event.timestamp if hasattr(event, "timestamp") else time.time(),
            event_type=event.type.value,
            object_id=getattr(event, "object_id", "") or "",
            payload=payload,
        )
        # Route metrics events to the appropriate CSV files
        if event.type == EventType.MONITOR_METRICS:
            self._write_monitor_csvs(payload)

    # ── Direct recording (called from inference thread) ────────────────

    def record(
        self,
        event_type: str,
        object_id: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Synchronous recording — called from within the inference thread for
        per-layer events (load start/done, compute start/done, released).
        """
        self._write_event(
            ts=time.time(),
            event_type=event_type,
            object_id=object_id,
            payload=payload or {},
        )
        # Route latency events to latency.csv
        if event_type in ("LAYER_LOAD_DONE", "LAYER_COMPUTE_DONE"):
            p = payload or {}
            self._write_latency_row(
                layer_index=p.get("layer_index", -1),
                layer_name=p.get("layer_name", ""),
                event_type=event_type,
                latency_ms=p.get("latency_ms", 0.0),
                vram_used_gb=p.get("vram_used_gb", 0.0),
            )

    # ── Internal writers (all thread-safe via lock) ────────────────────

    def _write_event(
        self,
        ts: float,
        event_type: str,
        object_id: str,
        payload: Dict[str, Any],
    ) -> None:
        line = orjson.dumps({
            "ts":        round(ts, 6),
            "type":      event_type,
            "object_id": object_id,
            "payload":   payload,
        }).decode() + "\n"
        with self._lock:
            if not self._closed:
                self._events_fh.write(line)

    def _write_monitor_csvs(self, p: Dict[str, Any]) -> None:
        ts = p.get("ts", time.time())
        with self._lock:
            if self._closed:
                return
            if "gpu_util_pct" in p:
                self._gpu_csv.writerow({
                    "ts":           round(ts, 3),
                    "gpu_util_pct": p.get("gpu_util_pct", ""),
                    "vram_used_gb": p.get("vram_used_gb", 0.0),
                    "vram_free_gb": p.get("vram_free_gb", 0.0),
                })
            self._ram_csv.writerow({
                "ts":           round(ts, 3),
                "ram_used_gb":  p.get("ram_used_gb", 0.0),
                "ram_free_gb":  p.get("ram_free_gb", 0.0),
                "ram_util_pct": p.get("ram_util_pct", 0.0),
            })
            self._vram_csv.writerow({
                "ts":           round(ts, 3),
                "vram_used_gb": p.get("vram_used_gb", 0.0),
                "vram_free_gb": p.get("vram_free_gb", 0.0),
            })

    def _write_latency_row(
        self,
        layer_index: int,
        layer_name: str,
        event_type: str,
        latency_ms: float,
        vram_used_gb: float,
    ) -> None:
        with self._lock:
            if not self._closed:
                self._latency_csv.writerow({
                    "ts":          round(time.time(), 6),
                    "layer_index": layer_index,
                    "layer_name":  layer_name,
                    "event_type":  event_type,
                    "latency_ms":  round(latency_ms, 3),
                    "vram_used_gb": round(vram_used_gb, 4),
                })

    # ── Flush + close ──────────────────────────────────────────────────

    def flush(self) -> None:
        with self._lock:
            for fh in (
                self._events_fh, self._gpu_fh, self._ram_fh,
                self._vram_fh,   self._latency_fh,
            ):
                fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for fh in (
                self._events_fh, self._gpu_fh, self._ram_fh,
                self._vram_fh,   self._latency_fh,
            ):
                fh.flush()
                fh.close()
        logger.info("Recorder closed and all files flushed.")

    # ── Replay helper ──────────────────────────────────────────────────

    @staticmethod
    def replay(run_dir: Path) -> list[dict]:
        """
        Load and return all events from events.json in timestamp order.
        Each event is a dict: {ts, type, object_id, payload}.
        """
        path = run_dir / "events.json"
        if not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                events.append(orjson.loads(line))
        events.sort(key=lambda e: e["ts"])
        return events
