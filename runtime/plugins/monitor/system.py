"""
SystemMonitor — collects GPU, VRAM, RAM, and PCIe bandwidth metrics.

Runs as a background asyncio task.
Publishes MONITOR_METRICS events to the Event Bus at each sample interval.

Rule: does NOT call Memory Controller directly. Only publishes to Event Bus.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import psutil
import torch

from runtime.event_bus import Event, EventBus, EventType
from runtime.plugins.monitor.base import MonitorPlugin

logger = logging.getLogger("gamr.monitor")


class SystemMonitor(MonitorPlugin):
    """
    Samples hardware metrics and publishes MONITOR_METRICS events.

    Metrics per sample:
      gpu_util_pct    — GPU core utilisation 0–100%
      vram_used_gb    — VRAM used (GB)
      vram_free_gb    — VRAM free (GB)
      ram_used_gb     — RAM used (GB)
      ram_free_gb     — RAM free (GB)
      ram_util_pct    — RAM utilisation 0–100%
      h2d_bw_gbps     — H2D PCIe bandwidth estimate (GB/s) — measured per layer
      ts              — epoch timestamp of the sample
    """

    def __init__(
        self,
        event_bus: EventBus,
        device: str = "cuda:0",
        interval_s: float = 0.5,
    ) -> None:
        self._bus      = event_bus
        self._device   = device
        self._interval = interval_s
        self._task: Optional[asyncio.Task] = None
        self._running  = False
        self._is_cuda  = "cuda" in device

        # pynvml handle (optional — falls back to torch if unavailable)
        self._nvml_handle = None
        if self._is_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                idx = int(device.split(":")[-1]) if ":" in device else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            except Exception:
                logger.warning("pynvml not available; falling back to torch for GPU metrics.")

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="gamr.monitor")
        logger.info(f"SystemMonitor started (interval={self._interval}s, device={self._device})")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("SystemMonitor stopped.")

    async def _loop(self) -> None:
        while self._running:
            try:
                snap = self.snapshot()
                await self._bus.publish(Event(
                    type=EventType.MONITOR_METRICS,
                    payload=snap,
                ))
            except Exception as exc:
                logger.error(f"Monitor sample error: {exc!r}")
            await asyncio.sleep(self._interval)

    # ── Snapshot ───────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Thread-safe metrics collection."""
        m: Dict[str, Any] = {"ts": time.time()}

        # RAM
        vm = psutil.virtual_memory()
        m["ram_used_gb"]  = round(vm.used  / (1024**3), 2)
        m["ram_free_gb"]  = round(vm.available / (1024**3), 2)
        m["ram_util_pct"] = round(vm.percent, 1)

        if not self._is_cuda:
            m.update({"gpu_util_pct": 0, "vram_used_gb": 0.0, "vram_free_gb": 0.0})
            return m

        # GPU + VRAM
        if self._nvml_handle is not None:
            try:
                import pynvml
                util   = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem    = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                m["gpu_util_pct"]  = util.gpu
                m["vram_used_gb"]  = round(mem.used  / (1024**3), 3)
                m["vram_free_gb"]  = round(mem.free  / (1024**3), 3)
            except Exception:
                m.update(self._torch_gpu_metrics())
        else:
            m.update(self._torch_gpu_metrics())

        return m

    def _torch_gpu_metrics(self) -> Dict[str, Any]:
        try:
            allocated = torch.cuda.memory_allocated(self._device)
            reserved  = torch.cuda.memory_reserved(self._device)
            total     = torch.cuda.get_device_properties(self._device).total_memory
            return {
                "gpu_util_pct": None,  # torch cannot query util%
                "vram_used_gb": round(allocated / (1024**3), 3),
                "vram_free_gb": round((total - reserved) / (1024**3), 3),
            }
        except Exception:
            return {"gpu_util_pct": None, "vram_used_gb": 0.0, "vram_free_gb": 0.0}
