"""MonitorPlugin — base interface for all monitor implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class MonitorPlugin(ABC):
    """
    All monitors implement this interface.

    Rule from the plan: Monitor NEVER calls Memory Controller directly.
    All observations are published to the Event Bus.
    The Memory Controller subscribes to MONITOR_METRICS events.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start sampling in the background (async task)."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the background sampling task gracefully."""

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a single point-in-time snapshot of all metrics.
        Must be safe to call from any thread (pynvml + psutil are thread-safe).
        """
