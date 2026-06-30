"""
AdaptiveScheduler — Phase 5.

Exposes the same interface as StaticPrefetchScheduler but its prefetch_depth
is mutable: the Decision Engine inside MemoryController changes it at runtime
based on observed latency metrics.

Key contract:
  - set_prefetch_depth(n)  ← Decision Engine calls this
  - prefetch_depth         ← Decision Engine reads current value
  - All other methods identical to StaticPrefetchScheduler

The scheduler itself has NO decision logic — it only executes the current
depth setting. Decisions belong exclusively to the Decision Engine.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from runtime.memory.memory_object import WeightObject
from runtime.plugins.scheduler.base import SchedulerPlugin

logger = logging.getLogger("gamr.scheduler.adaptive")

_MIN_DEPTH = 1
_MAX_DEPTH = 4   # RTX 3050: 4 layers × ~100 MB = ~400 MB peak VRAM overhead


class AdaptiveScheduler(SchedulerPlugin):
    """
    Prefetch scheduler whose depth is dynamically adjusted by the Decision Engine.

    The Decision Engine observes GPU stall events (idle time between layer
    compute steps) and adjusts prefetch_depth up or down to minimise stalls
    without exceeding the VRAM budget.
    """

    def __init__(self, initial_depth: int = 1) -> None:
        if not (_MIN_DEPTH <= initial_depth <= _MAX_DEPTH):
            raise ValueError(
                f"initial_depth must be {_MIN_DEPTH}–{_MAX_DEPTH}, got {initial_depth}"
            )
        self._prefetch_depth = initial_depth
        self._queue: deque[WeightObject] = deque()

    # ── Depth management (called by Decision Engine) ───────────────────

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    def set_prefetch_depth(self, new_depth: int) -> bool:
        """
        Attempt to change prefetch depth. Clamps to [MIN, MAX].
        Returns True if the value actually changed.
        """
        clamped = max(_MIN_DEPTH, min(_MAX_DEPTH, new_depth))
        if clamped == self._prefetch_depth:
            return False
        old = self._prefetch_depth
        self._prefetch_depth = clamped
        logger.info(f"AdaptiveScheduler: prefetch_depth {old} → {clamped}")
        return True

    def min_depth(self) -> int:
        return _MIN_DEPTH

    def max_depth(self) -> int:
        return _MAX_DEPTH

    # ── Queue interface (identical to StaticPrefetchScheduler) ─────────

    def enqueue(self, obj: WeightObject) -> None:
        self._queue.append(obj)

    def has_next(self) -> bool:
        return bool(self._queue)

    def next(self) -> WeightObject:
        if not self._queue:
            raise StopIteration("AdaptiveScheduler queue is empty.")
        return self._queue.popleft()

    def peek(self) -> Optional[WeightObject]:
        return self._queue[0] if self._queue else None

    def remaining(self) -> int:
        return len(self._queue)

    def prefetch_window(self) -> list[WeightObject]:
        """Return next prefetch_depth objects without removing them."""
        return list(self._queue)[: self._prefetch_depth]
