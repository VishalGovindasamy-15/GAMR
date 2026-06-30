"""
StaticPrefetchScheduler — overlaps GPU computation with layer loading.

While the GPU computes layer N, the next K layers are being copied
from CPU RAM to GPU VRAM on a dedicated CUDA copy stream.

This is the key pipeline optimization that reduces GPU idle time:

  FIFO:           |load 0|compute 0|load 1|compute 1|load 2|compute 2|
  StaticPrefetch: |load 0|compute 0|compute 1|compute 2|
                          |load 1  |load 2  |load 3  |
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from runtime.memory.memory_object import WeightObject
from runtime.plugins.scheduler.base import SchedulerPlugin

logger = logging.getLogger("gamr.scheduler")


class StaticPrefetchScheduler(SchedulerPlugin):
    """
    Returns layers in FIFO order but tracks a look-ahead window.

    The Memory Controller reads `prefetch_window()` to know which
    layers to start copying to the device before they are needed.
    """

    def __init__(self, prefetch_depth: int = 1) -> None:
        if prefetch_depth < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {prefetch_depth}")
        self._prefetch_depth = prefetch_depth
        self._queue: deque[WeightObject] = deque()

    @property
    def prefetch_depth(self) -> int:
        return self._prefetch_depth

    def enqueue(self, obj: WeightObject) -> None:
        self._queue.append(obj)
        logger.debug(f"Enqueued {obj!r} (depth={len(self._queue)})")

    def has_next(self) -> bool:
        return len(self._queue) > 0

    def next(self) -> WeightObject:
        if not self._queue:
            raise StopIteration("StaticPrefetchScheduler queue is empty.")
        obj = self._queue.popleft()
        logger.debug(f"Scheduling {obj!r} (remaining={len(self._queue)})")
        return obj

    def peek(self) -> Optional[WeightObject]:
        return self._queue[0] if self._queue else None

    def remaining(self) -> int:
        return len(self._queue)

    def prefetch_window(self) -> list[WeightObject]:
        """
        Return the next `prefetch_depth` objects in the queue (without removing them).
        The Memory Controller should start loading these immediately.
        """
        window = list(self._queue)[:self._prefetch_depth]
        logger.debug(f"Prefetch window: {[o.layer_name for o in window]}")
        return window
