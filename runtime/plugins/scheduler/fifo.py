"""FIFOScheduler — returns WeightObjects in layer-index order."""
from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from runtime.memory.memory_object import WeightObject
from runtime.plugins.scheduler.base import SchedulerPlugin

logger = logging.getLogger("gamr.scheduler")


class FIFOScheduler(SchedulerPlugin):
    """First-in, first-out. Layer 0 → Layer 1 → … → Layer N."""

    def __init__(self) -> None:
        self._queue: deque[WeightObject] = deque()

    def enqueue(self, obj: WeightObject) -> None:
        self._queue.append(obj)
        logger.debug(f"Enqueued {obj!r} (queue depth={len(self._queue)})")

    def has_next(self) -> bool:
        return len(self._queue) > 0

    def next(self) -> WeightObject:
        if not self._queue:
            raise StopIteration("FIFOScheduler queue is empty.")
        obj = self._queue.popleft()
        logger.debug(f"Scheduling {obj!r} (remaining={len(self._queue)})")
        return obj

    def peek(self) -> Optional[WeightObject]:
        return self._queue[0] if self._queue else None

    def remaining(self) -> int:
        return len(self._queue)
