"""SchedulerPlugin — base interface for all scheduler implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from runtime.memory.memory_object import WeightObject


class SchedulerPlugin(ABC):
    """All schedulers implement this interface. Swap without touching runtime."""

    @abstractmethod
    def enqueue(self, obj: WeightObject) -> None:
        """Add a WeightObject to the scheduler queue."""

    @abstractmethod
    def has_next(self) -> bool:
        """Return True if there is at least one object ready to stream."""

    @abstractmethod
    def next(self) -> WeightObject:
        """Return the next WeightObject to load. Raises StopIteration when empty."""

    @abstractmethod
    def peek(self) -> Optional[WeightObject]:
        """Return the next object without removing it. None if empty."""

    @abstractmethod
    def remaining(self) -> int:
        """Number of objects not yet returned."""
