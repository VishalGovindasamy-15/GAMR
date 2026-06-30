"""
FixedPool — fixed-budget RAM and VRAM pools.

Budget is calculated once at startup (fraction * free_memory).
Does NOT dynamically resize. Phase 6 adds DynamicPool.
"""
from __future__ import annotations

import logging

from runtime.memory.memory_object import MemoryObject
from runtime.plugins.pool.base import PoolPlugin

logger = logging.getLogger("gamr.pool")


class FixedPool(PoolPlugin):
    """Fixed-budget pool. Used for both RAM and VRAM in the POC."""

    def __init__(self, budget_bytes: int, name: str = "pool") -> None:
        self._budget = budget_bytes
        self._used = 0
        self._name = name
        logger.info(
            f"{name}: budget={budget_bytes / (1024**3):.2f} GB"
        )

    def budget_bytes(self) -> int:
        return self._budget

    def used_bytes(self) -> int:
        return self._used

    def free_bytes(self) -> int:
        return self._budget - self._used

    def can_fit(self, obj: MemoryObject) -> bool:
        return obj.size_bytes <= self.free_bytes()

    def allocate(self, obj: MemoryObject) -> None:
        if not self.can_fit(obj):
            raise MemoryError(
                f"{self._name}: cannot fit {obj.size_bytes} bytes "
                f"(free={self.free_bytes()}, budget={self._budget})"
            )
        self._used += obj.size_bytes
        logger.debug(
            f"{self._name}: allocated {obj.size_bytes // 1024}KB "
            f"({self._used / self._budget:.1%} used)"
        )

    def free(self, obj: MemoryObject) -> None:
        self._used = max(0, self._used - obj.size_bytes)
        logger.debug(
            f"{self._name}: freed {obj.size_bytes // 1024}KB "
            f"({self._used / self._budget:.1%} used)"
        )


def make_ram_pool(free_ram_bytes: int, fraction: float) -> FixedPool:
    budget = int(free_ram_bytes * fraction)
    return FixedPool(budget_bytes=budget, name="RAMPool")


def make_vram_pool(free_vram_bytes: int, fraction: float) -> FixedPool:
    budget = int(free_vram_bytes * fraction)
    return FixedPool(budget_bytes=budget, name="VRAMPool")
