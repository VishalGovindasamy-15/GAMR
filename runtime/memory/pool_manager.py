"""
MemoryPoolManager — coordinates the RAM pool and VRAM pool together.

Sits between the Memory Controller and the pool plugins.
The Memory Controller calls MemoryPoolManager; it never touches pools directly.

In Phase 6: DynamicPool drops in here — no controller changes needed.
"""
from __future__ import annotations

import logging
from typing import Dict

from runtime.memory.memory_object import Location, MemoryObject
from runtime.plugins.pool.base import PoolPlugin

logger = logging.getLogger("gamr.memory.pool_manager")


class MemoryPoolManager:
    """
    Unified interface to all memory pools.

    POC pools:
      ram_pool  — FixedPool (25% of free RAM at startup)
      vram_pool — FixedPool (80% of free VRAM at startup)

    Phase 6: DynamicPool replaces FixedPool — no other changes required.
    """

    def __init__(self, ram_pool: PoolPlugin, vram_pool: PoolPlugin) -> None:
        self._pools: Dict[Location, PoolPlugin] = {
            Location.RAM:  ram_pool,
            Location.VRAM: vram_pool,
        }

    # ── Query ──────────────────────────────────────────────────────────

    def can_load_to_ram(self, obj: MemoryObject) -> bool:
        return self._pools[Location.RAM].can_fit(obj)

    def can_load_to_vram(self, obj: MemoryObject) -> bool:
        return self._pools[Location.VRAM].can_fit(obj)

    def ram_free_bytes(self) -> int:
        return self._pools[Location.RAM].free_bytes()

    def vram_free_bytes(self) -> int:
        return self._pools[Location.VRAM].free_bytes()

    def ram_utilization(self) -> float:
        return self._pools[Location.RAM].utilization()

    def vram_utilization(self) -> float:
        return self._pools[Location.VRAM].utilization()

    # ── Allocation ─────────────────────────────────────────────────────

    def allocate_ram(self, obj: MemoryObject) -> None:
        self._pools[Location.RAM].allocate(obj)
        logger.debug(f"RAM allocated for {obj!r} ({self.ram_utilization():.1%} used)")

    def allocate_vram(self, obj: MemoryObject) -> None:
        self._pools[Location.VRAM].allocate(obj)
        logger.debug(f"VRAM allocated for {obj!r} ({self.vram_utilization():.1%} used)")

    def free_ram(self, obj: MemoryObject) -> None:
        self._pools[Location.RAM].free(obj)
        logger.debug(f"RAM freed for {obj!r} ({self.ram_utilization():.1%} used)")

    def free_vram(self, obj: MemoryObject) -> None:
        self._pools[Location.VRAM].free(obj)
        logger.debug(f"VRAM freed for {obj!r} ({self.vram_utilization():.1%} used)")

    # ── Status snapshot ────────────────────────────────────────────────

    def status(self) -> dict:
        ram  = self._pools[Location.RAM]
        vram = self._pools[Location.VRAM]
        return {
            "ram_budget_gb":  round(ram.budget_bytes()  / (1024**3), 2),
            "ram_used_gb":    round(ram.used_bytes()    / (1024**3), 2),
            "ram_util_pct":   round(ram.utilization()   * 100, 1),
            "vram_budget_gb": round(vram.budget_bytes() / (1024**3), 2),
            "vram_used_gb":   round(vram.used_bytes()   / (1024**3), 2),
            "vram_util_pct":  round(vram.utilization()  * 100, 1),
        }
