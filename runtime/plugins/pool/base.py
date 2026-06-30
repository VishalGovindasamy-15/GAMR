"""PoolPlugin — base interface for RAM and VRAM pool implementations."""
from __future__ import annotations

from abc import ABC, abstractmethod

from runtime.memory.memory_object import MemoryObject


class PoolPlugin(ABC):
    """Budget-aware memory pool. Swap FixedPool for DynamicPool in Phase 6."""

    @abstractmethod
    def budget_bytes(self) -> int:
        """Total byte budget for this pool."""

    @abstractmethod
    def used_bytes(self) -> int:
        """Bytes currently allocated."""

    @abstractmethod
    def free_bytes(self) -> int:
        """Bytes remaining in budget."""

    @abstractmethod
    def can_fit(self, obj: MemoryObject) -> bool:
        """Return True if obj.size_bytes fits in the remaining budget."""

    @abstractmethod
    def allocate(self, obj: MemoryObject) -> None:
        """Mark obj.size_bytes as used. Raises if budget exceeded."""

    @abstractmethod
    def free(self, obj: MemoryObject) -> None:
        """Release obj.size_bytes back to the budget."""

    def utilization(self) -> float:
        """0.0–1.0 fraction of budget used."""
        b = self.budget_bytes()
        return self.used_bytes() / b if b > 0 else 0.0
