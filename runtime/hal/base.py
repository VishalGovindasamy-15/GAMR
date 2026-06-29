"""
HardwareBackend — abstract interface for all compute backends.

Runtime code only speaks this interface.
CUDA / CPU / ROCm / TPU are implementation details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class HardwareBackend(ABC):
    """Abstract hardware backend. Never instantiate directly."""

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @abstractmethod
    def device_name(self) -> str:
        """Human-readable device name, e.g. 'NVIDIA GeForce RTX 3050'."""

    @abstractmethod
    def device(self) -> str:
        """Torch device string, e.g. 'cuda:0' or 'cpu'."""

    # ------------------------------------------------------------------
    # Memory queries
    # ------------------------------------------------------------------

    @abstractmethod
    def total_vram_bytes(self) -> int:
        """Total VRAM capacity in bytes (RAM on CPU backend)."""

    @abstractmethod
    def free_vram_bytes(self) -> int:
        """Free VRAM in bytes (RAM on CPU backend)."""

    @abstractmethod
    def total_ram_bytes(self) -> int:
        """Total system RAM in bytes."""

    @abstractmethod
    def free_ram_bytes(self) -> int:
        """Free (available) system RAM in bytes."""

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    @abstractmethod
    def gpu_utilization_percent(self) -> float:
        """GPU compute utilisation 0–100. Returns 0.0 on CPU backend."""

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    @abstractmethod
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to this backend's compute device (non-blocking)."""

    @abstractmethod
    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor back to CPU RAM (non-blocking)."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all pending operations on the device are complete."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def free_vram_gb(self) -> float:
        return self.free_vram_bytes() / (1024 ** 3)

    def free_ram_gb(self) -> float:
        return self.free_ram_bytes() / (1024 ** 3)

    def total_vram_gb(self) -> float:
        return self.total_vram_bytes() / (1024 ** 3)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device()!r}, "
            f"vram={self.total_vram_gb():.1f}GB)"
        )
