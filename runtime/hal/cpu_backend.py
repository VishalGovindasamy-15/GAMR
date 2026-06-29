"""
CPUBackend — CPU-only fallback backend.

Uses system RAM as the compute device.
For development and testing on machines without a GPU.
"""
from __future__ import annotations

import logging

import psutil
import torch

from runtime.hal.base import HardwareBackend

logger = logging.getLogger("gamr.hal.cpu")


class CPUBackend(HardwareBackend):
    """
    CPU-only backend.

    VRAM queries return RAM values (RAM acts as the 'device memory').
    GPU utilisation always returns 0.
    """

    def __init__(self) -> None:
        logger.warning(
            "CPUBackend active — no GPU acceleration. "
            "Use this backend for testing only."
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def device_name(self) -> str:
        return "CPU (no GPU)"

    def device(self) -> str:
        return "cpu"

    # ------------------------------------------------------------------
    # Memory queries — RAM acts as VRAM substitute
    # ------------------------------------------------------------------

    def total_vram_bytes(self) -> int:
        return psutil.virtual_memory().total

    def free_vram_bytes(self) -> int:
        return psutil.virtual_memory().available

    def total_ram_bytes(self) -> int:
        return psutil.virtual_memory().total

    def free_ram_bytes(self) -> int:
        return psutil.virtual_memory().available

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    def gpu_utilization_percent(self) -> float:
        return 0.0

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to("cpu")

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to("cpu")

    def synchronize(self) -> None:
        pass  # No-op on CPU
