"""
CUDABackend — NVIDIA GPU backend.

Wraps PyTorch CUDA and pynvml for accurate memory reporting.
Falls back gracefully if pynvml is unavailable.
"""
from __future__ import annotations

import logging
from typing import Optional

import psutil
import torch

from runtime.hal.base import HardwareBackend

logger = logging.getLogger("gamr.hal.cuda")


class CUDABackend(HardwareBackend):
    """CUDA implementation. Targets NVIDIA RTX 3050 (and any CUDA GPU)."""

    def __init__(self, device_index: int = 0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available on this system. Use CPUBackend instead."
            )
        self._index = device_index
        self._device_str = f"cuda:{device_index}"
        self._nvml = None
        self._nvml_handle = None
        self._init_nvml()
        logger.info(
            f"CUDABackend ready: {self.device_name()} | "
            f"VRAM {self.total_vram_gb():.1f} GB"
        )

    # ------------------------------------------------------------------
    # pynvml initialisation (best-effort)
    # ------------------------------------------------------------------

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._index)
            logger.debug("pynvml initialised — accurate VRAM reporting enabled.")
        except Exception as exc:
            logger.warning(
                f"pynvml unavailable ({exc}). "
                "Falling back to torch memory APIs (less accurate)."
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def device_name(self) -> str:
        return torch.cuda.get_device_name(self._index)

    def device(self) -> str:
        return self._device_str

    # ------------------------------------------------------------------
    # Memory queries
    # ------------------------------------------------------------------

    def total_vram_bytes(self) -> int:
        if self._nvml_handle:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).total)
        return torch.cuda.get_device_properties(self._index).total_memory

    def free_vram_bytes(self) -> int:
        if self._nvml_handle:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle).free)
        # Torch fallback: reserved - allocated
        torch.cuda.synchronize(self._index)
        return (
            torch.cuda.memory_reserved(self._index)
            - torch.cuda.memory_allocated(self._index)
        )

    def total_ram_bytes(self) -> int:
        return psutil.virtual_memory().total

    def free_ram_bytes(self) -> int:
        return psutil.virtual_memory().available

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    def gpu_utilization_percent(self) -> float:
        if self._nvml_handle:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu)
        return 0.0

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self._device_str, non_blocking=True)

    def to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to("cpu", non_blocking=True)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self._index)
