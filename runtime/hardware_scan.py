"""
Hardware scanner — detects GPU, VRAM, RAM, and storage.

Produces HardwareInfo which is serialised to hardware.json on every run.
This file is the ground truth record of what machine the run executed on.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import orjson
import psutil

logger = logging.getLogger("gamr.hardware")


@dataclass
class HardwareInfo:
    # GPU
    gpu_name: Optional[str]
    cuda_available: bool
    cuda_version: Optional[str]
    vram_total_bytes: int
    vram_free_bytes: int

    # RAM
    ram_total_bytes: int
    ram_free_bytes: int

    # Storage
    ssd_path: str
    ssd_total_bytes: int
    ssd_free_bytes: int

    # System
    python_version: str
    platform: str
    cpu_count: int

    # Convenience GB fields (computed)
    vram_total_gb: float = 0.0
    ram_total_gb: float = 0.0

    def __post_init__(self) -> None:
        self.vram_total_gb = round(self.vram_total_bytes / (1024 ** 3), 2)
        self.ram_total_gb = round(self.ram_total_bytes / (1024 ** 3), 2)


def scan_hardware(ssd_path: str = "/") -> HardwareInfo:
    """
    Scan all hardware and return a HardwareInfo dataclass.
    Works with or without a GPU — gracefully degrades to CPU-only info.
    """
    # ── GPU ───────────────────────────────────────────────────────────
    cuda_available = False
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    vram_total = 0
    vram_free = 0

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_free = vram_total  # rough before any allocation

            # Prefer pynvml for accurate free VRAM
            try:
                import pynvml  # type: ignore
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_total = int(mem.total)
                vram_free = int(mem.free)
            except Exception:
                pass  # fallback: vram_free stays as total estimate
    except ImportError:
        logger.warning("torch not importable — GPU detection skipped.")

    # ── RAM ───────────────────────────────────────────────────────────
    vm = psutil.virtual_memory()

    # ── Storage ───────────────────────────────────────────────────────
    try:
        disk = psutil.disk_usage(ssd_path)
        ssd_total = disk.total
        ssd_free = disk.free
    except Exception:
        ssd_total = 0
        ssd_free = 0

    info = HardwareInfo(
        gpu_name=gpu_name,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        vram_total_bytes=vram_total,
        vram_free_bytes=vram_free,
        ram_total_bytes=vm.total,
        ram_free_bytes=vm.available,
        ssd_path=ssd_path,
        ssd_total_bytes=ssd_total,
        ssd_free_bytes=ssd_free,
        python_version=platform.python_version(),
        platform=platform.platform(),
        cpu_count=psutil.cpu_count(logical=True) or 0,
    )

    logger.info(
        f"Hardware scan complete — "
        f"GPU: {gpu_name or 'None'} | CUDA: {cuda_available} | "
        f"VRAM: {info.vram_total_gb:.1f} GB | RAM: {info.ram_total_gb:.1f} GB"
    )
    return info


def save_hardware_json(info: HardwareInfo, run_dir: Path) -> None:
    """Write hardware.json to the run directory."""
    out = run_dir / "hardware.json"
    out.write_bytes(orjson.dumps(asdict(info), option=orjson.OPT_INDENT_2))
    logger.info(f"hardware.json saved → {out}")
