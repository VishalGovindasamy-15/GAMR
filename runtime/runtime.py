"""
GAMR Runtime — main orchestrator.

Week 1 scope:
  Start → Load Config → Detect Hardware → Save Artifacts → Exit

The runtime is intentionally thin. All logic lives in plugins and managers.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import orjson

from runtime.config import load_config
from runtime.hardware_scan import save_hardware_json, scan_hardware
from runtime.logger import setup_logging
from runtime.run_manager import create_run_dir

logger = logging.getLogger("gamr.runtime")

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      GAMR — Galaxy Adaptive Memory Runtime  v0.1.0          ║
║      Adaptive Hierarchical AI Memory Manager (AHAMM)        ║
╚══════════════════════════════════════════════════════════════╝
"""


async def run() -> int:
    """Main async entry point. Returns exit code."""

    # ── 1. Load config (before logging so we have log_level) ─────────
    config = load_config("configs/runtime.yaml")

    # ── 2. Create run directory ───────────────────────────────────────
    run_dir = create_run_dir(Path(config.output_dir))

    # ── 3. Setup structured logging ───────────────────────────────────
    setup_logging(run_dir, config.log_level)

    logger.info(_BANNER.strip())
    logger.info(f"Run directory : {run_dir}")
    logger.info(f"Scheduler     : {config.scheduler}")
    logger.info(f"Validate      : {config.validate_output}")

    # ── 4. Save config snapshot ───────────────────────────────────────
    config_snapshot = run_dir / "config.json"
    config_snapshot.write_bytes(
        orjson.dumps(config.model_dump(), option=orjson.OPT_INDENT_2)
    )
    logger.info(f"Config snapshot → {config_snapshot}")

    # ── 5. Hardware scan ──────────────────────────────────────────────
    logger.info("Scanning hardware...")
    hw = scan_hardware(ssd_path=str(Path(config.output_dir).anchor))
    save_hardware_json(hw, run_dir)

    # ── 6. Select HAL backend ─────────────────────────────────────────
    if hw.cuda_available:
        from runtime.hal.cuda_backend import CUDABackend
        hal = CUDABackend()
    else:
        from runtime.hal.cpu_backend import CPUBackend
        hal = CPUBackend()

    logger.info(f"HAL backend   : {hal}")
    logger.info(
        f"Free VRAM     : {hal.free_vram_gb():.2f} GB  |  "
        f"Free RAM: {hal.free_ram_gb():.2f} GB"
    )

    # ── Week 1 milestone complete ─────────────────────────────────────
    logger.info("=" * 62)
    logger.info("Week 1 milestone complete.")
    logger.info("Hardware detected. Config loaded. Logging active.")
    logger.info("Next: Week 2 — MemoryObject + Event Bus + TinyLlama.")
    logger.info("=" * 62)

    return 0


def main() -> None:
    code = asyncio.run(run())
    sys.exit(code)


if __name__ == "__main__":
    main()
