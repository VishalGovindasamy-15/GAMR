"""
GAMR Runtime — main orchestrator (Phase 4: Monitor + Recorder wired in).

Flow:
  Load Config → Detect Hardware → Create Run Dir → Setup Logging
  → Select HAL → Start Monitor + Recorder → Run Streaming Inference (thread)
  → Stop Monitor → Validate vs HF Reference → Close Recorder
  → Save All Artifacts → Exit

Monitor → Event Bus → Memory Controller (never direct calls).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import orjson
import torch

from recorder.recorder import Recorder
from runtime.config import load_config
from runtime.controllers.memory_controller import MemoryController
from runtime.event_bus import Event, EventBus, EventType
from runtime.hal.base import HardwareBackend
from runtime.hardware_scan import save_hardware_json, scan_hardware
from runtime.logger import setup_logging
from runtime.memory.object_manager import ObjectManager
from runtime.plugins.monitor.system import SystemMonitor
from runtime.plugins.pool.fixed import make_ram_pool, make_vram_pool
from runtime.plugins.scheduler.fifo import FIFOScheduler
from runtime.run_manager import create_run_dir
from validation.engine import ValidationEngine
from validation.report import write_report

logger = logging.getLogger("gamr.runtime")

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║      GAMR — Galaxy Adaptive Memory Runtime  v0.1.0          ║
║      Adaptive Hierarchical AI Memory Manager (AHAMM)        ║
╚══════════════════════════════════════════════════════════════╝
"""


def _select_hal(hw) -> HardwareBackend:
    if hw.cuda_available:
        from runtime.hal.cuda_backend import CUDABackend
        return CUDABackend()
    from runtime.hal.cpu_backend import CPUBackend
    return CPUBackend()


async def run() -> int:
    # ── 1. Config ─────────────────────────────────────────────────────
    config = load_config("configs/runtime.yaml")
    model_cfg = _load_model_config()

    # ── 2. Run directory + logging ────────────────────────────────────
    run_dir = create_run_dir(Path(config.output_dir))
    setup_logging(run_dir, config.log_level)

    logger.info(_BANNER.strip())
    logger.info(f"Run directory : {run_dir}")

    # ── 3. Save config snapshot ───────────────────────────────────────
    (run_dir / "config.json").write_bytes(
        orjson.dumps(config.model_dump(), option=orjson.OPT_INDENT_2)
    )

    # ── 4. Hardware scan + HAL ────────────────────────────────────────
    hw = scan_hardware(ssd_path=str(Path(config.output_dir).anchor))
    save_hardware_json(hw, run_dir)
    hal = _select_hal(hw)
    logger.info(f"HAL: {hal}")

    # ── 5. Build pools ────────────────────────────────────────────────
    ram_pool  = make_ram_pool(hw.ram_free_bytes,  config.pool.ram_fraction())
    vram_pool = make_vram_pool(hw.vram_free_bytes, config.pool.vram_fraction())

    # ── 6. Core objects ───────────────────────────────────────────────
    event_bus      = EventBus()
    object_manager = ObjectManager()
    scheduler      = FIFOScheduler()

    # ── 7. Phase 4: Recorder + Monitor ───────────────────────────────
    recorder = Recorder(run_dir, event_bus)
    monitor  = SystemMonitor(event_bus, device=hal.device(), interval_s=0.5)

    controller = MemoryController(
        hal=hal,
        object_manager=object_manager,
        scheduler=scheduler,
        ram_pool=ram_pool,
        vram_pool=vram_pool,
        event_bus=event_bus,
        recorder=recorder,
    )

    # ── 8. Publish RUNTIME_STARTED ────────────────────────────────────
    await event_bus.publish(Event(type=EventType.RUNTIME_STARTED))
    await monitor.start()

    # ── 9. Streaming inference (in thread so monitor loop keeps running)
    model_path = model_cfg.get("local_path", config.model_path)
    prompt     = model_cfg.get("test_prompt", "<|system|>\nYou are a helpful assistant.<|user|>\nWhat is 2 + 2?<|assistant|>\n")
    max_tokens = 20

    logger.info(f"Model path : {model_path}")
    logger.info(f"Prompt     : {prompt!r}")

    if not Path(model_path).exists():
        logger.error(
            f"Model not found at {model_path}. "
            "Download with: huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 "
            "--local-dir ./models/TinyLlama-1.1B"
        )
        await monitor.stop()
        recorder.close()
        return 1

    if "cuda" in hal.device():
        torch.cuda.reset_peak_memory_stats(hal.device())

    gamr_text, gamr_token_ids, metrics = await asyncio.to_thread(
        controller.run_streaming_inference,
        model_path,
        prompt,
        max_tokens,
    )
    logger.info(f"GAMR output: {gamr_text!r}")

    # ── 10. Stop monitor ──────────────────────────────────────────────
    await monitor.stop()

    # ── 11. Validation ────────────────────────────────────────────────
    result = None
    if config.validate_output:
        logger.info("Running validation against HF reference...")
        validator = ValidationEngine(
            model_path=model_path,
            device=hal.device(),
            max_new_tokens=max_tokens,
        )
        result = validator.compare(
            prompt=prompt,
            gamr_text=gamr_text,
            gamr_tokens=gamr_token_ids,
            gamr_time_s=metrics["gen_time_s"],
        )
        write_report(result, run_dir)
        metrics["validation_passed"] = result.passed

    metrics["peak_vram_gb"] = round(metrics.get("peak_vram_bytes", 0) / (1024**3), 3)

    # ── 12. Save metrics ──────────────────────────────────────────────
    (run_dir / "metrics.json").write_bytes(
        orjson.dumps(metrics, option=orjson.OPT_INDENT_2)
    )

    # ── 13. Publish RUNTIME_STOPPED + close recorder ──────────────────
    await event_bus.publish(Event(
        type=EventType.RUNTIME_STOPPED,
        payload={"exit_code": 0 if (result is None or result.passed) else 1},
    ))
    # Short pause for any fire-and-forget event tasks to complete
    await asyncio.sleep(0.05)
    recorder.close()

    # ── 14. Final status ──────────────────────────────────────────────
    logger.info("=" * 62)
    if result:
        status = "PASS ✅" if result.passed else "FAIL ❌"
        logger.info(f"Validation  : {status}")
    logger.info(f"Peak VRAM   : {metrics.get('peak_vram_gb', 0):.3f} GB")
    logger.info(f"Gen time    : {metrics['gen_time_s']:.2f}s")
    logger.info(f"Run saved   : {run_dir}")
    logger.info("=" * 62)

    return 0 if (result is None or result.passed) else 1


def _load_model_config() -> dict:
    import yaml
    p = Path("configs/model.yaml")
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f).get("model", {})


def main() -> None:
    sys.exit(asyncio.run(run()))


if __name__ == "__main__":
    main()
