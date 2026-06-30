"""
MemoryController — single authority for all memory decisions.

POC (Week 3): thin coordinator — streams layers FIFO, no adaptation.
Phase 5: gains adaptive logic (prefetch tuning, rollback).

The streaming inference strategy:
  1. Model weights stay on CPU (RAM tier).
  2. For each layer: move to GPU → forward pass → move back to CPU.
  3. Peak VRAM = one layer's weights + the hidden states buffer.
"""
from __future__ import annotations

import logging
import time
from typing import List

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from runtime.event_bus import Event, EventBus, EventType
from runtime.hal.base import HardwareBackend
from runtime.memory.memory_object import ObjectState, WeightObject
from runtime.memory.object_manager import ObjectManager
from runtime.plugins.pool.base import PoolPlugin
from runtime.plugins.scheduler.base import SchedulerPlugin

logger = logging.getLogger("gamr.controller")


class _StreamingLayerWrapper(nn.Module):
    """
    Wraps a transformer decoder layer.
    On forward(): moves to GPU → computes → moves back to CPU.
    This is the RAM → VRAM → GPU_ACTIVE → RAM transition per layer.
    """

    def __init__(self, layer: nn.Module, device: str) -> None:
        super().__init__()
        self._layer = layer
        self._device = device

    def forward(self, *args, **kwargs):
        # Move all tensor args to device
        args = tuple(
            a.to(self._device) if isinstance(a, torch.Tensor) else a
            for a in args
        )
        kwargs = {
            k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }
        self._layer.to(self._device)
        output = self._layer(*args, **kwargs)
        self._layer.to("cpu")
        torch.cuda.empty_cache()
        return output


class MemoryController:
    """
    Coordinates the streaming inference loop.

    Owns: object_manager, scheduler, ram_pool, vram_pool, event_bus.
    In the POC it is a thin coordinator.
    In Phase 5 it gains a Decision Engine for adaptive tuning.
    """

    def __init__(
        self,
        hal: HardwareBackend,
        object_manager: ObjectManager,
        scheduler: SchedulerPlugin,
        ram_pool: PoolPlugin,
        vram_pool: PoolPlugin,
        event_bus: EventBus,
    ) -> None:
        self.hal = hal
        self.manager = object_manager
        self.scheduler = scheduler
        self.ram_pool = ram_pool
        self.vram_pool = vram_pool
        self.bus = event_bus
        self._device = hal.device()

    # ── Inference ─────────────────────────────────────────────────────

    def run_streaming_inference(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Full streaming inference:
        1. Load model to CPU (RAM tier).
        2. Wrap each decoder layer with _StreamingLayerWrapper.
        3. Run model.generate() — transformers handles masking.
        4. Each layer move: CPU → GPU → compute → CPU = RAM→VRAM→GPU→RAM.
        5. Return generated text.
        """
        logger.info("Loading model to CPU RAM (SSD → RAM step)...")
        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()

        load_time = time.time() - t0
        logger.info(f"Model loaded to CPU in {load_time:.1f}s")

        # ── Wrap decoder layers ───────────────────────────────────────
        original_layers = list(model.model.layers)
        model.model.layers = nn.ModuleList([
            _StreamingLayerWrapper(layer, self._device)
            for layer in original_layers
        ])
        # Move non-layer components to device (they stay there)
        model.model.embed_tokens.to(self._device)
        model.model.norm.to(self._device)
        model.lm_head.to(self._device)

        logger.info(
            f"Streaming inference: {len(original_layers)} layers, "
            f"device={self._device}"
        )

        # ── Publish start event ───────────────────────────────────────
        # (synchronous emit — event_bus is async but we call bus directly here)

        # ── Generate ─────────────────────────────────────────────────
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self._device)

        t_gen = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        gen_time = time.time() - t_gen

        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_token_ids = output_ids[0].tolist()
        logger.info(f"Generation complete in {gen_time:.2f}s | {len(output_token_ids)} tokens")

        # ── Peak VRAM ─────────────────────────────────────────────────
        peak_vram = torch.cuda.max_memory_allocated(self._device) if "cuda" in self._device else 0
        logger.info(f"Peak VRAM used: {peak_vram / (1024**3):.3f} GB")

        # ── Restore model ─────────────────────────────────────────────
        model.model.layers = nn.ModuleList(original_layers)
        model.cpu()
        del model
        torch.cuda.empty_cache()

        return result, output_token_ids, {
            "load_time_s": round(load_time, 2),
            "gen_time_s": round(gen_time, 2),
            "peak_vram_bytes": peak_vram,
            "output_tokens": len(output_token_ids),
        }
