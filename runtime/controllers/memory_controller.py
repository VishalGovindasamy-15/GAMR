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
from typing import List, Optional

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

# Late import to avoid circular dependency
def _get_recorder_type():
    try:
        from recorder.recorder import Recorder
        return Recorder
    except ImportError:
        return None


class _StreamingLayerWrapper(nn.Module):
    """
    Wraps a transformer decoder layer.
    On forward(): moves to GPU → computes → moves back to CPU.
    This is the RAM → VRAM → GPU_ACTIVE → RAM transition per layer.
    Records per-layer latencies to the Recorder (Phase 4).
    """

    def __init__(
        self,
        layer: nn.Module,
        device: str,
        layer_index: int = -1,
        recorder=None,  # Optional[Recorder] — avoids circular import type hint
    ) -> None:
        super().__init__()
        self._layer       = layer
        self._device      = device
        self._layer_index = layer_index
        self._recorder    = recorder

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
        # H2D load (SSD → VRAM)
        t_load = time.time()
        self._layer.to(self._device)
        load_ms = (time.time() - t_load) * 1000

        # GPU compute
        t_compute = time.time()
        output = self._layer(*args, **kwargs)
        if "cuda" in self._device:
            torch.cuda.synchronize(self._device)
        compute_ms = (time.time() - t_compute) * 1000

        # VRAM after compute
        vram_gb = 0.0
        if "cuda" in self._device:
            vram_gb = torch.cuda.memory_allocated(self._device) / (1024**3)

        # Record events
        if self._recorder is not None:
            self._recorder.record(
                "LAYER_LOAD_DONE",
                payload={
                    "layer_index": self._layer_index,
                    "layer_name":  f"decoder.{self._layer_index}",
                    "latency_ms":  round(load_ms, 3),
                    "vram_used_gb": round(vram_gb, 4),
                },
            )
            self._recorder.record(
                "LAYER_COMPUTE_DONE",
                payload={
                    "layer_index": self._layer_index,
                    "layer_name":  f"decoder.{self._layer_index}",
                    "latency_ms":  round(compute_ms, 3),
                    "vram_used_gb": round(vram_gb, 4),
                },
            )

        # Move layer back to CPU
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
        recorder=None,  # Optional[Recorder]
    ) -> None:
        self.hal = hal
        self.manager = object_manager
        self.scheduler = scheduler
        self.ram_pool = ram_pool
        self.vram_pool = vram_pool
        self.bus = event_bus
        self.recorder = recorder
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
            _StreamingLayerWrapper(layer, self._device, layer_index=i, recorder=self.recorder)
            for i, layer in enumerate(original_layers)
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
            "scheduler": "fifo",
        }

    # ── Prefetch inference (Phase 3) ───────────────────────────────────

    def run_streaming_inference_prefetch(
        self,
        model_path: str,
        prompt: str,
        max_new_tokens: int = 50,
        prefetch_depth: int = 1,
    ) -> tuple[str, list[int], dict]:
        """
        Streaming inference with pipeline overlap.

        Uses two CUDA streams:
          copy_stream  — H2D transfer of next layer (non-blocking)
          compute_stream — GPU computation of current layer

        While layer N computes on the GPU, layer N+1 is being copied to the
        GPU in the background. When N finishes, N+1 is already resident.

        This is the Phase 3 optimization. The model weights stay on CPU RAM
        between steps (same as FIFO) but GPU utilization is higher because
        there is no idle wait for H2D transfers.
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

        device = self._device
        is_cuda = "cuda" in device

        # ── CUDA streams for pipeline overlap ─────────────────────────
        if is_cuda:
            copy_stream    = torch.cuda.Stream()
            compute_stream = torch.cuda.default_stream()
        else:
            copy_stream = compute_stream = None

        layers = list(model.model.layers)
        n = len(layers)
        logger.info(
            f"Prefetch inference: {n} layers, depth={prefetch_depth}, device={device}"
        )

        class _PrefetchWrapper(nn.Module):
            def __init__(self_w, idx: int) -> None:
                super().__init__()
                self_w._idx = idx

            def forward(self_w, *args, **kwargs):
                # Move args to device
                args = tuple(
                    a.to(device) if isinstance(a, torch.Tensor) else a for a in args
                )
                kwargs = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in kwargs.items()
                }

                # Make default stream wait for any pending H2D copy on copy_stream.
                # For layers 1-21: the previous layer's wrapper submitted their H2D
                # copy here — we must wait before using their weights.
                if is_cuda:
                    torch.cuda.current_stream().wait_stream(copy_stream)

                # Ensure this layer is on device.
                # - Layers 1-21: no-op (prefetch from previous step already done)
                # - Layer 0 on token > 1: synchronous H2D (no prefetch queued for it)
                layers[self_w._idx].to(device)

                # Start prefetching next layer(s) concurrently with compute.
                # These run on copy_stream, independent of the default stream.
                if is_cuda:
                    for offset in range(1, prefetch_depth + 1):
                        ni = self_w._idx + offset
                        if ni < n:
                            with torch.cuda.stream(copy_stream):
                                layers[ni].to(device, non_blocking=True)

                # Compute (default stream)
                output = layers[self_w._idx](*args, **kwargs)

                # Move current layer back to CPU
                layers[self_w._idx].to("cpu")
                if is_cuda:
                    torch.cuda.empty_cache()

                return output

        # Wrap all decoder layers
        model.model.layers = nn.ModuleList([_PrefetchWrapper(i) for i in range(n)])

        # Move non-layer components to device (stay there)
        model.model.embed_tokens.to(device)
        model.model.norm.to(device)
        model.lm_head.to(device)

        # Pre-load first `prefetch_depth` layers before generation starts
        if is_cuda:
            for i in range(min(prefetch_depth, n)):
                with torch.cuda.stream(copy_stream):
                    layers[i].to(device, non_blocking=True)
            copy_stream.synchronize()
        else:
            for i in range(min(prefetch_depth, n)):
                layers[i].to(device)

        # ── Generate ──────────────────────────────────────────────────
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

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
        logger.info(f"Prefetch generation done in {gen_time:.2f}s | {len(output_token_ids)} tokens")

        # ── Peak VRAM ─────────────────────────────────────────────────
        peak_vram = torch.cuda.max_memory_allocated(device) if is_cuda else 0
        logger.info(f"Peak VRAM used: {peak_vram / (1024**3):.3f} GB")

        # ── Cleanup ───────────────────────────────────────────────────
        model.model.layers = nn.ModuleList(layers)
        model.cpu()
        del model
        if is_cuda:
            torch.cuda.empty_cache()

        return result, output_token_ids, {
            "load_time_s":    round(load_time, 2),
            "gen_time_s":     round(gen_time, 2),
            "peak_vram_bytes": peak_vram,
            "output_tokens":  len(output_token_ids),
            "scheduler":      f"static_prefetch_d{prefetch_depth}",
        }
