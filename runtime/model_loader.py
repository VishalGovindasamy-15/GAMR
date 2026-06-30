"""
ModelLoader — discovers and registers WeightObjects from a local model directory.

Each transformer layer becomes one WeightObject in SSD_COLD state.
Embedding, norm, and lm_head are also registered as special WeightObjects.
Weights are NOT loaded into RAM yet — that happens during streaming.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

from runtime.memory.memory_object import WeightObject
from runtime.memory.object_manager import ObjectManager

logger = logging.getLogger("gamr.model_loader")


# Keys that belong to a specific decoder layer
_LAYER_PREFIX = "model.layers."

# Non-layer component names (order matters for inference)
_COMPONENT_KEYS = {
    "embed":   "model.embed_tokens.weight",
    "norm":    "model.norm.weight",
    "lm_head": "lm_head.weight",
}


class ModelLoader:
    """
    Scans a safetensors model directory, sizes each layer, and registers
    WeightObjects with the ObjectManager. Nothing is loaded to RAM yet.
    """

    def __init__(self, model_path: str, object_manager: ObjectManager) -> None:
        self.model_path = Path(model_path)
        self.manager = object_manager
        self._safetensors_file: Path = self._find_safetensors()
        self._key_sizes: Dict[str, int] = self._index_key_sizes()
        logger.info(
            f"ModelLoader: {self._safetensors_file.name} "
            f"({self._safetensors_file.stat().st_size / (1024**3):.2f} GB)"
        )

    # ── Discovery ─────────────────────────────────────────────────────

    def _find_safetensors(self) -> Path:
        candidates = list(self.model_path.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(
                f"No .safetensors file found in {self.model_path}"
            )
        # Prefer model.safetensors if present
        for c in candidates:
            if c.name == "model.safetensors":
                return c
        return candidates[0]

    def _index_key_sizes(self) -> Dict[str, int]:
        """Build a map of tensor_key → size_in_bytes (without loading tensors)."""
        sizes: Dict[str, int] = {}
        with safe_open(str(self._safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                sizes[key] = t.numel() * t.element_size()
        return sizes

    def _layer_count(self) -> int:
        indices = set()
        for key in self._key_sizes:
            if key.startswith(_LAYER_PREFIX):
                idx = int(key.split(".")[2])
                indices.add(idx)
        return len(indices)

    # ── Registration ──────────────────────────────────────────────────

    def register_all(self) -> List[WeightObject]:
        """
        Create and register WeightObjects for every layer + components.
        Returns them in inference order: [embed, layer_0, …, layer_N, norm, lm_head].
        """
        objects: List[WeightObject] = []

        # Embedding
        objects.append(self._register_component("embed", "model.embed_tokens.weight"))

        # Decoder layers
        n_layers = self._layer_count()
        logger.info(f"Model has {n_layers} decoder layers.")
        for i in range(n_layers):
            size = self._layer_size(i)
            w = WeightObject.create(
                layer_index=i,
                size_bytes=size,
                layer_name=f"model.layers.{i}",
            )
            self.manager.register(w)
            objects.append(w)
            logger.debug(f"Registered layer {i}: {size // (1024**2)} MB")

        # Norm + LM head
        objects.append(self._register_component("norm", "model.norm.weight"))
        objects.append(self._register_component("lm_head", "lm_head.weight"))

        logger.info(
            f"Registered {len(objects)} WeightObjects "
            f"({n_layers} decoder layers + 3 components)"
        )
        return objects

    def _layer_size(self, layer_index: int) -> int:
        prefix = f"{_LAYER_PREFIX}{layer_index}."
        return sum(v for k, v in self._key_sizes.items() if k.startswith(prefix))

    def _register_component(self, name: str, key_prefix: str) -> WeightObject:
        size = sum(v for k, v in self._key_sizes.items() if k.startswith(key_prefix))
        w = WeightObject.create(layer_index=-1, size_bytes=size, layer_name=name)
        self.manager.register(w)
        logger.debug(f"Registered component '{name}': {size // 1024} KB")
        return w

    # ── Tensor loading (called by Memory Controller at stream time) ────

    def load_tensors(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """
        Load all tensors for a layer_name from disk → CPU RAM.
        This is the SSD → RAM step.
        """
        result: Dict[str, torch.Tensor] = {}
        with safe_open(str(self._safetensors_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(layer_name) or key == layer_name:
                    result[key] = f.get_tensor(key)
        if not result:
            raise KeyError(f"No tensors found for layer_name={layer_name!r}")
        return result

    @property
    def safetensors_path(self) -> str:
        return str(self._safetensors_file)
