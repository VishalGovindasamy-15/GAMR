"""
MemoryObject — the universal data unit in GAMR.

Every piece of data the runtime manages is a MemoryObject.
POC: WeightObject (model layer weights).
Future: KVCacheObject, ActivationObject, GradientObject, OptimizerStateObject.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch


class ObjectType(Enum):
    WEIGHT          = "WEIGHT"
    KV_CACHE        = "KV_CACHE"        # future
    ACTIVATION      = "ACTIVATION"      # future
    GRADIENT        = "GRADIENT"        # future
    OPTIMIZER_STATE = "OPTIMIZER_STATE" # future


class Location(Enum):
    SSD  = "SSD"
    RAM  = "RAM"
    VRAM = "VRAM"
    GPU  = "GPU"
    NONE = "NONE"  # released


class ObjectState(Enum):
    """
    POC active:   SSD_COLD, RAM_READY, VRAM_READY, GPU_ACTIVE, RELEASED
    Reserved:     PREFETCHING, RAM_COLD, RAM_HOT, VRAM_EVICTION_QUEUE
    """
    # ── POC active ────────────────────────
    SSD_COLD            = "SSD_COLD"
    RAM_READY           = "RAM_READY"
    VRAM_READY          = "VRAM_READY"
    GPU_ACTIVE          = "GPU_ACTIVE"
    RELEASED            = "RELEASED"

    # ── Reserved (defined, not used in POC)
    PREFETCHING         = "PREFETCHING"
    RAM_COLD            = "RAM_COLD"
    RAM_HOT             = "RAM_HOT"
    VRAM_EVICTION_QUEUE = "VRAM_EVICTION_QUEUE"


@dataclass
class MemoryObject:
    """
    Base class for all objects GAMR manages.

    All 8 fields are defined now — priority and reference_count
    are unused in the POC but are here to prevent restructuring later.
    """
    id:              str
    type:            ObjectType
    location:        Location
    state:           ObjectState
    size_bytes:      int
    priority:        int    # 0 = lowest; higher = keep longer (future)
    timestamp:       float  # epoch time of last state transition
    reference_count: int    # consumers holding a reference (future)

    @staticmethod
    def make_id() -> str:
        return str(uuid.uuid4())

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_active(self) -> bool:
        return self.state != ObjectState.RELEASED

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.id[:8]}…, "
            f"state={self.state.value}, "
            f"location={self.location.value}, "
            f"size={self.size_bytes // 1024}KB)"
        )


@dataclass
class WeightObject(MemoryObject):
    """One transformer layer's weights. Streamed SSD→RAM→VRAM→GPU→Released."""
    layer_index: int = 0
    layer_name:  str = ""
    tensor:      Optional[torch.Tensor] = None

    @classmethod
    def create(
        cls,
        layer_index: int,
        size_bytes: int,
        layer_name: str = "",
    ) -> "WeightObject":
        """Factory — creates a WeightObject in SSD_COLD state."""
        return cls(
            id=cls.make_id(),
            type=ObjectType.WEIGHT,
            location=Location.SSD,
            state=ObjectState.SSD_COLD,
            size_bytes=size_bytes,
            priority=0,
            timestamp=time.time(),
            reference_count=0,
            layer_index=layer_index,
            layer_name=layer_name or f"layer_{layer_index}",
        )
