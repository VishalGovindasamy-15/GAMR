"""
ObjectManager — owns all MemoryObject state transitions.

Only ObjectManager (called by the Memory Controller) may change object state.
All other components read state; only ObjectManager writes it.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

from runtime.memory.memory_object import (
    Location, MemoryObject, ObjectState, WeightObject,
)

logger = logging.getLogger("gamr.memory.object_manager")

# ── Valid transitions ──────────────────────────────────────────────────────────
_VALID: Dict[ObjectState, Set[ObjectState]] = {
    ObjectState.SSD_COLD: {
        ObjectState.RAM_READY,
        ObjectState.PREFETCHING,   # Phase 3: async H2D copy started
        ObjectState.RELEASED,
    },
    ObjectState.PREFETCHING: {
        ObjectState.VRAM_READY,    # Phase 3: copy complete → ready
        ObjectState.RAM_READY,     # copy landed in RAM en route to VRAM
        ObjectState.SSD_COLD,      # prefetch cancelled
        ObjectState.RELEASED,
    },
    ObjectState.RAM_READY: {
        ObjectState.VRAM_READY, ObjectState.SSD_COLD, ObjectState.RELEASED,
    },
    ObjectState.VRAM_READY: {
        ObjectState.GPU_ACTIVE, ObjectState.RAM_READY, ObjectState.RELEASED,
    },
    ObjectState.GPU_ACTIVE: {
        ObjectState.VRAM_READY, ObjectState.RAM_READY, ObjectState.RELEASED,
    },
    ObjectState.RELEASED: set(),  # terminal
    # reserved future states
    ObjectState.RAM_COLD: {ObjectState.RAM_HOT, ObjectState.SSD_COLD, ObjectState.RELEASED},
    ObjectState.RAM_HOT:  {ObjectState.VRAM_READY, ObjectState.RAM_COLD, ObjectState.RELEASED},
    ObjectState.VRAM_EVICTION_QUEUE: {ObjectState.RAM_READY, ObjectState.RELEASED},
}

_STATE_TO_LOCATION: Dict[ObjectState, Location] = {
    ObjectState.SSD_COLD:            Location.SSD,
    ObjectState.PREFETCHING:         Location.SSD,
    ObjectState.RAM_COLD:            Location.RAM,
    ObjectState.RAM_READY:           Location.RAM,
    ObjectState.RAM_HOT:             Location.RAM,
    ObjectState.VRAM_READY:          Location.VRAM,
    ObjectState.VRAM_EVICTION_QUEUE: Location.VRAM,
    ObjectState.GPU_ACTIVE:          Location.GPU,
    ObjectState.RELEASED:            Location.NONE,
}


class ObjectManager:
    """Registry and state machine for all MemoryObjects."""

    def __init__(self) -> None:
        self._objects: Dict[str, MemoryObject] = {}

    # ── Registry ──────────────────────────────────────────────────────

    def register(self, obj: MemoryObject) -> None:
        if obj.id in self._objects:
            raise ValueError(f"Object {obj.id!r} is already registered.")
        self._objects[obj.id] = obj
        logger.debug(f"Registered {obj!r}")

    def get(self, object_id: str) -> Optional[MemoryObject]:
        return self._objects.get(object_id)

    def all_objects(self) -> List[MemoryObject]:
        return list(self._objects.values())

    def objects_in_state(self, state: ObjectState) -> List[MemoryObject]:
        return [o for o in self._objects.values() if o.state == state]

    def count(self) -> int:
        return len(self._objects)

    # ── Transitions ───────────────────────────────────────────────────

    def transition(self, object_id: str, new_state: ObjectState) -> MemoryObject:
        """
        Move object to new_state. Raises ValueError on illegal transition.
        Updates location and timestamp automatically.
        """
        obj = self._objects.get(object_id)
        if obj is None:
            raise KeyError(f"No object with id={object_id!r}")

        allowed = _VALID.get(obj.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid transition {obj.state.value} → {new_state.value} "
                f"(allowed: {[s.value for s in allowed]})"
            )

        old = obj.state
        obj.state    = new_state
        obj.location = _STATE_TO_LOCATION[new_state]
        obj.timestamp = time.time()

        if new_state == ObjectState.RELEASED and isinstance(obj, WeightObject):
            obj.tensor = None

        logger.debug(f"{obj.__class__.__name__}({object_id[:8]}…) {old.value} → {new_state.value}")
        return obj

    def release(self, object_id: str) -> None:
        """Transition to RELEASED and remove from registry."""
        self.transition(object_id, ObjectState.RELEASED)
        del self._objects[object_id]
        logger.debug(f"Object {object_id[:8]}… removed from registry.")
