"""
EventBus — async typed publish/subscribe.

Core flow:
  SSD_READ_DONE → RAM_READY → VRAM_COPY_DONE → GPU_COMPUTE_STARTED
                → GPU_COMPUTE_DONE → PREFETCH_NEXT

Rule: Monitor → Event Bus → Memory Controller (never direct calls).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("gamr.event_bus")


class EventType(Enum):
    # Storage
    SSD_READ_STARTED    = "SSD_READ_STARTED"
    SSD_READ_DONE       = "SSD_READ_DONE"
    # RAM
    RAM_READY           = "RAM_READY"
    # VRAM
    VRAM_COPY_STARTED   = "VRAM_COPY_STARTED"
    VRAM_COPY_DONE      = "VRAM_COPY_DONE"
    # GPU
    GPU_COMPUTE_STARTED = "GPU_COMPUTE_STARTED"
    GPU_COMPUTE_DONE    = "GPU_COMPUTE_DONE"
    # Scheduling
    PREFETCH_NEXT       = "PREFETCH_NEXT"
    # Lifecycle
    STATE_CHANGED       = "STATE_CHANGED"
    OBJECT_RELEASED     = "OBJECT_RELEASED"
    # Monitor (Phase 4)
    MONITOR_METRICS     = "MONITOR_METRICS"
    # Runtime
    RUNTIME_STARTED     = "RUNTIME_STARTED"
    RUNTIME_STOPPED     = "RUNTIME_STOPPED"


@dataclass
class Event:
    type:      EventType
    object_id: str = ""
    timestamp: float = field(default_factory=time.time)
    payload:   Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        obj = f" obj={self.object_id[:8]}…" if self.object_id else ""
        return f"Event({self.type.value}{obj} t={self.timestamp:.3f})"


Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async pub/sub event bus. Handlers are async coroutines.
    subscribe(handler, event_type=None) → wildcard (receives all events).
    publish() is fire-and-forget via asyncio.create_task().
    publish_and_wait() awaits all handlers sequentially (for tests).
    """

    def __init__(self) -> None:
        self._handlers: Dict[Optional[EventType], List[Handler]] = {}

    def subscribe(self, handler: Handler, event_type: Optional[EventType] = None) -> None:
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        label = event_type.value if event_type else "ALL"
        logger.debug(f"Subscribed {handler.__name__} → {label}")

    def unsubscribe(self, handler: Handler, event_type: Optional[EventType] = None) -> None:
        key = event_type
        if key in self._handlers:
            self._handlers[key] = [h for h in self._handlers[key] if h is not handler]

    def _collect_handlers(self, event: Event) -> List[Handler]:
        handlers: List[Handler] = []
        if event.type in self._handlers:
            handlers.extend(self._handlers[event.type])
        if None in self._handlers:
            handlers.extend(self._handlers[None])
        return handlers

    async def publish(self, event: Event) -> None:
        """Fire-and-forget. Schedules handlers as asyncio Tasks."""
        handlers = self._collect_handlers(event)
        if not handlers:
            logger.debug(f"No handlers for {event!r}")
            return
        for h in handlers:
            asyncio.create_task(_safe_call(h, event), name=f"gamr.{event.type.value}")
        logger.debug(f"Published {event!r} → {len(handlers)} handler(s)")

    async def publish_and_wait(self, event: Event) -> None:
        """Await all handlers sequentially. Use in tests."""
        for h in self._collect_handlers(event):
            await _safe_call(h, event)


async def _safe_call(handler: Handler, event: Event) -> None:
    try:
        await handler(event)
    except Exception as exc:
        logger.error(f"Handler {handler.__name__} raised {exc!r} on {event!r}")
