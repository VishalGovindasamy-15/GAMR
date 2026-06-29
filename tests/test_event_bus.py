"""Tests for the async EventBus."""
import asyncio
import pytest

from runtime.event_bus import Event, EventBus, EventType


@pytest.fixture
def bus():
    return EventBus()


class TestSubscribeAndPublish:
    async def test_specific_handler_called(self, bus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(handler, EventType.RAM_READY)
        await bus.publish_and_wait(Event(type=EventType.RAM_READY, object_id="abc"))
        assert len(received) == 1
        assert received[0].type == EventType.RAM_READY

    async def test_specific_handler_not_called_for_other_type(self, bus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(handler, EventType.RAM_READY)
        await bus.publish_and_wait(Event(type=EventType.GPU_COMPUTE_DONE))
        assert received == []

    async def test_wildcard_receives_all(self, bus):
        received = []

        async def wildcard(e: Event):
            received.append(e.type)

        bus.subscribe(wildcard)
        await bus.publish_and_wait(Event(type=EventType.SSD_READ_DONE))
        await bus.publish_and_wait(Event(type=EventType.GPU_COMPUTE_DONE))
        assert EventType.SSD_READ_DONE in received
        assert EventType.GPU_COMPUTE_DONE in received

    async def test_multiple_handlers_same_type(self, bus):
        calls = []

        async def h1(e): calls.append("h1")
        async def h2(e): calls.append("h2")

        bus.subscribe(h1, EventType.VRAM_COPY_DONE)
        bus.subscribe(h2, EventType.VRAM_COPY_DONE)
        await bus.publish_and_wait(Event(type=EventType.VRAM_COPY_DONE))
        assert "h1" in calls
        assert "h2" in calls

    async def test_no_handlers_no_error(self, bus):
        # Should not raise even if no handlers registered
        await bus.publish_and_wait(Event(type=EventType.PREFETCH_NEXT))

    async def test_event_payload_passed_through(self, bus):
        received = []

        async def handler(e: Event):
            received.append(e.payload)

        bus.subscribe(handler, EventType.SSD_READ_DONE)
        await bus.publish_and_wait(
            Event(type=EventType.SSD_READ_DONE, payload={"latency_ms": 8.3})
        )
        assert received[0]["latency_ms"] == 8.3

    async def test_object_id_passed_through(self, bus):
        received = []

        async def handler(e: Event):
            received.append(e.object_id)

        bus.subscribe(handler, EventType.RAM_READY)
        await bus.publish_and_wait(Event(type=EventType.RAM_READY, object_id="my-obj-id"))
        assert received[0] == "my-obj-id"

    async def test_bad_handler_does_not_crash_bus(self, bus):
        """A handler that raises must not prevent other handlers from running."""
        results = []

        async def bad(e: Event):
            raise RuntimeError("intentional error")

        async def good(e: Event):
            results.append("ok")

        bus.subscribe(bad, EventType.RUNTIME_STARTED)
        bus.subscribe(good, EventType.RUNTIME_STARTED)
        await bus.publish_and_wait(Event(type=EventType.RUNTIME_STARTED))
        assert "ok" in results


class TestUnsubscribe:
    async def test_unsubscribe_stops_calls(self, bus):
        received = []

        async def handler(e: Event):
            received.append(e)

        bus.subscribe(handler, EventType.RAM_READY)
        bus.unsubscribe(handler, EventType.RAM_READY)
        await bus.publish_and_wait(Event(type=EventType.RAM_READY))
        assert received == []


class TestEventRepr:
    def test_repr_with_object_id(self):
        e = Event(type=EventType.GPU_COMPUTE_DONE, object_id="abcdef12")
        r = repr(e)
        assert "GPU_COMPUTE_DONE" in r
        assert "abcdef" in r

    def test_repr_without_object_id(self):
        e = Event(type=EventType.RUNTIME_STARTED)
        r = repr(e)
        assert "RUNTIME_STARTED" in r
