"""Tests for SystemMonitor and MonitorPlugin base."""
import asyncio
import pytest

from runtime.event_bus import EventBus, EventType
from runtime.plugins.monitor.base import MonitorPlugin
from runtime.plugins.monitor.system import SystemMonitor


class TestMonitorPlugin:
    def test_monitor_plugin_is_abstract(self):
        """Cannot instantiate MonitorPlugin directly."""
        with pytest.raises(TypeError):
            MonitorPlugin()  # type: ignore


class TestSystemMonitorSnapshot:
    def test_snapshot_returns_dict(self):
        bus = EventBus()
        m = SystemMonitor(bus, device="cpu", interval_s=1.0)
        snap = m.snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_has_ram_keys(self):
        bus = EventBus()
        m = SystemMonitor(bus, device="cpu", interval_s=1.0)
        snap = m.snapshot()
        for key in ("ts", "ram_used_gb", "ram_free_gb", "ram_util_pct"):
            assert key in snap, f"Missing key: {key}"

    def test_snapshot_ram_values_positive(self):
        bus = EventBus()
        m = SystemMonitor(bus, device="cpu", interval_s=1.0)
        snap = m.snapshot()
        assert snap["ram_used_gb"] > 0
        assert snap["ram_free_gb"] >= 0
        assert 0 <= snap["ram_util_pct"] <= 100

    def test_snapshot_ts_is_recent(self):
        import time
        bus = EventBus()
        m = SystemMonitor(bus, device="cpu", interval_s=1.0)
        before = time.time()
        snap = m.snapshot()
        after  = time.time()
        assert before <= snap["ts"] <= after


class TestSystemMonitorLifecycle:
    def test_start_and_stop(self):
        async def _run():
            bus = EventBus()
            m = SystemMonitor(bus, device="cpu", interval_s=0.05)
            await m.start()
            assert m._running is True
            await asyncio.sleep(0.2)  # let it sample a few times
            await m.stop()
            assert m._running is False

        asyncio.run(_run())

    def test_monitor_publishes_events(self):
        collected = []

        async def _run():
            bus = EventBus()

            async def handler(event):
                collected.append(event)

            bus.subscribe(handler, EventType.MONITOR_METRICS)
            m = SystemMonitor(bus, device="cpu", interval_s=0.05)
            await m.start()
            await asyncio.sleep(0.25)
            await m.stop()

        asyncio.run(_run())
        # Should have collected at least 2 metric events in 250ms with 50ms interval
        assert len(collected) >= 2

    def test_events_have_monitor_type(self):
        events = []

        async def _run():
            bus = EventBus()

            async def handler(event):
                events.append(event)

            bus.subscribe(handler)  # wildcard
            m = SystemMonitor(bus, device="cpu", interval_s=0.05)
            await m.start()
            await asyncio.sleep(0.15)
            await m.stop()

        asyncio.run(_run())
        monitor_events = [e for e in events if e.type == EventType.MONITOR_METRICS]
        assert len(monitor_events) >= 1
        # Each event has ts in payload
        for e in monitor_events:
            assert "ts" in e.payload
            assert "ram_used_gb" in e.payload
