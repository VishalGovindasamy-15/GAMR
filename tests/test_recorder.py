"""Tests for Recorder — Phase 4 full implementation."""
import asyncio
import time
from pathlib import Path

import orjson
import pytest

from recorder.recorder import Recorder
from runtime.event_bus import Event, EventBus, EventType


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def recorder(run_dir: Path) -> Recorder:
    bus = EventBus()
    r = Recorder(run_dir, bus)
    yield r
    if not r._closed:
        r.close()


class TestRecorderFiles:
    def test_creates_events_json(self, recorder, run_dir):
        recorder.close()
        assert (run_dir / "events.json").exists()

    def test_creates_all_csv_files(self, recorder, run_dir):
        recorder.close()
        for name in ("gpu.csv", "ram.csv", "vram.csv", "latency.csv"):
            assert (run_dir / name).exists(), f"Missing: {name}"

    def test_csv_files_have_headers(self, recorder, run_dir):
        recorder.close()
        gpu_header = (run_dir / "gpu.csv").read_text().splitlines()[0]
        assert "gpu_util_pct" in gpu_header
        ram_header = (run_dir / "ram.csv").read_text().splitlines()[0]
        assert "ram_used_gb" in ram_header


class TestRecorderSyncRecord:
    def test_record_writes_to_events_json(self, recorder, run_dir):
        recorder.record("LAYER_LOAD_DONE", payload={"layer_index": 0, "latency_ms": 5.0})
        recorder.close()
        lines = (run_dir / "events.json").read_text().splitlines()
        assert len(lines) == 1
        data = orjson.loads(lines[0])
        assert data["type"] == "LAYER_LOAD_DONE"

    def test_record_latency_event_writes_to_latency_csv(self, recorder, run_dir):
        recorder.record(
            "LAYER_LOAD_DONE",
            payload={"layer_index": 3, "layer_name": "decoder.3", "latency_ms": 12.5, "vram_used_gb": 0.1},
        )
        recorder.close()
        content = (run_dir / "latency.csv").read_text()
        assert "decoder.3" in content
        assert "LAYER_LOAD_DONE" in content

    def test_record_compute_event_also_writes_to_latency_csv(self, recorder, run_dir):
        recorder.record(
            "LAYER_COMPUTE_DONE",
            payload={"layer_index": 5, "layer_name": "decoder.5", "latency_ms": 8.1, "vram_used_gb": 0.2},
        )
        recorder.close()
        content = (run_dir / "latency.csv").read_text()
        assert "LAYER_COMPUTE_DONE" in content

    def test_multiple_records_produce_multiple_lines(self, recorder, run_dir):
        for i in range(10):
            recorder.record("LAYER_LOAD_DONE", payload={"layer_index": i})
        recorder.close()
        lines = [l for l in (run_dir / "events.json").read_text().splitlines() if l.strip()]
        assert len(lines) == 10

    def test_event_json_is_valid_ndjson(self, recorder, run_dir):
        recorder.record("LAYER_LOAD_DONE", payload={"layer_index": 0})
        recorder.record("LAYER_COMPUTE_DONE", payload={"layer_index": 0})
        recorder.close()
        for line in (run_dir / "events.json").read_text().splitlines():
            d = orjson.loads(line)
            assert "ts" in d
            assert "type" in d

    def test_event_payload_stored(self, recorder, run_dir):
        recorder.record("LAYER_LOAD_DONE", object_id="abc123", payload={"layer_index": 7, "latency_ms": 99.9})
        recorder.close()
        data = orjson.loads((run_dir / "events.json").read_text().splitlines()[0])
        assert data["payload"]["layer_index"] == 7
        assert data["object_id"] == "abc123"


class TestRecorderBusSubscription:
    def test_monitor_event_writes_to_ram_csv(self, recorder, run_dir):
        async def _run():
            bus = recorder._bus
            await bus.publish_and_wait(Event(
                type=EventType.MONITOR_METRICS,
                payload={
                    "ts": time.time(),
                    "ram_used_gb": 4.0,
                    "ram_free_gb": 8.0,
                    "ram_util_pct": 33.0,
                    "gpu_util_pct": 75,
                    "vram_used_gb": 1.5,
                    "vram_free_gb": 4.5,
                },
            ))
            recorder.close()
            content = (run_dir / "ram.csv").read_text()
            assert "4.0" in content

        asyncio.run(_run())

    def test_monitor_event_writes_to_events_json(self, recorder, run_dir):
        async def _run():
            bus = recorder._bus
            await bus.publish_and_wait(Event(
                type=EventType.MONITOR_METRICS,
                payload={"ts": time.time(), "ram_used_gb": 4.0, "ram_free_gb": 8.0, "ram_util_pct": 30.0},
            ))
            recorder.close()
            lines = [l for l in (run_dir / "events.json").read_text().splitlines() if l.strip()]
            assert len(lines) >= 1
            types = [orjson.loads(l)["type"] for l in lines]
            assert "MONITOR_METRICS" in types

        asyncio.run(_run())


class TestRecorderReplay:
    def test_replay_returns_sorted_events(self, recorder, run_dir):
        recorder.record("LAYER_LOAD_DONE",    payload={"layer_index": 0})
        recorder.record("LAYER_COMPUTE_DONE", payload={"layer_index": 0})
        recorder.record("LAYER_LOAD_DONE",    payload={"layer_index": 1})
        recorder.close()
        events = Recorder.replay(run_dir)
        assert len(events) == 3
        ts_list = [e["ts"] for e in events]
        assert ts_list == sorted(ts_list)

    def test_replay_empty_if_no_events(self, run_dir):
        bus = EventBus()
        r = Recorder(run_dir, bus)
        r.close()
        events = Recorder.replay(run_dir)
        assert events == []

    def test_replay_event_structure(self, recorder, run_dir):
        recorder.record("LAYER_LOAD_DONE", object_id="xyz", payload={"layer_index": 2})
        recorder.close()
        events = Recorder.replay(run_dir)
        e = events[0]
        assert "ts" in e
        assert "type" in e
        assert "object_id" in e
        assert "payload" in e
        assert e["type"] == "LAYER_LOAD_DONE"
        assert e["object_id"] == "xyz"


class TestRecorderClose:
    def test_close_is_idempotent(self, recorder):
        recorder.close()
        recorder.close()  # should not raise

    def test_record_after_close_is_no_op(self, recorder):
        recorder.close()
        recorder.record("LAYER_LOAD_DONE")  # should not raise
