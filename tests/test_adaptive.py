"""Tests for Phase 5 — AdaptiveScheduler + DecisionEngine."""
import pytest

from runtime.plugins.scheduler.adaptive import AdaptiveScheduler
from runtime.memory.memory_object import WeightObject
from runtime.controllers.decision_engine import DecisionEngine, RunMetrics


# ── Helpers ────────────────────────────────────────────────────────────────

def _make(i: int) -> WeightObject:
    return WeightObject.create(layer_index=i, size_bytes=1024)


def _metrics(
    gen_time_s: float = 5.0,
    mean_load_ms: float = 10.0,
    mean_compute_ms: float = 5.0,
    peak_vram_gb: float = 0.35,
    prefetch_depth: int = 1,
    run_id: str = "test_run",
) -> RunMetrics:
    return RunMetrics(
        run_id=run_id,
        gen_time_s=gen_time_s,
        mean_load_ms=mean_load_ms,
        mean_compute_ms=mean_compute_ms,
        peak_vram_gb=peak_vram_gb,
        prefetch_depth=prefetch_depth,
    )


# ── AdaptiveScheduler ──────────────────────────────────────────────────────

class TestAdaptiveScheduler:
    def test_initial_depth_stored(self):
        s = AdaptiveScheduler(initial_depth=2)
        assert s.prefetch_depth == 2

    def test_invalid_initial_depth_raises(self):
        with pytest.raises(ValueError):
            AdaptiveScheduler(initial_depth=0)

    def test_set_prefetch_depth_changes_value(self):
        s = AdaptiveScheduler(initial_depth=1)
        changed = s.set_prefetch_depth(3)
        assert changed is True
        assert s.prefetch_depth == 3

    def test_set_prefetch_depth_no_change_returns_false(self):
        s = AdaptiveScheduler(initial_depth=2)
        changed = s.set_prefetch_depth(2)
        assert changed is False

    def test_depth_clamped_at_max(self):
        s = AdaptiveScheduler(initial_depth=4)  # max
        s.set_prefetch_depth(999)
        assert s.prefetch_depth == s.max_depth()

    def test_depth_clamped_at_min(self):
        s = AdaptiveScheduler(initial_depth=1)  # min
        s.set_prefetch_depth(-5)
        assert s.prefetch_depth == s.min_depth()

    def test_fifo_order_preserved(self):
        s = AdaptiveScheduler(initial_depth=1)
        for i in range(5):
            s.enqueue(_make(i))
        result = [s.next().layer_index for _ in range(5)]
        assert result == list(range(5))

    def test_prefetch_window_uses_current_depth(self):
        s = AdaptiveScheduler(initial_depth=2)
        for i in range(5):
            s.enqueue(_make(i))
        window = s.prefetch_window()
        assert len(window) == 2
        s.set_prefetch_depth(4)
        window2 = s.prefetch_window()
        assert len(window2) == 4

    def test_next_on_empty_raises(self):
        s = AdaptiveScheduler(initial_depth=1)
        with pytest.raises(StopIteration):
            s.next()

    def test_peek_does_not_consume(self):
        s = AdaptiveScheduler(initial_depth=1)
        s.enqueue(_make(9))
        assert s.peek().layer_index == 9
        assert s.remaining() == 1


# ── RunMetrics ─────────────────────────────────────────────────────────────

class TestRunMetrics:
    def test_idle_ratio_high_load(self):
        m = _metrics(mean_load_ms=40.0, mean_compute_ms=10.0)
        # load=40 / (40+10) = 0.8
        assert abs(m.idle_ratio - 0.8) < 1e-6

    def test_idle_ratio_low_load(self):
        m = _metrics(mean_load_ms=1.0, mean_compute_ms=50.0)
        assert m.idle_ratio < 0.05

    def test_idle_ratio_zero_when_no_load(self):
        m = _metrics(mean_load_ms=0.0, mean_compute_ms=0.0)
        assert m.idle_ratio == 0.0

    def test_as_dict_has_required_keys(self):
        m = _metrics()
        d = m.as_dict()
        for key in ("run_id", "gen_time_s", "mean_load_ms", "mean_compute_ms",
                    "peak_vram_gb", "prefetch_depth", "idle_ratio", "ts"):
            assert key in d


# ── DecisionEngine ─────────────────────────────────────────────────────────

class TestDecisionEngineIncrease:
    def test_high_idle_triggers_increase(self):
        """idle_ratio > 0.40 → depth should increase."""
        s = AdaptiveScheduler(initial_depth=1)
        de = DecisionEngine(scheduler=s)
        # mean_load=50ms, mean_compute=10ms → idle=50/60≈83%
        m = _metrics(mean_load_ms=50.0, mean_compute_ms=10.0, gen_time_s=5.0)
        decision = de.observe_and_decide(m)
        assert decision["action"] == "increase"
        assert decision["new_depth"] == 2

    def test_at_max_depth_increase_is_no_change(self):
        s = AdaptiveScheduler(initial_depth=4)  # max
        de = DecisionEngine(scheduler=s)
        m = _metrics(mean_load_ms=50.0, mean_compute_ms=10.0)
        decision = de.observe_and_decide(m)
        # clamped at max — no actual change
        assert decision["new_depth"] == 4


class TestDecisionEngineDecrease:
    def test_low_idle_triggers_decrease(self):
        """idle_ratio < 0.10 → depth should decrease."""
        s = AdaptiveScheduler(initial_depth=3)
        de = DecisionEngine(scheduler=s)
        # mean_load=1ms, mean_compute=50ms → idle=1/51≈2%
        m = _metrics(mean_load_ms=1.0, mean_compute_ms=50.0, gen_time_s=5.0)
        decision = de.observe_and_decide(m)
        assert decision["action"] == "decrease"
        assert decision["new_depth"] == 2

    def test_at_min_depth_decrease_is_no_change(self):
        s = AdaptiveScheduler(initial_depth=1)  # min
        de = DecisionEngine(scheduler=s)
        m = _metrics(mean_load_ms=1.0, mean_compute_ms=50.0)
        decision = de.observe_and_decide(m)
        assert decision["new_depth"] == 1


class TestDecisionEngineStable:
    def test_mid_idle_is_no_change(self):
        """idle_ratio between 0.10 and 0.40 → stable."""
        s = AdaptiveScheduler(initial_depth=2)
        de = DecisionEngine(scheduler=s)
        # mean_load=15ms, mean_compute=60ms → idle=15/75=20%
        m = _metrics(mean_load_ms=15.0, mean_compute_ms=60.0)
        decision = de.observe_and_decide(m)
        assert decision["action"] == "no_change"
        assert decision["new_depth"] == 2


class TestDecisionEngineRollback:
    def test_rollback_fires_when_gen_time_does_not_improve(self):
        """
        Scenario:
          Run 1: gen=5.0s, high idle → DE increases depth 1→2
          Run 2-4: gen=5.1s (worse) → rollback should fire on cycle 3+
        """
        s = AdaptiveScheduler(initial_depth=1)
        de = DecisionEngine(scheduler=s)

        # Run 1: high idle → increase
        m1 = _metrics(gen_time_s=5.0, mean_load_ms=50.0, mean_compute_ms=10.0, prefetch_depth=1)
        d1 = de.observe_and_decide(m1)
        assert d1["action"] == "increase"
        assert s.prefetch_depth == 2

        # Runs 2-4: same or worse gen time at depth=2
        rollback_fired = False
        for i in range(3):
            mi = _metrics(gen_time_s=5.2, mean_load_ms=15.0, mean_compute_ms=60.0, prefetch_depth=2)
            di = de.observe_and_decide(mi)
            if di["rollback_fired"]:
                rollback_fired = True
                break

        assert rollback_fired, "Rollback should have fired after gen time did not improve"
        assert s.prefetch_depth == 1, "Depth should be rolled back to 1"

    def test_rollback_does_not_fire_when_gen_time_improves(self):
        """If gen_time improves after a depth increase, no rollback."""
        s = AdaptiveScheduler(initial_depth=1)
        de = DecisionEngine(scheduler=s)

        # Run 1: high idle → increase
        m1 = _metrics(gen_time_s=5.0, mean_load_ms=50.0, mean_compute_ms=10.0)
        de.observe_and_decide(m1)

        # Runs 2-4: gen time clearly improved (faster)
        for i in range(3):
            mi = _metrics(gen_time_s=3.5, mean_load_ms=15.0, mean_compute_ms=60.0, prefetch_depth=2)
            di = de.observe_and_decide(mi)
            assert di["rollback_fired"] is False


class TestDecisionEngineHistory:
    def test_history_grows_with_observations(self):
        s = AdaptiveScheduler(initial_depth=2)
        de = DecisionEngine(scheduler=s)
        for i in range(5):
            de.observe_and_decide(_metrics(run_id=f"run_{i}"))
        assert len(de.history()) == 5

    def test_last_n_gen_times(self):
        s = AdaptiveScheduler(initial_depth=2)
        de = DecisionEngine(scheduler=s)
        gen_times = [4.0, 4.5, 3.9, 4.1, 3.7]
        for gt in gen_times:
            de.observe_and_decide(_metrics(gen_time_s=gt))
        last_3 = de.last_n_gen_times(3)
        assert len(last_3) == 3
        assert last_3 == gen_times[-3:]


class TestDecisionEngineRecorder:
    def test_records_decision_events(self, tmp_path):
        from runtime.event_bus import EventBus
        from recorder.recorder import Recorder
        bus = EventBus()
        rec = Recorder(tmp_path, bus)
        s = AdaptiveScheduler(initial_depth=1)
        de = DecisionEngine(scheduler=s, recorder=rec)
        m = _metrics(mean_load_ms=50.0, mean_compute_ms=10.0)
        de.observe_and_decide(m)
        rec.close()
        import orjson
        events = [orjson.loads(l) for l in (tmp_path / "events.json").read_text().splitlines() if l.strip()]
        decision_events = [e for e in events if e["type"] == "DECISION_ENGINE"]
        assert len(decision_events) >= 1
        assert "action" in decision_events[0]["payload"]


class TestMemoryControllerAdaptiveIntegration:
    def test_controller_with_fifo_has_no_decision_engine(self):
        from runtime.controllers.memory_controller import MemoryController
        from runtime.event_bus import EventBus
        from runtime.memory.object_manager import ObjectManager
        from runtime.plugins.pool.fixed import FixedPool
        from runtime.plugins.scheduler.fifo import FIFOScheduler
        from runtime.hal.cpu_backend import CPUBackend
        bus = EventBus()
        hal = CPUBackend()
        ctrl = MemoryController(
            hal=hal,
            object_manager=ObjectManager(),
            scheduler=FIFOScheduler(),
            ram_pool=FixedPool(1024**3, "RAM"),
            vram_pool=FixedPool(1024**3, "VRAM"),
            event_bus=bus,
        )
        assert ctrl.decision_engine is None

    def test_controller_with_adaptive_has_decision_engine(self):
        from runtime.controllers.memory_controller import MemoryController
        from runtime.event_bus import EventBus
        from runtime.memory.object_manager import ObjectManager
        from runtime.plugins.pool.fixed import FixedPool
        from runtime.hal.cpu_backend import CPUBackend
        bus = EventBus()
        hal = CPUBackend()
        ctrl = MemoryController(
            hal=hal,
            object_manager=ObjectManager(),
            scheduler=AdaptiveScheduler(initial_depth=1),
            ram_pool=FixedPool(1024**3, "RAM"),
            vram_pool=FixedPool(1024**3, "VRAM"),
            event_bus=bus,
        )
        assert ctrl.decision_engine is not None

    def test_adaptive_loop_requires_adaptive_scheduler(self):
        from runtime.controllers.memory_controller import MemoryController
        from runtime.event_bus import EventBus
        from runtime.memory.object_manager import ObjectManager
        from runtime.plugins.pool.fixed import FixedPool
        from runtime.plugins.scheduler.fifo import FIFOScheduler
        from runtime.hal.cpu_backend import CPUBackend
        ctrl = MemoryController(
            hal=CPUBackend(),
            object_manager=ObjectManager(),
            scheduler=FIFOScheduler(),
            ram_pool=FixedPool(1024**3, "RAM"),
            vram_pool=FixedPool(1024**3, "VRAM"),
            event_bus=EventBus(),
        )
        with pytest.raises(RuntimeError, match="AdaptiveScheduler"):
            ctrl.run_adaptive_loop("model_path", "prompt", n_runs=1)
