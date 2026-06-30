"""Tests for StaticPrefetchScheduler."""
import pytest

from runtime.memory.memory_object import WeightObject
from runtime.plugins.scheduler.static_prefetch import StaticPrefetchScheduler


def _make(i: int) -> WeightObject:
    return WeightObject.create(layer_index=i, size_bytes=1024)


@pytest.fixture
def sched():
    return StaticPrefetchScheduler(prefetch_depth=2)


class TestStaticPrefetchScheduler:
    def test_prefetch_depth_stored(self):
        s = StaticPrefetchScheduler(prefetch_depth=3)
        assert s.prefetch_depth == 3

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            StaticPrefetchScheduler(prefetch_depth=0)

    def test_empty_has_no_next(self, sched):
        assert sched.has_next() is False

    def test_enqueue_and_remaining(self, sched):
        sched.enqueue(_make(0))
        assert sched.remaining() == 1

    def test_fifo_order_preserved(self, sched):
        for i in range(5):
            sched.enqueue(_make(i))
        result = [sched.next().layer_index for _ in range(5)]
        assert result == list(range(5))

    def test_prefetch_window_size(self, sched):
        for i in range(5):
            sched.enqueue(_make(i))
        sched.next()  # consume layer 0 — next() is now 1
        window = sched.prefetch_window()
        assert len(window) == 2  # prefetch_depth=2

    def test_prefetch_window_does_not_consume(self, sched):
        for i in range(3):
            sched.enqueue(_make(i))
        sched.prefetch_window()
        assert sched.remaining() == 3

    def test_prefetch_window_respects_queue_end(self):
        s = StaticPrefetchScheduler(prefetch_depth=5)
        s.enqueue(_make(0))
        s.enqueue(_make(1))
        # Only 2 items but depth=5 — window should return only what's there
        window = s.prefetch_window()
        assert len(window) == 2

    def test_prefetch_window_empty_when_queue_empty(self, sched):
        assert sched.prefetch_window() == []

    def test_peek_returns_next_without_removing(self, sched):
        sched.enqueue(_make(7))
        assert sched.peek().layer_index == 7
        assert sched.remaining() == 1

    def test_next_on_empty_raises(self, sched):
        with pytest.raises(StopIteration):
            sched.next()


class TestPrefetchObjectManagerTransitions:
    """Verify PREFETCHING state is now a valid transition from SSD_COLD."""

    def test_ssd_cold_to_prefetching(self):
        from runtime.memory.object_manager import ObjectManager
        from runtime.memory.memory_object import ObjectState
        mgr = ObjectManager()
        w = WeightObject.create(layer_index=0, size_bytes=1)
        mgr.register(w)
        mgr.transition(w.id, ObjectState.PREFETCHING)
        assert w.state == ObjectState.PREFETCHING

    def test_prefetching_to_vram_ready(self):
        from runtime.memory.object_manager import ObjectManager
        from runtime.memory.memory_object import ObjectState
        mgr = ObjectManager()
        w = WeightObject.create(layer_index=0, size_bytes=1)
        mgr.register(w)
        mgr.transition(w.id, ObjectState.PREFETCHING)
        mgr.transition(w.id, ObjectState.VRAM_READY)
        assert w.state == ObjectState.VRAM_READY

    def test_prefetching_cancelled_back_to_ssd_cold(self):
        from runtime.memory.object_manager import ObjectManager
        from runtime.memory.memory_object import ObjectState
        mgr = ObjectManager()
        w = WeightObject.create(layer_index=0, size_bytes=1)
        mgr.register(w)
        mgr.transition(w.id, ObjectState.PREFETCHING)
        mgr.transition(w.id, ObjectState.SSD_COLD)
        assert w.state == ObjectState.SSD_COLD


class TestPoolManagerPrefetchQueue:
    def test_mark_prefetching_adds_to_queue(self):
        from runtime.memory.pool_manager import MemoryPoolManager
        from runtime.plugins.pool.fixed import FixedPool
        pm = MemoryPoolManager(
            FixedPool(1024**3, "RAM"), FixedPool(1024**3, "VRAM")
        )
        pm.mark_prefetching("obj-123")
        assert pm.is_prefetching("obj-123") is True
        assert pm.prefetch_in_flight() == 1

    def test_mark_prefetch_done_removes(self):
        from runtime.memory.pool_manager import MemoryPoolManager
        from runtime.plugins.pool.fixed import FixedPool
        pm = MemoryPoolManager(
            FixedPool(1024**3, "RAM"), FixedPool(1024**3, "VRAM")
        )
        pm.mark_prefetching("obj-123")
        pm.mark_prefetch_done("obj-123")
        assert pm.is_prefetching("obj-123") is False
        assert pm.prefetch_in_flight() == 0

    def test_status_includes_prefetch_field(self):
        from runtime.memory.pool_manager import MemoryPoolManager
        from runtime.plugins.pool.fixed import FixedPool
        pm = MemoryPoolManager(
            FixedPool(1024**3, "RAM"), FixedPool(1024**3, "VRAM")
        )
        assert "prefetch_in_flight" in pm.status()
