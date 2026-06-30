"""Tests for FIFOScheduler."""
import pytest

from runtime.memory.memory_object import WeightObject
from runtime.plugins.scheduler.fifo import FIFOScheduler


@pytest.fixture
def scheduler():
    return FIFOScheduler()


def _make_weight(layer_index: int) -> WeightObject:
    return WeightObject.create(layer_index=layer_index, size_bytes=1024)


class TestFIFOScheduler:
    def test_empty_has_no_next(self, scheduler):
        assert scheduler.has_next() is False

    def test_enqueue_increments_remaining(self, scheduler):
        scheduler.enqueue(_make_weight(0))
        assert scheduler.remaining() == 1

    def test_fifo_order(self, scheduler):
        for i in range(5):
            scheduler.enqueue(_make_weight(i))
        indices = [scheduler.next().layer_index for _ in range(5)]
        assert indices == list(range(5))

    def test_next_on_empty_raises(self, scheduler):
        with pytest.raises(StopIteration):
            scheduler.next()

    def test_peek_does_not_consume(self, scheduler):
        w = _make_weight(0)
        scheduler.enqueue(w)
        peeked = scheduler.peek()
        assert peeked is w
        assert scheduler.remaining() == 1  # still in queue

    def test_peek_empty_returns_none(self, scheduler):
        assert scheduler.peek() is None

    def test_has_next_false_after_exhausted(self, scheduler):
        scheduler.enqueue(_make_weight(0))
        scheduler.next()
        assert scheduler.has_next() is False

    def test_remaining_decrements(self, scheduler):
        for i in range(3):
            scheduler.enqueue(_make_weight(i))
        scheduler.next()
        assert scheduler.remaining() == 2
