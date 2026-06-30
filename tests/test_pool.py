"""Tests for FixedPool."""
import pytest

from runtime.memory.memory_object import WeightObject
from runtime.plugins.pool.fixed import FixedPool, make_ram_pool, make_vram_pool


@pytest.fixture
def pool():
    return FixedPool(budget_bytes=10 * 1024 * 1024, name="TestPool")  # 10 MB


def _obj(size_bytes: int) -> WeightObject:
    return WeightObject.create(layer_index=0, size_bytes=size_bytes)


class TestFixedPool:
    def test_budget_matches_init(self, pool):
        assert pool.budget_bytes() == 10 * 1024 * 1024

    def test_initially_empty(self, pool):
        assert pool.used_bytes() == 0
        assert pool.free_bytes() == pool.budget_bytes()

    def test_can_fit_small_object(self, pool):
        obj = _obj(1024)
        assert pool.can_fit(obj) is True

    def test_cannot_fit_oversized_object(self, pool):
        obj = _obj(pool.budget_bytes() + 1)
        assert pool.can_fit(obj) is False

    def test_allocate_updates_used(self, pool):
        obj = _obj(1024 * 1024)  # 1 MB
        pool.allocate(obj)
        assert pool.used_bytes() == 1024 * 1024

    def test_allocate_over_budget_raises(self, pool):
        pool.allocate(_obj(pool.budget_bytes()))
        with pytest.raises(MemoryError):
            pool.allocate(_obj(1))

    def test_free_restores_budget(self, pool):
        obj = _obj(1024 * 1024)
        pool.allocate(obj)
        pool.free(obj)
        assert pool.used_bytes() == 0
        assert pool.free_bytes() == pool.budget_bytes()

    def test_utilization_zero_initially(self, pool):
        assert pool.utilization() == 0.0

    def test_utilization_after_half_fill(self, pool):
        obj = _obj(pool.budget_bytes() // 2)
        pool.allocate(obj)
        assert abs(pool.utilization() - 0.5) < 0.01


class TestPoolFactories:
    def test_make_ram_pool(self):
        p = make_ram_pool(free_ram_bytes=8 * 1024**3, fraction=0.25)
        assert p.budget_bytes() == int(8 * 1024**3 * 0.25)

    def test_make_vram_pool(self):
        p = make_vram_pool(free_vram_bytes=6 * 1024**3, fraction=0.80)
        assert p.budget_bytes() == int(6 * 1024**3 * 0.80)
