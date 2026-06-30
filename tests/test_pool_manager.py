"""Tests for MemoryPoolManager."""
import pytest

from runtime.memory.memory_object import WeightObject
from runtime.memory.pool_manager import MemoryPoolManager
from runtime.plugins.pool.fixed import FixedPool


def _make_pool(budget_mb: int, name: str) -> FixedPool:
    return FixedPool(budget_bytes=budget_mb * 1024 * 1024, name=name)


def _obj(size_mb: int) -> WeightObject:
    return WeightObject.create(layer_index=0, size_bytes=size_mb * 1024 * 1024)


@pytest.fixture
def pool_manager():
    return MemoryPoolManager(
        ram_pool=_make_pool(100, "RAM"),
        vram_pool=_make_pool(50, "VRAM"),
    )


class TestMemoryPoolManager:
    def test_can_load_to_ram_when_space_available(self, pool_manager):
        assert pool_manager.can_load_to_ram(_obj(10)) is True

    def test_cannot_load_to_ram_when_full(self, pool_manager):
        assert pool_manager.can_load_to_ram(_obj(200)) is False

    def test_can_load_to_vram_when_space_available(self, pool_manager):
        assert pool_manager.can_load_to_vram(_obj(10)) is True

    def test_cannot_load_to_vram_when_full(self, pool_manager):
        assert pool_manager.can_load_to_vram(_obj(100)) is False

    def test_allocate_ram_reduces_free(self, pool_manager):
        obj = _obj(20)
        free_before = pool_manager.ram_free_bytes()
        pool_manager.allocate_ram(obj)
        assert pool_manager.ram_free_bytes() == free_before - obj.size_bytes

    def test_free_ram_restores_budget(self, pool_manager):
        obj = _obj(20)
        pool_manager.allocate_ram(obj)
        pool_manager.free_ram(obj)
        assert pool_manager.ram_utilization() == 0.0

    def test_allocate_vram_reduces_free(self, pool_manager):
        obj = _obj(10)
        free_before = pool_manager.vram_free_bytes()
        pool_manager.allocate_vram(obj)
        assert pool_manager.vram_free_bytes() == free_before - obj.size_bytes

    def test_free_vram_restores_budget(self, pool_manager):
        obj = _obj(10)
        pool_manager.allocate_vram(obj)
        pool_manager.free_vram(obj)
        assert pool_manager.vram_utilization() == 0.0

    def test_status_contains_all_keys(self, pool_manager):
        s = pool_manager.status()
        for key in ("ram_budget_gb", "ram_used_gb", "ram_util_pct",
                    "vram_budget_gb", "vram_used_gb", "vram_util_pct"):
            assert key in s, f"Missing key: {key}"

    def test_initial_utilization_is_zero(self, pool_manager):
        assert pool_manager.ram_utilization() == 0.0
        assert pool_manager.vram_utilization() == 0.0

    def test_ram_and_vram_pools_are_independent(self, pool_manager):
        pool_manager.allocate_ram(_obj(50))
        # VRAM should be unaffected
        assert pool_manager.vram_utilization() == 0.0
