"""Tests for ObjectManager state machine."""
import time
import pytest
import torch

from runtime.memory.memory_object import Location, ObjectState, WeightObject
from runtime.memory.object_manager import ObjectManager


@pytest.fixture
def manager():
    return ObjectManager()


@pytest.fixture
def weight(manager):
    w = WeightObject.create(layer_index=0, size_bytes=1024)
    manager.register(w)
    return w


class TestRegistry:
    def test_register_and_get(self, manager):
        w = WeightObject.create(layer_index=0, size_bytes=1)
        manager.register(w)
        assert manager.get(w.id) is w

    def test_get_unknown_returns_none(self, manager):
        assert manager.get("nonexistent") is None

    def test_register_duplicate_raises(self, manager, weight):
        with pytest.raises(ValueError, match="already registered"):
            manager.register(weight)

    def test_count(self, manager):
        for i in range(3):
            manager.register(WeightObject.create(layer_index=i, size_bytes=1))
        assert manager.count() == 3

    def test_objects_in_state(self, manager):
        for i in range(4):
            manager.register(WeightObject.create(layer_index=i, size_bytes=1))
        assert len(manager.objects_in_state(ObjectState.SSD_COLD)) == 4

    def test_all_objects(self, manager, weight):
        objs = manager.all_objects()
        assert weight in objs


class TestTransitions:
    def test_ssd_cold_to_ram_ready(self, manager, weight):
        manager.transition(weight.id, ObjectState.RAM_READY)
        assert weight.state == ObjectState.RAM_READY

    def test_full_lifecycle(self, manager, weight):
        manager.transition(weight.id, ObjectState.RAM_READY)
        manager.transition(weight.id, ObjectState.VRAM_READY)
        manager.transition(weight.id, ObjectState.GPU_ACTIVE)
        manager.transition(weight.id, ObjectState.RELEASED)
        assert weight.state == ObjectState.RELEASED

    def test_invalid_transition_raises(self, manager, weight):
        with pytest.raises(ValueError, match="Invalid transition"):
            manager.transition(weight.id, ObjectState.GPU_ACTIVE)

    def test_released_is_terminal(self, manager, weight):
        manager.transition(weight.id, ObjectState.RAM_READY)
        manager.transition(weight.id, ObjectState.RELEASED)
        with pytest.raises(ValueError):
            manager.transition(weight.id, ObjectState.RAM_READY)

    def test_unknown_id_raises_key_error(self, manager):
        with pytest.raises(KeyError):
            manager.transition("ghost", ObjectState.RAM_READY)

    def test_location_updated_with_state(self, manager, weight):
        manager.transition(weight.id, ObjectState.RAM_READY)
        assert weight.location == Location.RAM
        manager.transition(weight.id, ObjectState.VRAM_READY)
        assert weight.location == Location.VRAM
        manager.transition(weight.id, ObjectState.GPU_ACTIVE)
        assert weight.location == Location.GPU

    def test_timestamp_updated_on_transition(self, manager, weight):
        old_ts = weight.timestamp
        time.sleep(0.02)
        manager.transition(weight.id, ObjectState.RAM_READY)
        assert weight.timestamp > old_ts

    def test_release_clears_tensor(self, manager, weight):
        weight.tensor = torch.zeros(10)
        manager.transition(weight.id, ObjectState.RAM_READY)
        manager.transition(weight.id, ObjectState.RELEASED)
        assert weight.tensor is None

    def test_evict_back_to_ram(self, manager, weight):
        """VRAM_READY → RAM_READY is a valid eviction path."""
        manager.transition(weight.id, ObjectState.RAM_READY)
        manager.transition(weight.id, ObjectState.VRAM_READY)
        manager.transition(weight.id, ObjectState.RAM_READY)
        assert weight.state == ObjectState.RAM_READY


class TestRelease:
    def test_release_removes_from_registry(self, manager, weight):
        manager.release(weight.id)
        assert manager.get(weight.id) is None
        assert manager.count() == 0

    def test_release_sets_state(self, manager, weight):
        manager.release(weight.id)
        assert weight.state == ObjectState.RELEASED
