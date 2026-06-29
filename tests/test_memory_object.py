"""Tests for MemoryObject, WeightObject, and enums."""
import time
import pytest

from runtime.memory.memory_object import (
    Location, MemoryObject, ObjectState, ObjectType, WeightObject,
)


class TestObjectState:
    def test_poc_states_exist(self):
        for name in ("SSD_COLD", "RAM_READY", "VRAM_READY", "GPU_ACTIVE", "RELEASED"):
            assert hasattr(ObjectState, name)

    def test_reserved_states_exist(self):
        for name in ("PREFETCHING", "RAM_COLD", "RAM_HOT", "VRAM_EVICTION_QUEUE"):
            assert hasattr(ObjectState, name)


class TestWeightObjectCreate:
    def test_starts_in_ssd_cold(self):
        w = WeightObject.create(layer_index=0, size_bytes=1024)
        assert w.state == ObjectState.SSD_COLD
        assert w.location == Location.SSD

    def test_type_is_weight(self):
        w = WeightObject.create(layer_index=0, size_bytes=1024)
        assert w.type == ObjectType.WEIGHT

    def test_layer_index_set(self):
        w = WeightObject.create(layer_index=7, size_bytes=512)
        assert w.layer_index == 7

    def test_auto_layer_name(self):
        w = WeightObject.create(layer_index=3, size_bytes=1)
        assert "3" in w.layer_name

    def test_custom_layer_name(self):
        w = WeightObject.create(layer_index=0, size_bytes=1, layer_name="model.embed")
        assert w.layer_name == "model.embed"

    def test_unique_ids(self):
        ids = {WeightObject.create(layer_index=i, size_bytes=1).id for i in range(20)}
        assert len(ids) == 20

    def test_all_eight_fields_present(self):
        w = WeightObject.create(layer_index=0, size_bytes=1)
        for attr in ("id", "type", "location", "state", "size_bytes",
                     "priority", "timestamp", "reference_count"):
            assert hasattr(w, attr), f"Missing field: {attr}"

    def test_tensor_is_none_initially(self):
        w = WeightObject.create(layer_index=0, size_bytes=1)
        assert w.tensor is None

    def test_is_active_before_release(self):
        w = WeightObject.create(layer_index=0, size_bytes=1)
        assert w.is_active() is True

    def test_age_seconds_positive(self):
        w = WeightObject.create(layer_index=0, size_bytes=1)
        time.sleep(0.02)
        assert w.age_seconds() > 0

    def test_repr_contains_state_and_class(self):
        w = WeightObject.create(layer_index=0, size_bytes=4096)
        r = repr(w)
        assert "WeightObject" in r
        assert "SSD_COLD" in r
