"""Tests for MemoryController (no model required — tests structure and wiring)."""
import pytest

from runtime.controllers.memory_controller import MemoryController, _StreamingLayerWrapper
from runtime.event_bus import EventBus
from runtime.hal.cpu_backend import CPUBackend
from runtime.memory.object_manager import ObjectManager
from runtime.memory.pool_manager import MemoryPoolManager
from runtime.plugins.pool.fixed import FixedPool
from runtime.plugins.scheduler.fifo import FIFOScheduler


@pytest.fixture
def controller():
    hal = CPUBackend()
    ram_pool  = FixedPool(budget_bytes=2 * 1024**3, name="RAM")
    vram_pool = FixedPool(budget_bytes=1 * 1024**3, name="VRAM")
    return MemoryController(
        hal=hal,
        object_manager=ObjectManager(),
        scheduler=FIFOScheduler(),
        ram_pool=ram_pool,
        vram_pool=vram_pool,
        event_bus=EventBus(),
    )


class TestMemoryControllerInit:
    def test_controller_instantiates(self, controller):
        assert controller is not None

    def test_device_is_cpu(self, controller):
        assert controller._device == "cpu"

    def test_has_all_required_attributes(self, controller):
        for attr in ("hal", "manager", "scheduler", "ram_pool", "vram_pool", "bus"):
            assert hasattr(controller, attr), f"Missing attribute: {attr}"


class TestStreamingLayerWrapper:
    def test_wrapper_instantiates(self):
        import torch.nn as nn
        layer = nn.Linear(10, 10)
        wrapper = _StreamingLayerWrapper(layer, "cpu")
        assert wrapper is not None

    def test_wrapper_forward_runs_on_cpu(self):
        import torch
        import torch.nn as nn
        layer = nn.Linear(4, 4)
        wrapper = _StreamingLayerWrapper(layer, "cpu")
        x = torch.randn(1, 4)
        out = wrapper(x)
        assert out.shape == (1, 4)
        # Layer must be back on CPU after forward
        for p in layer.parameters():
            assert p.device.type == "cpu"
