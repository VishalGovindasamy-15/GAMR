"""Hardware Abstraction Layer — runtime never calls CUDA directly."""
from runtime.hal.base import HardwareBackend

__all__ = ["HardwareBackend"]
