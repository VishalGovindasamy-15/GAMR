"""Tests for hardware scanner."""
from dataclasses import asdict
import pytest

from runtime.hardware_scan import HardwareInfo, scan_hardware, save_hardware_json
import orjson


class TestHardwareInfo:
    def test_gb_fields_computed(self):
        info = HardwareInfo(
            gpu_name="Test GPU",
            cuda_available=False,
            cuda_version=None,
            vram_total_bytes=4 * 1024 ** 3,
            vram_free_bytes=2 * 1024 ** 3,
            ram_total_bytes=16 * 1024 ** 3,
            ram_free_bytes=8 * 1024 ** 3,
            ssd_path="/",
            ssd_total_bytes=512 * 1024 ** 3,
            ssd_free_bytes=256 * 1024 ** 3,
            python_version="3.11.0",
            platform="Linux",
            cpu_count=8,
        )
        assert info.vram_total_gb == 4.0
        assert info.ram_total_gb == 16.0


class TestScanHardware:
    def test_returns_hardware_info(self):
        info = scan_hardware(ssd_path="/")
        assert isinstance(info, HardwareInfo)
        assert info.ram_total_bytes > 0
        assert info.cpu_count > 0
        assert info.python_version != ""

    def test_cuda_fields_consistent(self):
        info = scan_hardware()
        if not info.cuda_available:
            assert info.gpu_name is None
            assert info.vram_total_bytes == 0
        else:
            assert info.gpu_name is not None
            assert info.vram_total_bytes > 0


class TestSaveHardwareJson:
    def test_writes_valid_json(self, tmp_path):
        info = scan_hardware(ssd_path="/")
        save_hardware_json(info, tmp_path)
        hardware_file = tmp_path / "hardware.json"
        assert hardware_file.exists()
        parsed = orjson.loads(hardware_file.read_bytes())
        assert "ram_total_bytes" in parsed
        assert "cuda_available" in parsed
        assert "platform" in parsed
