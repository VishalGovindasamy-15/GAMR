"""Tests for runtime config loading and validation."""
from pathlib import Path
import pytest
import tempfile
import yaml

from runtime.config import load_config, RuntimeConfig, PoolConfig


class TestPoolConfig:
    def test_valid_percent(self):
        p = PoolConfig(ram_budget="25%", vram_budget="80%")
        assert p.ram_fraction() == 0.25
        assert p.vram_fraction() == 0.80

    def test_invalid_format_raises(self):
        with pytest.raises(Exception):
            PoolConfig(ram_budget="25")

    def test_zero_percent_raises(self):
        with pytest.raises(Exception):
            PoolConfig(ram_budget="0%")


class TestRuntimeConfig:
    def test_defaults(self):
        cfg = RuntimeConfig()
        assert cfg.scheduler == "fifo"
        assert cfg.prefetch_depth == 1
        assert cfg.validate_output is True
        assert cfg.log_level == "INFO"

    def test_prefetch_zero_raises(self):
        with pytest.raises(Exception):
            RuntimeConfig(prefetch_depth=0)

    def test_invalid_scheduler_raises(self):
        with pytest.raises(Exception):
            RuntimeConfig(scheduler="unknown")


class TestLoadConfig:
    def test_missing_file_returns_defaults(self):
        cfg = load_config("nonexistent_path.yaml")
        assert isinstance(cfg, RuntimeConfig)

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "runtime.yaml"
        data = {
            "runtime": {
                "scheduler": "fifo",
                "prefetch_depth": 2,
                "log_level": "DEBUG",
                "pool": {"ram_budget": "30%", "vram_budget": "70%"},
            }
        }
        config_file.write_text(yaml.dump(data))
        cfg = load_config(config_file)
        assert cfg.scheduler == "fifo"
        assert cfg.prefetch_depth == 2
        assert cfg.log_level == "DEBUG"
        assert cfg.pool.ram_fraction() == 0.30
