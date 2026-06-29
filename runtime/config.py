"""
GAMR runtime configuration — validated with pydantic.

All tunables live in configs/runtime.yaml.
No constants in source code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class PoolConfig(BaseModel):
    ram_budget: str = "25%"
    vram_budget: str = "80%"

    @field_validator("ram_budget", "vram_budget")
    @classmethod
    def must_be_percent(cls, v: str) -> str:
        if not v.endswith("%"):
            raise ValueError(f"Budget must be a percentage string like '25%', got: {v!r}")
        pct = float(v[:-1])
        if not 0 < pct <= 100:
            raise ValueError(f"Budget percentage must be between 0 and 100, got: {pct}")
        return v

    def ram_fraction(self) -> float:
        return float(self.ram_budget[:-1]) / 100.0

    def vram_fraction(self) -> float:
        return float(self.vram_budget[:-1]) / 100.0


class RuntimeConfig(BaseModel):
    model_path: str = "/app/models/TinyLlama-1.1B"
    block_granularity: Literal["layer"] = "layer"
    pool: PoolConfig = PoolConfig()
    scheduler: Literal["fifo", "static_prefetch", "adaptive"] = "fifo"
    prefetch_depth: int = 1
    validate_output: bool = True
    output_dir: str = "/app/runs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("prefetch_depth")
    @classmethod
    def prefetch_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"prefetch_depth must be >= 1, got {v}")
        return v


def load_config(config_path: str | Path = "configs/runtime.yaml") -> RuntimeConfig:
    """Load and validate runtime config from YAML. Returns defaults if file missing."""
    path = Path(config_path)
    if not path.exists():
        return RuntimeConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    runtime_data = data.get("runtime", {})
    return RuntimeConfig(**runtime_data)
