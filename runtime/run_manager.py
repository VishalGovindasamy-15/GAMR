"""
Run directory manager.

Every docker compose up creates a new numbered folder:
    runs/Run_001/
    runs/Run_002/
    ...

Nothing is ever overwritten.
"""
from __future__ import annotations

import re
from pathlib import Path


def create_run_dir(output_dir: str | Path) -> Path:
    """
    Create the next Run_NNN directory inside output_dir.

    Thread-safe for single-process use.
    Returns the created Path.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    existing_nums = [
        int(m.group(1))
        for d in base.iterdir()
        if d.is_dir() and (m := re.match(r"^Run_(\d{3})$", d.name))
    ]

    next_num = (max(existing_nums) + 1) if existing_nums else 1
    run_dir = base / f"Run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
