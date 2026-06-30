"""
ValidationReport — writes the validation result to summary.md and metrics.json.
"""
from __future__ import annotations

from pathlib import Path

import orjson

from validation.engine import ValidationResult


def write_report(result: ValidationResult, run_dir: Path) -> None:
    """Write summary.md and update metrics.json with validation outcome."""

    # ── summary.md ────────────────────────────────────────────────────
    status = "PASS ✅" if result.passed else "FAIL ❌"
    md = f"""# GAMR Run Report

## Validation: {status}

| Field | Value |
|---|---|
| Prompt | `{result.prompt}` |
| Reference time | {result.reference_time_s:.2f}s |
| GAMR time | {result.gamr_time_s:.2f}s |
| Tokens matched | {"Yes" if result.passed else f"No — first mismatch at index {result.mismatch_index}"} |

### Reference output
```
{result.reference_text}
```

### GAMR output
```
{result.gamr_text}
```
"""
    (run_dir / "summary.md").write_text(md, encoding="utf-8")

    # ── metrics.json ──────────────────────────────────────────────────
    metrics = {
        "validation_passed": result.passed,
        "reference_time_s": result.reference_time_s,
        "gamr_time_s": result.gamr_time_s,
        "mismatch_index": result.mismatch_index,
        "prompt": result.prompt,
    }
    (run_dir / "metrics.json").write_bytes(
        orjson.dumps(metrics, option=orjson.OPT_INDENT_2)
    )
