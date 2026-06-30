"""
DecisionEngine — the adaptive brain of the Memory Controller (Phase 5).

Lives inside the Memory Controller. The only component allowed to call
AdaptiveScheduler.set_prefetch_depth().

Algorithm (one parameter per cycle, change and observe):
  1. After each inference run, read the observed mean load latency and
     mean compute latency from the latency.csv (or in-memory metrics).
  2. Compute the GPU idle ratio:
       idle_ratio = mean_load_ms / (mean_load_ms + mean_compute_ms)
  3. If idle_ratio > IDLE_HIGH_THRESHOLD → increase prefetch_depth by 1
     If idle_ratio < IDLE_LOW_THRESHOLD  → decrease prefetch_depth by 1
     Otherwise                           → no change
  4. Record the decision to events.json with payload.
  5. After N_OBSERVE cycles, evaluate if gen_time improved.
     If NOT improved → rollback to previous depth.

Rollback policy:
  - Keep a window of the last ROLLBACK_WINDOW run gen_times.
  - If the moving average has not improved after a depth increase, revert.

Fitted params:
  - Loaded from configs/fitted_params.json (Phase 1 benchmarks, do not redo).
  - Used to compute the *expected* stall probability at each depth.
  - In Phase 6 this feeds into risk_model.py — for now it is advisory.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

logger = logging.getLogger("gamr.decision_engine")

# ── Thresholds ──────────────────────────────────────────────────────────────
_IDLE_HIGH   = 0.40   # GPU idle > 40% of step time → increase prefetch depth
_IDLE_LOW    = 0.10   # GPU idle < 10% → VRAM pressure, decrease depth
_ROLLBACK_WINDOW  = 3  # Number of consecutive cycles before evaluating rollback
_MIN_IMPROVEMENT  = 0.02  # 2% gen_time improvement required to keep a change


class RunMetrics:
    """Lightweight value object for one completed inference run."""
    __slots__ = ("run_id", "gen_time_s", "mean_load_ms", "mean_compute_ms",
                 "peak_vram_gb", "prefetch_depth", "ts")

    def __init__(
        self,
        run_id: str,
        gen_time_s: float,
        mean_load_ms: float,
        mean_compute_ms: float,
        peak_vram_gb: float,
        prefetch_depth: int,
    ) -> None:
        self.run_id          = run_id
        self.gen_time_s      = gen_time_s
        self.mean_load_ms    = mean_load_ms
        self.mean_compute_ms = mean_compute_ms
        self.peak_vram_gb    = peak_vram_gb
        self.prefetch_depth  = prefetch_depth
        self.ts              = time.time()

    @property
    def idle_ratio(self) -> float:
        total = self.mean_load_ms + self.mean_compute_ms
        if total == 0:
            return 0.0
        return self.mean_load_ms / total

    def as_dict(self) -> dict:
        return {
            "run_id":          self.run_id,
            "gen_time_s":      round(self.gen_time_s, 3),
            "mean_load_ms":    round(self.mean_load_ms, 3),
            "mean_compute_ms": round(self.mean_compute_ms, 3),
            "peak_vram_gb":    round(self.peak_vram_gb, 4),
            "prefetch_depth":  self.prefetch_depth,
            "idle_ratio":      round(self.idle_ratio, 4),
            "ts":              round(self.ts, 3),
        }


class DecisionEngine:
    """
    Adaptive prefetch-depth tuner.

    Owned exclusively by MemoryController — nothing else should call it.
    """

    def __init__(
        self,
        scheduler,          # AdaptiveScheduler — typed loosely to avoid circular import
        params_path: Optional[Path] = None,
        recorder=None,      # Recorder (Optional) — records DECISION events
    ) -> None:
        self._scheduler = scheduler
        self._recorder  = recorder
        self._history: deque[RunMetrics] = deque(maxlen=20)
        self._rollback_window: deque[float] = deque(maxlen=_ROLLBACK_WINDOW)
        self._prev_depth: Optional[int] = None
        self._prev_mean_gen: Optional[float] = None
        self._cycles_since_change = 0

        # Load Phase 1 fitted params (advisory, not required for POC)
        self._fitted = {}
        if params_path and params_path.exists():
            try:
                self._fitted = json.loads(params_path.read_text())
                logger.info(f"Loaded fitted params from {params_path}")
            except Exception as exc:
                logger.warning(f"Could not load fitted params: {exc!r}")

    # ── Main entry point ───────────────────────────────────────────────

    def observe_and_decide(self, metrics: RunMetrics) -> dict:
        """
        Called once per completed inference run.

        1. Record the metrics.
        2. Evaluate idle_ratio → propose depth change.
        3. Check rollback condition.
        4. Apply or reject the change.
        5. Return a decision record (written to events.json by the controller).

        Returns:
            dict: Decision record with keys: action, old_depth, new_depth,
                  idle_ratio, reason, rollback_fired.
        """
        self._history.append(metrics)
        self._rollback_window.append(metrics.gen_time_s)
        self._cycles_since_change += 1

        current_depth = self._scheduler.prefetch_depth
        decision = {
            "action":        "no_change",
            "old_depth":     current_depth,
            "new_depth":     current_depth,
            "idle_ratio":    round(metrics.idle_ratio, 4),
            "gen_time_s":    round(metrics.gen_time_s, 3),
            "rollback_fired": False,
            "reason":        "",
        }

        # ── Rollback check ─────────────────────────────────────────────
        if (
            self._prev_depth is not None
            and self._cycles_since_change >= _ROLLBACK_WINDOW
            and self._prev_mean_gen is not None
        ):
            current_mean = sum(self._rollback_window) / len(self._rollback_window)
            improvement  = (self._prev_mean_gen - current_mean) / self._prev_mean_gen
            if improvement < _MIN_IMPROVEMENT:
                # The depth change did not help — roll back
                rolled = self._scheduler.set_prefetch_depth(self._prev_depth)
                if rolled:
                    decision["action"]        = "rollback"
                    decision["new_depth"]     = self._prev_depth
                    decision["rollback_fired"]= True
                    decision["reason"] = (
                        f"Gen time did not improve by ≥{_MIN_IMPROVEMENT:.0%} "
                        f"after {_ROLLBACK_WINDOW} cycles "
                        f"(improvement={improvement:.2%}). Reverting depth."
                    )
                    logger.info(
                        f"DecisionEngine ROLLBACK: depth {current_depth} → {self._prev_depth} "
                        f"(improvement={improvement:.2%})"
                    )
                    self._record_decision(decision, metrics)
                    self._prev_depth = None
                    self._cycles_since_change = 0
                    return decision

        # ── Adaptation decision ────────────────────────────────────────
        idle = metrics.idle_ratio
        proposed_depth = current_depth

        if idle > _IDLE_HIGH:
            proposed_depth = current_depth + 1
            reason = f"GPU idle ratio {idle:.1%} > {_IDLE_HIGH:.0%} threshold — increase depth to overlap H2D"
        elif idle < _IDLE_LOW:
            proposed_depth = current_depth - 1
            reason = f"GPU idle ratio {idle:.1%} < {_IDLE_LOW:.0%} threshold — decrease depth to reduce VRAM pressure"
        else:
            reason = f"GPU idle ratio {idle:.1%} within [{_IDLE_LOW:.0%}, {_IDLE_HIGH:.0%}] — stable"

        changed = self._scheduler.set_prefetch_depth(proposed_depth)
        if changed:
            decision["action"]   = "increase" if proposed_depth > current_depth else "decrease"
            decision["new_depth"]= self._scheduler.prefetch_depth
            decision["reason"]   = reason
            self._prev_depth     = current_depth
            self._prev_mean_gen  = sum(self._rollback_window) / len(self._rollback_window)
            self._cycles_since_change = 0
            logger.info(
                f"DecisionEngine: {decision['action']} depth "
                f"{current_depth} → {decision['new_depth']} | {reason}"
            )
        else:
            decision["reason"] = reason

        self._record_decision(decision, metrics)
        return decision

    # ── History ───────────────────────────────────────────────────────

    def history(self) -> list[dict]:
        return [m.as_dict() for m in self._history]

    def last_n_gen_times(self, n: int) -> list[float]:
        return [m.gen_time_s for m in list(self._history)[-n:]]

    # ── Internal ──────────────────────────────────────────────────────

    def _record_decision(self, decision: dict, metrics: RunMetrics) -> None:
        if self._recorder is None:
            return
        self._recorder.record(
            "DECISION_ENGINE",
            payload={**decision, **metrics.as_dict()},
        )
