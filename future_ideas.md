# Future Ideas

> **Rule:** If a feature is not required to make TinyLlama stream correctly
> and produce identical outputs, it goes here — not in source code.

---

## Deferred from POC

- **Phase E / Training support** — LoRA, gradient management, optimizer states,
  activation pools. Not until inference is validated.
- **Risk model (`risk_model.py`)** — lognormal P(stall) from fitted SSD
  parameters. Phase 6.
- **Dynamic block size** — benchmark grouped layers vs. single layers.
  Phase 6+.
- **Dynamic memory budgets** — shrink/grow RAM and VRAM pools under system
  pressure. Phase 6.
- **Live web dashboard** — real-time charts for GPU util, RAM, VRAM, SSD queue.
  Phase 6.
- **KV Cache management** — stream conversation memory through the hierarchy.
  Phase 6.
- **ROCm backend** — AMD GPU support via `hal/rocm_backend.py`. After POC.
- **TPU backend** — future hardware target.
- **Research platform** — log reader, run comparator, parameter sweep engine,
  automated report generator. Phase 7+.
- **`RAM_COLD` / `RAM_HOT` / `VRAM_EVICTION`** — additional block states.
  Add only when needed.
- **Experiment engine** — automatically launch many Docker runs with different
  configs and compare results. Phase 7+.

---

## Notes

Add new ideas here as they come up during development.
