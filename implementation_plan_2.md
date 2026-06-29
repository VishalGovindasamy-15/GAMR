# GAMR — Galaxy Adaptive Memory Runtime
## Implementation Plan · FINAL · FROZEN

> **Architecture is frozen. Every new idea goes to `future_ideas.md`. Build in order.**

---

## What We Are Building

**GAMR** is an **Adaptive Hierarchical AI Memory Manager (AHAMM)**.

It treats the entire machine (SSD → RAM → VRAM → GPU) as one intelligent memory hierarchy for running AI models. Instead of loading everything into VRAM, it streams model layers block-by-block, observes hardware through an independent monitor, routes all observations through an event bus, and lets the **Memory Controller** — the single authority — decide where every object lives.

**First model:** `TinyLlama-1.1B`  
**Block granularity:** One layer = one MemoryObject (no grouping)  
**Target GPU:** NVIDIA RTX 3050  
**Validation:** Token-by-token comparison against Hugging Face reference  
**Prior work:** Fitted lognormal SSD model + linear PCIe model from Phase 1 benchmarks → reused directly in Phase 6 (risk-aware scheduling).

---

## What Is NOT In The POC

Deferred. Do not implement until streaming runtime is working and validated.

| Deferred Item | Lives In |
|---|---|
| Training / LoRA / Gradients / Optimizer States | Phase 7+ |
| `risk_model.py` | Phase 6 |
| Dynamic block size | Phase 6 |
| Dynamic memory budgets | Phase 6 |
| Live Dashboard | Phase 6 |
| `RAM_COLD`, `RAM_HOT`, `VRAM_EVICTION` states | Add only when needed |
| KV Cache management | Phase 6 |
| Activation pool | Phase 7+ |

---

## Final Architecture

```
Docker
    │
    ▼
Runtime  (thin orchestrator — load config, init, run, exit)
    │
    ▼
Memory Manager
    │
    ├── Memory Controller        ← single decision authority
    │       └── Decision Engine  ← inside Memory Controller, not separate
    │
    ├── Memory Pool Manager      ← RAM pool, VRAM pool, pinned memory
    │
    ├── Memory Object Manager    ← state machine per MemoryObject
    │
    └── plugins/
            ├── scheduler/       ← FIFO → StaticPrefetch → Adaptive
            ├── pool/            ← FixedPool → DynamicPool
            └── monitor/         ← collectors (GPU, RAM, SSD, PCIe)

Monitor  (independent — never calls Memory Controller directly)
    │
    ▼
Event Bus  (async pub/sub)
    │
    ▼
Memory Controller

HAL (Hardware Abstraction Layer)
    ├── CUDABackend  (RTX 3050)
    └── CPUBackend   (fallback)
```

### Data flow — inference

```
Runtime → Memory Controller → Scheduler → Memory Object Manager → HAL → GPU
```

### Monitoring flow

```
Monitor → Event Bus → Memory Controller
```

Monitor **never** calls Memory Controller directly. Everything is event-driven.

---

## Memory Controller — The Brain

The Memory Controller is the **only** component that changes memory state.

**It owns:**

| Responsibility | Detail |
|---|---|
| Current State | Knows where every MemoryObject lives right now |
| Memory Budget | Tracks RAM pool usage, VRAM pool usage |
| Current Policy | Which scheduler plugin is active |
| Adaptation | Changes one parameter per cycle (Phase 6) |
| Rollback | Reverts if metric does not improve (Phase 6) |
| Decision History | Log of every parameter change and outcome |

**In Phase 1–5:** Memory Controller is a thin coordinator — it routes events and calls the FIFO scheduler.  
**In Phase 6:** Memory Controller gains adaptive logic. No other component changes.

```
Monitor
    │  (via Event Bus)
    ▼
Memory Controller
    │
    ├── Scheduler Plugin
    ├── Memory Pool Manager
    └── HAL
```

---

## MemoryObject — Everything Is A MemoryObject

Not `WeightBlock`. Not `LayerBlock`. **MemoryObject**.

### Base class

```python
@dataclass
class MemoryObject:
    id:              str          # unique identifier
    type:            ObjectType   # WEIGHT | KV_CACHE | ACTIVATION | GRADIENT | OPTIMIZER
    location:        Location     # SSD | RAM | VRAM | GPU
    state:           ObjectState  # SSD_COLD | RAM_READY | VRAM_READY | GPU_ACTIVE | RELEASED
    size_bytes:      int
    priority:        int          # 0 = lowest, higher = keep longer
    timestamp:       float        # last state transition (epoch seconds)
    reference_count: int          # how many consumers hold a reference
```

> `priority` and `reference_count` are defined now even if not used until Phase 6. No restructuring later.

### First concrete subclass (POC only)

```python
@dataclass
class WeightObject(MemoryObject):
    layer_index: int
    tensor:      Optional[torch.Tensor] = None
```

Later additions (not in POC): `KVCacheObject`, `ActivationObject`, `GradientObject`, `OptimizerStateObject` — all inherit `MemoryObject`.

---

## Memory Object State Machine (POC States)

```
SSD_COLD
    │
    ▼
RAM_READY
    │
    ▼
VRAM_READY
    │
    ▼
GPU_ACTIVE
    │
    ▼
RELEASED
```

States `RAM_COLD`, `RAM_HOT`, `VRAM_EVICTION_QUEUE` are reserved in the enum but not transitioned in the POC.

---

## Event Flow

```
SSD_DONE → RAM_READY → VRAM_COPY → GPU_START → GPU_DONE → PREFETCH_NEXT
```

No polling loops. All async `asyncio` pub/sub through the Event Bus.

---

## Hardware Abstraction Layer (HAL)

Runtime never calls CUDA directly.

```python
class HardwareBackend(ABC):
    def to_device(self, tensor: Tensor, device: str) -> Tensor: ...
    def free_vram_bytes(self) -> int: ...
    def free_ram_bytes(self) -> int: ...
    def transfer_async(self, tensor: Tensor, dst: str) -> Awaitable: ...
    def device_name(self) -> str: ...
```

Implementations: `CUDABackend` (RTX 3050), `CPUBackend` (fallback, uses RAM as VRAM substitute for testing).

---

## Project Structure

```
gamr/
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── future_ideas.md             ← every deferred idea lives here, not in source
│
├── runtime/
│   ├── __init__.py
│   ├── runtime.py              ← thin orchestrator
│   ├── event_bus.py            ← async pub/sub
│   ├── hardware_scan.py        ← detect GPU, VRAM, RAM, SSD
│   │
│   ├── hal/
│   │   ├── base.py
│   │   ├── cuda_backend.py
│   │   └── cpu_backend.py
│   │
│   ├── memory/
│   │   ├── memory_object.py    ← MemoryObject base + WeightObject
│   │   ├── object_manager.py   ← state machine transitions
│   │   └── pool_manager.py     ← RAM pool + VRAM pool (fixed budgets in POC)
│   │
│   ├── controllers/            ← reserved folder
│   │   ├── __init__.py
│   │   └── memory_controller.py ← single decision authority + decision engine
│   │
│   ├── model_loader.py         ← load safetensors, emit WeightObjects
│   │
│   └── plugins/
│       ├── __init__.py
│       ├── scheduler/
│       │   ├── base.py         ← SchedulerPlugin interface
│       │   └── fifo.py         ← FIFOScheduler (Phase 3)
│       ├── pool/
│       │   ├── base.py         ← PoolPlugin interface
│       │   └── fixed.py        ← FixedRAMPool + FixedVRAMPool
│       └── monitor/
│           ├── base.py         ← MonitorPlugin interface
│           └── system.py       ← GPU, RAM, SSD, PCIe collectors
│
├── validation/
│   ├── __init__.py
│   ├── engine.py               ← HF reference run + GAMR run + compare
│   └── report.py               ← PASS / FAIL + token diff
│
├── recorder/
│   ├── __init__.py
│   └── recorder.py             ← black-box flight recorder
│
├── configs/
│   ├── runtime.yaml
│   ├── scheduler.yaml
│   └── model.yaml
│
├── runs/                       ← every docker run saves here
│   └── Run_001/
│       ├── config.json
│       ├── hardware.json
│       ├── runtime.log
│       ├── scheduler.log
│       ├── monitor.log
│       ├── metrics.json
│       ├── events.json
│       ├── gpu.csv
│       ├── ram.csv
│       ├── vram.csv
│       ├── latency.csv
│       └── summary.md
│
└── tests/
    ├── test_memory_object.py
    ├── test_object_manager.py
    ├── test_pool_manager.py
    ├── test_memory_controller.py
    ├── test_scheduler.py
    ├── test_event_bus.py
    └── test_validation.py
```

---

## Configuration — No Constants In Source

Everything lives in `configs/runtime.yaml`. Runtime source contains zero magic numbers.

```yaml
runtime:
  model_path: /app/models/TinyLlama-1.1B
  block_granularity: layer       # one layer = one MemoryObject
  pool:
    ram_budget: 25%              # of free RAM at startup
    vram_budget: 80%             # of free VRAM
  scheduler: fifo
  prefetch_depth: 1
  validate: true
  output_dir: /app/runs
  log_level: INFO
```

---

## Docker Strategy

**Docker automates execution, monitoring, logging, packaging — not decisions.**

```
docker compose up
    ↓ Detect Hardware
    ↓ Read Config
    ↓ Initialize Memory Controller
    ↓ Load TinyLlama → WeightObjects
    ↓ Stream Layers via FIFO Scheduler
    ↓ Generate Response
    ↓ Validate vs. HF Reference → PASS / FAIL
    ↓ Save Run_NNN/  (logs, metrics, events, summary)
    ↓ Exit
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Model loading | `transformers` + `safetensors` |
| GPU compute | `torch` (CUDA 12+) |
| Async runtime | `asyncio` |
| System metrics | `psutil` |
| GPU metrics | `pynvml` |
| Config + validation | `PyYAML` + `pydantic-settings` |
| Serialization | `orjson` |
| Container | Docker + `docker-compose` |
| Testing | `pytest` + `pytest-asyncio` |

---

## 7-Phase Roadmap

### Phase 0 — Infrastructure
**No AI yet. Just the skeleton.**

- [ ] `docker-compose.yml` + `Dockerfile`
- [ ] `configs/runtime.yaml` (pydantic-validated config model)
- [ ] Structured logging (`runtime.log`, `scheduler.log`, `monitor.log`)
- [ ] `hardware_scan.py` — GPU name, VRAM total, RAM total, SSD path
- [ ] `hal/cuda_backend.py` + `hal/cpu_backend.py`
- [ ] Run output directory: `runs/Run_NNN/` with auto-increment
- [ ] `future_ideas.md` created
- [ ] **Milestone:** `docker compose up` → detects hardware → writes `hardware.json` → exits cleanly

---

### Phase 1 — MemoryObject + Event Bus
**Define the data model and communication backbone.**

- [ ] `memory/memory_object.py` — `MemoryObject` (all 8 fields) + `WeightObject`
- [ ] `ObjectState` enum (all states defined, POC subset active)
- [ ] `ObjectType` enum
- [ ] `event_bus.py` — async pub/sub, typed events
- [ ] **Milestone:** Unit tests pass for `MemoryObject` creation, state transitions, event publishing/subscribing

---

### Phase 2 — TinyLlama + Validation
**Stream one layer at a time. Prove correctness.**

- [ ] `model_loader.py` — load TinyLlama safetensors, emit `WeightObject` per layer
- [ ] `memory/object_manager.py` — state machine transitions per MemoryObject
- [ ] `memory/pool_manager.py` — `FixedRAMPool` + `FixedVRAMPool` (25% free RAM, 80% VRAM)
- [ ] `controllers/memory_controller.py` — thin coordinator (Phase 2: no adaptation yet)
- [ ] `plugins/scheduler/fifo.py` — `FIFOScheduler`
- [ ] `runtime.py` — thin orchestrator
- [ ] `validation/engine.py` + `validation/report.py`
- [ ] **Milestone:** `docker compose up` → streams all layers → generates response → **PASS** validation → peak VRAM < full-model baseline

---

### Phase 3 — Pipeline (Prefetch)
**Overlap loading and computation.**

- [ ] `plugins/scheduler/static_prefetch.py` — `StaticPrefetchScheduler`
- [ ] `PREFETCHING` state activated in object manager
- [ ] Prefetch queue in pool manager
- [ ] **Milestone:** Time-to-first-token measurably lower than FIFO. Validation still **PASS**.

---

### Phase 4 — Monitor + Recorder
**Collect everything. Make no decisions. Build the black box.**

- [ ] `plugins/monitor/system.py` — GPU util %, VRAM, RAM, SSD read latency per object, PCIe bandwidth
- [ ] `recorder/recorder.py` — timestamped event log written to `events.json`, `gpu.csv`, `ram.csv`, `vram.csv`, `latency.csv`
- [ ] `summary.md` generator
- [ ] Monitor → Event Bus → Memory Controller flow verified (no direct calls)
- [ ] **Milestone:** Every run produces complete `Run_NNN/` folder. Events are replay-able from `events.json`.

---

### Phase 5 — Memory Controller (Adaptive Mode)
**Memory Controller gains the ability to change one parameter per cycle.**

- [ ] Load fitted lognormal SSD parameters from Phase 1 benchmarks (reuse files, do not redo)
- [ ] Decision Engine inside `memory_controller.py` — reads monitor events, selects parameter to adjust
- [ ] Rollback logic — reverts if metric does not improve after N cycles
- [ ] `plugins/scheduler/adaptive.py` — `AdaptiveScheduler`
- [ ] Decision history written to `events.json`
- [ ] **Milestone:** Adaptive controller adjusts prefetch depth across 10 runs. Rollback fires correctly. No correctness regression.

---

### Phase 6+ — Future *(not POC)*

- Dynamic RAM/VRAM budgets
- `risk_model.py` (lognormal P(stall))
- Risk-aware scheduler
- Live dashboard
- KV Cache management
- Training / LoRA / Gradients / Optimizer States
- Research platform (log reader, run comparator)

---

## Week-by-Week Build Order

| Week | Phase | Deliverable |
|---|---|---|
| **1** | 0 | Docker, HAL, config, logging, hardware scan |
| **2** | 1 | MemoryObject, ObjectState, Event Bus, unit tests |
| **3** | 2 | TinyLlama streaming, FIFO, Validation **PASS** |
| **4** | 3 | Static prefetch, pipeline overlap, latency measurement |
| **5** | 4 | Monitor, recorder, full `Run_NNN/` folder |
| **6** | 5 | Memory Controller adaptive mode, rollback logic |
| **7+** | 6 | Risk model, dynamic budgets, dashboard, training |

---

## The One Rule

> **If a feature is not required to make TinyLlama stream correctly and produce identical outputs, it does not belong in the POC. Write it in `future_ideas.md` instead.**

---

## Verification Plan

| Phase | Test | Pass Criteria |
|---|---|---|
| 0 | `docker compose up` cold start | Exits cleanly, `hardware.json` written |
| 1 | Unit tests | All state/event tests green |
| 2 | Validation engine | Token output matches HF reference exactly; peak VRAM < full-model |
| 3 | Latency benchmark | Time-to-first-token lower than Phase 2 baseline |
| 4 | Replay test | `events.json` can reconstruct full run timeline |
| 5 | Adaptation test | Controller adjusts prefetch; rollback fires; correctness maintained |
