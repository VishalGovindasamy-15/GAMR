# GAMR — Galaxy Adaptive Memory Runtime
## Implementation Plan

**Full project name:** GAMR / HAMR — *Galaxy (or Hierarchical) Adaptive Memory Runtime*  
**Core concept:** An **Adaptive Hierarchical AI Memory Manager (AHAMM)** that treats the whole computer (SSD → RAM → VRAM → GPU) as one intelligent, continuously adapting memory hierarchy for running AI models.

---

## What I Understand

The idea has three layers:

| Layer | What it means |
|---|---|
| **Core insight** | AI runtimes today assume all model weights fit in VRAM. GAMR breaks that assumption by streaming blocks through SSD → RAM → VRAM → GPU on demand, adapting in real-time. |
| **Distinctive claim** | It's not just streaming (llama.cpp does that). The distinctive part is the *adaptive controller* that continuously observes hardware conditions, respects dynamic memory budgets, and decides where every kind of AI data lives (weights, KV cache, activations, gradients, optimizer states). |
| **Research context** | Your experiments already showed: SSD latency is stochastic (lognormal), PCIe is comparatively stable, GPU compute is fast relative to storage. So the runtime's adaptive logic should focus on SSD→RAM scheduling. |

The **immediate goal** is a **Proof of Concept (PoC)**: stream a model's layers, produce correct outputs, reduce VRAM requirement. Adaptation comes after.

---

## Architecture Overview

```
AI Model
    │
    ▼
Adaptive Hierarchical Memory Manager (AHAMM)
    │
    ├── Runtime Controller
    │       ├── System Monitor      ← observes hardware
    │       └── Decision Engine     ← changes one parameter at a time
    │
    ├── Memory State Manager
    │       ├── Block Manager       ← every block has a state machine
    │       ├── Cache Manager       ← RAM cache + VRAM cache with dynamic budgets
    │       └── Scheduler           ← FIFO → Static Prefetch → Adaptive → Risk-Aware
    │
    └── Model Loader
            └── Layer Streamer      ← SSD → RAM → VRAM → GPU pipeline
```

**Every block has exactly one state at any time:**

```
SSD → RAM_COLD → RAM_HOT → VRAM_READY → GPU_ACTIVE → Released
```

**Everything is event-driven** — no polling loops.

---

## Three Independent Systems (Your Own Recommendation)

| System | Responsibility | Never Does |
|---|---|---|
| **System 1 — Runtime** | Executes AI models using the AHAMM | Analyze, change itself |
| **System 2 — Monitor** | Collects every metric, writes black-box logs | Make decisions |
| **System 3 — Research Platform** | Reads logs, compares runs, generates reports | Touch the runtime |

---

## Project Structure

```
gamr/
│
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
│
├── runtime/                       ← System 1
│   ├── __init__.py
│   ├── runtime.py                 ← main entrypoint
│   ├── block_manager.py           ← state machine per block
│   ├── cache_manager.py           ← RAM + VRAM budgets
│   ├── scheduler.py               ← FIFO → Adaptive
│   ├── model_loader.py            ← splits model into blocks
│   ├── event_bus.py               ← event-driven core
│   └── hardware_scan.py           ← detect GPU, RAM, SSD
│
├── monitor/                       ← System 2
│   ├── __init__.py
│   ├── monitor.py                 ← main monitor loop
│   ├── collectors/
│   │   ├── gpu_collector.py       ← GPU util, VRAM, compute latency
│   │   ├── ram_collector.py       ← free RAM, pinned memory
│   │   ├── ssd_collector.py       ← read latency, bandwidth, queue depth
│   │   └── pcie_collector.py     ← transfer latency, bandwidth
│   └── recorder.py                ← black-box flight recorder
│
├── research/                      ← System 3 (Phase D/E only)
│   ├── __init__.py
│   ├── log_reader.py
│   ├── run_comparator.py
│   └── report_generator.py
│
├── dashboard/                     ← Web UI (live monitoring)
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── configs/
│   ├── runtime.yaml
│   ├── scheduler.yaml
│   └── model.yaml
│
├── runs/                          ← every run saves here
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
    ├── test_block_manager.py
    ├── test_scheduler.py
    ├── test_cache_manager.py
    └── test_event_bus.py
```

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem, PyTorch integration |
| **Model loading** | `transformers` + `safetensors` | Standard, layer-level access |
| **GPU ops** | `torch` (CUDA) | Standard |
| **Async runtime** | `asyncio` | Event-driven, no polling |
| **System metrics** | `psutil` | RAM, CPU |
| **GPU metrics** | `pynvml` or `nvidia-ml-py` | VRAM, GPU util, temp |
| **SSD metrics** | `psutil` + `iostat` | Read latency, bandwidth |
| **Serialization** | `orjson` | Fast JSON for event logs |
| **Config** | `PyYAML` + `pydantic` | Validated config |
| **Dashboard** | HTML + vanilla JS + SSE | Real-time, no framework needed |
| **Container** | Docker + `docker-compose` | Reproducible runs |
| **Testing** | `pytest` | Unit tests per component |

---

## Development Phases

### Phase A — Core Streaming PoC ✦ Start Here

**Goal:** Stream a model's layers from SSD → RAM → VRAM → GPU, produce correct output, without loading the full model into VRAM.

**Model:** `TinyLlama-1.1B` (small, fast, no VRAM pressure, easy to verify correctness)

**What gets built:**

| File | What it does |
|---|---|
| `hardware_scan.py` | Detect available GPU, total/free RAM, VRAM, SSD paths |
| `model_loader.py` | Load `safetensors` file, split into N blocks of layers |
| `block_manager.py` | State machine: `SSD → RAM_COLD → RAM_HOT → VRAM_READY → GPU_ACTIVE → Released` |
| `cache_manager.py` | Maintain RAM cache (fixed size), VRAM cache (fixed size), evict LRU |
| `scheduler.py` | FIFO: enqueue next block, load it, run it, unload previous |
| `event_bus.py` | Simple async publish/subscribe — `SSD_DONE`, `RAM_READY`, `VRAM_COPY`, `GPU_START`, `GPU_DONE` |
| `runtime.py` | Orchestrate all of the above, run a single inference |

**Verification:** Run TinyLlama through GAMR and directly through `transformers`. Compare token outputs — must be identical.

---

### Phase B — Pipeline Overlap

**Goal:** While the GPU is computing block N, prefetch block N+1 from SSD → RAM → VRAM. Overlap loading and computation.

**What gets added:**

| File | Change |
|---|---|
| `scheduler.py` | Add `StaticPrefetchScheduler` — always prefetch next K blocks |
| `cache_manager.py` | Add prefetch queue logic |
| `block_manager.py` | Add `PREFETCHING` state |

**Verification:** Measure time-to-first-token with FIFO vs. Static Prefetch. Expect improvement.

---

### Phase C — Real-Time Monitoring

**Goal:** Every hardware metric gets collected in a background loop and written to the black-box recorder. The live dashboard shows them.

**What gets built:**

| File | What it does |
|---|---|
| `collectors/gpu_collector.py` | GPU util %, VRAM used/free, compute latency, temperature |
| `collectors/ram_collector.py` | Free RAM, cache usage, pinned memory |
| `collectors/ssd_collector.py` | Read latency (per-block measured), bandwidth, queue depth |
| `collectors/pcie_collector.py` | Transfer latency, PCIe bandwidth |
| `monitor.py` | Aggregates all collectors, emits timestamped metrics |
| `recorder.py` | Writes all events + metrics to `events.json`, `gpu.csv`, `ram.csv`, etc. |
| `dashboard/` | HTML page with live SSE charts — GPU util, RAM, VRAM, SSD queue, prefetch depth |

**Every run produces:**
```
Run_NNN/
├── config.json       ← exact config used
├── hardware.json     ← detected hardware
├── runtime.log
├── scheduler.log
├── monitor.log
├── metrics.json      ← summary numbers
├── events.json       ← full event timeline (replay-able)
├── gpu.csv
├── ram.csv
├── vram.csv
├── latency.csv
└── summary.md        ← human-readable report
```

---

### Phase D — Adaptive Controller

**Goal:** The runtime observes metrics and changes one parameter at a time. If it improved → keep. Otherwise → rollback.

**Parameters the controller may change:**

- Block size
- RAM cache size
- VRAM cache size
- Prefetch depth
- Eviction policy
- Pinned memory pool size

**Rule:** Change only **one** parameter per adaptation cycle.

**What gets built:**

| File | What it does |
|---|---|
| `decision_engine.py` | Reads monitor state, picks one parameter to adjust |
| `risk_model.py` | Models SSD stall probability using fitted lognormal (from your earlier experiments) |
| `scheduler.py` | Add `AdaptiveScheduler` and `RiskAwareScheduler` |
| `cache_manager.py` | Dynamic budgets — shrink when system RAM drops, grow when free |

**Dynamic budget example:**
```
Total RAM = 16 GB → Available = 9 GB → Safe Budget = 6 GB → GAMR Cache = 6 GB
# Another process starts → Free RAM = 3 GB → Shrink Cache → Continue Running
```

---

### Phase E — Training & Fine-Tuning Support

**Goal:** Extend the memory manager to handle not just weights, but also gradients, optimizer states, and activations during training and LoRA fine-tuning.

> [!NOTE]
> This is a research hypothesis — whether it improves training performance or scalability needs experimental validation.

**What gets extended:**

- `block_manager.py` — new block types: `GradientBlock`, `OptimizerStateBlock`, `ActivationBlock`
- `cache_manager.py` — separate budget pools per block type
- `scheduler.py` — handle backward pass ordering
- `research/` — experiment engine, parameter sweeps, run comparisons

---

## Docker Strategy

Docker's responsibility is narrow: **reproducible execution environment**, not a decision-maker.

```yaml
# docker-compose.yml
services:
  runtime:
    build: .
    volumes:
      - ./runs:/app/runs
      - ./configs:/app/configs
      - ./models:/app/models   ← mount model files here
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  dashboard:
    build: ./dashboard
    ports:
      - "8080:8080"
    depends_on:
      - runtime
```

**Every `docker compose up` will:**
1. Detect hardware
2. Read config
3. Load model
4. Run runtime
5. Collect logs
6. Save metrics + timeline
7. Exit cleanly

Docker does **not** modify algorithms. That is the user's job.

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Model for PoC:** I plan to use `TinyLlama-1.1B` because it's small enough to verify correctness easily. Do you want to use a different model?

> [!IMPORTANT]
> **Q2 — Block granularity:** Should blocks be individual transformer layers (one layer = one block), or groups of layers (e.g., 4 layers per block)? Individual layers give maximum flexibility; groups give lower overhead. We can start with individual layers.

> [!IMPORTANT]
> **Q3 — Dashboard:** Do you want the live web dashboard in Phase A/B, or only from Phase C onward? It's easier to build it alongside the monitor.

> [!IMPORTANT]
> **Q4 — Hardware:** Do you have an NVIDIA GPU available in this environment? The PoC can run CPU-only (using RAM as VRAM substitute) if needed for initial testing.

> [!WARNING]
> **Q5 — Existing benchmarking work:** From conversation `ef22269e`, you already have stochastic memory benchmarks (lognormal SSD model, fitted PCIe linear model). Should I incorporate those fitted parameters directly into Phase D's risk model, or rebuild from scratch?

---

## Verification Plan

### Phase A
- Run TinyLlama via GAMR and via raw `transformers`
- Assert token-by-token output is identical
- Assert peak VRAM usage is lower than full-model load

### Phase B
- Measure time-to-first-token: FIFO vs. StaticPrefetch
- Assert no correctness regression

### Phase C
- Verify all CSV/JSON files are written after each run
- Visually confirm dashboard shows live updates

### Phase D
- Run 10 inference sessions; verify controller adjusts prefetch depth
- Verify rollback fires when GPU idle does not improve

### Phase E
- Fine-tune TinyLlama on a small dataset via GAMR
- Assert loss decreases correctly vs. standard training

---

## First Milestone (Your Suggestion, Adopted)

```
docker compose up
    ↓ Detect Hardware
    ↓ Run TinyLlama
    ↓ Stream Layers
    ↓ Generate Response
    ↓ Save runtime.log, metrics.json, events.json, summary.md
    ↓ Stop
```

**This is the definition of "Phase A Done."**

