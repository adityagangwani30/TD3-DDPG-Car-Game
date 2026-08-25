# Project Architecture

This document explains the overall repository structure, code organization, and data flow.

---

## High-Level Flow

```
Environment (car.py + environment.py)
        │
        ▼
  TD3/DDPG Agent (td3_agent.py / ddpg_agent.py)
        │
        ▼
  Training & Eval Loop (train.py)
        │
        ▼
  Experiment Runner (run_experiments.py)  ──►  Logs (logs_v2/)
        │                                       │
        ▼                                       ▼
  Models (models_v2/)                  Report & Plot Engines (generate_td3_ddpg_report.py & plot_metrics.py)
                                                │
                                                ▼
                                        Results (results_v2/)
                                                │
                                                ▼
                                        Research Paper / Report
```

---

## Canonical Codebase Architecture

The repository maintains a **single canonical implementation** for training, evaluation, metrics tracking, aggregation, and plotting. The same scripts operate on both legacy and camera-ready datasets via path parameters:

- **Camera-Ready V2 (Default):** `--logs-dir logs_v2 --models-dir models_v2 --results-dir results_v2`
- **Legacy Accepted Paper (Historical):** `--logs-dir logs --models-dir models --results-dir results`

### Core Scripts

| File | Role |
|------|------|
| `preflight_checks.py` | Standalone 16-point sanity suite validating state independence, physics feasibility, and matrix enumeration before long training runs. |
| `main.py` | Single-agent training, evaluation, and demo mode. Accepts `--algo {td3,ddpg}`, `--mode {train,eval,demo}`, and CLI flags. |
| `run_experiments.py` | Canonical batch runner for the 72-run factorial grid ($2 \text{ algos} \times 4 \text{ rewards} \times 3 \text{ noises} \times 3 \text{ seeds}$). Triggers 20-episode deterministic evaluation upon run completion. |
| `eval_models.py` | Multi-model evaluation and comparison. Loads checkpoints, auto-detects conditions, and executes deterministic evaluation episodes. |
| `plot_metrics.py` | Publication-quality plotting engine. Exports high-resolution (300 DPI PNG) and vector PDF figures across training dynamics, performance, and noise degradation. |
| `generate_td3_ddpg_report.py` | Master research report and table generator. Aggregates seed data into CSV tables (`condition_results.csv`, `condition_aggregate.csv`, `td3_vs_ddpg_comparisons.csv`) and compiles the comprehensive research PDF. |

### Agent Implementations

| File | Role |
|------|------|
| `td3_agent.py` | TD3 agent: Actor, twin Critic, target networks, delayed policy updates, target action smoothing. |
| `ddpg_agent.py` | DDPG agent: Actor, single Critic, target networks. Shared API with TD3. |
| `replay_buffer.py` | Experience replay buffer backed by pre-allocated NumPy arrays. |

### Environment

| File | Role |
|------|------|
| `environment.py` | Gym-style `CarRacingEnv` class. Handles step transitions, reward modes (R1–R4), off-track detection, and Pygame lifecycle. |
| `car.py` | Vehicle dynamics (bicycle model), raycast sensor casting, state array generation (with independent `.copy()` arrays), collision handling. |
| `lap_timer.py` | Finish-line crossing geometry and lap timer. |

### Infrastructure

| File | Role |
|------|------|
| `config.py` | Central configuration — hyperparameters, physics constants, 72-run grid, default paths (`logs_v2`, `models_v2`, `results_v2`). |
| `metrics_tracker.py` | Episode-level metrics logging to JSONL format. |
| `utils.py` | Helpers: global seed setting, headless display detection, Pygame initialization, track mask loading. |

---

## Directory Hierarchy

```
logs_v2/
├── td3/
│   ├── R1_N1/
│   │   ├── seed_0/
│   │   │   ├── metadata.json
│   │   │   ├── training_log.jsonl
│   │   │   ├── evaluation_log.jsonl
│   │   │   └── evaluation_summary.json
│   │   ├── seed_42/
│   │   └── seed_123/
│   └── ...
└── ddpg/
    └── ...

models_v2/
├── td3/
└── ddpg/

results_v2/
├── aggregated/
├── per_seed/
├── tables/
└── plots/
    ├── training/
    ├── performance/
    ├── td3_vs_ddpg/
    ├── noise/
    └── reward_tradeoff/
```

*(Legacy data in `logs/`, `models/`, and `results/` is preserved untouched for historical reference).*
- `{algo}_best_avg100.pth` — best 100-episode rolling average
- `{algo}_ep{N}.pth` — periodic checkpoints every 100 episodes

For experiments, the path includes the experiment tag and seed:
```
models/{algo}/{experiment_tag}/seed_{seed}/{algo}_best.pth
```

---

## How Colab Execution Fits In

The Colab notebooks (`colab_demo_*.ipynb`):

1. Clone the repository
2. Install dependencies (`pip install -r requirements.txt`)
3. Automatically detect headless mode via `detect_headless_environment()` in `utils.py`
4. Set SDL environment variables for off-screen rendering
5. Run training and experiments in headless mode
6. Generate plots from the resulting logs
7. Provide download links for results as ZIP files

The code is identical between local and Colab execution — the headless detection and dummy Pygame driver handle the differences transparently.

---

## Further Reading

- [training_and_evaluation.md](training_and_evaluation.md) — Detailed commands for training and evaluation
- [experiment_design.md](experiment_design.md) — How the 12-experiment grid is structured
- [environment_design.md](environment_design.md) — The racing environment details
