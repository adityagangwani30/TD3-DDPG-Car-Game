# Training and Evaluation

This document explains how to train, evaluate, and run experiments using the canonical shared scripts available in this repository.

---

## Pre-Flight Verification

Before running long training experiments, verify environment dynamics, replay transition integrity, and configuration validity:

```bash
python preflight_checks.py
```

All 16 sanity checks must pass.

---

## Types of Runs

| Run Type | Script | Purpose |
|----------|--------|---------|
| Pre-Flight Sanity | `preflight_checks.py` | 16-point automated verification suite |
| Single training run | `main.py` | Train one agent with default settings |
| Batch experiment run | `run_experiments.py` | Run all 12 configurations for one algorithm across 3 seeds |
| Evaluation/demo run | `main.py` / `eval_models.py` | Evaluate a trained checkpoint (deterministic, 20 eps) |
| Plot generation | `plot_metrics.py` | Generate publication-grade plots (PNG + PDF) |
| Research Report & Tables | `generate_td3_ddpg_report.py` | Aggregate metrics across seeds and export CSV tables & PDF report |

---

## Single Training Run

Train a single agent with default settings:

```bash
# Train TD3 (2,000 episodes, 600 max steps, with GUI rendering)
python main.py --algo td3 --mode train

# Train DDPG in headless mode (faster, no display)
python main.py --algo ddpg --mode train --headless

# Custom episode and step counts
python main.py --algo td3 --mode train --max-episodes 2000 --max-steps 600 --headless

# Resume training from latest checkpoint
python main.py --algo td3 --mode train --resume

# Resume from a specific checkpoint
python main.py --algo td3 --mode train --checkpoint models_v2/td3/td3_ep500.pth
```

A single training run uses the default reward mode (`shaped` / R2) and the default sensor noise from `config.py`. It trains from scratch unless `--resume` or `--checkpoint` is specified.

---

## Batch Experiment Run

Run all 12 configurations (4 reward × 3 noise) across 3 seeds for one algorithm:

```bash
# Run all TD3 experiments (writes to logs_v2/ and models_v2/ by default)
python run_experiments.py --algo td3 --headless

# Run all DDPG experiments
python run_experiments.py --algo ddpg --headless

# Run a specific seed only
python run_experiments.py --algo td3 --seed 42 --headless

# Resume interrupted experiments (skips runs where 2000 eps AND evaluation are complete)
python run_experiments.py --algo td3 --resume --headless

# Explicitly specify custom log and model directories
python run_experiments.py --algo td3 --logs-dir logs_v2 --models-dir models_v2 --headless
```

The batch runner:
- Iterates through all 12 experiments defined in `config.EXPERIMENTS`
- For each experiment, trains with seeds [0, 42, 123] (unless `--seed` overrides)
- Automatically triggers a **20-episode deterministic evaluation** (`add_noise=False`) on the best checkpoint upon completing 2,000 episodes
- Creates isolated output directories per `(algo, experiment, seed)`
- With `--resume`, skips experiments only when both 2,000 episodes and evaluation summary are complete

---

## Evaluation and Demo

Evaluate a trained model without exploration noise:

```bash
# Evaluate TD3 with best checkpoint (20 deterministic episodes)
python main.py --algo td3 --mode eval --eval-episodes 20 --render

# Evaluate with a specific checkpoint
python main.py --algo td3 --mode eval --checkpoint models_v2/td3/td3_best.pth --render

# Run interactive demo (2 episodes, always rendered)
python main.py --algo td3 --mode demo
```

Evaluation mode:
- Loads the best available checkpoint (or a specified one)
- Runs deterministic actions (`agent.select_action(state, add_noise=False)`)
- Emits `evaluation_log.jsonl` and `evaluation_summary.json`
- Reports average reward, crash rate, lap completion rate, distance traveled, and lap times

### Multi-Model Comparison

```bash
# Evaluate a specific model file
python eval_models.py --model td3_best.pth --episodes 20

# Evaluate across a models directory
python eval_models.py --models-dir models_v2 --episodes 20 --headless
```

---

## Result Aggregation and Plot Generation

### 1. Research Report and CSV Table Aggregation

```bash
# Generate comprehensive CSV tables and PDF research report for Camera-Ready V2
python generate_td3_ddpg_report.py --logs-dir logs_v2 --results-dir results_v2 --strict

# Run on legacy logs for historical analysis
python generate_td3_ddpg_report.py --logs-dir logs --results-dir results
```

Outputs generated in `results_v2/tables/`:
- `condition_results.csv`: Per-seed metrics (72 rows)
- `condition_aggregate.csv`: Aggregated across seeds with Mean, SD, SEM, and 95% CI (24 rows)
- `td3_vs_ddpg_comparisons.csv`: Direct pairwise head-to-head differences within condition (12 rows)
- `noise_degradation.csv`: Performance degradation across sensor noise levels
- `reward_tradeoff.csv`: Reward shaping aggressiveness sweep
- `training_curves.csv`: Complete episodic training trajectory records

### 2. Publication Figures

```bash
# Generate all publication-quality comparison plots (PNG + PDF)
python plot_metrics.py --logs-dir logs_v2 --results-dir results_v2 --compare-algos
```

Plots are saved under `results_v2/plots/` in high-resolution PNG (300 DPI) and vector PDF formats.

---

## Directory Hierarchy

```
logs_v2/
└── {algo}/                  # td3 or ddpg
    └── {experiment_tag}/    # e.g., R1_N1, R2_N2
        └── seed_{seed}/     # seed_0, seed_42, seed_123
            ├── metadata.json
            ├── training_log.jsonl
            ├── evaluation_log.jsonl
            └── evaluation_summary.json

models_v2/
└── {algo}/
    └── {experiment_tag}_seed_{seed}_best.pth

results_v2/
├── aggregated/              # JSON aggregates
├── per_seed/                # Individual seed JSON summaries
├── tables/                  # Formatted CSV tables
└── plots/                   # Publication figures (PNG + PDF)
```

*(Legacy data in `logs/`, `models/`, and `results/` remains preserved and read-only).*

```
models/{algo}/{experiment_tag}/seed_{seed}/{algo}_best.pth
```

Checkpoint types:
- `{algo}_best.pth` — Best single-episode reward
- `{algo}_best_avg100.pth` — Best 100-episode rolling average
- `{algo}_ep{N}.pth` — Periodic checkpoints every 100 episodes

### Results/Plots

```
results/plots/{algo}/individual/{experiment}/reward_vs_episodes.png
results/plots/comparison/{experiment}_reward_comparison.png
results/grouped/{algo}_{noise}_reward.png
```

---

## Training Behavior

### Fresh Start vs Resume

- **Fresh start** (default): Agent is initialized with random weights. Training starts from episode 1.
- **Resume** (`--resume`): The runner searches for the latest checkpoint in the model directory, loads weights, reads the log to determine the last completed episode, and continues from there. If no checkpoint is found, training starts from scratch with a warning.

### When Models Are Saved

During training:
- **Best model** is updated whenever a new highest single-episode reward is achieved
- **Best avg100 model** is updated whenever the 100-episode rolling average reaches a new high
- **Periodic checkpoints** are saved every 100 episodes

### Why Results Can Vary Between Runs

Even with the same configuration and seed, results may differ between runs due to:
- Different hardware (CPU vs GPU, different GPU models)
- Non-deterministic CUDA operations
- Python hash randomization
- Operating system threading differences

Setting `torch.backends.cudnn.deterministic = True` (done in `utils.py`) helps but does not guarantee bit-exact reproducibility.

---

## Further Reading

- [experiment_design.md](experiment_design.md) — The 12-experiment factorial grid
- [project_architecture.md](project_architecture.md) — How the code is organized
- [results_interpretation.md](results_interpretation.md) — How to read generated plots
- [reproducibility.md](reproducibility.md) — Why results vary
