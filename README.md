# 🏎️ TD3 & DDPG Self-Driving Car — Reinforcement Learning Simulation

> A deep reinforcement learning project that trains autonomous driving agents using **Twin Delayed DDPG (TD3)** and **Deep Deterministic Policy Gradient (DDPG)** in a custom-built 2D car racing environment — with structured experiments comparing reward shaping strategies and sensor noise robustness.

[![Open TD3 Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-DDPG-Car-Game/blob/main/colab_demo_td3.ipynb)
[![Open DDPG Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-DDPG-Car-Game/blob/main/colab_demo_ddpg.ipynb)

---

## 📌 What This Project Does

Self-driving cars require agents that can make **continuous decisions** — smooth steering angles and precise throttle control — rather than choosing from a fixed set of discrete actions. This project tackles that challenge by:

1. **Building a custom 2D racing environment** from scratch using Pygame, complete with physics, raycasting sensors, and lap detection.
2. **Implementing two state-of-the-art actor-critic algorithms** — TD3 and DDPG — in PyTorch to train agents that learn to drive autonomously.
3. **Running a structured 24-experiment research grid** across 4 reward shaping strategies × 3 sensor noise levels × 2 algorithms to systematically study how design choices affect learning.

The result is a complete, reproducible RL pipeline — from environment design to training, evaluation, logging, and visualization.

---

## ✨ Key Features

- **Two RL Algorithms** — Full implementations of TD3 (with twin critics, delayed updates, target smoothing) and DDPG (single-critic baseline) for head-to-head comparison.
- **Custom 2D Environment** — Pygame-based oval track with realistic bicycle-model physics, speed-dependent steering, and collision detection.
- **Raycasting Sensor System** — 3-ray distance sensors with configurable Gaussian noise for robust perception testing.
- **4 Reward Modes (R1–R4)** — From basic survival rewards to heavily shaped incentives for speed, stability, and lap completion.
- **3 Noise Levels (N1–N3)** — Evaluate agent robustness under clean, moderate, and noisy sensor readings.
- **Multi-Seed Experiments** — Each configuration runs across 3 seeds (0, 42, 123) for statistical reliability.
- **Structured Logging** — JSONL-format training logs with per-episode metrics for easy analysis.
- **Visualization Pipeline** — Automated reward curves, crash rates, and cross-algorithm comparison plots.
- **Headless Training** — Full off-screen execution support for cloud, Kaggle, Colab, and CI environments.
- **Resume & Checkpoint System** — Training can be interrupted and resumed from the latest saved checkpoint.

---

## 🏗️ Project Structure

```text
TD3-DDPG-Car-Game/
│
├── main.py                  # Entry point — train, evaluate, or demo a single agent
├── train.py                 # Core training loop with exploration decay and metrics
├── run_experiments.py       # Automated experiment runner for the full R×N grid
├── eval_models.py           # Multi-model evaluation and comparison tool
│
├── td3_agent.py             # TD3 actor-critic networks and training logic
├── ddpg_agent.py            # DDPG actor-critic networks and training logic
├── replay_buffer.py         # Fixed-size circular replay buffer (NumPy-backed)
│
├── environment.py           # Gym-style racing environment with rewards and rendering
├── car.py                   # Car physics, raycasting sensors, and state output
├── lap_timer.py             # Finish-line crossing detection and lap timing
│
├── config.py                # All hyperparameters, constants, and experiment grid
├── utils.py                 # Seed management, asset generation, Pygame helpers
├── metrics_tracker.py       # Per-step and per-episode metrics with JSONL logging
├── plot_metrics.py          # Plotting utilities for reward curves and comparisons
│
├── colab_demo_td3.ipynb     # Colab notebook for TD3 experiments
├── colab_demo_ddpg.ipynb    # Colab notebook for DDPG experiments
├── colab_demo_both.ipynb    # Combined notebook for full experiment suite
│
├── assets/                  # Auto-generated track and car sprite images
├── logs/                    # Training logs organized by algo/config/seed
├── models/                  # Saved model checkpoints (best, periodic, avg100)
├── results/                 # Generated plots and analysis outputs
└── requirements.txt         # Python dependencies
```

---

## 🧠 Algorithms Implemented

### TD3 — Twin Delayed Deep Deterministic Policy Gradient

TD3 addresses the overestimation bias and instability issues present in DDPG through three key innovations:

| Technique | What It Does |
|-----------|-------------|
| **Twin Critics** | Maintains two independent Q-networks and takes the minimum Q-value to compute targets, reducing overestimation bias. |
| **Delayed Policy Updates** | Updates the actor network only every 2 critic updates, allowing the critic to stabilize before the policy changes. |
| **Target Policy Smoothing** | Adds clipped noise to target actions during critic updates, preventing the policy from exploiting sharp peaks in the Q-function. |

### DDPG — Deep Deterministic Policy Gradient

DDPG is the foundational algorithm for continuous control with deep RL:

| Component | Description |
|-----------|-------------|
| **Deterministic Policy** | The actor outputs a single action (no sampling), making it efficient for continuous spaces. |
| **Single Critic** | One Q-network estimates action values — simpler but more prone to overestimation. |
| **Soft Target Updates** | Target networks are updated via Polyak averaging (τ = 0.005) for training stability. |

### Why Both?

Including both algorithms enables a direct, controlled comparison. TD3 is expected to be more stable and consistent, while DDPG serves as a simpler baseline. The structured experiment grid reveals exactly where and why these differences emerge.

---

## 🌍 Environment Details

### Simulation

The environment is a **custom Pygame-based 2D racing track** — an elliptical oval where the car must learn to navigate continuously without going off-road.

| Property | Value |
|----------|-------|
| Screen Size | 1200 × 800 pixels |
| Track Shape | Elliptical oval with inner/outer boundaries |
| Physics Model | Bicycle model with friction and speed-dependent steering |
| Max Speed | 8.0 units/step |
| Termination | Off-track collision or stuck for 180+ steps |

### State Space (7-dimensional)

The agent observes a compact state vector at each step:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | `x` | [0, 1] | Normalized horizontal position |
| 1 | `y` | [0, 1] | Normalized vertical position |
| 2 | `speed` | [0, 1] | Normalized current speed |
| 3 | `angle` | [0, 1] | Normalized heading angle |
| 4 | `sensor_left` | [0, 1] | Distance to track edge at −45° |
| 5 | `sensor_front` | [0, 1] | Distance to track edge at 0° |
| 6 | `sensor_right` | [0, 1] | Distance to track edge at +45° |

### Action Space (2-dimensional, continuous)

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [−1, +1] | Left/right turn intensity |
| Throttle | [0, +1] | Forward acceleration (no reverse) |

### Sensor System

The car uses **3 raycasting sensors** that emit rays at −45°, 0°, and +45° relative to the car's heading. Each ray marches pixel-by-pixel (step size = 2) up to 200 pixels, returning the normalized distance to the nearest off-track boundary.

Configurable **Gaussian noise** is added to sensor readings to simulate real-world perception uncertainty.

### Reward Configurations

| Mode | Code | Description |
|------|------|-------------|
| **R1** — Basic | `basic` | Flat +0.05 survival bonus per step; −5.0 crash penalty. No shaping. |
| **R2** — Shaped | `shaped` | Adds speed bonus (+0.15), lap completion bonus (+15.0), and steering penalty. |
| **R3** — Modified | `modified` | Enhanced R2 with higher speed bonus (+0.18), stability bonuses, and anti-idle penalty. |
| **R4** — Tuned | `tuned` | Aggressive shaping with strong speed incentive (+0.25), high lap bonus (+18.0), and movement rewards. |

### Sensor Noise Levels

| Level | Code | Std Dev | Description |
|-------|------|---------|-------------|
| **N1** | `0.00` | 0.0 | Perfect sensors — no noise |
| **N2** | `0.02` | 0.02 | Mild noise — slight perception jitter |
| **N3** | `0.05` | 0.05 | Heavy noise — significant reading uncertainty |

---

## 🔬 Experiment Setup

### Design

The experiment grid systematically tests how **reward shaping** and **sensor noise** interact with each algorithm:

```
Experiments = 4 Reward Modes × 3 Noise Levels × 2 Algorithms = 24 configurations
Each configuration × 3 Seeds = 72 total training runs
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes per run | 2,000 |
| Max steps per episode | 300 |
| Replay buffer capacity | 200,000 |
| Batch size | 256 |
| Training starts after | 5,000 steps |
| Network architecture | 2 hidden layers, 64 units each |
| Learning rate (actor & critic) | 3 × 10⁻⁴ |
| Discount factor (γ) | 0.99 |
| Soft update rate (τ) | 0.005 |
| Exploration noise | 0.1, decaying at 0.9999/episode |
| Seeds | 0, 42, 123 |

### Output Organization

Experiment outputs are cleanly isolated by algorithm, configuration, and seed:

```
logs/
└── td3/
    ├── R1_N1/
    │   ├── seed_0/training_log.jsonl
    │   ├── seed_42/training_log.jsonl
    │   └── seed_123/training_log.jsonl
    ├── R1_N2/
    │   └── ...
    └── R4_N3/
        └── ...

models/
└── td3/
    └── R1_N1/
        └── seed_42/
            ├── td3_R1_N1_best.pth
            ├── td3_R1_N1_best_avg100.pth
            └── td3_R1_N1_ep500.pth
```

---

## 📊 Results & Insights

Key observations from the experiment suite:

| Finding | Detail |
|---------|--------|
| **TD3 is more stable** | Twin critics and delayed updates produce smoother reward curves with less variance across seeds. |
| **DDPG learns faster initially** | Simpler architecture can converge faster in low-noise conditions, but often becomes unstable later. |
| **Reward shaping matters significantly** | R4 (tuned) produces the fastest convergence, while R1 (basic) struggles to learn meaningful behavior. |
| **Noise degrades DDPG more than TD3** | Higher sensor noise (N3) causes more crashes and reward variance in DDPG; TD3 is more robust. |
| **Multi-seed averaging is essential** | Individual runs can vary widely — 3-seed averages reveal the true trend. |

Use the built-in plotting tools to generate reward curves, crash rate comparisons, and cross-algorithm analysis from your own training logs.

---

## 🚀 How to Run

### Prerequisites

- Python 3.10+
- pip

### Local Setup

```bash
# Clone the repository
git clone https://github.com/adityagangwani30/TD3-DDPG-Car-Game.git
cd TD3-DDPG-Car-Game

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Train a Single Agent

```bash
# Train TD3 with default settings (GUI mode)
python main.py --algo td3 --mode train

# Train DDPG in headless mode
python main.py --algo ddpg --mode train --headless

# Train with custom episode count
python main.py --algo td3 --mode train --max-episodes 2000 --max-steps 300

# Resume training from latest checkpoint
python main.py --algo td3 --mode train --resume
```

### Run the Full Experiment Grid

```bash
# Run all 12 TD3 experiments (4 reward × 3 noise) with headless rendering
python run_experiments.py --algo td3 --max-episodes 2000 --max-steps 300 --headless

# Run all 12 DDPG experiments
python run_experiments.py --algo ddpg --max-episodes 2000 --max-steps 300 --headless

# Run a specific seed only
python run_experiments.py --algo td3 --seed 42 --headless

# Resume interrupted experiments (skips completed runs)
python run_experiments.py --algo td3 --resume --headless
```

### Evaluate & Demo

```bash
# Evaluate a trained model (10 episodes)
python main.py --algo td3 --mode eval --render

# Run a quick visual demo
python main.py --algo td3 --mode demo

# Compare multiple checkpoints
python eval_models.py --model "td3_*.pth" --episodes 10
```

### Plot Results

```bash
# Generate comparison plots across all experiments
python plot_metrics.py --compare-algos

# See all plotting options
python plot_metrics.py -h
```

### Run on Google Colab / Kaggle

1. Open the appropriate Colab badge link at the top of this README.
2. The notebook handles environment setup, dependency installation, and headless configuration automatically.
3. Training runs entirely off-screen — results and plots are saved to the notebook's file system.
4. For Kaggle, enable GPU acceleration and upload the repository files, then run the notebook cells in order.

---

## 📂 Outputs

| Directory | Contents |
|-----------|----------|
| `logs/` | JSONL training logs with per-episode reward, length, crashes, laps, speed stats, and exploration noise. |
| `models/` | PyTorch checkpoint files — best model, best rolling-100 average, and periodic saves every 100 episodes. |
| `results/` | Generated matplotlib plots — reward curves, crash rates, algorithm comparisons, and per-experiment analysis. |

---

## ⚡ Performance Optimizations

The codebase includes several performance optimizations that improve training speed **without changing any results**:

| Optimization | Location | Impact |
|-------------|----------|--------|
| Pre-allocated state buffer | `car.py` | Eliminates repeated NumPy array construction every step |
| Cached trig computations | `car.py` | Reduces redundant `math.cos/sin` calls in raycasting |
| Headless render fast-path | `environment.py` | Skips all rendering work when display is disabled |
| CPU tensor skip | `td3_agent.py`, `ddpg_agent.py` | Avoids unnecessary `.cpu()` transfer when already on CPU |
| Faster gradient clearing | `td3_agent.py`, `ddpg_agent.py` | Uses `zero_grad(set_to_none=True)` for faster gradient reset |
| Running accumulators | `metrics_tracker.py` | Replaces per-step list appends with O(1) running sums |

These optimizations yield an estimated **1.5×–2.5× speedup** in headless training mode while producing **identical training results**.

---

## 🔮 Future Improvements

- **Additional Algorithms** — Implement SAC (Soft Actor-Critic) and PPO for entropy-regularized and on-policy baselines.
- **Environment Complexity** — Add track variants (S-curves, chicanes), dynamic obstacles, and weather/visibility effects.
- **Multi-Agent Racing** — Introduce competitive scenarios where multiple agents race simultaneously.
- **Hyperparameter Tuning** — Integrate Optuna or Ray Tune for automated hyperparameter search.
- **Richer Evaluation** — Add TensorBoard integration, real-time dashboards, and standardized benchmark scripts.
- **Transfer Learning** — Train on one track variant and evaluate generalization to unseen tracks.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

- **Report Issues** — Open an issue for bugs, crashes, or unexpected behavior.
- **Suggest Features** — Propose new algorithms, environment features, or analysis tools.
- **Submit PRs** — Fork the repo, make your changes, and submit a pull request.
- **Improve Experiments** — Suggestions for better reward functions, noise models, or evaluation metrics are especially appreciated.

---

## 📝 License

Please check repository licensing details before external reuse or redistribution.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core language |
| **PyTorch** | Neural network implementation and training |
| **NumPy** | Numerical computation, replay buffer, state processing |
| **Pygame** | 2D environment rendering, physics simulation, raycasting |
| **Matplotlib** | Training visualization and result plotting |
