# 🏎️ TD3 vs DDPG for Autonomous Driving — Trade-off Analysis

> A reinforcement learning study comparing **Twin Delayed DDPG (TD3)** and **Deep Deterministic Policy Gradient (DDPG)** for continuous control in a custom 2D racing environment. This project demonstrates that **higher reward does not necessarily imply safer or more stable policies** — through systematic multi-metric evaluation across 12 experimental configurations.

---

## 🎯 Core Premise

In reinforcement learning for autonomous driving, we often face a fundamental trade-off:

- **High Reward** may correlate with aggressive driving that crashes frequently  
- **Low Crash Rate** may come from conservative policies that complete fewer laps  
- **Stable Learning** may not achieve peak performance in the short term  

This project systematically explores these trade-offs by training TD3 and DDPG agents across varying reward functions and sensor noise levels, then comparing them across **multiple metrics** — not just average reward.

---

## ✨ Key Features

- **TD3 vs DDPG Comparison** — Full implementations of both algorithms with identical hyperparameters for fair comparison.
- **Multi-Metric Evaluation** — Assess performance on reward, crash rate, stability (variance), and convergence — no single winner.
- **4 Reward Modes (R1–R4)** — Basic survival, shaped incentives, modified tuning, and aggressive shaping to study reward design effects.
- **3 Sensor Noise Levels (N1–N3)** — Test robustness under clean, mild, and heavy sensor noise (0.0, 0.02, 0.05 std).
- **12 Experimental Configurations** — Full factorial grid (4 reward × 3 noise × 2 algorithms) with 3 random seeds each.
- **Custom 2D Racing Environment** — Pygame-based physics simulation with speed-dependent steering, raycasting sensors, and lap timing.
- **Automated Analysis Pipeline** — JSONL logging, checkpoint management, and plotting for reproducible comparisons.
- **Headless & Colab Support** — Train in the cloud, locally, or interactively without modification.

---

## 🏗️ Project Structure

```
TD3-DDPG-Car-Game/
│
├── main.py                  # Single-agent training, evaluation, and demo
├── train.py                 # Core training loop with logging and metrics
├── run_experiments.py       # Batch runner for the full 12-experiment grid
├── eval_models.py           # Multi-model evaluation and comparison
│
├── td3_agent.py             # TD3 implementation (twin critics, delayed policy)
├── ddpg_agent.py            # DDPG implementation (single critic baseline)
├── replay_buffer.py         # Fixed-capacity circular replay buffer
│
├── environment.py           # Gym-style racing environment with rewards
├── car.py                   # Physics simulation, raycasting, state management
├── lap_timer.py             # Lap detection and timing logic
│
├── config.py                # Hyperparameters and experiment configurations
├── metrics_tracker.py       # Episode metrics logging (JSONL format)
├── plot_metrics.py          # Visualization utilities
├── utils.py                 # Helper functions (seeds, assets, rendering)
│
├── colab_demo_td3.ipynb     # Notebook: TD3 experiments
├── colab_demo_ddpg.ipynb    # Notebook: DDPG experiments
├── colab_demo_both.ipynb    # Notebook: Full comparative suite
│
├── models/                  # Trained model checkpoints (algo/config/seed)
├── logs/                    # Training logs (algo/config/seed/training_log.jsonl)
├── results/
│   ├── plots/               # Generated matplotlib figures
│   │   ├── comparison/      # TD3 vs DDPG comparisons
│   │   ├── td3/             # TD3 per-configuration analysis
│   │   └── ddpg/            # DDPG per-configuration analysis
│   ├── grouped/             # Grouped results by noise level
│   └── aggregate/           # Aggregated metrics across seeds
│
├── assets/                  # Generated track and car sprites
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🧠 Algorithms

### TD3 — Twin Delayed Deep Deterministic Policy Gradient

TD3 addresses three key weaknesses in DDPG through targeted innovations:

| Innovation | Problem Solved | Impact |
|-----------|--------------|--------|
| **Twin Critics** | Single Q-network can overestimate action values, leading to divergence | Takes minimum of two independent critics to reduce bias |
| **Delayed Policy Updates** | Frequent policy updates can chase an unstable critic | Updates policy only every 2 critic updates for stability |
| **Target Policy Smoothing** | Critic exploits sharp peaks in Q-function | Adds clipped noise to target actions during updates |

**Result:** More stable learning curves, fewer divergence episodes, better long-term performance.

### DDPG — Deep Deterministic Policy Gradient

DDPG is a simpler, foundational actor-critic algorithm:

- **Deterministic Actor:** Outputs a single action per state (no sampling).  
- **Single Critic:** Simpler network, but more prone to overestimation.  
- **Soft Target Updates:** Polyak averaging (τ = 0.005) for stability.

**Trade-off:** DDPG can converge faster initially but often becomes unstable in later episodes, especially under noise.

### Comparison Philosophy

Both algorithms use **identical hyperparameters and training procedures**. The experiment grid isolates the algorithmic differences from confounding factors. This enables direct attribution of performance differences to the algorithms themselves, not their hyperparameters.

---

## 🌍 Environment

### The Track

A 2D elliptical oval racing circuit rendered in Pygame with:

- **Dimensions:** 1200 × 800 pixels  
- **Physics:** Bicycle model with speed-dependent steering, friction, and acceleration limits  
- **Sensors:** 3 raycasting rays (−45°, 0°, +45° relative to heading) that measure distance to track boundaries  
- **Dynamics:** Max speed 8.0 units/step, steering limited by speed, continuous throttle control  

### State Space (7-dimensional)

| Feature | Range | Meaning |
|---------|-------|---------|
| Position X | [0, 1] | Normalized horizontal location |
| Position Y | [0, 1] | Normalized vertical location |
| Speed | [0, 1] | Current velocity (normalized) |
| Heading | [0, 1] | Angle facing (normalized to [0, 2π]) |
| Sensor Left | [0, 1] | Distance to boundary at −45° |
| Sensor Front | [0, 1] | Distance to boundary at 0° |
| Sensor Right | [0, 1] | Distance to boundary at +45° |

### Action Space (2-dimensional, continuous)

| Action | Range | Description |
|--------|-------|-------------|
| Steering | [−1, +1] | Turn intensity (left to right) |
| Throttle | [0, +1] | Acceleration (no reverse) |

### Reward System

Four reward modes explore the space of possible design choices:

| Mode | Code | Design | Use Case |
|------|------|--------|----------|
| **R1** | `basic` | +0.05 per step, −5.0 crash | Minimal shaping; pure survival |
| **R2** | `shaped` | Adds lap bonus (+15.0), speed bonus (+0.15), steering penalty | Standard shaped incentives |
| **R3** | `modified` | Enhanced R2: higher speed bonus (+0.18), stability bonuses | Mid-level tuning |
| **R4** | `tuned` | Aggressive shaping: max speed bonus (+0.25), max lap bonus (+18.0), anti-idle penalty | Maximum shaping |

### Sensor Noise

Robustness is tested under three noise regimes applied to sensor readings:

| Level | Std Dev | Interpretation |
|-------|---------|-----------------|
| **N1** | 0.00 | Perfect sensors — clean observations |
| **N2** | 0.02 | Mild noise — realistic perception jitter |
| **N3** | 0.05 | Heavy noise — significant uncertainty |

---

---

## 🔬 Experiments

### Design Rationale

The experiment grid uses a **full factorial design** to isolate how reward shaping and noise robustness interact with algorithmic choice:

$$\text{Experiments} = 4 \text{ Reward Modes} \times 3 \text{ Noise Levels} \times 2 \text{ Algorithms} = 24 \text{ configurations}$$

Each configuration is trained with 3 independent random seeds (0, 42, 123) for statistical validity.

### Configuration

| Parameter | Setting |
|-----------|---------|
| Episodes per run | 2,000 |
| Max steps per episode | 300 |
| Replay buffer capacity | 200,000 |
| Batch size | 256 |
| Training starts after | 5,000 steps |
| Network architecture | 2 hidden layers, 64 units each |
| Learning rates (actor & critic) | 3 × 10⁻⁴ |
| Discount factor (γ) | 0.99 |
| Soft update rate (τ) | 0.005 |
| Exploration noise | 0.1, decaying at 0.9999/episode |
| Random seeds | 0, 42, 123 |

### Output Organization

Logs and models are organized hierarchically for easy access:

```
logs/
├── td3/
│   ├── basic_noise_0.00/
│   │   ├── seed_0/training_log.jsonl
│   │   ├── seed_42/training_log.jsonl
│   │   └── seed_123/training_log.jsonl
│   └── tuned_noise_0.05/
│       └── ...
└── ddpg/
    └── ...

models/
├── td3/
│   ├── basic_noise_0.00/
│   │   ├── seed_0/
│   │   │   ├── td3_best.pth
│   │   │   ├── td3_best_avg100.pth
│   │   │   └── td3_ep500.pth
│   │   └── ...
└── ddpg/
    └── ...
```

---

## 📊 Metrics

All agents are evaluated on a consistent set of metrics — no single number captures the full picture:

### 1. Average Reward (mean ± std across seeds)

The primary signal of learning progress. Higher values indicate agents that accumulate more reward.

- **Interpretation:** Reward captures incentives defined by the reward function.  
- **Caveat:** Depends heavily on reward shaping; higher reward ≠ safer driving.

### 2. Crash Rate (%)

Percentage of episodes in which the agent leaves the track.

- **Interpretation:** Safety proxy; lower is better.  
- **Caveat:** Reward-based agents may crash if crashes are insufficiently penalized.

### 3. Stability (Reward Variance)

Standard deviation of rewards within a training run, averaged across seeds.

- **Interpretation:** Learning consistency; lower variance = more predictable behavior.  
- **Caveat:** Stable doesn't always mean good — a stable bad policy is still bad.

### 4. Convergence Episode

The episode at which the agent reaches 50% of its final average reward (over last 100 episodes).

- **Interpretation:** Speed of learning; earlier = faster.  
- **Caveat:** Faster convergence may indicate overoptimization to current reward, not true learning.

### 5. Lap Completion Rate

Percentage of episodes in which the agent completes at least one full lap.

- **Interpretation:** Task success; higher is better.  
- **Caveat:** With strong lap bonuses (R3, R4), lap rate should be high regardless of driving quality.

---

## 🎯 Key Insights & Trade-offs

This section summarizes patterns observed across the 24 configurations. **No single algorithm or configuration dominates all metrics.**

### Finding 1: Reward-Safety Trade-off

Configurations with the highest rewards (R4) often exhibit higher crash rates compared to conservatively-shaped rewards (R1, R2).

- **Why:** Aggressive reward shaping incentivizes risky behaviors (e.g., high speed near edges).  
- **Implication:** For safety-critical applications, use modestly-shaped rewards and accept lower peak performance.

### Finding 2: Stability Advantage for TD3

TD3 generally produces lower reward variance and fewer divergence episodes compared to DDPG, especially under noise.

- **Why:** Twin critics and delayed updates dampen Q-function estimation errors.  
- **Implication:** For reproducible, reliable policies, TD3 is the better choice.

### Finding 3: Noise Robustness

Sensor noise (N3) degrades both algorithms, but DDPG's single critic is more sensitive to perception errors.

- **Why:** DDPG's critic lacks the error-dampening of TD3's twin design.  
- **Implication:** In real-world scenarios with noisy sensors, TD3 provides more robust policies.

### Finding 4: Reward Shaping is Non-Monotonic

Increasing reward shaping (R1 → R4) does not monotonically improve performance across all metrics.

- **Why:** Over-shaping can introduce spurious incentives that conflict with the true task (racing).  
- **Implication:** Tuning reward functions requires empirical evaluation, not just intuition.

### Finding 5: Seed Variability is Significant

Differences within a configuration (across seeds) are often comparable to differences between configurations.

- **Why:** Stochastic initialization and replay buffer order introduce substantial variance.  
- **Implication:** Always report mean ± std; single-run conclusions are unreliable.

---

## 📈 Visualizations

The plotting pipeline generates figures for analysis:

### Comparison Plots (`results/plots/comparison/`)

- **Reward vs Crash Rate (Scatter):** Reveals the reward-safety trade-off across configurations.  
- **TD3 vs DDPG (Side-by-Side):** Direct algorithmic comparison at each noise level.  
- **Convergence Curves:** Learning trajectory for representative configurations.

### Per-Algorithm Plots (`results/plots/td3/` and `results/plots/ddpg/`)

- **Reward Curves (line plots):** Average reward ± std across seeds.  
- **Crash Rate Heatmap:** Crash % organized by reward mode and noise level.  
- **Stability Analysis:** Reward variance across configurations.

### Grouped Plots (`results/grouped/`)

- **By Noise Level:** All 4 reward modes at a fixed noise level (N1, N2, N3).  
- **By Reward Mode:** All 3 noise levels at a fixed reward mode (R1–R4).

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
# Train TD3 with default settings (2000 episodes, rendered)
python main.py --algo td3 --mode train

# Train DDPG in headless mode (faster, no display)
python main.py --algo ddpg --mode train --headless

# Custom episode and step counts
python main.py --algo td3 --mode train --max-episodes 1000 --max-steps 300 --headless

# Resume training from latest checkpoint
python main.py --algo td3 --mode train --resume
```

### Run Full Experiment Suite

```bash
# Run all 12 TD3 configurations (4 reward × 3 noise, headless)
python run_experiments.py --algo td3 --headless

# Run all 12 DDPG configurations
python run_experiments.py --algo ddpg --headless

# Run a specific seed
python run_experiments.py --algo td3 --seed 42 --headless

# Resume interrupted experiments (skips completed runs)
python run_experiments.py --algo td3 --resume --headless
```

### Evaluation & Visualization

```bash
# Evaluate a trained model (10 episodes, with rendering)
python main.py --algo td3 --mode eval --render

# Run interactive demo
python main.py --algo td3 --mode demo

# Generate all plots from existing logs
python plot_metrics.py --compare-algos

# Compare specific checkpoints
python eval_models.py
```

### Running on Google Colab / Kaggle

1. Open the Colab badge link at the top of this README.  
2. The notebook handles dependencies, environment setup, and headless configuration.  
3. Run cells sequentially — training executes in background with periodic updates.  
4. Plots and logs are saved to the notebook's file system.

---

## 📂 Outputs

| Directory | Contents |
|-----------|----------|
| `logs/` | JSONL training logs; one file per (algo, config, seed) tuple with per-episode reward, crashes, laps, speed, and noise decay. |
| `models/` | PyTorch `.pth` checkpoints; best episode, best 100-episode average, and periodic saves. |
| `results/plots/` | Matplotlib figures organized by algorithm and comparison type. |
| `results/grouped/` | Analysis plots organized by reward mode or noise level. |
| `results/aggregate/` | Summary statistics and CSV exports. |

---

## 🔑 Key Hyperparameters

Tuning order (most to least impactful):

1. **Reward Shaping:** Changes exploration incentives fundamentally.  
2. **Sensor Noise:** Affects difficulty and algorithm robustness.  
3. **Exploration Noise Decay:** Controls convergence speed vs. stability.  
4. **Network Size (64 → 128):** Increases capacity at computational cost.  
5. **Learning Rate:** Small adjustments help; large changes destabilize training.

To experiment with these, modify `config.py` before training.

---

## ⚙️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **RL Framework** | PyTorch (2.0+) |
| **Numerics** | NumPy |
| **Physics Simulation** | Pygame |
| **Visualization** | Matplotlib |
| **Logging** | JSONL (text-based) |

---

## 📖 Citation & Attribution

If you use this codebase in research, please cite the authors and acknowledge the TD3 and DDPG original papers:

- **TD3:** [Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477)  
- **DDPG:** [Continuous Control with Deep Reinforcement Learning (Lillicrap et al., 2015)](https://arxiv.org/abs/1509.02971)

---

## 📝 Notes

- **Trade-offs, Not Winners:** This study emphasizes the absence of a universally optimal configuration. Choose based on your priority (reward, safety, or stability).  
- **Reproducibility:** All random seeds, hyperparameters, and data are logged for full reproducibility.  
- **Extension:** The modular structure makes it easy to add new reward modes, algorithms, or environment variants.  
- **Feedback:** If you identify bugs, inconsistencies, or have suggestions for the experiment design, please open an issue.
