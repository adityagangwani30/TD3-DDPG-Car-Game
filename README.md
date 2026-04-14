# TD3 Self-Driving Car: Autonomous Racing via Reinforcement Learning

**A research-grade reinforcement learning project exploring continuous control, reward design, and sensor robustness in autonomous driving.**

> This repository presents a complete implementation of **Twin Delayed Deep Deterministic Policy Gradient (TD3)** applied to a simulated 2D racing environment. The project is designed for learning RL fundamentals while maintaining research reproducibility through structured experiments, ablation studies, and comprehensive metrics tracking.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-Car-Game/blob/main/colab_demo.ipynb)

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
- [Research Objective](#research-objective)
- [Key Contributions](#key-contributions)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [Experiment Setup](#experiment-setup)
- [Results & Metrics](#results--metrics)
- [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview & Motivation

### The Problem

Autonomous driving requires learning **complex control policies** from high-dimensional sensory input. Key challenges include:

- **Continuous action spaces**: steering and throttle are continuous values, not discrete choices
- **Local optima**: early training often leads to local minima (e.g., driving in circles)
- **Robustness**: policies must handle imperfect sensor readings and environmental variations
- **Sample efficiency**: training should require reasonable computational resources

### The Solution

This project uses **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**, an off-policy actor-critic algorithm that addresses these challenges through:

- Twin Q-networks to reduce overestimation bias
- Delayed policy updates for stability
- Target policy smoothing to improve robustness
- Experience replay for sample efficiency

We focus on a **simplified top-down racing task** to isolate the core control problem and enable fast experimentation.

### Why This Project?

This repository balances **educational clarity** with **research rigor**:

- **For learners**: clean, modular code with thorough comments explaining RL fundamentals
- **For researchers**: reproducible experiments with isolated logging, hyperparameter grids, and metrics tracking
- **For practitioners**: extensible framework for testing new reward functions and sensor configurations

---

## Research Objective

### Primary Questions

This project investigates how two factors influence TD3 learning in continuous control tasks:

1. **Reward Shaping**: How does reward design impact convergence speed, policy quality, and stability?
2. **Sensor Robustness**: How does sensor noise affect learning success and final policy performance?

### Hypotheses

- Well-designed reward functions should accelerate convergence and improve final policy quality
- Sensor noise degrades performance, but robust rewards can mitigate this degradation
- The interaction between reward mode and noise level is non-trivial and worth studying empirically

---

## Key Contributions

1. **Multi-mode Reward System**: Three distinct reward formulations (basic, shaped, modified) for systematic ablation studies
2. **Sensor Noise Framework**: Configurable Gaussian noise injection for robustness analysis
3. **Reproducible Experiments**: Deterministic seeding and isolated experiment directories for clean comparative analysis
4. **Complete Metrics Suite**: Episode rewards, crash rates, lap times, steering smoothness, and phase-space analysis
5. **Modular Codebase**: Clear separation between environment, agent, training, and evaluation logic

---

## Methodology

### 1. Environment Design

#### Simulation Model

The environment simulates a car on a 2D oval track rendered from above:

- **Track**: Oval boundary defined by outer radius (480×320 px) and inner radius (320×180 px)
- **Car dynamics**: Simplified kinematic model with friction and speed-dependent steering
- **Start condition**: Fixed position and heading for deterministic episode initialization
- **Episode termination**: Off-track, collision, or max steps (configurable)

#### Physics

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Max speed | 8.0 units/frame | Controls timescale and learning difficulty |
| Acceleration | 0.3 units/frame² | Throttle responsiveness |
| Friction | 0.05 | Realistic deceleration |
| Turn rate | 4.0°/frame | Base steering responsiveness |
| Speed-turn factor | 0.5 | Reduces turning at high speed: `turn_angle *= (1 - speed/max_speed * 0.5)` |

### 2. State Space

**Observation vector: 7 dimensions** (4 proprioceptive + 3 exteroceptive)

```
State = [x_normalized, y_normalized, speed_normalized, heading_normalized, 
         sensor_1_distance, sensor_2_distance, sensor_3_distance]
```

**Components**:
- `x, y`: Normalized vehicle position (divide by screen dimensions)
- `speed`: Normalized velocity (divide by max speed)
- `heading`: Normalized heading angle (divide by 2π)
- `sensor_*`: Normalized ray-cast distances (divide by max sensor distance = 200 px)

**Why?** Normalization helps neural networks learn faster and improves generalization.

### 3. Action Space

**Action vector: 2 dimensions** (continuous, constrained)

```
Action = [steering, throttle]
  where steering ∈ [-1, 1]  (left/right)
        throttle ∈ [0, 1]   (forward only, no reverse)
```

**Mapping to control**:
- Steering: `turn_angle = steering * CAR_TURN_RATE * speed_factor`
- Throttle: `new_speed = current_speed + throttle * CAR_ACCELERATION - friction`

### 4. Reward Function

The reward function drives all learning. We implement **three modes** for ablation studies:

#### Mode 1: Basic Reward (Minimal Shaping)

```
R_basic = +1.0 per step (survival bonus)
        - 10.0 if off-track
        - 5.0 if collision / stuck
        + 50.0 for lap completion
        - 0.01 * |steering| (steering penalty)
```

**Pros**: Simple, easy to understand  
**Cons**: Often leads to reward hacking or poor driving (e.g., driving in circles)

#### Mode 2: Shaped Reward (Recommended)

```
R_shaped = +1.0 per step (survival bonus)
         + 0.5 * speed (reward forward motion)
         + 1.0 * (progress_made) (reward lap progress)
         - 10.0 if off-track
         - 5.0 if collision / stuck
         + 100.0 for lap completion
         - 0.02 * |steering| (steering penalty)
```

**Design rationale**:
- Encourages consistent forward motion (avoids stuck loops)
- Rewards incremental lap progress (guidance signal)
- Heavy lap bonus incentivizes goal completion
- Steering penalty promotes smooth, efficient control

**Pros**: Better-shaped exploration signal  
**Cons**: More hyperparameters to tune

#### Mode 3: Modified Reward (Enhanced Shaping)

```
R_modified = R_shaped + adaptive_noise_bonus
```

Adds robustness-aware shaping to encourage policies that are more resilient to sensor uncertainty.

**Design rationale**: When sensor noise is high, the agent should learn smoother, less reactive policies

### 5. TD3 Algorithm (High-Level)

**Twin Delayed DDPG** ([Fujimoto et al., 2018](https://arxiv.org/abs/1802.09477)) improves DDPG through:

#### Key Components

| Component | Purpose |
|-----------|---------|
| **Actor Network** | Maps state → action (policy) |
| **Critic Networks (×2)** | Maps (state, action) → Q-value estimate |
| **Target Networks** | Slow-moving copies for stable TD targets |
| **Replay Buffer** | Stores (state, action, reward, next_state, done) tuples |

#### Training Algorithm

1. **Collect experience**: Current policy samples transitions into replay buffer
2. **Sample mini-batch**: Draw random sample from replay buffer (breaks correlation)
3. **Compute TD targets** (using target networks):
   ```
   y = reward + γ * Q_target(s', μ_target(s') + ε)  [where ε ~ N(0, σ²)]
   ```
4. **Update critics**: Minimize MSE between Q predictions and TD targets
5. **Delayed policy update** (every d steps):
   - Compute actor gradient using first critic
   - Update target networks (exponential moving average)

#### Why TD3 Works Well for This Task

- **Twin Q-networks**: Reduce overestimation of Q-values → more conservative, stable learning
- **Delayed updates**: Gives critics time to stabilize before updating policy → prevents instability
- **Target smoothing**: Adds noise to prevent deterministic overfitting → improves robustness

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/adityagangwani30/TD3-Car-Game.git
cd td3-car-game
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- `torch>=2.0`: Deep learning framework
- `pygame>=2.0`: Environment rendering
- `numpy`: Numerical operations
- (Optional) `matplotlib`: Plotting utilities (used by `plot_metrics.py`)

### 4. First Run

The first execution will auto-generate assets (track and car sprites):

```bash
python main.py --mode demo
```

If assets are created successfully, you should see:
```
✓ Assets exist in: ./assets/
```

---

## Quick Start

### Try the Demo (2-3 minutes)

```bash
python main.py --mode demo
```

Runs 3 evaluation episodes using a pre-trained checkpoint (if available). Perfect for verifying installation.

### Train from Scratch (30 minutes to several hours)

```bash
python main.py --mode train
```

Trains a new agent. By default:
- Uses **shaped reward mode**
- Runs for **100 episodes**
- Renders the environment in real-time (set `RENDER_DURING_TRAINING = False` in config.py for faster training)
- Saves checkpoints every 10 episodes

### Evaluate a Checkpoint

```bash
python main.py --mode eval --checkpoint models/td3_best.pth --eval-episodes 10
```

Runs 10 evaluation episodes (no learning) and reports metrics.

---

## How to Run

### Training Commands

```bash
# Train from scratch (default)
python main.py --mode train

# Resume from latest checkpoint
python main.py --mode train --resume

# Resume from specific checkpoint
python main.py --mode train --checkpoint models/td3_ep500.pth

# Train in headless mode (for servers / Colab)
python main.py --mode train --headless
```

### Evaluation Commands

```bash
# Evaluate with latest checkpoint (auto-detected)
python main.py --mode eval

# Evaluate specific checkpoint for 20 episodes
python main.py --mode eval --checkpoint models/td3_best.pth --eval-episodes 20

# Headless evaluation
python main.py --mode eval --headless --eval-episodes 10
```

### Demo Commands

```bash
# Quick demo (3 episodes)
python main.py --mode demo

# Extended demo (10 episodes)
python main.py --mode demo --eval-episodes 10

# Demo in headless mode
python main.py --mode demo --headless
```

### Google Colab

Use the provided notebook for a one-click Colab experience:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-Car-Game/blob/main/colab_demo.ipynb)

The notebook automatically:
- Clones the repo
- Installs dependencies
- Checks GPU availability
- Runs a headless demo with visual output

---

## Experiment Setup

### Structured Ablation Studies

The project supports **systematic experimentation** via `run_experiments.py`. This enables fair comparison of different reward modes and sensor noise levels.

### Experiment Grid

We study 9 configurations (3 reward modes × 3 noise levels):

| Reward Mode | Noise Level | Experiment ID |
|-------------|-------------|---------------|
| basic | 0.00 | `basic_noise_0.00` |
| basic | 0.02 | `basic_noise_0.02` |
| basic | 0.05 | `basic_noise_0.05` |
| shaped | 0.00 | `shaped_noise_0.00` |
| shaped | 0.02 | `shaped_noise_0.02` |
| shaped | 0.05 | `shaped_noise_0.05` |
| modified | 0.00 | `modified_noise_0.00` |
| modified | 0.02 | `modified_noise_0.02` |
| modified | 0.05 | `modified_noise_0.05` |

#### Sensor Noise Mechanism

During environment steps, raw sensor distances are corrupted by Gaussian noise:

```
corrupted_distance = raw_distance + N(0, std²)
corrupted_distance = max(0, min(MAX_DIST, corrupted_distance))  # Clamp to valid range
```

This simulates real-world sensor imperfections (e.g., lidar noise, camera calibration errors).

### Running Experiments

#### Full Experiment Suite (All 9 Configurations)

```bash
python run_experiments.py
```

This sequentially trains 9 agents with different configurations. Results are isolated:
- **Logs**: `logs/{experiment_id}/training_log.jsonl`
- **Models**: `models/{experiment_id}/td3_best.pth`, etc.

Typical runtime: 4-8 hours on a modern GPU

#### Quick Validation (Small Experiment)

```bash
python run_experiments.py --max-experiments 2 --max-episodes 10 --max-steps 50 --headless
```

Tests the infrastructure with 2 experiments, 10 episodes each, 50 steps/episode. Runtime: ~2 minutes.

#### Custom Experiments

Edit the `EXPERIMENTS` dictionary in `config.py` to test custom configurations:

```python
EXPERIMENTS = {
    "my_exp_1": {"reward_mode": "shaped", "sensor_noise_std": 0.03},
    "my_exp_2": {"reward_mode": "modified", "sensor_noise_std": 0.01},
}
```

Then run:
```bash
python run_experiments.py
```

---

## Results & Metrics

### Metrics Tracked

Each training episode logs the following metrics to `logs/{experiment_id}/training_log.jsonl`:

| Metric | Description |
|--------|-------------|
| `episode` | Episode number (0-indexed) |
| `reward_total` | Total reward accumulated in the episode |
| `reward_rolling_avg_100` | Average reward over last 100 episodes (smoothed metric) |
| `episode_length` | Number of steps before episode termination |
| `laps_completed` | Integer lap count |
| `collisions` | Number of collision/off-track events |
| `avg_speed` | Mean velocity during the episode |
| `steering_smoothness` | Measure of steering angle changes (lower = smoother) |
| `experiment_name` | Experiment identifier (e.g., `shaped_noise_0.02`) |
| `reward_mode` | Reward mode used (basic / shaped / modified) |
| `sensor_noise_std` | Applied sensor noise standard deviation |
| `seed` | Random seed for reproducibility |

### Interpretation Guide

**Success indicators**:
- `reward_rolling_avg_100` increases over time → agent is learning
- `collisions` decreases as training progresses → agent learns to avoid crashes
- `laps_completed` increases → agent learns to navigate successfully
- `steering_smoothness` improves → policy becomes more stable

**Problem signs**:
- `reward_rolling_avg_100` plateaus early → may indicate reward design issue
- `collisions` remain constant → agent not learning caution
- `avg_speed` = 0 for many episodes → agent stuck (frozen policy or bad initialization)

### Evaluating Checkpoints

Compare learned models across experiments:

```bash
python eval_models.py --episodes 20
```

This runs all saved checkpoints in `models/` for 20 evaluation episodes each and reports comparative statistics.

---

## Visualization

### 1. GUI Mode (Local)

When running locally with a display:

```bash
python main.py --mode train
```

A Pygame window opens showing:
- **Track rendering**: Oval with road/grass boundaries
- **Car visualization**: Small rectangle with heading indicator
- **Ray sensors**: Cyan lines showing sensor rays, red dots at endpoints
- **HUD (Head-Up Display)**: Real-time metrics overlay (FPS, speed, episode reward, etc.)

### 2. Headless Mode (Servers / Colab)

When running without a display:

```bash
python main.py --mode train --headless
```

- No Pygame window opens
- Simulation runs 3-4× faster (no rendering overhead)
- Metrics still logged to files for post-hoc analysis
- Colab notebook captures preview images

### 3. Metrics Visualization

After training, generate plots:

```bash
# Plot single experiment
python plot_metrics.py --log-dir logs --experiment shaped_noise_0.02

# Compare all experiments
python plot_metrics.py --log-dir logs --compare

# Compare specific subset
python plot_metrics.py --log-dir logs --experiments shaped_noise_0.00 shaped_noise_0.02 shaped_noise_0.05
```

Generated plots include:
- **Reward curve**: Episode reward vs. training step
- **Crash rate**: Collision count vs. training step
- **Lap completions**: Cumulative laps vs. training step
- **Cross-experiment comparison**: Side-by-side analysis of different configurations

---

## Project Structure

```
td3-car-game/
├── main.py                      # CLI entry point (train / eval / demo modes)
├── train.py                     # Training and evaluation loops
├── environment.py               # CarRacingEnv: RL task + reward + rendering
├── car.py                       # Car dynamics and sensor raycasting
├── td3_agent.py                 # Actor/Critic networks and TD3 algorithm
├── replay_buffer.py             # Off-policy experience replay buffer
├── config.py                    # Hyperparameters, paths, constants
├── utils.py                     # Helpers: seeding, asset generation, pygame init
├── lap_timer.py                 # Lap timing and cross-finish-line detection
├── metrics_tracker.py           # Metrics logging (JSONL) and statistics
├── plot_metrics.py              # Generate plots from training logs
├── eval_models.py               # Compare saved checkpoints
├── run_experiments.py           # Run experiment grid sequentially
├── colab_demo.ipynb             # Google Colab notebook
├── requirements.txt             # Python dependencies
├── assets/                      # Generated track and car images
│   ├── track.png                # Oval track rendering
│   └── car.png                  # Car sprite
├── models/                      # Saved checkpoints
│   ├── td3_best.pth             # Best model overall
│   ├── td3_best_avg100.pth      # Best by rolling avg reward
│   ├── td3_ep100.pth            # Checkpoint at episode 100
│   └── {experiment_id}/         # Experiment-specific models
└── logs/                        # Training logs
    ├── training_log.jsonl       # Global training log
    └── {experiment_id}/         # Experiment-specific logs
        └── training_log.jsonl
```

### File Descriptions

| File | Purpose |
|------|---------|
| **main.py** | CLI interface with modes: train, eval, demo. Provides `--mode`, `--checkpoint`, `--resume`, `--headless` flags |
| **train.py** | Contains `train()` and `evaluate()` functions; implements core learning loop |
| **environment.py** | Implements `CarRacingEnv` (Gym-like interface); handles physics, rendering, reward computation |
| **car.py** | Implements `Car` class; simulates physics and raycasting for sensors |
| **td3_agent.py** | Implements `TD3Agent` with actor/critic networks; contains TD3 training logic |
| **replay_buffer.py** | Implements `ReplayBuffer` for off-policy experience storage |
| **config.py** | Central configuration: hyperparameters, experiment grid, physics constants |
| **utils.py** | Helper functions: deterministic seeding, asset generation, pygame initialization |
| **lap_timer.py** | Helper to detect lap completion using finish-line crossing |
| **metrics_tracker.py** | Logs episode statistics to JSONL for analysis and plotting |
| **plot_metrics.py** | Reads JSONL logs and generates matplotlib figures |
| **eval_models.py** | Loads all checkpoints and runs evaluation benchmark |
| **run_experiments.py** | Orchestrates the full experiment grid (9 configurations) sequentially |

---

## Configuration

### Key Hyperparameters

All configuration is centralized in `config.py`. Important sections:

#### Training (TD3)

```python
LEARNING_RATE_ACTOR = 0.0001       # Actor learning rate
LEARNING_RATE_CRITIC = 0.001       # Critic learning rate
GAMMA = 0.99                       # Discount factor
TAU = 0.005                        # Target network soft update rate
BATCH_SIZE = 64                    # Mini-batch size
REPLAY_BUFFER_SIZE = 100_000       # Max stored transitions
TRAIN_FREQ = 1                     # Update frequency (steps per update)
UPDATE_FREQ_ACTOR = 2              # Delay actor updates
```

#### Environment

```python
SCREEN_WIDTH = 1200               # Pixel width
SCREEN_HEIGHT = 800               # Pixel height
NUM_SENSORS = 3                   # Ray sensors per observation
SENSOR_MAX_DIST = 200             # Maximum sensor range
SENSOR_ANGLES = [-45, 0, 45]     # Sensor directions (degrees)
```

#### Reward Function

```python
# Mode-specific reward constants
REWARD_ALIVE = 1.0
REWARD_FORWARD = 0.5
REWARD_LAP = 100.0
PENALTY_CRASH = -10.0
PENALTY_STUCK = -5.0
PENALTY_STEERING = -0.02           # Per radian
```

### Modifying Experiments

Edit `EXPERIMENTS` dict in `config.py`:

```python
EXPERIMENTS = {
    "basic_noise_0.00": {
        "reward_mode": "basic",
        "sensor_noise_std": 0.0,
    },
    "shaped_noise_0.02": {
        "reward_mode": "shaped",
        "sensor_noise_std": 0.02,
    },
    # ... add more configurations
}
```

---

## Limitations

### Environmental

- **Simple track**: Single oval; no complex geometry or intersections
- **Idealized physics**: Kinematic model; no wheels, suspension, or friction details
- **Limited observation**: Only ray sensors; no camera or lidar; no odometry uncertainty
- **No dynamics randomization**: Physics constants are fixed; no variation in friction, wind, etc.

### Algorithmic

- **Off-policy only**: Only TD3; no on-policy methods (PPO, TRPO) for comparison
- **Single architecture**: Fixed network sizes; no hyperparameter search
- **No domain randomization**: Track and car properties are fixed deterministically

### Experimental

- **Limited noise model**: Only Gaussian sensor noise; no other uncertainty types
- **Small scale**: 9 total experiments; limited statistical coverage
- **No significance testing**: Results are single runs per configuration

### Generalization

- **Narrow task**: Trained policies unlikely to transfer beyond this specific track
- **Simulation gap**: No sim-to-real transfer analysis
- **Limited curriculum**: No progression from easy to hard tasks

---

## Future Work

### Short-term Extensions

1. **More reward modes**: Implement inverse RL or imitation learning
2. **Additional baselines**: Compare TD3 against SAC, PPO, TRPO
3. **Hyperparameter sweep**: Systematic tuning via grid or Bayesian search
4. **Statistical analysis**: Repeat experiments multiple times, compute confidence intervals

### Medium-term Research

1. **Domain randomization**: Vary track geometry, physics, sensor noise at runtime
2. **Curriculum learning**: Progressive difficulty scheduling (small track → large, low noise → high)
3. **Multi-task learning**: Train single agent for multiple tracks/conditions
4. **Interpretability**: Visualize learned policies (attention maps, state importance)

### Long-term Vision

1. **Sim-to-real**: Validate learned policies on real robots or higher-fidelity simulators
2. **Multi-agent**: Competitive or cooperative multi-car scenarios
3. **Learning from demonstrations**: Combine TD3 with behavioral cloning
4. **Meta-learning**: Learn to adapt quickly to new environments

---

## References

- Fujimoto, S., Hoof, H., & Mnih, D. (2018). **Addressing Function Approximation Error in Actor-Critic Methods**. *ICML 2018*. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)
- Lillicrap, T., Hunt, J. J., et al. (2015). **Continuous Control with Deep Reinforcement Learning**. *ICLR 2016*. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Mnih, V., Badia, A. P., et al. (2016). **Asynchronous Methods for Deep Reinforcement Learning**. *ICML 2016*. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783)

---

## Contributing

Contributions are welcome! Areas of interest:

- Bug fixes and code cleanup
- Additional reward formulations
- New environment variants
- Visualization improvements
- Documentation enhancements

Please ensure code follows the existing style and includes docstrings.

---

## License

This project is released under the **MIT License**. See LICENSE file for details.

---

## Citation

If you use this project in research, please cite:

```bibtex
@software{td3_car_game,
  title = {TD3 Self-Driving Car: Autonomous Racing via Reinforcement Learning},
  author = {Aditya Gangwani},
  year = {2026},
  url = {https://github.com/adityagangwani30/TD3-Car-Game}
}
```

---

## Contact & Support

For questions, issues, or suggestions:
- Open a GitHub issue
- Contact the maintainer via email or social media

Happy learning! 🚀
