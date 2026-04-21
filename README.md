# TD3 and DDPG Self-Driving Car Simulation using Reinforcement Learning

Train and evaluate continuous-control RL agents in a custom 2D car racing environment with reproducible reward and sensor-noise experiments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-Car-Game/blob/main/colab_demo.ipynb)

## Overview
This project is a practical reinforcement learning playground for autonomous driving concepts. A virtual car learns to steer and throttle on an oval track using deep RL, with support for both TD3 and DDPG.

The core problem is continuous control under imperfect sensing. Instead of choosing from fixed actions, the agent must output smooth steering and acceleration values at each step. The repository is designed to help you understand how reward design and sensor noise affect learning behavior, stability, and final performance.

Why this project is interesting:
- It combines algorithm implementation and experimentation in one codebase.
- It is easy to run locally or in Colab.
- It includes structured experiments you can reproduce, compare, and extend.

## Features
- TD3 and DDPG agent implementations in PyTorch.
- Continuous-control car racing environment with sensor-based state input.
- Reward shaping system with 4 modes (`R1` to `R4`).
- Sensor-noise experiments with 3 levels (`N1` to `N3`).
- Full experiment grid across two algorithms (24 total runs).
- Training logs, model checkpoints, and plotting utilities.
- Headless execution support for cloud/Kaggle/CI-style environments.
- Colab notebook for fast setup and demonstration workflow.

## Tech Stack
- Python
- PyTorch
- NumPy
- Pygame
- Matplotlib

## Project Structure
- `main.py`: Main entry point for training, evaluation, and demo modes.
- `train.py`: Core training and evaluation loops.
- `td3_agent.py`: TD3 actor-critic implementation.
- `ddpg_agent.py`: DDPG actor-critic implementation.
- `environment.py`: Car racing environment, dynamics, and reward handling.
- `run_experiments.py`: Sequential experiment runner for reward/noise combinations.
- `plot_metrics.py`: Generates reward, crash, and comparison plots from logs.
- `logs/`: Per-run metrics logs (`training_log.jsonl`).
- `models/`: Saved checkpoints per algorithm and experiment.
- `results/`: Generated plots and analysis outputs.
- `colab_demo.ipynb`: Notebook-based setup and run flow.

## How It Works
### State Space
At each step, the agent receives a compact state vector containing:
- Car position and heading (normalized).
- Current speed.
- Distance readings from track sensors.

This gives the policy enough context to stay on track and move efficiently.

### Action Space
The policy outputs two continuous actions:
- Steering in `[-1, 1]` (left/right control).
- Throttle in `[0, 1]` (forward acceleration).

This mirrors real continuous driving behavior better than discrete actions.

### Reward System
The reward function balances progress and control quality:
- Positive signal for staying alive and moving forward.
- Bonus for lap completion.
- Penalties for collisions/off-track behavior.
- Steering/behavior shaping terms depending on reward mode.

You can switch between `R1` to `R4` to test how reward design changes learning.

### Training Loop
A standard off-policy RL loop is used:
1. Reset environment and roll out an episode.
2. Store transitions in replay buffer.
3. Sample mini-batches and update actor/critic networks.
4. Save logs and checkpoints periodically.
5. Evaluate trends using plots and summary metrics.

## Running the Project
### A. Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### B. Train
```bash
python main.py --algo td3 --mode train
```

Useful options:
```bash
python main.py --algo td3 --mode train --headless
python main.py --algo td3 --mode train --max-episodes 500
python main.py --algo ddpg --mode train
```

### C. Run Experiments
```bash
# TD3: runs the configured reward/noise grid
python run_experiments.py --algo td3 --max-episodes 5000 --headless

# DDPG: run the same grid for comparison
python run_experiments.py --algo ddpg --max-episodes 5000 --headless
```

### D. Plot Results
```bash
python plot_metrics.py --compare-algos
```

Tip: if your local CLI options differ by branch/version, run:
```bash
python plot_metrics.py -h
```

## Experiments
The experiment design is a grid over reward shaping and sensor noise:

| Dimension | Variants | Description |
|---|---|---|
| Reward Mode | `R1`, `R2`, `R3`, `R4` | Different reward shaping strategies |
| Sensor Noise | `N1`, `N2`, `N3` | Increasing observation noise levels |

Per algorithm:
- 4 reward modes x 3 noise levels = 12 experiments.

Across both TD3 and DDPG:
- 12 x 2 = 24 total experiments.

Experiment outputs are isolated by algorithm and configuration under `logs/` and `models/`.

## Results (Simplified)
Typical observations from this setup:
- TD3 is usually more stable during long training runs.
- DDPG can learn well but is often more sensitive to noise and hyperparameters.
- Reward shaping has a major effect on convergence speed and policy smoothness.
- Higher sensor noise generally increases variance and crash risk.

Your exact rankings can vary by seed and training budget, which is why the structured experiment pipeline is important.

## Why This Project Matters
This repository is useful for building real RL engineering skills:
- Understand continuous control in a hands-on environment.
- Learn how algorithm choice changes behavior (TD3 vs DDPG).
- Practice experiment design, reproducibility, and result interpretation.
- Build a portfolio project that demonstrates practical ML implementation, not just theory.

## Future Improvements
- Add Soft Actor-Critic (SAC) for stronger entropy-regularized baselines.
- Expand environment complexity (track variants, dynamic obstacles, weather/noise profiles).
- Add multi-agent racing scenarios.
- Add richer evaluation dashboards and experiment tracking integrations.
- Provide preconfigured benchmark scripts for standardized comparisons.

## Contribution
Contributions are welcome.

- Open an issue for bugs, ideas, or improvements.
- Submit a pull request for fixes and new features.
- Suggestions for better experiments and visualizations are especially appreciated.

## License
Check repository licensing details before external reuse or redistribution.
