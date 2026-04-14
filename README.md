<<<<<<< HEAD
# TD3 Self-Driving Car

A beginner-friendly reinforcement learning project that trains an autonomous car to drive around a simple oval track using **TD3** (Twin Delayed Deep Deterministic Policy Gradient), **PyTorch**, and **Pygame**.

## 🚀 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adityagangwani30/TD3-Car-Game/blob/main/colab_demo.ipynb)

Use the notebook above for a one-click Colab experience. It clones this repository, installs dependencies, checks for GPU support, and runs a short demo so you can verify everything quickly.

Because Colab is headless, it does not open a normal desktop pygame window. Instead, the notebook runs the simulation and displays a saved preview frame inline.

### Quick Demo

If you want a short run locally, use demo mode:

```bash
python main.py --mode demo
```

This uses a saved checkpoint if one is available and finishes in a few minutes.

In Colab, the same demo mode runs headless and saves a preview image for inline display.

### Train From Scratch or Resume Training

By default, training starts from scratch:

```bash
python main.py
```

To continue from the latest saved checkpoint, use:

```bash
python main.py --mode train --resume
```

To resume from a specific checkpoint, use:

```bash
python main.py --mode train --checkpoint models/td3_ep900.pth
```

### Colab vs Local

- **Colab**: best for a fast demo, GPU checks, and a browser-based walkthrough
- **Local**: best for longer training runs and interactive rendering

## 🚀 Project Overview

This project simulates a small top-down driving environment where an RL agent learns to control a car using continuous steering and throttle commands.

It solves a simple but useful control problem:

- keep the car on the track
- avoid crashing or getting stuck
- complete laps as efficiently as possible

TD3 is used because it works well for **continuous action spaces**, which makes it a good fit for driving control.

## 🧠 About TD3 Algorithm

TD3 is an off-policy actor-critic algorithm designed for stable learning in continuous-control tasks.

### Core ideas

- **Actor**: learns the policy, meaning it decides what action to take in each state.
- **Critic**: estimates how good an action is by predicting the Q-value.
- **Target networks**: slower-moving copies of the actor and critic used for stable training.
- **Delayed updates**: the actor is updated less often than the critic.
- **Target policy smoothing**: small noise is added to target actions to reduce overestimation.

### Why TD3 instead of DDPG?

TD3 improves on DDPG by reducing common training problems such as Q-value overestimation and unstable policy updates. In practice, this usually means:

- more stable learning
- less noisy value estimates
- better performance on continuous control tasks

## 🎮 Environment & Game Description

The environment is a simple 2D oval track viewed from above.

### How it works

- The car starts at a fixed position and heading.
- The agent controls two continuous actions:
	- **steering** in `[-1, 1]`
	- **throttle** in `[0, 1]` for forward motion only
- The car moves with a lightweight physics model.
- Ray sensors detect how far the track boundary is in front of the car.

### State space

The observation contains:

- normalized `x` position
- normalized `y` position
- normalized speed
- normalized heading angle
- 3 normalized sensor distances

Total state size: **9 dimensions**.

### Action space

The agent outputs 2 continuous values:

- `steering`: left/right turning control
- `throttle`: forward acceleration only

### Goals and constraints

- Stay on the road
- Move forward smoothly
- Complete laps
- Avoid getting stuck

The project intentionally stays simple so the RL problem remains easy to understand and debug.

## 🏆 Reward Function

The reward is designed to encourage safe forward driving and lap completion, not just raw speed.

Current structure:

- small positive reward for surviving each step
- extra reward for moving instead of remaining stuck
- large bonus for completing a lap
- penalty for sharp steering
- negative reward when the car goes off track or gets stuck

### Why this structure?

The reward needs to avoid “reward hacking,” where the agent learns to collect reward without actually driving well.

This design keeps the objective clear:

- drive forward
- stay on the track
- finish laps

## 📊 Metrics & Evaluation

The project tracks training metrics to make learning easier to inspect.

### Metrics tracked

- episode reward
- average reward over recent episodes
- episode length
- lap completions
- collision/off-track count
- average speed
- steering smoothness
- exploration noise level
- replay buffer size

### How performance is evaluated

Performance is judged using:

- total episode reward
- number of completed laps
- crash rate
- average episode length
- stability of learning over time

Training logs are written in JSON Lines format so they can be plotted or analyzed later.

## 🛠️ Tech Stack

- **Python** - main programming language
- **PyTorch** - neural networks and RL training
- **Pygame** - environment rendering and visualization
- **NumPy** - numerical operations and replay buffer storage

## 📂 Project Structure

```text
td3-car-game/
├── main.py              # Entry point for training and evaluation
├── train.py             # Training loop and evaluation logic
├── environment.py       # Driving environment, reward function, rendering
├── car.py               # Car physics and sensor raycasting
├── td3_agent.py         # Actor, critic, and TD3 training logic
├── replay_buffer.py     # Experience replay memory
├── config.py            # Hyperparameters and project settings
├── utils.py             # Asset generation and helper functions
├── lap_timer.py         # Lap timing and finish-line tracking
├── metrics_tracker.py   # Training metrics and logging
├── plot_metrics.py      # Plot training logs
├── eval_models.py       # Compare saved models
├── assets/              # Generated images for track and car
├── models/              # Saved checkpoints
├── logs/                # Training logs and plots
└── requirements.txt     # Python dependencies
```

### File roles

- `main.py`: starts the app in training or evaluation mode
- `train.py`: collects experience, trains the TD3 agent, saves checkpoints
- `environment.py`: simulates the car-driving task and computes rewards
- `car.py`: handles movement, steering, and raycasting sensors
- `td3_agent.py`: defines actor/critic networks and TD3 updates
- `replay_buffer.py`: stores past transitions for off-policy learning
- `config.py`: central place for hyperparameters and environment settings
- `utils.py`: generates assets and handles shared helper functions
- `lap_timer.py`: keeps lap time logic separate from the environment
- `metrics_tracker.py`: records and summarizes training statistics
- `plot_metrics.py`: creates charts from saved logs
- `eval_models.py`: runs benchmark-style evaluation across checkpoints

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd td3-car-game
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
=======
# TD3 Self-Driving Car for Research on Reward Design and Sensor Robustness

This repository presents a reinforcement learning framework for autonomous driving on a 2-D racing track using **Twin Delayed Deep Deterministic Policy Gradient (TD3)**. The project is designed to be beginner-friendly for implementation and reproducible enough for research reporting.

The core focus is adaptive decision-making under changing reward formulations and sensor uncertainty. The codebase supports structured experiments, per-experiment logging, model isolation, and plotting for comparative analysis.

## Project Description

The agent observes vehicle state and ray-based distance sensors, and outputs continuous steering/throttle actions. Training is based on TD3 with replay buffers, delayed policy updates, target smoothing, and gradient clipping.

Key characteristics:
- Continuous control with TD3 (actor-critic)
- Multiple reward modes for ablation studies
- Sensor-noise injection for robustness analysis
- Reproducibility controls via deterministic seeding
- Structured experiment runner with isolated outputs

## Research Objective

This project studies how **reward shaping** and **sensor noise** affect policy learning quality, safety, and stability in an autonomous-driving task.

Primary questions:
- How much does reward design impact convergence speed and driving quality?
- How robust is TD3 performance under progressively noisier sensor readings?
- Which reward-noise combinations provide the best trade-off between reward, crash rate, and lap completion?

## Experiment Setup

Experiments are configured in `config.py` through a grid of reward modes and sensor-noise levels.

Reward modes:
- `basic`: alive bonus + crash penalty (minimal shaping)
- `shaped`: progress-oriented shaping with movement/lap incentives
- `modified`: enhanced shaped reward with additional stability bias

Sensor noise levels:
- `0.00`
- `0.02`
- `0.05`

Total combinations:
- 3 reward modes x 3 noise levels = **9 experiments**

Each experiment runs with:
- Unique experiment identifier
- Dedicated model directory
- Dedicated log directory
- Independent seed configuration

## Project Structure

```text
TD3-Car-Game/
├── main.py                  # Main CLI entry: train / eval / demo
├── run_experiments.py       # Sequential research experiment runner
├── config.py                # Global configuration, experiments, hyperparameters
├── environment.py           # RL environment, reward modes, runtime noise control
├── car.py                   # Vehicle dynamics and sensor simulation
├── td3_agent.py             # TD3 actor/critic implementation
├── replay_buffer.py         # Off-policy replay memory
├── train.py                 # Training/evaluation loops and checkpoint logic
├── metrics_tracker.py       # Per-episode metrics logging (JSONL)
├── plot_metrics.py          # Per-experiment and cross-experiment plotting
├── eval_models.py           # Evaluate and compare trained checkpoints
├── lap_timer.py             # Lap timing and finish-line crossing detection
├── utils.py                 # Seeding, asset generation, pygame initialization
├── requirements.txt         # Python dependencies
├── assets/                  # Generated/loaded sprites and track assets
├── logs/                    # Experiment logs (JSONL + plots)
└── models/                  # Saved checkpoints (experiment-specific folders)
```

## Observation and Action Spaces

State vector (`4 + NUM_SENSORS`):
- Normalized position `(x, y)`
- Normalized speed
- Normalized heading angle
- Sensor distances (normalized ray distances)

Action vector (`2` continuous values):
- Steering in `[-1, 1]`
- Throttle in `[0, 1]`

## Installation
>>>>>>> dd7ca13774afbb2769a861a5c54b1c48e33d362d

```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
### 4. Run the project once

The first run will generate the track and car assets automatically if they are missing.

## ▶️ How to Run

### Train the agent

```bash
python main.py
```

### Evaluate a trained checkpoint

```bash
python main.py --mode eval --checkpoint models/td3_best.pth
```

### Compare multiple saved models

```bash
python eval_models.py --episodes 10
```

### Plot training metrics

=======
## Quick Start

Train:
```bash
python main.py --mode train
```

Evaluate:
```bash
python main.py --mode eval --eval-episodes 10
```

Demo playback:
>>>>>>> dd7ca13774afbb2769a861a5c54b1c48e33d362d
```bash
python main.py --mode demo --eval-episodes 3
```

<<<<<<< HEAD
## 📈 Results / Observations

Because this is a basic RL project, results depend on random seeds, training duration, and reward tuning. In general, a successful run should show:

- rising average episode reward over time
- fewer crashes as training progresses
- more frequent lap completions
- smoother steering behavior

### Limitations

- the physics model is intentionally simple
- the track is basic and not highly realistic
- the agent only sees ray sensor distances, not a full map
- results may vary significantly between runs

## ✨ Features

- TD3 implementation for continuous control
- simple top-down driving environment
- forward-only car motion
- speed-dependent steering
- noisy ray sensors for robustness
- reward shaping for safe racing behavior
- model checkpointing
- evaluation mode
- metrics logging and plotting
- lightweight and beginner-friendly structure

## 🔮 Future Improvements

Possible next steps that stay realistic and simple:

- add more track layouts
- add a clearer difficulty selector
- compare reward variations systematically
- add a small evaluation dashboard
- store training curves automatically after each run

## 🤝 Contributing

Contributions are welcome if they keep the project simple and readable.

Good contribution ideas:

- improve documentation
- add new plots for training analysis
- refine the reward function
- test different physics constants

## 📜 License

No explicit license file is included in the repository at the moment. If you plan to publish or share the project, add a license that matches your intended usage.

## Notes

- The code is intentionally kept beginner-friendly rather than highly optimized.
- Assets are generated automatically when missing.
- Training is easier to inspect if rendering is disabled during long runs.
=======
Headless execution (server/Colab):
```bash
python main.py --mode train --headless
```

## How to Run Experiments

Run all configured experiments sequentially:
```bash
python run_experiments.py
```

Headless experiment sweep:
```bash
python run_experiments.py --headless
```

Validation run (short, partial):
```bash
python run_experiments.py --max-experiments 2 --max-episodes 1 --max-steps 10 --headless
```

Outputs are isolated by experiment tag (example: `R2_N3`):
- Logs: `logs/R2_N3/training_log.jsonl`
- Models: `models/R2_N3/*.pth`

## Results and Evaluation

### Metrics Logged

Each episode logs research-relevant metrics including:
- `reward_total`
- `reward_rolling_avg_100`
- `collisions` (crashes)
- `laps_completed`
- `length` (episode steps)
- experiment metadata (`experiment_name`, `reward_mode`, `sensor_noise_std`, `seed`)

### Plotting and Comparison

Generate plots for individual experiments and comparisons:
```bash
python plot_metrics.py --log-dir logs --compare
```

Or target selected experiments:
```bash
python plot_metrics.py --log-dir logs --experiments R1_N1 R2_N1 R3_N1 --compare
```

Generated figures include:
- Reward vs Episodes
- Crash Rate vs Episodes
- Laps vs Episodes
- Cross-experiment comparison plots

Expected trends (typical):
- Reward increases with training (after early exploration)
- Crash rate decreases as policy stabilizes
- Laps completed increase as trajectory control improves
- Higher sensor noise generally reduces stability unless reward shaping is robust

## Configuration Notes

Important config groups in `config.py`:
- Training budget and optimization hyperparameters
- Reward constants and reward mode definitions
- Sensor layout and noise controls
- Experiment grid (`EXPERIMENTS`)
- Reproducibility defaults (`DEFAULT_SEED`, `EXPERIMENT_BASE_SEED`)

## Troubleshooting

Slow training:
- Disable frequent rendering via `RENDER_DURING_TRAINING = False`

No display environment:
- Use `--headless` in `main.py`, `run_experiments.py`, or `eval_models.py`

Evaluation checkpoint issues:
- Pass `--checkpoint path/to/model.pth` explicitly
- Or place default checkpoints in `models/`

## Future Work

Research-aligned extensions:
- Domain randomization (track geometry and friction variation)
- Curriculum learning across increasing noise/difficulty schedules
- Multi-sensor fusion beyond sparse ray casting
- Statistical significance testing across repeated seeds
- Sim-to-real inspired robustness benchmarks and perturbation suites
- Extended baselines (e.g., SAC/PPO) for comparative studies

## License

MIT License.

## Notes

- TD3 core logic is intentionally modular and isolated from environment code.
- Assets are auto-generated when missing.
- The repository is suitable for both learning-oriented implementation and experimental reporting workflows.
>>>>>>> dd7ca13774afbb2769a861a5c54b1c48e33d362d
