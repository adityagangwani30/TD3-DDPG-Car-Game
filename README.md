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

```bash
pip install -r requirements.txt
```

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
```bash
python main.py --mode demo --eval-episodes 3
```

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
