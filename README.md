# TD3 Self-Driving Car

A reinforcement learning simulation in which a car learns to drive around a racing circuit autonomously using the **Twin Delayed Deep Deterministic policy gradient (TD3)** algorithm. The simulation is built with **PyTorch** (neural networks) and **Pygame** (real-time visualisation).

---

## Table of Contents

1. [Project Description](#project-description)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Running the Simulation](#running-the-simulation)
7. [Configuration](#configuration)
8. [Future Improvements](#future-improvements)

---

## Project Description

The car starts with no knowledge of the track and must learn entirely through trial-and-error. At each simulation step the agent observes:

- its normalised `(x, y)` position on the screen
- its current speed and heading angle
- five **raycasting sensor** readings that measure the distance to the nearest track boundary in five directions

It then outputs a continuous **steering** value and a **throttle** value. Over thousands of episodes the TD3 algorithm refines the agent's neural-network policy until the car can navigate the circuit smoothly without crashing.

---

## Features

| Feature | Details |
|---|---|
| **TD3 reinforcement learning** | Off-policy actor-critic with clipped double-Q, delayed policy updates, and target policy smoothing |
| **Top-down racing simulation** | Real-time Pygame window at 60 FPS |
| **Raycasting sensor system** | Five sensors cast rays from the car; cyan lines visualise each ray on screen |
| **Lap timing system** | Step counter and cumulative episode reward tracked per lap |
| **Fastest lap tracking** | Best episode reward saved automatically as `models/td3_best.pth` |
| **Real-time HUD** | Displays episode number, step count, episode reward, speed, and heading angle |
| **Programmatic asset generation** | Track image and car sprite are auto-generated on first run if not present |
| **Periodic checkpointing** | Model weights saved every N episodes (configurable) |
| **GPU support** | Automatically uses CUDA if available, falls back to CPU |

---

## Project Structure

```
TD3 Car Game/
│
├── main.py            # Entry point – initialises Pygame, creates environment & agent, starts training
├── config.py          # Central configuration: all hyperparameters, physics constants, file paths
├── environment.py     # Gym-style RL environment: reset(), step(), render(), reward computation
├── car.py             # Car physics model + raycasting sensor system
├── td3_agent.py       # TD3 algorithm: Actor, Critic networks and training logic
├── replay_buffer.py   # Fixed-size circular experience replay buffer
├── train.py           # Training loop: episode rollouts, buffer storage, agent updates
├── utils.py           # Helpers: track generation, road-mask building, Pygame text rendering
│
├── assets/
│   ├── track.png      # Auto-generated oval racing circuit (created on first run)
│   └── car.png        # Auto-generated top-down car sprite (created on first run)
│
├── models/
│   ├── td3_best.pth   # Weights of the best-performing episode so far
│   └── td3_ep<N>.pth  # Periodic checkpoint every SAVE_MODEL_EVERY episodes
│
└── requirements.txt   # Python dependencies
```

### File Summaries

| File | Responsibility |
|---|---|
| `main.py` | Calls `pygame.init()`, detects GPU, wires the environment and agent together, then delegates to `train()`. |
| `config.py` | Single source of truth for every tunable parameter. Edit this file to change learning rates, sensor range, reward weights, etc. |
| `environment.py` | Implements `CarRacingEnv` with a `reset() / step() / render()` interface compatible with standard RL workflows. Handles reward shaping and the HUD overlay. |
| `car.py` | Implements `Car`: bicycle-model physics (speed, steering, friction) plus per-frame raycasting across five angular directions. |
| `td3_agent.py` | Defines the `Actor` (policy) and `Critic` (twin Q-networks) PyTorch modules and the `TD3Agent` class that owns the training step. |
| `replay_buffer.py` | `ReplayBuffer` stores `(s, a, r, s', done)` transitions in pre-allocated NumPy arrays and supports uniform random sampling. |
| `train.py` | Outer `train()` function: episode loop, transition collection, agent updates, logging, and model saving. |
| `utils.py` | Procedural track and car-sprite generation, road-pixel mask computation, and Pygame text helpers. |

---

## How It Works

### TD3 Algorithm

TD3 (Fujimoto et al., 2018) is an off-policy actor-critic algorithm for **continuous action spaces**. It addresses the over-estimation bias present in DDPG through three key techniques:

1. **Clipped Double-Q Learning**  
   Two independent critic networks (`Q1`, `Q2`) estimate the value of the next state. The *minimum* of their predictions is used as the training target, preventing over-optimistic value estimates.

2. **Delayed Policy Updates**  
   The actor (policy) network is updated less frequently than the critics (every `POLICY_DELAY` critic steps). This keeps the value estimates stable before the policy learns from them.

3. **Target Policy Smoothing**  
   Small clipped Gaussian noise is added to the target action when computing the critic target. This regularises the Q-function and prevents the policy from exploiting narrow peaks in the value landscape.

### Training Flow

```
┌─────────────────────────────────────────────────────────┐
│  Episode loop                                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Step loop                                        │  │
│  │  1. Agent selects action  (Actor + noise)         │  │
│  │  2. Environment executes action                   │  │
│  │     a. Update car physics                         │  │
│  │     b. Cast raycasting sensors                    │  │
│  │     c. Check collision → done flag                │  │
│  │     d. Compute reward                             │  │
│  │  3. Store transition in ReplayBuffer              │  │
│  │  4. Sample mini-batch → update Critic networks    │  │
│  │  5. Every POLICY_DELAY steps → update Actor       │  │
│  │  6. Soft-update target networks                   │  │
│  │  7. Render frame                                  │  │
│  └───────────────────────────────────────────────────┘  │
│  Log episode stats; save model if best / checkpoint     │
└─────────────────────────────────────────────────────────┘
```

### State Vector (9 values)

| Index | Value | Normalisation |
|---|---|---|
| 0 | Car X position | ÷ screen width |
| 1 | Car Y position | ÷ screen height |
| 2 | Speed | ÷ `CAR_MAX_SPEED` |
| 3 | Heading angle | ÷ 360° |
| 4–8 | Sensor distances (×5) | ÷ `SENSOR_MAX_DIST` |

### Action Vector (2 values)

| Index | Value | Range |
|---|---|---|
| 0 | Steering | `[-1, 1]` (−1 = hard left, +1 = hard right) |
| 1 | Throttle | `[-1, 1]` in network output → remapped to `[0, 1]` in environment |

### Reward Function

| Condition | Reward |
|---|---|
| Staying on track (per step) | `+REWARD_ALIVE` (default `+0.1`) |
| Forward speed | `+REWARD_VELOCITY_SCALE × (speed / max_speed)` |
| Excessive steering | `−REWARD_STEERING_PENALTY × |steering|` |
| Crash (off track) | `REWARD_CRASH` (default `−10.0`) |

---

## Installation

**Prerequisites:** Python 3.10+

```bash
pip install torch pygame numpy
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

> **GPU acceleration:** If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch first by following the instructions at [pytorch.org](https://pytorch.org/get-started/locally/). The simulation will automatically use the GPU when available.

---

## Running the Simulation

```bash
python main.py
```

On the **first run**, the track image and car sprite are generated automatically and saved to the `assets/` directory.

The **Pygame window** opens immediately and you can watch the agent learn in real time. The terminal also prints a log line after every episode:

```
Episode     1 | Steps   312 | Reward   -89.23 | Avg100   -89.23 | Buffer     312
Episode     2 | Steps    54 | Reward   -51.10 | Avg100   -70.17 | Buffer     366
...
```

Press **Ctrl+C** or close the window to stop training gracefully.

---

## Configuration

All tuneable parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|---|---|---|
| `MAX_EPISODES` | `5000` | Total training episodes |
| `MAX_STEPS_PER_EPISODE` | `2000` | Maximum steps before forced episode end |
| `BATCH_SIZE` | `256` | Mini-batch size for network updates |
| `BUFFER_CAPACITY` | `1 000 000` | Maximum replay buffer size |
| `GAMMA` | `0.99` | Discount factor |
| `TAU` | `0.005` | Soft target update rate (Polyak averaging) |
| `ACTOR_LR` / `CRITIC_LR` | `3e-4` | Adam learning rates |
| `POLICY_DELAY` | `2` | Critic updates per actor update |
| `SENSOR_MAX_DIST` | `200` | Maximum sensor ray length (pixels) |
| `SAVE_MODEL_EVERY` | `100` | Checkpoint frequency (episodes) |

---

## Future Improvements

- **Better track generation** – Replace the simple oval with a realistic F1-style circuit featuring hairpins, chicanes, and varying corner radii
- **Improved reward shaping** – Add checkpoint-based progress rewards to guide the agent more efficiently
- **Multiple cars** – Run several agents in parallel to speed up experience collection
- **Curriculum learning** – Start with a wide, easy track and gradually narrow it as the agent improves
- **Export and replay** – Save trained models and replay recorded episodes for analysis
- **Lap timing** – Track and display actual lap times rather than step counts
- **Observation enhancement** – Include angular velocity and look-ahead track curvature in the state vector
- **Hyperparameter tuning** – Integrate Optuna or Ray Tune for automated hyperparameter search

---

## References

- Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML 2018. [[arXiv:1802.09477]](https://arxiv.org/abs/1802.09477)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Pygame Documentation](https://www.pygame.org/docs/)
