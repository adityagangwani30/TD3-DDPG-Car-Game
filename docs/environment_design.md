# Environment Design

This document explains the custom 2D racing environment used to train and evaluate the TD3 and DDPG agents.

---

## Overview

The environment is a **2D elliptical oval racing circuit** built with Pygame. It provides a Gym-style API (`reset()`, `step()`, `render()`) and simulates a top-down car with physics, raycasting sensors, and lap detection.

The environment is implemented across three files:

- `environment.py` — Main environment class (`CarRacingEnv`)
- `car.py` — Car physics, sensors, and state construction
- `lap_timer.py` — Lap crossing detection and timing

---

## The Track

- **Shape:** Elliptical oval (not a circle — the horizontal and vertical radii differ)
- **Screen size:** 1200 × 800 pixels
- **Track center:** (600, 400)
- **Outer ellipse:** 480 × 320 pixel radii
- **Inner ellipse:** 320 × 180 pixel radii
- **Road surface:** Dark gray (60, 60, 60) — used for the track mask
- **Grass:** Green (34, 139, 34) — off-track areas
- **Borders:** White elliptical outlines for visual clarity
- **Finish line:** Vertical checkered line at the track center x-coordinate, spanning the lower half of the track

The track image and car sprite are **programmatically generated** at runtime if they do not already exist (see `utils.py`). No external image assets are required.

---

## State Space (7-dimensional)

The agent observes a 7-dimensional continuous vector at every time step:

| Index | Feature | Range | How It Is Computed |
|-------|---------|-------|-------------------|
| 0 | Position X | [0, 1] | `car.x / SCREEN_WIDTH` (1200) |
| 1 | Position Y | [0, 1] | `car.y / SCREEN_HEIGHT` (800) |
| 2 | Speed | [0, 1] | `car.speed / CAR_MAX_SPEED` (8.0) |
| 3 | Heading | [0, 1] | `car.angle / 360.0` |
| 4 | Sensor Left | [0, 1] | Distance to boundary at −45° from heading |
| 5 | Sensor Front | [0, 1] | Distance to boundary at 0° (straight ahead) |
| 6 | Sensor Right | [0, 1] | Distance to boundary at +45° from heading |

All values are normalized to approximately [0, 1], which is beneficial for neural network training.

### Sensor Details

The three raycasting sensors emit rays from the car's center:

- **Angles:** −45°, 0°, +45° relative to the car's heading
- **Maximum range:** 200 pixels
- **Step size:** Every 2 pixels (for speed; ~1% accuracy trade-off)
- **Hit detection:** A ray terminates when it reaches an off-track pixel (non-road color) or exits the screen bounds
- **Normalization:** `hit_distance / SENSOR_MAX_DIST` → values close to 0 mean the boundary is near; values close to 1 mean the road ahead is clear

---

## Action Space (2-dimensional, continuous)

| Index | Action | Range | Description |
|-------|--------|-------|-------------|
| 0 | Steering | [−1, +1] | Negative = left, positive = right |
| 1 | Throttle | [0, +1] | Acceleration magnitude (no reverse) |

The actor network outputs values in [−1, +1] via tanh. The throttle is clipped to [0, +1] by the environment's `_parse_action()` method.

---

## Car Physics

The car uses a simplified **bicycle model** implemented in `car.py`:

- **Speed-dependent steering:** At higher speeds, the car turns less sharply. Controlled by `CAR_TURN_SPEED_FACTOR = 0.5`:
  ```
  speed_factor = 1.0 - (speed / max_speed) * 0.5
  adjusted_steering = steering * speed_factor
  ```

- **Acceleration:** `speed += throttle * CAR_ACCELERATION` (0.3 per step)
- **Friction:** `speed -= CAR_FRICTION * speed` (5% per step)
- **Max speed:** 8.0 units/step
- **Position update:** Forward motion based on heading angle:
  ```
  x += speed * cos(angle)
  y -= speed * sin(angle)
  ```

- **Car dimensions:** 20 × 40 pixels (width × height)
- **Starting position:** (600, 620), angle 0°

---

## Episode Termination

An episode ends under any of three conditions:

### 1. Off-Track (Crash)

The car's center pixel is checked against the track mask. If `track_mask[x, y]` is False (the pixel is not road-colored), the episode ends with `termination_reason = "off_track"`.

### 2. Stuck

If the car's speed stays below `STUCK_SPEED_THRESHOLD` (0.15) for `STUCK_STEP_LIMIT` (180) consecutive steps, the episode ends with `termination_reason = "stuck"`. This prevents the agent from learning to sit still and collect survival rewards.

### 3. Max Steps

Each episode has a maximum of `MAX_STEPS_PER_EPISODE` (600) steps. If the agent survives to the end without crashing or getting stuck, the episode simply ends — this is handled by the training loop in `train.py`, not by the environment itself. (The 600-step horizon provides ample headroom over the theoretical minimum of 382 steps needed to complete a full 2068.8-pixel centerline lap at full acceleration).

---

## Lap Completion Logic

Lap detection is handled by `LapTimer` in `lap_timer.py`:

1. A **finish line** spans vertically from the inner to outer track boundary at `x = TRACK_CENTER_X` (600)
2. The car must cross this line from **left to right** (`prev_x < 600 <= curr_x`)
3. Both the previous and current y-positions must be within the finish line's y-range
4. At least `MIN_LAP_STEPS` (120) steps must have elapsed since the last crossing — this prevents false positives at the start of an episode or from oscillating near the line

When a lap is completed:
- Lap time is recorded (steps / FPS)
- Best lap time is updated
- Lap count increments
- Reward bonuses are applied (depending on the reward mode)

---

## Why This Environment?

This environment is useful for testing autonomous decision-making because:

- **Continuous control challenge:** The agent must produce precise steering and throttle values, not pick from discrete options
- **Safety-critical dynamics:** Crashing (going off-track) is a clear failure mode with real consequences
- **Multi-objective tension:** Maximizing speed increases reward but also increases crash risk
- **Sensor noise testing:** The raycasting sensors can be injected with configurable noise, testing robustness
- **Controlled complexity:** The oval track is simple enough to learn in thousands of episodes but complex enough to reveal differences between algorithms and reward designs
- **Deterministic physics:** Given the same inputs, the environment produces the same outputs (modulo sensor noise), enabling reproducible experiments

---

## Further Reading

- [reward_system.md](reward_system.md) — The four reward modes used in this environment
- [observation_noise.md](observation_noise.md) — How sensor noise is applied
- [td3_explanation.md](td3_explanation.md) — The agent that interacts with this environment
