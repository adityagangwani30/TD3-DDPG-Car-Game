# TD3 Self-Driving Car

A small reinforcement learning project where a car learns to drive around a simple oval track using TD3, PyTorch, and Pygame.

## Overview

The agent observes:

- normalized car position `(x, y)`
- normalized speed
- normalized heading angle
- five ray sensors pointing around the car

The agent outputs:

- steering in `[-1, 1]`
- throttle in `[-1, 1]`, remapped to `[0, 1]` inside the environment

The environment is intentionally simple:

- episode ends if the car center goes off track
- episode also ends if the car stays nearly stationary for too long
- reward is `alive_bonus + speed_bonus - steering_penalty`
- crash gives a terminal penalty

Lap timing is tracked only for display in the HUD.

## Project Structure

```text
TD3 Car Game/
|-- main.py
|-- config.py
|-- environment.py
|-- car.py
|-- td3_agent.py
|-- replay_buffer.py
|-- train.py
|-- utils.py
|-- requirements.txt
|-- assets/
|   |-- track.png
|   `-- car.png
`-- models/
```

## Main Components

- `main.py`: starts Pygame, creates the environment and agent, and launches training
- `config.py`: central place for hyperparameters and environment settings
- `environment.py`: reset/step/render logic, reward calculation, done logic, and lap timing display
- `car.py`: simple top-down car physics and ray sensor casting
- `td3_agent.py`: actor, critic, and TD3 training step
- `replay_buffer.py`: fixed-size replay memory
- `train.py`: episode loop, replay insertion, network updates, and terminal logging
- `utils.py`: asset generation, track mask creation, and text drawing helpers

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Start training:

```bash
python main.py
```

## Training Output

The terminal logs a compact summary per episode:

```text
Episode     1 | Length   150 | Reward   +59.52 | Avg100   +59.52 | End max_steps
```

Where:

- `Length` is the number of steps in the episode
- `Reward` is the total episode reward
- `Avg100` is the moving average reward over the last 100 episodes
- `End` is the termination reason: `off_track`, `stuck`, or `max_steps`

By default, rendering is not shown every episode so training can run faster.

## Key Configuration

Important settings in `config.py`:

- `MAX_EPISODES`
- `MAX_STEPS_PER_EPISODE`
- `BUFFER_CAPACITY`
- `TRAINING_START`
- `BATCH_SIZE`
- `HIDDEN_DIM_1`, `HIDDEN_DIM_2`
- `RENDER_EVERY_EPISODES`
- `STUCK_SPEED_THRESHOLD`, `STUCK_STEP_LIMIT`

## Notes

- The TD3 core logic is standard and intentionally kept separate from the environment logic.
- Assets are generated automatically if they are missing.
- Best and last lap times persist across episode resets for display.
