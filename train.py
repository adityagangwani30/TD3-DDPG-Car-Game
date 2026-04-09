"""
train.py - Training loop for the TD3 self-driving car.

Orchestrates episode collection, replay-buffer storage, and TD3 network
updates while rendering every frame so the user can watch the agent learn.
"""

import os

import numpy as np

from config import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    MODEL_DIR,
    SAVE_MODEL_EVERY,
    TRAINING_START,
)
from environment import CarRacingEnv
from replay_buffer import ReplayBuffer
from td3_agent import TD3Agent


def _format_lap_time(value: float | None) -> str:
    """Format lap times consistently in terminal logs."""
    if value is None:
        return "--.--s"
    return f"{value:5.2f}s"


def train(env: CarRacingEnv, agent: TD3Agent):
    """Run the main training loop."""
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_reward = -float("inf")
    reward_history: list[float] = []

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            action = agent.select_action(state, add_noise=True)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.is_ready(TRAINING_START):
                agent.train(replay_buffer, BATCH_SIZE)

            env.render()
            state = next_state

            if done:
                break

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-100:])

        print(
            f"Episode {episode:>5d} | "
            f"Steps {step:>5d} | "
            f"Reward {episode_reward:>+8.2f} | "
            f"Laps {env.laps_completed:>2d} | "
            f"LastLap {_format_lap_time(env.last_lap_time)} | "
            f"BestLap {_format_lap_time(env.best_lap_time)} | "
            f"Avg100 {avg_reward:>+8.2f} | "
            f"Buffer {len(replay_buffer):>7d}"
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(MODEL_DIR, "td3_best.pth"))

        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(MODEL_DIR, f"td3_ep{episode}.pth"))

    print("\n[train] Training complete.")
    env.close()
