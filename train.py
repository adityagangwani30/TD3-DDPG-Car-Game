"""
train.py – Training loop for the TD3 self-driving car.

Orchestrates episode collection, replay-buffer storage, and TD3 network
updates while rendering every frame so the user can watch the agent learn.
"""

import os
import numpy as np

from config import (
    MAX_EPISODES, MAX_STEPS_PER_EPISODE,
    BATCH_SIZE, BUFFER_CAPACITY, TRAINING_START,
    SAVE_MODEL_EVERY, MODEL_DIR,
)
from environment import CarRacingEnv
from td3_agent import TD3Agent
from replay_buffer import ReplayBuffer


def train(env: CarRacingEnv, agent: TD3Agent):
    """Run the main training loop.

    Args:
        env:   Initialised CarRacingEnv.
        agent: TD3Agent instance.
    """
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_reward = -float("inf")
    reward_history: list[float] = []

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # Select action (with exploration noise during training)
            action = agent.select_action(state, add_noise=True)

            # Environment step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)

            # Train once the buffer has enough samples
            if replay_buffer.is_ready(TRAINING_START):
                agent.train(replay_buffer, BATCH_SIZE)

            # Render the frame
            env.render()

            state = next_state

            if done:
                break

        # ----- End of episode bookkeeping -----
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-100:])

        print(
            f"Episode {episode:>5d} | "
            f"Steps {step:>5d} | "
            f"Reward {episode_reward:>+8.2f} | "
            f"Avg100 {avg_reward:>+8.2f} | "
            f"Buffer {len(replay_buffer):>7d}"
        )

        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(MODEL_DIR, "td3_best.pth"))

        # Periodic checkpoint
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(MODEL_DIR, f"td3_ep{episode}.pth"))

    print("\n[train] Training complete.")
    env.close()
