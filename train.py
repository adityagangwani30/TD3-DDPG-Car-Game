"""
train.py - Training loop for the TD3 self-driving car.

Orchestrates episode collection, replay-buffer storage, TD3 network updates,
exploration noise decay, and comprehensive metrics tracking.
"""

import os

import numpy as np

from config import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    EXPLORATION_NOISE,
    EXPLORATION_NOISE_DECAY,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    MODEL_DIR,
    RENDER_DURING_TRAINING,
    RENDER_EVERY_EPISODES,
    SAVE_MODEL_EVERY,
    TRAINING_START,
)
from environment import CarRacingEnv
from metrics_tracker import MetricsTracker
from replay_buffer import ReplayBuffer
from td3_agent import TD3Agent


def _should_render_episode(episode: int) -> bool:
    """Return True when this episode should be rendered."""
    if RENDER_EVERY_EPISODES <= 0:
        return False
    return episode == 1 or episode % RENDER_EVERY_EPISODES == 0


def train(env: CarRacingEnv, agent: TD3Agent):
    """Run the main training loop with exploration decay and metrics tracking."""
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    metrics = env.metrics or MetricsTracker()
    os.makedirs(MODEL_DIR, exist_ok=True)

    best_reward = -float("inf")
    best_reward_per_100 = -float("inf")
    reward_history = []
    exploration_noise = EXPLORATION_NOISE

    for episode in range(1, MAX_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        termination_reason = "max_steps"
        render_enabled = _should_render_episode(episode) and not RENDER_DURING_TRAINING

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            # Use decaying exploration noise
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.is_ready(TRAINING_START):
                agent.train(replay_buffer, BATCH_SIZE)

            env.render(enabled=render_enabled, limit_fps=False)
            state = next_state

            if done:
                termination_reason = info["termination_reason"]
                break

        # Decay exploration noise
        exploration_noise *= EXPLORATION_NOISE_DECAY
        exploration_noise = max(exploration_noise, 0.01)  # Minimum noise floor

        reward_history.append(episode_reward)
        avg_reward_100 = np.mean(reward_history[-100:])

        # Compute episode summary for metrics
        episode_summary = metrics.get_episode_summary(episode)
        episode_summary["exploration_noise"] = exploration_noise
        episode_summary["replay_buffer_size"] = len(replay_buffer)
        metrics.log_episode(episode_summary)

        # Print summary
        metrics.print_summary(episode, episode_summary, avg_reward_100)

        # Save best model by individual episode reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(MODEL_DIR, "td3_best.pth"))

        # Save best model by rolling 100-episode average
        if avg_reward_100 > best_reward_per_100:
            best_reward_per_100 = avg_reward_100
            agent.save(os.path.join(MODEL_DIR, "td3_best_avg100.pth"))

        # Periodic checkpoint
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(MODEL_DIR, f"td3_ep{episode}.pth"))

    print("\n[train] Training complete.")
    print(f"[train] Best episode reward: {best_reward:.2f}")
    print(f"[train] Best 100-episode average: {best_reward_per_100:.2f}")
    print(f"[train] Models saved to: {MODEL_DIR}")
    env.close()


def evaluate(env: CarRacingEnv, agent: TD3Agent, num_episodes: int = 10, 
             render: bool = True, checkpoint_path: str | None = None) -> dict:
    """
    Evaluate a trained agent without exploration noise.
    
    Args:
        env: The environment to evaluate in
        agent: The agent to evaluate
        num_episodes: Number of evaluation episodes
        render: Whether to render the episodes
        checkpoint_path: Path to load checkpoint from (optional)
    
    Returns:
        Dictionary with evaluation statistics
    """
    if checkpoint_path:
        agent.load(checkpoint_path)

    results = {
        "total_rewards": [],
        "episode_lengths": [],
        "laps_completed": [],
        "crashes": 0,
        "avg_reward": 0.0,
        "avg_length": 0.0,
        "avg_laps": 0.0,
    }

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_laps = 0

        while not done:
            # Deterministic action (no noise)
            action = agent.select_action(state, add_noise=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if info.get("lap_completed", False):
                episode_laps += 1

            if render:
                env.render(enabled=True, limit_fps=True)

        results["total_rewards"].append(episode_reward)
        results["episode_lengths"].append(episode_length)
        results["laps_completed"].append(episode_laps)
        if info.get("termination_reason") == "off_track":
            results["crashes"] += 1

        print(f"Eval Episode {ep+1}/{num_episodes} | Reward: {episode_reward:+7.2f} | "
              f"Length: {episode_length:4d} | Laps: {episode_laps:2d}")

    # Compute statistics
    results["avg_reward"] = np.mean(results["total_rewards"])
    results["avg_length"] = np.mean(results["episode_lengths"])
    results["avg_laps"] = np.mean(results["laps_completed"])
    results["crash_rate"] = results["crashes"] / num_episodes

    print("\n=== Evaluation Summary ===")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Episode Length: {results['avg_length']:.1f}")
    print(f"Average Laps per Episode: {results['avg_laps']:.1f}")
    print(f"Crash Rate: {results['crash_rate']:.1%}")

    return results
