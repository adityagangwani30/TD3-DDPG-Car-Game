"""
train.py - Training loop for deterministic policy-gradient car training.

Orchestrates episode collection, replay-buffer storage, network updates,
exploration noise decay, and comprehensive metrics tracking.
"""

import os
import json

import numpy as np
import pygame

from config import (
    BATCH_SIZE,
    BUFFER_CAPACITY,
    DEFAULT_SEED,
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
from ddpg_agent import DDPGAgent
from td3_agent import TD3Agent
from utils import set_global_seed


def _load_existing_progress(log_file: str) -> tuple[int, list[float]]:
    """Return (last_episode, reward_history) from an existing JSONL metrics log."""
    if not os.path.exists(log_file):
        return 0, []

    last_episode = 0
    reward_history: list[float] = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode = int(payload.get("episode", 0) or 0)
                if episode > 0:
                    last_episode = max(last_episode, episode)
                if "reward_total" in payload:
                    reward_history.append(float(payload.get("reward_total", 0.0)))
    except OSError:
        return 0, []

    return last_episode, reward_history


def _should_render_episode(episode: int) -> bool:
    """Return True when this episode should be rendered."""
    if RENDER_EVERY_EPISODES <= 0:
        return True
    return episode == 1 or episode % RENDER_EVERY_EPISODES == 0


def train(
    env: CarRacingEnv,
    algo: str,
    device: str = "cpu",
    model_dir: str | None = None,
    run_label: str | None = None,
    checkpoint_path: str | None = None,
    require_checkpoint: bool = False,
    experiment_name: str = "default",
    seed: int | None = None,
    max_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
):
    """Run the main training loop with exploration decay and metrics tracking."""
    return train_with_config(
        env,
        algo=algo,
        device=device,
        model_dir=model_dir,
        run_label=run_label,
        checkpoint_path=checkpoint_path,
        require_checkpoint=require_checkpoint,
        experiment_name=experiment_name,
        seed=seed,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
    )


def train_with_config(
    env: CarRacingEnv,
    algo: str,
    device: str = "cpu",
    model_dir: str | None = None,
    run_label: str | None = None,
    checkpoint_path: str | None = None,
    require_checkpoint: bool = False,
    experiment_name: str = "default",
    seed: int | None = None,
    max_episodes: int | None = None,
    max_steps_per_episode: int | None = None,
):
    """Run training with optional custom output directory and run label."""
    prefix = f"[{run_label}] " if run_label else ""
    resolved_seed = DEFAULT_SEED if seed is None else int(seed)
    set_global_seed(resolved_seed)

    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)
    metrics = env.metrics or MetricsTracker(
        experiment_name=experiment_name,
        reward_mode=getattr(env, "reward_mode", None),
        sensor_noise_std=getattr(env, "sensor_noise_std", None),
        seed=resolved_seed,
    )
    if algo == "td3":
        agent = TD3Agent(device=device)
    elif algo == "ddpg":
        agent = DDPGAgent(device=device)
    else:
        raise ValueError("Unsupported algorithm")

    if checkpoint_path:
        try:
            agent.load(checkpoint_path)
            print(f"{prefix}[train] Loaded checkpoint: {checkpoint_path}")
        except (RuntimeError, KeyError, FileNotFoundError) as exc:
            message = f"{prefix}[train] Could not load checkpoint '{checkpoint_path}': {exc}"
            if require_checkpoint:
                raise RuntimeError(message) from exc
            print(message)
            print(f"{prefix}[train] Continuing with freshly initialized weights.")

    target_model_dir = model_dir or os.path.join(MODEL_DIR, algo)
    os.makedirs(target_model_dir, exist_ok=True)
    print(
        f"{prefix}[train] Experiment: {experiment_name} | "
        f"Reward mode: {env.reward_mode} | Sensor noise: {env.sensor_noise_std:.3f} | Seed: {resolved_seed}"
    )
    print(f"{prefix}[train] Starting training loop. Models -> {target_model_dir}")

    model_prefix = f"{experiment_name}_" if experiment_name and experiment_name != "default" else ""
    total_episodes = max_episodes if max_episodes is not None else MAX_EPISODES
    steps_per_episode = (
        max_steps_per_episode if max_steps_per_episode is not None else MAX_STEPS_PER_EPISODE
    )

    start_episode = 1
    loaded_reward_history: list[float] = []
    existing_log_file = getattr(metrics, "log_file", "")
    if checkpoint_path and existing_log_file:
        last_logged_episode, loaded_reward_history = _load_existing_progress(existing_log_file)
        if last_logged_episode >= total_episodes:
            print(
                f"{prefix}[train] Existing log already reached episode {last_logged_episode}/{total_episodes}. "
                "Skipping training."
            )
            env.close()
            return
        if last_logged_episode > 0:
            start_episode = last_logged_episode + 1
            print(
                f"{prefix}[train] Resuming from episode {start_episode}/{total_episodes} "
                f"(detected {last_logged_episode} completed episodes in log)."
            )

    best_reward = -float("inf")
    best_reward_per_100 = -float("inf")
    reward_history = loaded_reward_history.copy()
    exploration_noise = EXPLORATION_NOISE

    for episode in range(start_episode, total_episodes + 1):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_laps = 0
        episode_crashes = 0
        termination_reason = "max_steps"
        render_enabled = _should_render_episode(episode) and RENDER_DURING_TRAINING

        for step in range(1, steps_per_episode + 1):
            # Use decaying exploration noise
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if info.get("lap_completed", False):
                episode_laps += 1

            replay_buffer.add(state, action, reward, next_state, done)

            if replay_buffer.is_ready(TRAINING_START):
                agent.train(replay_buffer, BATCH_SIZE)

            env.render(enabled=render_enabled, limit_fps=False)
            state = next_state

            if done:
                termination_reason = info.get("termination_reason", "unknown")
                if termination_reason == "off_track":
                    episode_crashes = 1
                break

        # Decay exploration noise
        exploration_noise *= EXPLORATION_NOISE_DECAY
        exploration_noise = max(exploration_noise, 0.01)  # Minimum noise floor

        reward_history.append(episode_reward)
        avg_reward_100 = np.mean(reward_history[-100:])

        # Compute episode summary for metrics
        episode_summary = metrics.get_episode_summary(episode)
        # Keep core paper/report metrics consistent even if env.metrics is disabled.
        episode_summary["reward_total"] = float(episode_reward)
        episode_summary["length"] = int(episode_length)
        episode_summary["laps_completed"] = int(episode_laps)
        episode_summary["collisions"] = int(episode_crashes)
        episode_summary["termination_reason"] = termination_reason
        episode_summary["reward_rolling_avg_100"] = float(avg_reward_100)
        episode_summary["exploration_noise"] = exploration_noise
        episode_summary["replay_buffer_size"] = len(replay_buffer)
        metrics.log_episode(episode_summary)

        # Print summary
        if run_label:
            print(f"[{run_label}] ", end="")
        metrics.print_summary(episode, episode_summary, avg_reward_100)

        # Save best model by individual episode reward
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(target_model_dir, f"{model_prefix}{algo}_best.pth"))

        # Save best model by rolling 100-episode average
        if avg_reward_100 > best_reward_per_100:
            best_reward_per_100 = avg_reward_100
            agent.save(os.path.join(target_model_dir, f"{model_prefix}{algo}_best_avg100.pth"))

        # Periodic checkpoint
        if episode % SAVE_MODEL_EVERY == 0:
            agent.save(os.path.join(target_model_dir, f"{model_prefix}{algo}_ep{episode}.pth"))

    print(f"\n{prefix}[train] Training complete.")
    print(f"{prefix}[train] Best episode reward: {best_reward:.2f}")
    print(f"{prefix}[train] Best 100-episode average: {best_reward_per_100:.2f}")
    print(f"{prefix}[train] Models saved to: {target_model_dir}")
    env.close()


def evaluate(env: CarRacingEnv, agent, num_episodes: int = 10, 
             render: bool = True, checkpoint_path: str | None = None,
             preview_path: str | None = None) -> dict:
    """
    Evaluate a trained agent without exploration noise.
    
    Args:
        env: The environment to evaluate in
        agent: The agent to evaluate
        num_episodes: Number of evaluation episodes
        render: Whether to render the episodes
        checkpoint_path: Path to load checkpoint from (optional)
        preview_path: Path to save a preview frame (works in both GUI and headless modes)
    
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
        preview_saved = False

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
                # Save preview frame (works in both GUI and headless modes)
                if preview_path and not preview_saved:
                    env.save_frame(preview_path)
                    preview_saved = True

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
