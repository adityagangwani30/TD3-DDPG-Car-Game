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
    EVAL_EPISODES,
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

    target_log_dir = getattr(metrics, "log_dir", target_model_dir)
    os.makedirs(target_log_dir, exist_ok=True)
    metadata_payload = {
        "algorithm": algo,
        "experiment_name": experiment_name,
        "reward_mode": getattr(env, "reward_mode", "shaped"),
        "sensor_noise_std": float(getattr(env, "sensor_noise_std", 0.0)),
        "seed": resolved_seed,
        "max_episodes": total_episodes,
        "max_steps_per_episode": steps_per_episode,
        "eval_episodes": EVAL_EPISODES,
        "exploration_noise_initial": EXPLORATION_NOISE,
        "exploration_noise_decay": EXPLORATION_NOISE_DECAY,
        "exploration_noise_floor": 0.01,
        "batch_size": BATCH_SIZE,
        "buffer_capacity": BUFFER_CAPACITY,
        "warmup_steps": TRAINING_START,
        "throttle_mapping": "clip(action[1], 0, 1)",
        "steering_mapping": "[-1, 1]",
        "protocol_version": "V2_camera_ready",
    }
    with open(os.path.join(target_log_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_payload, f, indent=2)

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
        episode_speeds = []
        termination_reason = "max_steps"
        render_enabled = _should_render_episode(episode) and RENDER_DURING_TRAINING

        for step in range(1, steps_per_episode + 1):
            # Use decaying exploration noise
            action = agent.select_action(state, add_noise=True, noise_scale=exploration_noise)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            episode_speeds.append(info.get("speed", 0.0))
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
        episode_summary["distance_traveled"] = float(info.get("distance_traveled", np.sum(episode_speeds)))
        episode_summary["avg_speed"] = float(np.mean(episode_speeds)) if episode_speeds else 0.0
        episode_summary["termination_reason"] = termination_reason
        episode_summary["reward_rolling_avg_100"] = float(avg_reward_100)
        episode_summary["exploration_noise"] = exploration_noise
        episode_summary["replay_buffer_size"] = len(replay_buffer)
        episode_summary["algorithm"] = algo
        episode_summary["reward_mode"] = getattr(env, "reward_mode", "shaped")
        episode_summary["sensor_noise_std"] = float(getattr(env, "sensor_noise_std", 0.0))
        episode_summary["seed"] = resolved_seed
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

    # Run deterministic evaluation on the best checkpoint
    best_checkpoint = os.path.join(target_model_dir, f"{model_prefix}{algo}_best_avg100.pth")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = os.path.join(target_model_dir, f"{model_prefix}{algo}_best.pth")
    if not os.path.exists(best_checkpoint):
        best_checkpoint = None

    target_log_dir = getattr(metrics, "log_dir", target_model_dir)
    eval_log_file = os.path.join(target_log_dir, "evaluation_log.jsonl")
    eval_summary_file = os.path.join(target_log_dir, "evaluation_summary.json")

    eval_meta = {
        "algo": algo,
        "experiment_name": experiment_name,
        "reward_mode": getattr(env, "reward_mode", "shaped"),
        "sensor_noise_std": getattr(env, "sensor_noise_std", 0.0),
        "seed": resolved_seed,
        "checkpoint": best_checkpoint,
        "max_steps_per_episode": steps_per_episode,
    }

    print(f"{prefix}[train] Running {EVAL_EPISODES} deterministic evaluation episodes...")
    eval_results = evaluate(
        env,
        agent,
        num_episodes=EVAL_EPISODES,
        render=False,
        checkpoint_path=best_checkpoint,
        eval_log_path=eval_log_file,
        eval_summary_path=eval_summary_file,
        metadata=eval_meta,
    )

    env.close()
    return eval_results


def evaluate(
    env: CarRacingEnv,
    agent,
    num_episodes: int = EVAL_EPISODES,
    max_steps_per_episode: int = MAX_STEPS_PER_EPISODE,
    render: bool = False,
    checkpoint_path: str | None = None,
    preview_path: str | None = None,
    eval_log_path: str | None = None,
    eval_summary_path: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Evaluate a trained agent with DETERMINISTIC actions (exploration noise OFF).
    
    Args:
        env: The environment to evaluate in
        agent: The agent to evaluate
        num_episodes: Number of evaluation episodes (default: 20)
        max_steps_per_episode: Maximum steps per evaluation episode (default: 600)
        render: Whether to render the episodes
        checkpoint_path: Path to load checkpoint from (optional)
        preview_path: Path to save a preview frame (optional)
        eval_log_path: Path to save per-episode evaluation JSONL log (optional)
        eval_summary_path: Path to save aggregated evaluation JSON summary (optional)
        metadata: Extra metadata to record in summary (optional)
    
    Returns:
        Dictionary with full raw episode metrics and aggregated statistics.
    """
    if checkpoint_path:
        agent.load(checkpoint_path)

    raw_rewards = []
    raw_lengths = []
    raw_crashes = []
    raw_laps = []
    raw_distances = []
    raw_speeds = []
    raw_lap_times = []
    
    eval_logs = []
    preview_saved = False

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        episode_laps = 0
        episode_crashed = 0
        step_speeds = []
        best_ep_lap_time = None

        for step in range(1, max_steps_per_episode + 1):
            # Deterministic action (exploration noise strictly OFF)
            action = agent.select_action(state, add_noise=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            step_speeds.append(info.get("speed", 0.0))

            if info.get("lap_completed", False):
                episode_laps += 1
                lap_time = info.get("last_lap_time")
                if lap_time:
                    raw_lap_times.append(float(lap_time))
                    if best_ep_lap_time is None or lap_time < best_ep_lap_time:
                        best_ep_lap_time = lap_time

            if render:
                env.render(enabled=True, limit_fps=True)
                if preview_path and not preview_saved:
                    env.save_frame(preview_path)
                    preview_saved = True

            if done:
                break

        termination_reason = info.get("termination_reason", "max_steps")
        if termination_reason == "off_track":
            episode_crashed = 1

        dist = info.get("distance_traveled", float(np.sum(step_speeds)))
        avg_ep_speed = float(np.mean(step_speeds)) if step_speeds else 0.0

        raw_rewards.append(float(episode_reward))
        raw_lengths.append(int(episode_length))
        raw_crashes.append(int(episode_crashed))
        raw_laps.append(int(episode_laps))
        raw_distances.append(float(dist))
        raw_speeds.append(float(avg_ep_speed))

        ep_record = {
            "eval_episode": ep,
            "reward": float(episode_reward),
            "length": int(episode_length),
            "crashed": int(episode_crashed),
            "laps_completed": int(episode_laps),
            "distance_traveled": float(dist),
            "avg_speed": float(avg_ep_speed),
            "termination_reason": termination_reason,
            "best_lap_time": best_ep_lap_time,
        }
        eval_logs.append(ep_record)

    n = max(1, len(raw_rewards))
    crash_rate = float(np.mean(raw_crashes))
    lap_completion_rate = float(np.mean([1 if l > 0 else 0 for l in raw_laps]))
    
    summary = {
        "num_episodes": n,
        "crash_rate": crash_rate,
        "lap_completion_rate": lap_completion_rate,
        "total_laps_completed": int(np.sum(raw_laps)),
        "mean_laps_per_episode": float(np.mean(raw_laps)),
        "laps_std": float(np.std(raw_laps, ddof=1)) if n > 1 else 0.0,
        "distance_mean": float(np.mean(raw_distances)),
        "distance_std": float(np.std(raw_distances, ddof=1)) if n > 1 else 0.0,
        "distance_sem": float(np.std(raw_distances, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "avg_length": float(np.mean(raw_lengths)),
        "length_std": float(np.std(raw_lengths, ddof=1)) if n > 1 else 0.0,
        "length_sem": float(np.std(raw_lengths, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "avg_reward": float(np.mean(raw_rewards)),
        "reward_std": float(np.std(raw_rewards, ddof=1)) if n > 1 else 0.0,
        "reward_sem": float(np.std(raw_rewards, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "avg_speed": float(np.mean(raw_speeds)),
        "speed_std": float(np.std(raw_speeds, ddof=1)) if n > 1 else 0.0,
        "best_lap_time": min(raw_lap_times) if raw_lap_times else None,
        "avg_lap_time": float(np.mean(raw_lap_times)) if raw_lap_times else None,
        "raw_rewards": raw_rewards,
        "raw_lengths": raw_lengths,
        "raw_crashes": raw_crashes,
        "raw_laps": raw_laps,
        "raw_distances": raw_distances,
        "raw_speeds": raw_speeds,
        "raw_lap_times": raw_lap_times,
    }
    if metadata:
        summary["metadata"] = metadata

    if eval_log_path:
        os.makedirs(os.path.dirname(os.path.abspath(eval_log_path)), exist_ok=True)
        with open(eval_log_path, "w", encoding="utf-8") as f:
            for rec in eval_logs:
                f.write(json.dumps(rec) + "\n")

    if eval_summary_path:
        os.makedirs(os.path.dirname(os.path.abspath(eval_summary_path)), exist_ok=True)
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("\n=== Deterministic Evaluation Summary (Exploration Noise OFF) ===")
    print(f"Episodes Evaluated     : {n}")
    print(f"Crash Rate             : {summary['crash_rate']:.1%}")
    print(f"Lap Completion Rate    : {summary['lap_completion_rate']:.1%}")
    print(f"Total Laps Completed   : {summary['total_laps_completed']}")
    print(f"Mean Distance Traveled : {summary['distance_mean']:.1f} +/- {summary['distance_std']:.1f} px")
    print(f"Mean Episode Length    : {summary['avg_length']:.1f} +/- {summary['length_std']:.1f} steps")
    print(f"Mean Episode Reward    : {summary['avg_reward']:.2f} +/- {summary['reward_std']:.2f}")
    print(f"Mean Speed             : {summary['avg_speed']:.2f} px/step")
    if summary['best_lap_time'] is not None:
        print(f"Best Lap Time          : {summary['best_lap_time']:.2f}s")
    print("=================================================================\n")

    return summary

