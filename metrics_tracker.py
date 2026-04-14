"""
metrics_tracker.py - Tracks training metrics for analysis and debugging.

Records episode statistics, network performance, and training health.
"""

import json
import os
from collections import defaultdict
from typing import Any

import numpy as np

from config import LOGS_DIR


class MetricsTracker:
    """Tracks and logs training metrics to JSON Lines format."""

    def __init__(
        self,
        log_dir: str = LOGS_DIR,
        log_filename: str = "training_log.jsonl",
        experiment_name: str = "default",
        reward_mode: str | None = None,
        sensor_noise_std: float | None = None,
        seed: int | None = None,
    ):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, log_filename)
        self.experiment_name = experiment_name
        self.reward_mode = reward_mode
        self.sensor_noise_std = sensor_noise_std
        self.seed = seed
        
        # Current episode metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_steerings = []
        self.termination_counts = defaultdict(int)
        self.laps_completed = 0
        self.collisions = 0
        
        # Network metrics
        self.critic_losses = []
        self.actor_losses = []
        self.buffer_sizes = []

    def reset_episode(self):
        """Reset metrics for a new episode."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_speeds = []
        self.episode_steerings = []
        self.laps_completed = 0
        self.collisions = 0

    def log_step(self, reward: float, speed: float, steering: float, action: np.ndarray):
        """Log a single step within an episode."""
        self.episode_rewards.append(reward)
        self.episode_speeds.append(speed)
        self.episode_steerings.append(abs(steering))

    def log_termination(self, reason: str):
        """Log episode termination reason."""
        self.termination_counts[reason] += 1
        if reason == "off_track":
            self.collisions += 1

    def log_lap_completion(self, lap_time: float):
        """Log a completed lap."""
        self.laps_completed += 1

    def log_network_stats(self, critic_loss: float | None, actor_loss: float | None, 
                          buffer_size: int):
        """Log network training statistics."""
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        self.buffer_sizes.append(buffer_size)

    def get_episode_summary(self, episode: int) -> dict[str, Any]:
        """Compute summary statistics for the episode."""
        if not self.episode_rewards:
            return {
                "episode": episode,
                "reward_total": 0.0,
                "reward_mean": 0.0,
                "length": 0,
            }
        
        total_reward = sum(self.episode_rewards)
        return {
            "episode": episode,
            "reward_total": total_reward,
            "reward_mean": np.mean(self.episode_rewards),
            "reward_std": np.std(self.episode_rewards),
            "length": len(self.episode_rewards),
            "speed_mean": np.mean(self.episode_speeds),
            "speed_max": np.max(self.episode_speeds) if self.episode_speeds else 0.0,
            "steering_smooth": np.mean(self.episode_steerings),
            "laps_completed": self.laps_completed,
            "collisions": self.collisions,
            "termination_reason": max(self.termination_counts, key=self.termination_counts.get) 
                                 if self.termination_counts else "unknown",
            "critic_loss_mean": np.mean(self.critic_losses) if self.critic_losses else None,
            "actor_loss_mean": np.mean(self.actor_losses) if self.actor_losses else None,
            "buffer_size": self.buffer_sizes[-1] if self.buffer_sizes else 0,
        }

    def log_episode(self, episode_summary: dict[str, Any]):
        """Log episode to file and compute rolling statistics."""
        episode_summary = {
            "experiment_name": self.experiment_name,
            "reward_mode": self.reward_mode,
            "sensor_noise_std": self.sensor_noise_std,
            "seed": self.seed,
            **episode_summary,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(episode_summary) + "\n")

    @staticmethod
    def load_logs(log_dir: str = LOGS_DIR) -> list[dict]:
        """Load training logs from JSON Lines file."""
        log_file = os.path.join(log_dir, "training_log.jsonl")
        if not os.path.exists(log_file):
            return []
        
        logs = []
        with open(log_file, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return logs

    @staticmethod
    def get_rolling_stats(log_dir: str = LOGS_DIR, window: int = 100) -> dict[str, list]:
        """Compute rolling statistics from logs."""
        logs = MetricsTracker.load_logs(log_dir)
        if not logs:
            return {}
        
        episodes = [log["episode"] for log in logs]
        rewards = [log["reward_total"] for log in logs]
        lengths = [log["length"] for log in logs]
        collisions = [log["collisions"] for log in logs]
        
        rolling_avg_reward = [
            np.mean(rewards[max(0, i-window):i]) for i in range(1, len(rewards) + 1)
        ]
        rolling_avg_length = [
            np.mean(lengths[max(0, i-window):i]) for i in range(1, len(lengths) + 1)
        ]
        rolling_avg_collision = [
            np.mean(collisions[max(0, i-window):i]) for i in range(1, len(collisions) + 1)
        ]
        
        return {
            "episodes": episodes,
            "rolling_avg_reward": rolling_avg_reward,
            "rolling_avg_length": rolling_avg_length,
            "rolling_collision_rate": rolling_avg_collision,
        }

    def print_summary(self, episode: int, episode_summary: dict[str, Any], 
                     rolling_avg_reward: float):
        """Print formatted summary to console."""
        print(
            f"Episode {episode:>5d} | "
            f"Reward {episode_summary['reward_total']:>+8.2f} | "
            f"Avg100 {rolling_avg_reward:>+8.2f} | "
            f"Laps {episode_summary['laps_completed']:>2d} | "
            f"Crashes {episode_summary['collisions']:>2d} | "
            f"End {episode_summary['termination_reason']}"
        )
