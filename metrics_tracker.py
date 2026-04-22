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
        
        # Current episode metrics — use running accumulators instead of lists
        # to reduce per-step memory allocation and speed up summary computation.
        self._reward_sum = 0.0
        self._reward_sum_sq = 0.0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._steering_abs_sum = 0.0
        self._step_count = 0
        self.termination_counts: dict[str, int] = {}
        self.laps_completed = 0
        self.collisions = 0
        
        # Network metrics
        self.critic_losses = []
        self.actor_losses = []
        self.buffer_sizes = []

        # Cached JSON prefix fields (constant per tracker lifetime)
        self._json_prefix = {
            "experiment_name": self.experiment_name,
            "reward_mode": self.reward_mode,
            "sensor_noise_std": self.sensor_noise_std,
            "seed": self.seed,
        }

    def reset_episode(self):
        """Reset metrics for a new episode using running accumulators."""
        self._reward_sum = 0.0
        self._reward_sum_sq = 0.0
        self._speed_sum = 0.0
        self._speed_max = 0.0
        self._steering_abs_sum = 0.0
        self._step_count = 0
        self.termination_counts.clear()
        self.laps_completed = 0
        self.collisions = 0

    def log_step(self, reward: float, speed: float, steering: float, action: np.ndarray):
        """Log a single step within an episode using running accumulators."""
        self._reward_sum += reward
        self._reward_sum_sq += reward * reward
        self._speed_sum += speed
        if speed > self._speed_max:
            self._speed_max = speed
        self._steering_abs_sum += abs(steering)
        self._step_count += 1

    def log_termination(self, reason: str):
        """Log episode termination reason."""
        self.termination_counts[reason] = self.termination_counts.get(reason, 0) + 1
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
        """Compute summary statistics for the episode from running accumulators."""
        n = self._step_count
        if n == 0:
            return {
                "episode": episode,
                "reward_total": 0.0,
                "reward_mean": 0.0,
                "reward_std": 0.0,
                "length": 0,
                "speed_mean": 0.0,
                "speed_max": 0.0,
                "steering_smooth": 0.0,
                "laps_completed": 0,
                "collisions": 0,
                "termination_reason": "unknown",
                "critic_loss_mean": np.mean(self.critic_losses) if self.critic_losses else None,
                "actor_loss_mean": np.mean(self.actor_losses) if self.actor_losses else None,
                "buffer_size": self.buffer_sizes[-1] if self.buffer_sizes else 0,
            }
        
        inv_n = 1.0 / n
        reward_mean = self._reward_sum * inv_n
        # Compute std from running sums: std = sqrt(E[x^2] - E[x]^2)
        variance = (self._reward_sum_sq * inv_n) - (reward_mean * reward_mean)
        reward_std = variance ** 0.5 if variance > 0 else 0.0

        # Determine termination reason from counts
        if self.termination_counts:
            termination_reason = max(self.termination_counts, key=self.termination_counts.get)
        else:
            termination_reason = "unknown"

        return {
            "episode": episode,
            "reward_total": self._reward_sum,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "length": n,
            "speed_mean": self._speed_sum * inv_n,
            "speed_max": self._speed_max,
            "steering_smooth": self._steering_abs_sum * inv_n,
            "laps_completed": self.laps_completed,
            "collisions": self.collisions,
            "termination_reason": termination_reason,
            "critic_loss_mean": np.mean(self.critic_losses) if self.critic_losses else None,
            "actor_loss_mean": np.mean(self.actor_losses) if self.actor_losses else None,
            "buffer_size": self.buffer_sizes[-1] if self.buffer_sizes else 0,
        }

    def log_episode(self, episode_summary: dict[str, Any]):
        """Log episode to file and compute rolling statistics."""
        episode_summary = {
            **self._json_prefix,
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
