"""
plot_metrics.py - Plot training metrics from logs.

Visualizes training progress, rewards, and other statistics.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import LOGS_DIR


def load_logs(log_dir: str = LOGS_DIR) -> list[dict]:
    """Load training logs from JSON Lines file."""
    log_file = os.path.join(log_dir, "training_log.jsonl")
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return []
    
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(logs)} log entries from {log_file}")
    return logs


def plot_rewards(logs: list[dict], output_dir: str = LOGS_DIR):
    """Plot episode rewards with rolling average."""
    if not logs:
        print("No logs to plot")
        return
    
    episodes = [log["episode"] for log in logs]
    rewards = [log["reward_total"] for log in logs]
    
    # Compute moving average
    window = 100
    ma_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ma_episodes = episodes[window-1:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    plt.plot(ma_episodes, ma_rewards, 'r-', linewidth=2, label=f'MA-{window}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plot_rewards.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_collisions(logs: list[dict], output_dir: str = LOGS_DIR):
    """Plot crash/collision statistics."""
    if not logs:
        return
    
    episodes = [log["episode"] for log in logs]
    collisions = [log["collisions"] for log in logs]
    
    window = 50
    ma_collisions = np.convolve(collisions, np.ones(window)/window, mode='valid')
    ma_episodes = episodes[window-1:]
    
    plt.figure(figsize=(12, 6))
    plt.bar(episodes, collisions, alpha=0.5, label='Crashes')
    plt.plot(ma_episodes, ma_collisions, 'r-', linewidth=2, label=f'MA-{window}')
    plt.xlabel('Episode')
    plt.ylabel('Crashes per Episode')
    plt.title('Training Progress: Crashes')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plot_collisions.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_laps(logs: list[dict], output_dir: str = LOGS_DIR):
    """Plot laps completed per episode."""
    if not logs:
        return
    
    episodes = [log["episode"] for log in logs]
    laps = [log["laps_completed"] for log in logs]
    
    plt.figure(figsize=(12, 6))
    plt.scatter(episodes, laps, alpha=0.6, s=30)
    plt.xlabel('Episode')
    plt.ylabel('Laps Completed')
    plt.title('Training Progress: Laps Completed')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plot_laps.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_speed_and_smoothness(logs: list[dict], output_dir: str = LOGS_DIR):
    """Plot average speed and steering smoothness."""
    if not logs:
        return
    
    episodes = [log["episode"] for log in logs]
    speeds = [log.get("speed_mean", 0.0) for log in logs]
    smoothness = [log.get("steering_smooth", 0.0) for log in logs]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Speed plot
    ax1.plot(episodes, speeds, 'b-', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('Average Speed', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Training Progress: Speed and Steering Smoothness')
    
    # Smoothness plot
    ax2.plot(episodes, smoothness, 'g-', alpha=0.7, linewidth=1.5)
    ax2.set_ylabel('Avg Steering Magnitude', color='g')
    ax2.set_xlabel('Episode')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plot_speed_smoothness.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_exploration_decay(logs: list[dict], output_dir: str = LOGS_DIR):
    """Plot exploration noise decay over time."""
    if not logs or "exploration_noise" not in logs[0]:
        print("No exploration noise data in logs")
        return
    
    episodes = [log["episode"] for log in logs]
    noise = [log.get("exploration_noise", 0.1) for log in logs]
    
    plt.figure(figsize=(12, 6))
    plt.semilogy(episodes, noise, 'r-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Exploration Noise (log scale)')
    plt.title('Exploration Noise Decay')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plot_noise_decay.png")
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_all(log_dir: str = LOGS_DIR):
    """Generate all plots."""
    os.makedirs(log_dir, exist_ok=True)
    logs = load_logs(log_dir)
    
    if not logs:
        print("No data to plot")
        return
    
    print(f"\nGenerating plots from {len(logs)} episodes...")
    plot_rewards(logs, log_dir)
    plot_collisions(logs, log_dir)
    plot_laps(logs, log_dir)
    plot_speed_and_smoothness(logs, log_dir)
    plot_exploration_decay(logs, log_dir)
    print(f"\nAll plots saved to: {log_dir}")


if __name__ == "__main__":
    plot_all()
