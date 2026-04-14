"""
plot_metrics.py - Plot and compare training metrics for research reporting.

Supports:
  - Single-run plotting from one log directory
  - Per-experiment plotting from multiple experiment folders
  - Cross-experiment comparison plots
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import LOGS_DIR


def load_logs(log_dir: str) -> list[dict]:
    """Load JSONL logs from a directory. Returns an empty list when missing."""
    log_file = os.path.join(log_dir, "training_log.jsonl")
    if not os.path.exists(log_file):
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Compute rolling mean with a small-sample fallback for early episodes."""
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.array([np.mean(arr[: i + 1]) for i in range(len(arr))], dtype=float)

    kernel = np.ones(window, dtype=float) / float(window)
    core = np.convolve(arr, kernel, mode="valid")
    warmup = np.array([np.mean(arr[: i + 1]) for i in range(window - 1)], dtype=float)
    return np.concatenate([warmup, core])


def sanitize_name(name: str) -> str:
    """Create filesystem-safe names for output files."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return safe or "experiment"


def infer_experiment_name(logs: list[dict], fallback: str) -> str:
    """Pick an experiment name from logs or fallback to folder name."""
    if logs and logs[0].get("experiment_name"):
        return str(logs[0]["experiment_name"])
    return fallback


def extract_series(logs: list[dict], rolling_window: int) -> dict[str, np.ndarray]:
    """Extract robust metric series from logs, with compatibility fallbacks."""
    episodes = np.array([int(log.get("episode", i + 1)) for i, log in enumerate(logs)], dtype=int)
    reward_total = np.array([float(log.get("reward_total", 0.0)) for log in logs], dtype=float)
    crashes = np.array([float(log.get("collisions", 0.0)) for log in logs], dtype=float)
    laps = np.array([float(log.get("laps_completed", 0.0)) for log in logs], dtype=float)
    length = np.array([float(log.get("length", 0.0)) for log in logs], dtype=float)

    if logs and "reward_rolling_avg_100" in logs[0]:
        reward_rolling = np.array(
            [float(log.get("reward_rolling_avg_100", 0.0)) for log in logs], dtype=float
        )
    else:
        reward_rolling = rolling_mean(reward_total.tolist(), rolling_window)

    crash_rate_rolling = rolling_mean(crashes.tolist(), rolling_window)

    return {
        "episodes": episodes,
        "reward_total": reward_total,
        "reward_rolling": reward_rolling,
        "crashes": crashes,
        "crash_rate_rolling": crash_rate_rolling,
        "laps": laps,
        "length": length,
    }


def plot_experiment(logs: list[dict], output_dir: str, experiment_name: str, window: int):
    """Generate publication-ready plots for one experiment."""
    if not logs:
        print(f"[plot] No logs found for experiment: {experiment_name}")
        return

    os.makedirs(output_dir, exist_ok=True)
    safe_name = sanitize_name(experiment_name)
    series = extract_series(logs, rolling_window=window)

    episodes = series["episodes"]
    reward_total = series["reward_total"]
    reward_rolling = series["reward_rolling"]
    crashes = series["crashes"]
    crash_rate_rolling = series["crash_rate_rolling"]
    laps = series["laps"]

    plt.style.use("seaborn-v0_8-whitegrid")

    # Reward vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, reward_total, color="#4C72B0", alpha=0.35, linewidth=1.2, label="Episode reward")
    plt.plot(episodes, reward_rolling, color="#D62728", linewidth=2.2, label=f"Rolling avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Reward vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    reward_path = os.path.join(output_dir, f"{safe_name}_reward_vs_episodes.png")
    plt.savefig(reward_path, dpi=220)
    plt.close()

    # Crash rate vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, crashes, color="#8C564B", alpha=0.28, linewidth=1.0, label="Crashes per episode")
    plt.plot(
        episodes,
        crash_rate_rolling,
        color="#FF7F0E",
        linewidth=2.2,
        label=f"Rolling crash rate ({window})",
    )
    plt.xlabel("Episode")
    plt.ylabel("Crash count / rate")
    plt.title(f"Crash Rate vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    crash_path = os.path.join(output_dir, f"{safe_name}_crash_rate_vs_episodes.png")
    plt.savefig(crash_path, dpi=220)
    plt.close()

    # Laps vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(episodes, laps, color="#2CA02C", linewidth=1.8, label="Laps completed")
    plt.xlabel("Episode")
    plt.ylabel("Laps completed")
    plt.title(f"Laps vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    laps_path = os.path.join(output_dir, f"{safe_name}_laps_vs_episodes.png")
    plt.savefig(laps_path, dpi=220)
    plt.close()

    print(f"[plot] Saved: {reward_path}")
    print(f"[plot] Saved: {crash_path}")
    print(f"[plot] Saved: {laps_path}")


def resolve_experiment_dirs(base_log_dir: str, experiment_args: list[str] | None) -> dict[str, str]:
    """Resolve experiment names to absolute log directories."""
    base_path = Path(base_log_dir)

    if experiment_args:
        resolved = {}
        for raw in experiment_args:
            p = Path(raw)
            if p.is_absolute() and p.exists():
                exp_dir = p
                exp_name = p.name
            else:
                exp_dir = base_path / raw
                exp_name = raw
            resolved[exp_name] = str(exp_dir)
        return resolved

    direct_log = base_path / "training_log.jsonl"
    if direct_log.exists():
        return {base_path.name or "default": str(base_path)}

    resolved = {}
    if base_path.exists():
        for child in sorted(base_path.iterdir()):
            if child.is_dir() and (child / "training_log.jsonl").exists():
                resolved[child.name] = str(child)
    return resolved


def plot_comparison(experiment_logs: dict[str, list[dict]], output_dir: str, window: int):
    """Generate comparison plots across experiments."""
    if len(experiment_logs) < 2:
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # Comparison: reward
    plt.figure(figsize=(10.5, 6.0))
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        plt.plot(series["episodes"], series["reward_rolling"], linewidth=2.0, label=name)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling average reward ({window})")
    plt.title("Experiment Comparison: Reward")
    plt.legend(loc="best")
    plt.tight_layout()
    reward_path = os.path.join(output_dir, "comparison_reward_vs_episodes.png")
    plt.savefig(reward_path, dpi=220)
    plt.close()

    # Comparison: crash rate
    plt.figure(figsize=(10.5, 6.0))
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        plt.plot(series["episodes"], series["crash_rate_rolling"], linewidth=2.0, label=name)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling crash rate ({window})")
    plt.title("Experiment Comparison: Crash Rate")
    plt.legend(loc="best")
    plt.tight_layout()
    crash_path = os.path.join(output_dir, "comparison_crash_rate_vs_episodes.png")
    plt.savefig(crash_path, dpi=220)
    plt.close()

    # Comparison: laps
    plt.figure(figsize=(10.5, 6.0))
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        laps_rolling = rolling_mean(series["laps"].tolist(), window)
        plt.plot(series["episodes"], laps_rolling, linewidth=2.0, label=name)
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling laps completed ({window})")
    plt.title("Experiment Comparison: Laps")
    plt.legend(loc="best")
    plt.tight_layout()
    laps_path = os.path.join(output_dir, "comparison_laps_vs_episodes.png")
    plt.savefig(laps_path, dpi=220)
    plt.close()

    print(f"[plot] Saved: {reward_path}")
    print(f"[plot] Saved: {crash_path}")
    print(f"[plot] Saved: {laps_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot and compare TD3 training metrics")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOGS_DIR,
        help=f"Base log directory (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experiment folder names under --log-dir, or absolute experiment log paths",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate cross-experiment comparison plots when multiple experiments are loaded",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Rolling window size for smoothed metrics (default: 100)",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default=None,
        help="Output directory for comparison plots (default: <log-dir>/comparison)",
    )
    args = parser.parse_args()

    experiment_dirs = resolve_experiment_dirs(args.log_dir, args.experiments)
    if not experiment_dirs:
        print("[plot] No experiment logs found.")
        return

    loaded = {}
    for raw_name, exp_dir in experiment_dirs.items():
        logs = load_logs(exp_dir)
        if not logs:
            print(f"[plot] Skipping empty/missing logs: {exp_dir}")
            continue

        exp_name = infer_experiment_name(logs, fallback=raw_name)
        loaded[exp_name] = logs
        plot_experiment(logs, output_dir=exp_dir, experiment_name=exp_name, window=max(1, args.window))

    if not loaded:
        print("[plot] No valid logs loaded.")
        return

    if args.compare or len(loaded) > 1:
        comparison_out = args.comparison_output or os.path.join(args.log_dir, "comparison")
        plot_comparison(loaded, output_dir=comparison_out, window=max(1, args.window))

    print(f"[plot] Completed for {len(loaded)} experiment(s).")


if __name__ == "__main__":
    main()
