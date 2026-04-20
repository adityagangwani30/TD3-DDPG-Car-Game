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
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import LOGS_DIR


PLOT_DPI = 300
FONT_SIZE = 13
TITLE_SIZE = 15
LEGEND_SIZE = 11
LINE_WIDTH = 2.2
MARKER_SIZE = 6
MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")
COLORS = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
)


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


def configure_plot_style():
    """Apply consistent publication-style formatting across all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": LEGEND_SIZE,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        }
    )


def marker_every(num_points: int) -> int:
    """Pick marker interval to keep plots readable without clutter."""
    if num_points >= 500:
        return 10
    if num_points >= 100:
        return 5
    return max(1, num_points // 20)


def annotate_peak_and_final(x: np.ndarray, y: np.ndarray, color: str):
    """Highlight the maximum and final points for quick visual interpretation."""
    if len(x) == 0 or len(y) == 0:
        return
    peak_idx = int(np.argmax(y))
    plt.scatter(x[peak_idx], y[peak_idx], color=color, marker="*", s=130, zorder=5, label="Max")
    plt.scatter(x[-1], y[-1], color=color, marker="X", s=80, zorder=5, label="Final")


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

    configure_plot_style()
    markevery = marker_every(len(episodes))

    # Reward vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        episodes,
        reward_total,
        color=COLORS[0],
        alpha=0.4,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markevery=markevery,
        label="Episode reward",
    )
    plt.plot(
        episodes,
        reward_rolling,
        color=COLORS[1],
        linewidth=LINE_WIDTH,
        marker="s",
        markersize=MARKER_SIZE,
        markevery=markevery,
        label=f"Rolling avg ({window})",
    )
    annotate_peak_and_final(episodes, reward_rolling, color=COLORS[1])
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(f"Reward vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    reward_path = os.path.join(output_dir, f"{safe_name}_reward_vs_episodes.png")
    plt.savefig(reward_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # Crash rate vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        episodes,
        crashes,
        color=COLORS[5],
        alpha=0.35,
        linewidth=LINE_WIDTH,
        marker="^",
        markersize=MARKER_SIZE,
        markevery=markevery,
        label="Crashes per episode",
    )
    plt.plot(
        episodes,
        crash_rate_rolling,
        color=COLORS[3],
        linewidth=LINE_WIDTH,
        marker="D",
        markersize=MARKER_SIZE,
        markevery=markevery,
        label=f"Rolling crash rate ({window})",
    )
    annotate_peak_and_final(episodes, crash_rate_rolling, color=COLORS[3])
    plt.xlabel("Episode")
    plt.ylabel("Crash count / rate")
    plt.title(f"Crash Rate vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    crash_path = os.path.join(output_dir, f"{safe_name}_crash_rate_vs_episodes.png")
    plt.savefig(crash_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # Laps vs episodes
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        episodes,
        laps,
        color=COLORS[2],
        linewidth=LINE_WIDTH,
        marker="P",
        markersize=MARKER_SIZE,
        markevery=markevery,
        label="Laps completed",
    )
    annotate_peak_and_final(episodes, laps, color=COLORS[2])
    plt.xlabel("Episode")
    plt.ylabel("Laps completed")
    plt.title(f"Laps vs Episodes ({experiment_name})")
    plt.legend(loc="best")
    plt.tight_layout()
    laps_path = os.path.join(output_dir, f"{safe_name}_laps_vs_episodes.png")
    plt.savefig(laps_path, dpi=PLOT_DPI, bbox_inches="tight")
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
    configure_plot_style()
    marker_cycle = cycle(MARKERS)
    color_cycle = cycle(COLORS)

    # Comparison: reward
    plt.figure(figsize=(10.5, 6.0))
    marker_cycle = cycle(MARKERS)
    color_cycle = cycle(COLORS)
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        markevery = marker_every(len(series["episodes"]))
        plt.plot(
            series["episodes"],
            series["reward_rolling"],
            linewidth=LINE_WIDTH,
            marker=next(marker_cycle),
            markersize=MARKER_SIZE,
            markevery=markevery,
            color=next(color_cycle),
            label=name,
        )
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling average reward ({window})")
    plt.title("Experiment Comparison: Reward")
    plt.legend(loc="best")
    plt.tight_layout()
    reward_path = os.path.join(output_dir, "comparison_reward_vs_episodes.png")
    plt.savefig(reward_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # Comparison: crash rate
    plt.figure(figsize=(10.5, 6.0))
    marker_cycle = cycle(MARKERS)
    color_cycle = cycle(COLORS)
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        markevery = marker_every(len(series["episodes"]))
        plt.plot(
            series["episodes"],
            series["crash_rate_rolling"],
            linewidth=LINE_WIDTH,
            marker=next(marker_cycle),
            markersize=MARKER_SIZE,
            markevery=markevery,
            color=next(color_cycle),
            label=name,
        )
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling crash rate ({window})")
    plt.title("Experiment Comparison: Crash Rate")
    plt.legend(loc="best")
    plt.tight_layout()
    crash_path = os.path.join(output_dir, "comparison_crash_rate_vs_episodes.png")
    plt.savefig(crash_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()

    # Comparison: laps
    plt.figure(figsize=(10.5, 6.0))
    marker_cycle = cycle(MARKERS)
    color_cycle = cycle(COLORS)
    for name, logs in experiment_logs.items():
        series = extract_series(logs, rolling_window=window)
        laps_rolling = rolling_mean(series["laps"].tolist(), window)
        markevery = marker_every(len(series["episodes"]))
        plt.plot(
            series["episodes"],
            laps_rolling,
            linewidth=LINE_WIDTH,
            marker=next(marker_cycle),
            markersize=MARKER_SIZE,
            markevery=markevery,
            color=next(color_cycle),
            label=name,
        )
    plt.xlabel("Episode")
    plt.ylabel(f"Rolling laps completed ({window})")
    plt.title("Experiment Comparison: Laps")
    plt.legend(loc="best")
    plt.tight_layout()
    laps_path = os.path.join(output_dir, "comparison_laps_vs_episodes.png")
    plt.savefig(laps_path, dpi=PLOT_DPI, bbox_inches="tight")
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
        "--individual-output",
        type=str,
        default=os.path.join("results", "plots", "individual"),
        help="Output base directory for per-experiment plots (default: results/plots/individual)",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default=os.path.join("results", "plots", "comparison"),
        help="Output directory for comparison plots (default: results/plots/comparison)",
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
        individual_exp_dir = os.path.join(args.individual_output, sanitize_name(raw_name))
        os.makedirs(individual_exp_dir, exist_ok=True)
        plot_experiment(
            logs,
            output_dir=individual_exp_dir,
            experiment_name=exp_name,
            window=max(1, args.window),
        )

    if not loaded:
        print("[plot] No valid logs loaded.")
        return

    if args.compare or len(loaded) > 1:
        comparison_out = args.comparison_output
        plot_comparison(loaded, output_dir=comparison_out, window=max(1, args.window))

    print(f"[plot] Completed for {len(loaded)} experiment(s).")


if __name__ == "__main__":
    main()
