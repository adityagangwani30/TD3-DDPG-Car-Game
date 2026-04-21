"""plot_metrics.py - Plot publication-quality RL metrics."""

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import LOGS_DIR


PLOT_DPI = 300
FONT_SIZE = 13
TITLE_SIZE = 15
LEGEND_SIZE = 11
LINE_WIDTH = 2.5
RAW_LINE_WIDTH = 1.0
MARKER_SIZE = 5

ALGORITHMS = ("td3", "ddpg")
ALGO_COLORS = {
    "td3": "#1f77b4",
    "ddpg": "#ff7f0e",
}
REWARD_MODE_TO_LABEL = {
    "basic": "R1",
    "shaped": "R2",
    "modified": "R3",
    "tuned": "R4",
}
NOISE_TO_LABEL = {
    0.00: "N1",
    0.02: "N2",
    0.05: "N3",
}


def apply_styling():
    """Apply consistent publication-style formatting across all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": LEGEND_SIZE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


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


def format_experiment_label(experiment_name: str) -> str:
    """Convert verbose experiment names into compact R{reward}_N{noise} labels."""
    name = str(experiment_name).strip().lower()

    compact_match = re.search(r"r([1-4])_n([1-3])", name)
    if compact_match:
        return f"R{compact_match.group(1)}_N{compact_match.group(2)}"

    reward_match = re.search(r"(?:^|_)(basic|shaped|modified|tuned)(?:_|$)", name)
    noise_match = re.search(r"(0\.00|0\.02|0\.05)", name)

    reward_label = REWARD_MODE_TO_LABEL.get(reward_match.group(1), None) if reward_match else None
    noise_label = NOISE_TO_LABEL.get(float(noise_match.group(1)), None) if noise_match else None

    if reward_label and noise_label:
        return f"{reward_label}_{noise_label}"
    if reward_label:
        return reward_label
    if noise_label:
        return noise_label
    return sanitize_name(experiment_name)


def sanitize_name(name: str) -> str:
    """Create filesystem-safe names for output files."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return safe or "experiment"


def _log_file_path(base_log_dir: str, algo: str, experiment_id: str) -> Path:
    return Path(base_log_dir) / algo / experiment_id / "training_log.jsonl"


def load_logs_from_file(log_file: Path) -> list[dict]:
    """Load JSONL logs from a file. Returns an empty list when missing or invalid."""
    if not log_file.exists():
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs


def discover_experiment_ids(base_log_dir: str, algo: str) -> list[str]:
    """Discover experiment IDs under logs/{algo}/ that contain training logs."""
    algo_dir = Path(base_log_dir) / algo
    if not algo_dir.exists():
        return []

    ids = []
    for child in sorted(algo_dir.iterdir()):
        if child.is_dir() and (child / "training_log.jsonl").exists():
            ids.append(child.name)
    return ids


def resolve_experiment_ids(base_log_dir: str, algo: str, experiment_args: list[str] | None) -> list[str]:
    """Resolve explicit experiment IDs or discover them from logs/{algo}/."""
    if experiment_args:
        return [str(exp).strip() for exp in experiment_args]
    return discover_experiment_ids(base_log_dir, algo)


def extract_series(logs: list[dict], rolling_window: int) -> dict[str, np.ndarray]:
    """Extract robust metric series for plotting."""
    episodes = np.array([int(log.get("episode", i + 1)) for i, log in enumerate(logs)], dtype=int)
    reward_total = np.array([float(log.get("reward_total", 0.0)) for log in logs], dtype=float)
    crashes = np.array([float(log.get("collisions", 0.0)) for log in logs], dtype=float)
    laps = np.array([float(log.get("laps_completed", 0.0)) for log in logs], dtype=float)

    reward_smooth = np.array(
        [float(log.get("reward_rolling_avg_100", 0.0)) for log in logs],
        dtype=float,
    )
    if not np.any(reward_smooth):
        reward_smooth = rolling_mean(reward_total.tolist(), rolling_window)

    crash_smooth = rolling_mean(crashes.tolist(), rolling_window)
    laps_smooth = rolling_mean(laps.tolist(), rolling_window)

    return {
        "episodes": episodes,
        "reward_total": reward_total,
        "reward_smooth": reward_smooth,
        "crash_total": crashes,
        "crash_smooth": crash_smooth,
        "laps_total": laps,
        "laps_smooth": laps_smooth,
    }


def _plot_series_with_smoothing(
    ax,
    episodes: np.ndarray,
    raw_values: np.ndarray,
    smooth_values: np.ndarray,
    color: str,
    ylabel: str,
    label: str,
    raw_label: str | None = None,
    smooth_label: str | None = None,
    highlight_peak: bool = False,
):
    """Plot raw and smoothed series with consistent styling."""
    ax.plot(
        episodes,
        raw_values,
        linewidth=RAW_LINE_WIDTH,
        color=color,
        alpha=0.25,
        label=raw_label,
        zorder=1,
    )
    ax.plot(
        episodes,
        smooth_values,
        linewidth=LINE_WIDTH,
        color=color,
        alpha=0.95,
        marker="o",
        markersize=MARKER_SIZE,
        markevery=max(1, len(episodes) // 25),
        label=smooth_label or label,
        zorder=3,
    )

    if highlight_peak and len(episodes) and len(smooth_values):
        peak_idx = int(np.argmax(smooth_values))
        peak_x = episodes[peak_idx]
        peak_y = smooth_values[peak_idx]
        ax.scatter(peak_x, peak_y, s=90, color=color, marker="*", zorder=5)
        ax.annotate(
            "peak",
            xy=(peak_x, peak_y),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            color=color,
        )


def highlight_convergence(ax, episodes: np.ndarray, smoothed_reward: np.ndarray):
    """Shade a likely convergence region based on low rolling variance."""
    if len(episodes) < 10 or len(smoothed_reward) < 10:
        return

    window = max(10, min(25, len(smoothed_reward) // 8))
    rolling_std = np.array(
        [np.std(smoothed_reward[max(0, i - window + 1): i + 1]) for i in range(len(smoothed_reward))],
        dtype=float,
    )
    threshold = max(float(np.nanstd(smoothed_reward)) * 0.20, 0.25)
    stable = rolling_std < threshold

    start_idx = None
    streak = 0
    for idx, is_stable in enumerate(stable):
        streak = streak + 1 if is_stable else 0
        if streak >= 5:
            start_idx = idx - streak + 1
            break

    if start_idx is None:
        start_idx = int(len(episodes) * 0.7)

    start_idx = max(0, min(start_idx, len(episodes) - 1))
    ax.axvspan(episodes[start_idx], episodes[-1], alpha=0.10, color="grey")


def plot_individual_experiment(
    algo: str,
    experiment_id: str,
    logs: list[dict],
    output_dir: str,
    window: int,
    smooth: bool,
):
    """Generate reward, crash, and laps plots for one algorithm experiment."""
    if not logs:
        print(f"[plot][warn] Missing or empty logs for {algo}/{experiment_id}. Skipping.")
        return

    apply_styling()
    os.makedirs(output_dir, exist_ok=True)
    series = extract_series(logs, rolling_window=window)

    episodes = series["episodes"]
    reward_raw = series["reward_total"]
    reward_smooth = series["reward_smooth"]
    crash_raw = series["crash_total"]
    crash_smooth = series["crash_smooth"]
    laps_raw = series["laps_total"]
    laps_smooth = series["laps_smooth"]
    color = ALGO_COLORS[algo]
    label_name = format_experiment_label(experiment_id)

    reward_path = os.path.join(output_dir, "reward_vs_episodes.png")
    crash_path = os.path.join(output_dir, "crash_rate_vs_episodes.png")
    laps_path = os.path.join(output_dir, "laps_vs_episodes.png")

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    _plot_series_with_smoothing(
        ax,
        episodes,
        reward_raw,
        reward_smooth,
        color=color,
        ylabel="Reward",
        label=label_name,
        raw_label=f"{label_name} raw",
        smooth_label=f"{label_name} smoothed",
        highlight_peak=True,
    )
    highlight_convergence(ax, episodes, reward_smooth)
    ax.set_xlabel("Episodes", fontsize=13)
    ax.set_ylabel("Reward", fontsize=13)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(reward_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    _plot_series_with_smoothing(
        ax,
        episodes,
        crash_raw,
        crash_smooth,
        color=color,
        ylabel="Collisions",
        label=label_name,
        raw_label=f"{label_name} raw",
        smooth_label=f"{label_name} smoothed",
    )
    highlight_convergence(ax, episodes, reward_smooth)
    ax.set_xlabel("Episodes", fontsize=13)
    ax.set_ylabel("Collisions", fontsize=13)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(crash_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    _plot_series_with_smoothing(
        ax,
        episodes,
        laps_raw,
        laps_smooth,
        color=color,
        ylabel="Laps completed",
        label=label_name,
        raw_label=f"{label_name} raw",
        smooth_label=f"{label_name} smoothed",
    )
    highlight_convergence(ax, episodes, reward_smooth)
    ax.set_xlabel("Episodes", fontsize=13)
    ax.set_ylabel("Laps completed", fontsize=13)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(laps_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] Saved: {reward_path}")
    print(f"[plot] Saved: {crash_path}")
    print(f"[plot] Saved: {laps_path}")


def _plot_two_algo_metric(
    td3_series: dict[str, np.ndarray],
    ddpg_series: dict[str, np.ndarray],
    key: str,
    ylabel: str,
    output_path: str,
):
    """Plot TD3 vs DDPG for a single metric key."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.plot(
        td3_series["episodes"],
        td3_series[f"{key}_raw"],
        color=ALGO_COLORS["td3"],
        linewidth=RAW_LINE_WIDTH,
        alpha=0.25,
        label=None,
        zorder=1,
    )
    ax.plot(
        td3_series["episodes"],
        td3_series[key],
        color=ALGO_COLORS["td3"],
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markevery=max(1, len(td3_series["episodes"]) // 25),
        label="TD3",
        zorder=3,
    )
    ax.plot(
        ddpg_series["episodes"],
        ddpg_series[f"{key}_raw"],
        color=ALGO_COLORS["ddpg"],
        linewidth=RAW_LINE_WIDTH,
        alpha=0.25,
        label=None,
        zorder=1,
    )
    ax.plot(
        ddpg_series["episodes"],
        ddpg_series[key],
        color=ALGO_COLORS["ddpg"],
        linewidth=LINE_WIDTH,
        marker="s",
        markersize=MARKER_SIZE,
        markevery=max(1, len(ddpg_series["episodes"]) // 25),
        label="DDPG",
        zorder=3,
    )
    highlight_convergence(ax, td3_series["episodes"], td3_series["reward_smooth"])
    ax.set_xlabel("Episodes", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.legend(loc="best", frameon=False)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _aggregate_mean(series_list: list[np.ndarray]) -> np.ndarray:
    """Nan-aware mean across variable-length experiment series."""
    if not series_list:
        return np.array([])

    max_len = max(len(s) for s in series_list)
    stacked = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        stacked[i, : len(s)] = s
    return np.nanmean(stacked, axis=0)


def plot_algo_comparisons(base_log_dir: str, experiment_ids: list[str], output_dir: str, window: int, smooth: bool):
    """Generate per-experiment and aggregate TD3-vs-DDPG comparison plots."""
    apply_styling()
    os.makedirs(output_dir, exist_ok=True)
    per_experiment_dir = os.path.join(output_dir, "per_experiment")
    os.makedirs(per_experiment_dir, exist_ok=True)

    td3_reward_all = []
    ddpg_reward_all = []
    td3_crash_all = []
    ddpg_crash_all = []
    td3_laps_all = []
    ddpg_laps_all = []

    for exp_id in experiment_ids:
        td3_logs = load_logs_from_file(_log_file_path(base_log_dir, "td3", exp_id))
        ddpg_logs = load_logs_from_file(_log_file_path(base_log_dir, "ddpg", exp_id))

        if not td3_logs or not ddpg_logs:
            print(f"[plot][warn] Missing TD3/DDPG pair for {exp_id}. Skipping per-experiment comparison.")
            continue

        td3_series = extract_series(td3_logs, rolling_window=window)
        ddpg_series = extract_series(ddpg_logs, rolling_window=window)

        td3_reward_all.append(td3_series["reward_smooth"])
        ddpg_reward_all.append(ddpg_series["reward_smooth"])
        td3_crash_all.append(td3_series["crash_smooth"])
        ddpg_crash_all.append(ddpg_series["crash_smooth"])
        td3_laps_all.append(td3_series["laps_smooth"])
        ddpg_laps_all.append(ddpg_series["laps_smooth"])

        td3_plot = {
            **td3_series,
            "reward_plot": td3_series["reward_smooth"],
            "reward_plot_raw": td3_series["reward_total"],
            "crash_plot": td3_series["crash_smooth"],
            "crash_plot_raw": td3_series["crash_total"],
            "laps_plot": td3_series["laps_smooth"],
            "laps_plot_raw": td3_series["laps_total"],
        }
        ddpg_plot = {
            **ddpg_series,
            "reward_plot": ddpg_series["reward_smooth"],
            "reward_plot_raw": ddpg_series["reward_total"],
            "crash_plot": ddpg_series["crash_smooth"],
            "crash_plot_raw": ddpg_series["crash_total"],
            "laps_plot": ddpg_series["laps_smooth"],
            "laps_plot_raw": ddpg_series["laps_total"],
        }

        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="reward_plot",
            ylabel="Reward",
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_reward.png"),
        )
        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="crash_plot",
            ylabel="Collisions",
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_crash.png"),
        )
        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="laps_plot",
            ylabel="Laps completed",
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_laps.png"),
        )

    if not td3_reward_all or not ddpg_reward_all:
        print("[plot][warn] No complete TD3/DDPG experiment pairs found for aggregate comparison.")
        return

    td3_reward_avg = _aggregate_mean(td3_reward_all)
    ddpg_reward_avg = _aggregate_mean(ddpg_reward_all)
    td3_crash_avg = _aggregate_mean(td3_crash_all)
    ddpg_crash_avg = _aggregate_mean(ddpg_crash_all)
    td3_laps_avg = _aggregate_mean(td3_laps_all)
    ddpg_laps_avg = _aggregate_mean(ddpg_laps_all)

    reward_ep = np.arange(1, len(td3_reward_avg) + 1)
    crash_ep = np.arange(1, len(td3_crash_avg) + 1)
    laps_ep = np.arange(1, len(td3_laps_avg) + 1)

    _plot_two_algo_metric(
        {"episodes": reward_ep, "reward_plot": td3_reward_avg, "reward_plot_raw": td3_reward_avg},
        {"episodes": reward_ep, "reward_plot": ddpg_reward_avg, "reward_plot_raw": ddpg_reward_avg},
        key="reward_plot",
        ylabel="Average reward",
        output_path=os.path.join(output_dir, "td3_vs_ddpg_reward.png"),
    )
    _plot_two_algo_metric(
        {"episodes": crash_ep, "crash_plot": td3_crash_avg, "crash_plot_raw": td3_crash_avg},
        {"episodes": crash_ep, "crash_plot": ddpg_crash_avg, "crash_plot_raw": ddpg_crash_avg},
        key="crash_plot",
        ylabel="Average collisions",
        output_path=os.path.join(output_dir, "td3_vs_ddpg_crash.png"),
    )
    _plot_two_algo_metric(
        {"episodes": laps_ep, "laps_plot": td3_laps_avg, "laps_plot_raw": td3_laps_avg},
        {"episodes": laps_ep, "laps_plot": ddpg_laps_avg, "laps_plot_raw": ddpg_laps_avg},
        key="laps_plot",
        ylabel="Average laps completed",
        output_path=os.path.join(output_dir, "td3_vs_ddpg_laps.png"),
    )

    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_reward.png')}")
    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_crash.png')}")
    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_laps.png')}")


def main():
    parser = argparse.ArgumentParser(description="Plot TD3/DDPG training metrics")
    parser.add_argument(
        "--algo",
        choices=["td3", "ddpg"],
        default=None,
        help="Algorithm to plot in single-algorithm mode",
    )
    parser.add_argument(
        "--compare-algos",
        action="store_true",
        help="Generate TD3 vs DDPG comparison plots (ignores --algo)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=LOGS_DIR,
        help=f"Base logs directory (default: {LOGS_DIR})",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional experiment IDs, e.g. R1_N1 R1_N2",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Smoothing window size (default: 100)",
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Apply rolling smoothing to plotted curves",
    )
    parser.add_argument(
        "--individual-output",
        type=str,
        default=os.path.join("results", "plots"),
        help="Base output directory for individual plots (default: results/plots)",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default=os.path.join("results", "plots", "comparison"),
        help="Output directory for TD3 vs DDPG plots (default: results/plots/comparison)",
    )
    args = parser.parse_args()

    window = max(1, int(args.window))

    if not args.compare_algos and args.algo is None:
        raise ValueError("You must specify --algo {td3, ddpg}")

    if args.compare_algos:
        if args.experiments:
            experiment_ids = [str(exp).strip() for exp in args.experiments]
        else:
            td3_ids = set(discover_experiment_ids(args.log_dir, "td3"))
            ddpg_ids = set(discover_experiment_ids(args.log_dir, "ddpg"))
            experiment_ids = sorted(td3_ids.union(ddpg_ids))

        if not experiment_ids:
            print("[plot][warn] No experiments found for TD3/DDPG comparison.")
            return

        plot_algo_comparisons(
            base_log_dir=args.log_dir,
            experiment_ids=experiment_ids,
            output_dir=args.comparison_output,
            window=window,
            smooth=args.smooth,
        )
        print(f"[plot] Comparison mode complete for up to {len(experiment_ids)} experiments.")
        return

    algo = args.algo
    experiment_ids = resolve_experiment_ids(args.log_dir, algo, args.experiments)
    if not experiment_ids:
        print(f"[plot][warn] No experiment logs found under logs/{algo}/")
        return

    plotted = 0
    for exp_id in experiment_ids:
        log_file = _log_file_path(args.log_dir, algo, exp_id)
        logs = load_logs_from_file(log_file)
        if not logs:
            print(f"[plot][warn] Missing logs file: {log_file}")
            continue

        out_dir = os.path.join(args.individual_output, algo, "individual", sanitize_name(exp_id))
        plot_individual_experiment(
            algo=algo,
            experiment_id=exp_id,
            logs=logs,
            output_dir=out_dir,
            window=window,
            smooth=args.smooth,
        )
        plotted += 1

    if plotted == 0:
        print(f"[plot][warn] No valid {algo.upper()} logs were plotted.")
        return

    print(f"[plot] Completed {algo.upper()} plotting for {plotted} experiment(s).")


if __name__ == "__main__":
    main()
