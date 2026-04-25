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
FONT_SIZE = 12
TITLE_SIZE = 14
LEGEND_SIZE = 11
LINE_WIDTH = 2.5

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
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
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


def rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
    """Compute a NaN-aware moving average with a small-sample fallback."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([])

    window = max(1, int(window))
    smoothed = np.empty_like(arr, dtype=float)
    for idx in range(arr.size):
        start = max(0, idx - window + 1)
        window_values = arr[start : idx + 1]
        finite_values = window_values[np.isfinite(window_values)]
        smoothed[idx] = float(np.mean(finite_values)) if finite_values.size else np.nan
    return smoothed


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


def _experiment_log_dir(base_log_dir: str, algo: str, experiment_id: str) -> Path:
    return Path(base_log_dir) / algo / experiment_id


def _seed_log_files(base_log_dir: str, algo: str, experiment_id: str) -> list[Path]:
    """Return all available training logs for an experiment (seeded or legacy layout)."""
    exp_dir = _experiment_log_dir(base_log_dir, algo, experiment_id)
    if not exp_dir.exists():
        return []

    legacy_log = exp_dir / "training_log.jsonl"
    seed_logs = sorted(exp_dir.glob("seed_*/training_log.jsonl"))

    if seed_logs:
        return [p for p in seed_logs if p.is_file()]
    if legacy_log.exists():
        return [legacy_log]
    return []


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
        if not child.is_dir():
            continue
        has_legacy_log = (child / "training_log.jsonl").exists()
        has_seed_logs = any(child.glob("seed_*/training_log.jsonl"))
        if has_legacy_log or has_seed_logs:
            ids.append(child.name)
    return ids


def resolve_experiment_ids(base_log_dir: str, algo: str, experiment_args: list[str] | None) -> list[str]:
    """Resolve explicit experiment IDs or discover them from logs/{algo}/."""
    if experiment_args:
        return [str(exp).strip() for exp in experiment_args]
    return discover_experiment_ids(base_log_dir, algo)


def extract_series(logs: list[dict]) -> dict[str, np.ndarray]:
    """Extract raw metric series for plotting."""
    episodes = np.array([int(log.get("episode", i + 1)) for i, log in enumerate(logs)], dtype=int)
    reward_total = np.array([float(log.get("reward_total", 0.0)) for log in logs], dtype=float)
    crashes = np.array([float(log.get("collisions", 0.0)) for log in logs], dtype=float)
    laps = np.array([float(log.get("laps_completed", 0.0)) for log in logs], dtype=float)

    return {
        "episodes": episodes,
        "reward_total": reward_total,
        "crash_total": crashes,
        "laps_total": laps,
    }


def _aggregate_mean_std(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Nan-aware mean/std across variable-length series."""
    if not series_list:
        return np.array([]), np.array([])

    max_len = max(len(s) for s in series_list)
    stacked = np.full((len(series_list), max_len), np.nan, dtype=float)
    for i, s in enumerate(series_list):
        stacked[i, : len(s)] = s

    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, std


def load_experiment_series(
    base_log_dir: str,
    algo: str,
    experiment_id: str,
    rolling_window: int,
) -> dict[str, np.ndarray] | None:
    """Load and aggregate one experiment across all available seed logs."""
    log_files = _seed_log_files(base_log_dir, algo, experiment_id)
    if not log_files:
        return None

    series_per_seed = []
    for log_file in log_files:
        logs = load_logs_from_file(log_file)
        if logs:
            series_per_seed.append(extract_series(logs))

    if not series_per_seed:
        return None

    reward_total_mean, reward_total_std = _aggregate_mean_std(
        [series["reward_total"] for series in series_per_seed]
    )
    crash_total_mean, crash_total_std = _aggregate_mean_std(
        [series["crash_total"] for series in series_per_seed]
    )
    laps_total_mean, laps_total_std = _aggregate_mean_std(
        [series["laps_total"] for series in series_per_seed]
    )

    reward_smooth_mean = rolling_mean(reward_total_mean, rolling_window)
    crash_smooth_mean = rolling_mean(crash_total_mean, rolling_window)
    laps_smooth_mean = rolling_mean(laps_total_mean, rolling_window)

    episodes = np.arange(1, len(reward_total_mean) + 1)
    return {
        "episodes": episodes,
        "reward_total": reward_total_mean,
        "reward_total_std": reward_total_std,
        "reward_smooth": reward_smooth_mean,
        "crash_total": crash_total_mean,
        "crash_total_std": crash_total_std,
        "crash_smooth": crash_smooth_mean,
        "laps_total": laps_total_mean,
        "laps_total_std": laps_total_std,
        "laps_smooth": laps_smooth_mean,
    }


def _plot_mean_with_std(
    ax,
    episodes: np.ndarray,
    mean_values: np.ndarray,
    std_values: np.ndarray | None,
    color: str,
    label: str,
):
    """Plot a mean curve with an optional standard-deviation band."""
    ax.plot(
        episodes,
        mean_values,
        linewidth=LINE_WIDTH,
        color=color,
        alpha=0.98,
        label=label,
        zorder=3,
    )
    if std_values is not None and len(std_values) == len(mean_values):
        ax.fill_between(
            episodes,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )


def _format_plot_title(metric_name: str, context_label: str | None = None) -> str:
    title = f"{metric_name} vs Episodes"
    if context_label:
        return f"{title} ({context_label})"
    return title


def _set_plot_labels(ax, title: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel("Episodes", fontsize=FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.margins(x=0.02)


def _experiment_sort_key(experiment_id: str) -> tuple[int, int, str]:
    match = re.search(r"r(\d+)_n(\d+)", str(experiment_id).lower())
    if match:
        return int(match.group(1)), int(match.group(2)), str(experiment_id)
    return 99, 99, str(experiment_id)


def _sorted_experiment_ids(experiment_ids: list[str]) -> list[str]:
    return sorted({str(exp).strip() for exp in experiment_ids}, key=_experiment_sort_key)


def _experiment_color_map(experiment_ids: list[str]) -> dict[str, str]:
    ordered_ids = _sorted_experiment_ids(experiment_ids)
    if not ordered_ids:
        return {}

    cmap = plt.get_cmap("tab20", len(ordered_ids))
    return {experiment_id: cmap(index) for index, experiment_id in enumerate(ordered_ids)}


def find_convergence_episode(episodes: np.ndarray, smoothed_reward: np.ndarray) -> int | None:
    """Estimate the episode where the reward curve stabilizes."""
    if len(episodes) < 10 or len(smoothed_reward) < 10:
        return None

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
    return int(episodes[start_idx])


def add_convergence_marker(ax, episodes: np.ndarray, smoothed_reward: np.ndarray):
    """Add a dashed convergence line to a reward plot."""
    convergence_episode = find_convergence_episode(episodes, smoothed_reward)
    if convergence_episode is None:
        return

    ax.axvline(
        convergence_episode,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.6,
        alpha=0.95,
        label="Convergence",
        zorder=4,
    )


def _plot_aggregate_experiment_grid(
    algo: str,
    experiment_series: dict[str, dict[str, np.ndarray]],
    output_dir: str,
    window: int,
):
    """Plot all experiment curves for one algorithm across reward, crash, and laps."""
    if not experiment_series:
        print(f"[plot][warn] No {algo.upper()} experiment series available for aggregate plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    apply_styling()

    colors = _experiment_color_map(list(experiment_series.keys()))
    ordered_experiments = _sorted_experiment_ids(list(experiment_series.keys()))

    metric_specs = [
        (
            "reward_plot",
            "Average Reward",
            f"{algo.upper()} Performance Across All Experiments (Reward)",
            os.path.join(output_dir, f"{algo}_reward_all.png"),
        ),
        (
            "crash_plot",
            "Crash Rate (%)",
            f"{algo.upper()} Crash Rate Comparison Across Experiments",
            os.path.join(output_dir, f"{algo}_crash_all.png"),
        ),
        (
            "laps_plot",
            "Laps per Episode",
            f"{algo.upper()} Laps Comparison Across Experiments",
            os.path.join(output_dir, f"{algo}_laps_all.png"),
        ),
    ]

    for metric_key, ylabel, title, output_path in metric_specs:
        fig, ax = plt.subplots(figsize=(10, 6))
        for experiment_id in ordered_experiments:
            series = experiment_series[experiment_id]
            mean_values = series[metric_key]
            std_values = series.get(f"{metric_key}_std")
            label = format_experiment_label(experiment_id)
            color = colors.get(experiment_id, "#333333")
            ax.plot(
                series["episodes"],
                mean_values,
                color=color,
                linewidth=2.0,
                alpha=0.95,
                label=label,
                zorder=3,
            )
            if std_values is not None and len(std_values) == len(mean_values):
                ax.fill_between(
                    series["episodes"],
                    mean_values - std_values,
                    mean_values + std_values,
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                    zorder=2,
                )

        ax.set_title(title)
        ax.set_xlabel("Episodes", fontsize=FONT_SIZE)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        ax.margins(x=0.02)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=4,
            fontsize=9,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            edgecolor="#d0d0d0",
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        print(f"[plot] Saved: {output_path}")


def plot_individual_experiment(
    algo: str,
    experiment_id: str,
    series: dict[str, np.ndarray],
    output_dir: str,
    window: int,
    smooth: bool,
):
    """Generate reward, crash, and laps plots for one algorithm experiment."""
    if not series:
        print(f"[plot][warn] Missing or empty logs for {algo}/{experiment_id}. Skipping.")
        return

    apply_styling()
    os.makedirs(output_dir, exist_ok=True)

    episodes = series["episodes"]
    reward_mean = series["reward_smooth"]
    reward_std = series.get("reward_total_std")
    crash_mean = series["crash_smooth"]
    crash_std = series.get("crash_total_std")
    laps_mean = series["laps_smooth"]
    laps_std = series.get("laps_total_std")
    color = ALGO_COLORS[algo]
    label_name = format_experiment_label(experiment_id)
    reward_title = _format_plot_title("Reward", label_name)
    crash_title = _format_plot_title("Crash Rate", label_name)
    laps_title = _format_plot_title("Laps per Episode", label_name)

    reward_path = os.path.join(output_dir, "reward_vs_episodes.png")
    crash_path = os.path.join(output_dir, "crash_rate_vs_episodes.png")
    laps_path = os.path.join(output_dir, "laps_vs_episodes.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_std(
        ax,
        episodes,
        reward_mean,
        reward_std,
        color=color,
        label=label_name,
    )
    add_convergence_marker(ax, episodes, reward_mean)
    _set_plot_labels(ax, reward_title, "Average Reward")
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(reward_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_std(
        ax,
        episodes,
        crash_mean,
        crash_std,
        color=color,
        label=label_name,
    )
    _set_plot_labels(ax, crash_title, "Crash Rate (%)")
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(crash_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_std(
        ax,
        episodes,
        laps_mean,
        laps_std,
        color=color,
        label=label_name,
    )
    _set_plot_labels(ax, laps_title, "Laps per Episode")
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(laps_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    print(f"[plot] Saved: {reward_path}")
    print(f"[plot] Saved: {crash_path}")
    print(f"[plot] Saved: {laps_path}")


def _plot_two_algo_metric(
    td3_series: dict[str, np.ndarray],
    ddpg_series: dict[str, np.ndarray],
    key: str,
    ylabel: str,
    title: str,
    output_path: str,
):
    """Plot TD3 vs DDPG for a single metric key."""
    fig, ax = plt.subplots(figsize=(8, 5))
    td3_std = td3_series.get(f"{key}_std")
    ddpg_std = ddpg_series.get(f"{key}_std")
    _plot_mean_with_std(
        ax,
        td3_series["episodes"],
        td3_series[key],
        td3_std,
        color=ALGO_COLORS["td3"],
        label="TD3",
    )
    _plot_mean_with_std(
        ax,
        ddpg_series["episodes"],
        ddpg_series[key],
        ddpg_std,
        color=ALGO_COLORS["ddpg"],
        label="DDPG",
    )
    _set_plot_labels(ax, title, ylabel)
    if key == "reward_plot":
        add_convergence_marker(ax, td3_series["episodes"], td3_series[key])
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#d0d0d0")
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.08)
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
        td3_series = load_experiment_series(
            base_log_dir=base_log_dir,
            algo="td3",
            experiment_id=exp_id,
            rolling_window=window,
        )
        ddpg_series = load_experiment_series(
            base_log_dir=base_log_dir,
            algo="ddpg",
            experiment_id=exp_id,
            rolling_window=window,
        )

        if td3_series is None or ddpg_series is None:
            print(f"[plot][warn] Missing TD3/DDPG pair for {exp_id}. Skipping per-experiment comparison.")
            continue

        td3_reward_all.append(td3_series["reward_total"])
        ddpg_reward_all.append(ddpg_series["reward_total"])
        td3_crash_all.append(td3_series["crash_total"])
        ddpg_crash_all.append(ddpg_series["crash_total"])
        td3_laps_all.append(td3_series["laps_total"])
        ddpg_laps_all.append(ddpg_series["laps_total"])

        td3_plot = {
            **td3_series,
            "reward_plot": td3_series["reward_smooth"],
            "reward_plot_std": td3_series["reward_total_std"],
            "crash_plot": td3_series["crash_smooth"],
            "crash_plot_std": td3_series["crash_total_std"],
            "laps_plot": td3_series["laps_smooth"],
            "laps_plot_std": td3_series["laps_total_std"],
        }
        ddpg_plot = {
            **ddpg_series,
            "reward_plot": ddpg_series["reward_smooth"],
            "reward_plot_std": ddpg_series["reward_total_std"],
            "crash_plot": ddpg_series["crash_smooth"],
            "crash_plot_std": ddpg_series["crash_total_std"],
            "laps_plot": ddpg_series["laps_smooth"],
            "laps_plot_std": ddpg_series["laps_total_std"],
        }

        comparison_label = format_experiment_label(exp_id)
        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="reward_plot",
            ylabel="Average Reward",
            title=_format_plot_title("Reward", comparison_label),
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_reward.png"),
        )
        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="crash_plot",
            ylabel="Crash Rate (%)",
            title=_format_plot_title("Crash Rate", comparison_label),
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_crash.png"),
        )
        _plot_two_algo_metric(
            td3_plot,
            ddpg_plot,
            key="laps_plot",
            ylabel="Laps per Episode",
            title=_format_plot_title("Laps per Episode", comparison_label),
            output_path=os.path.join(per_experiment_dir, f"{sanitize_name(exp_id)}_td3_vs_ddpg_laps.png"),
        )

    if not td3_reward_all or not ddpg_reward_all:
        print("[plot][warn] No complete TD3/DDPG experiment pairs found for aggregate comparison.")
        return

    td3_reward_avg, td3_reward_std = _aggregate_mean_std(td3_reward_all)
    ddpg_reward_avg, ddpg_reward_std = _aggregate_mean_std(ddpg_reward_all)
    td3_crash_avg, td3_crash_std = _aggregate_mean_std(td3_crash_all)
    ddpg_crash_avg, ddpg_crash_std = _aggregate_mean_std(ddpg_crash_all)
    td3_laps_avg, td3_laps_std = _aggregate_mean_std(td3_laps_all)
    ddpg_laps_avg, ddpg_laps_std = _aggregate_mean_std(ddpg_laps_all)

    reward_ep = np.arange(1, len(td3_reward_avg) + 1)
    crash_ep = np.arange(1, len(td3_crash_avg) + 1)
    laps_ep = np.arange(1, len(td3_laps_avg) + 1)

    td3_reward_avg = rolling_mean(td3_reward_avg, window)
    ddpg_reward_avg = rolling_mean(ddpg_reward_avg, window)
    td3_crash_avg = rolling_mean(td3_crash_avg, window)
    ddpg_crash_avg = rolling_mean(ddpg_crash_avg, window)
    td3_laps_avg = rolling_mean(td3_laps_avg, window)
    ddpg_laps_avg = rolling_mean(ddpg_laps_avg, window)

    _plot_two_algo_metric(
        {"episodes": reward_ep, "reward_plot": td3_reward_avg, "reward_plot_std": td3_reward_std},
        {"episodes": reward_ep, "reward_plot": ddpg_reward_avg, "reward_plot_std": ddpg_reward_std},
        key="reward_plot",
        ylabel="Average Reward",
        title=_format_plot_title("Reward", "All Experiments"),
        output_path=os.path.join(output_dir, "td3_vs_ddpg_reward.png"),
    )
    _plot_two_algo_metric(
        {"episodes": crash_ep, "crash_plot": td3_crash_avg, "crash_plot_std": td3_crash_std},
        {"episodes": crash_ep, "crash_plot": ddpg_crash_avg, "crash_plot_std": ddpg_crash_std},
        key="crash_plot",
        ylabel="Crash Rate (%)",
        title=_format_plot_title("Crash Rate", "All Experiments"),
        output_path=os.path.join(output_dir, "td3_vs_ddpg_crash.png"),
    )
    _plot_two_algo_metric(
        {"episodes": laps_ep, "laps_plot": td3_laps_avg, "laps_plot_std": td3_laps_std},
        {"episodes": laps_ep, "laps_plot": ddpg_laps_avg, "laps_plot_std": ddpg_laps_std},
        key="laps_plot",
        ylabel="Laps per Episode",
        title=_format_plot_title("Laps per Episode", "All Experiments"),
        output_path=os.path.join(output_dir, "td3_vs_ddpg_laps.png"),
    )

    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_reward.png')}")
    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_crash.png')}")
    print(f"[plot] Saved: {os.path.join(output_dir, 'td3_vs_ddpg_laps.png')}")


def plot_all_experiment_aggregates(
    base_log_dir: str,
    experiment_ids: list[str],
    output_dir: str,
    window: int,
):
    """Generate aggregate TD3 and DDPG plots across all experiments."""
    ordered_experiments = _sorted_experiment_ids(experiment_ids)
    if not ordered_experiments:
        print("[plot][warn] No experiments available for aggregate plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for algo in ALGORITHMS:
        algo_series: dict[str, dict[str, np.ndarray]] = {}
        for exp_id in ordered_experiments:
            series = load_experiment_series(
                base_log_dir=base_log_dir,
                algo=algo,
                experiment_id=exp_id,
                rolling_window=window,
            )
            if series is None:
                continue
            algo_series[exp_id] = {
                **series,
                "reward_plot": series["reward_smooth"],
                "reward_plot_std": series["reward_total_std"],
                "crash_plot": series["crash_smooth"],
                "crash_plot_std": series["crash_total_std"],
                "laps_plot": series["laps_smooth"],
                "laps_plot_std": series["laps_total_std"],
            }

        _plot_aggregate_experiment_grid(
            algo=algo,
            experiment_series=algo_series,
            output_dir=output_dir,
            window=window,
        )


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
    parser.add_argument(
        "--aggregate-output",
        type=str,
        default=os.path.join("results", "aggregate"),
        help="Output directory for TD3/DDPG aggregate experiment plots (default: results/aggregate)",
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
        plot_all_experiment_aggregates(
            base_log_dir=args.log_dir,
            experiment_ids=experiment_ids,
            output_dir=args.aggregate_output,
            window=window,
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
        series = load_experiment_series(
            base_log_dir=args.log_dir,
            algo=algo,
            experiment_id=exp_id,
            rolling_window=window,
        )
        if series is None:
            print(f"[plot][warn] Missing logs for: {algo}/{exp_id}")
            continue

        out_dir = os.path.join(args.individual_output, algo, "individual", sanitize_name(exp_id))
        plot_individual_experiment(
            algo=algo,
            experiment_id=exp_id,
            series=series,
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
