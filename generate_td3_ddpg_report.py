"""generate_td3_ddpg_report.py - Build a PDF report for TD3 vs DDPG experiments.

Pipeline:
    logs -> metrics extraction -> NVIDIA AI analysis -> PDF report

The script is intentionally defensive:
- It skips missing or incomplete experiments.
- It never sends images to the LLM.
- It falls back to deterministic summaries when the NVIDIA API is unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from textwrap import fill
from typing import Any
from urllib import error, request
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_OUTPUT_FILE = Path("td3_ddpg_research_report.pdf")
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_LAST_N = 100
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_STABILITY_WINDOW = 10
DEFAULT_MAX_RETRIES = 3

ALGORITHMS = ("td3", "ddpg")
REWARD_LEVELS = ("R1", "R2", "R3", "R4")
NOISE_LEVELS = ("N1", "N2", "N3")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if it exists."""
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)


@dataclass
class SeedMetrics:
    """Metrics computed for one seed run of an experiment."""

    seed: str
    num_episodes: int
    avg_reward_last_n: float
    avg_crash_last_n: float
    avg_laps_last_n: float
    max_reward: float
    convergence_episode: int
    final_reward: float
    reward_std: float
    crash_std: float
    laps_std: float


@dataclass
class AlgorithmMetrics:
    """Aggregated metrics for one algorithm within one experiment."""

    algorithm: str
    experiment: str
    seeds: list[SeedMetrics] = field(default_factory=list)
    avg_reward_last_n_mean: float | None = None
    avg_reward_last_n_variance: float | None = None
    avg_crash_last_n_mean: float | None = None
    avg_crash_last_n_variance: float | None = None
    avg_laps_last_n_mean: float | None = None
    avg_laps_last_n_variance: float | None = None
    reward_std_mean: float | None = None
    reward_std_variance: float | None = None
    max_reward_mean: float | None = None
    max_reward_variance: float | None = None
    convergence_episode_mean: float | None = None
    convergence_episode_variance: float | None = None
    final_reward_mean: float | None = None
    final_reward_variance: float | None = None
    num_episodes_mean: float | None = None
    num_episodes_variance: float | None = None
    status: str = "missing"

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a compact serialisable view for LLM prompts."""
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "seed_count": len(self.seeds),
            "avg_reward_last_n_mean": self.avg_reward_last_n_mean,
            "avg_reward_last_n_variance": self.avg_reward_last_n_variance,
            "avg_crash_last_n_mean": self.avg_crash_last_n_mean,
            "avg_crash_last_n_variance": self.avg_crash_last_n_variance,
            "avg_laps_last_n_mean": self.avg_laps_last_n_mean,
            "avg_laps_last_n_variance": self.avg_laps_last_n_variance,
            "reward_std_mean": self.reward_std_mean,
            "reward_std_variance": self.reward_std_variance,
            "max_reward_mean": self.max_reward_mean,
            "max_reward_variance": self.max_reward_variance,
            "convergence_episode_mean": self.convergence_episode_mean,
            "convergence_episode_variance": self.convergence_episode_variance,
            "final_reward_mean": self.final_reward_mean,
            "final_reward_variance": self.final_reward_variance,
            "num_episodes_mean": self.num_episodes_mean,
            "num_episodes_variance": self.num_episodes_variance,
        }


@dataclass
class ExperimentReport:
    """All metrics for a single experiment across algorithms."""

    experiment: str
    td3: AlgorithmMetrics | None
    ddpg: AlgorithmMetrics | None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "td3": self.td3.to_prompt_dict() if self.td3 else None,
            "ddpg": self.ddpg.to_prompt_dict() if self.ddpg else None,
        }


def load_jsonl_logs(log_file: Path) -> list[dict[str, Any]]:
    """Load JSONL logs from disk while ignoring malformed lines."""
    if not log_file.exists():
        return []

    logs: list[dict[str, Any]] = []
    try:
        with log_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    logs.append(payload)
    except OSError:
        return []
    return logs


def rolling_mean(values: list[float], window: int) -> np.ndarray:
    """Compute a rolling mean with a warm-up period."""
    if not values:
        return np.array([], dtype=float)

    window = max(1, int(window))
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.array([np.mean(arr[: i + 1]) for i in range(len(arr))], dtype=float)

    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(arr, kernel, mode="valid")
    warmup = np.array([np.mean(arr[: i + 1]) for i in range(window - 1)], dtype=float)
    return np.concatenate([warmup, smoothed])


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float with a fallback."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Convert a value to int with a fallback."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def estimate_convergence_episode(
    rewards: list[float],
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
) -> int:
    """Estimate when reward stabilises using a rolling-window heuristic.

    The heuristic looks for the first point where two adjacent stability windows
    have both low variance and a small mean shift. If no stable region is found,
    it returns the final episode.
    """
    if not rewards:
        return 0

    if len(rewards) < max(rolling_window, stability_window * 2):
        return len(rewards)

    smooth = rolling_mean(rewards, rolling_window)
    overall_scale = max(1.0, float(np.std(rewards)), float(np.mean(np.abs(rewards))))
    std_threshold = max(0.08 * overall_scale, 0.5)
    mean_threshold = max(0.05 * overall_scale, 0.25)

    for idx in range(stability_window * 2, len(smooth) + 1):
        current = smooth[idx - stability_window : idx]
        previous = smooth[idx - (stability_window * 2) : idx - stability_window]
        if len(current) < stability_window or len(previous) < stability_window:
            continue

        current_std = float(np.std(current))
        previous_std = float(np.std(previous))
        current_mean = float(np.mean(current))
        previous_mean = float(np.mean(previous))

        if (
            current_std <= std_threshold
            and previous_std <= std_threshold
            and abs(current_mean - previous_mean) <= mean_threshold
        ):
            return idx

    return len(rewards)


def compute_seed_metrics(
    logs: list[dict[str, Any]],
    last_n: int = DEFAULT_LAST_N,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
) -> SeedMetrics | None:
    """Compute metrics for one seed run."""
    if not logs:
        return None

    ordered_logs = sorted(logs, key=lambda item: safe_int(item.get("episode"), 0))
    rewards = [safe_float(row.get("reward_total"), 0.0) for row in ordered_logs]
    crashes = [safe_float(row.get("collisions"), 0.0) for row in ordered_logs]
    laps = [safe_float(row.get("laps_completed"), 0.0) for row in ordered_logs]
    if not rewards:
        return None

    tail = min(last_n, len(rewards))
    tail_rewards = rewards[-tail:]
    tail_crashes = crashes[-tail:] if crashes else []
    tail_laps = laps[-tail:] if laps else []

    avg_reward_last_n = float(np.mean(tail_rewards))
    avg_crash_last_n = float(np.mean(tail_crashes)) * 100.0 if tail_crashes else 0.0
    avg_laps_last_n = float(np.mean(tail_laps)) if tail_laps else 0.0
    max_reward = float(np.max(rewards))
    convergence_episode = int(
        estimate_convergence_episode(
            rewards,
            rolling_window=rolling_window,
            stability_window=stability_window,
        )
    )
    reward_std = float(np.std(tail_rewards))
    crash_std = float(np.std(tail_crashes)) * 100.0 if tail_crashes else 0.0
    laps_std = float(np.std(tail_laps)) if tail_laps else 0.0
    final_reward = float(rewards[-1])
    seed_value = ordered_logs[-1].get("seed", ordered_logs[0].get("seed", "unknown"))

    return SeedMetrics(
        seed=str(seed_value),
        num_episodes=len(rewards),
        avg_reward_last_n=avg_reward_last_n,
        avg_crash_last_n=avg_crash_last_n,
        avg_laps_last_n=avg_laps_last_n,
        max_reward=max_reward,
        convergence_episode=convergence_episode,
        final_reward=final_reward,
        reward_std=reward_std,
        crash_std=crash_std,
        laps_std=laps_std,
    )


def variance_or_none(values: list[float]) -> float | None:
    """Return the population variance for a list, or None when unavailable."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(statistics.pvariance(values))


def mean_or_none(values: list[float]) -> float | None:
    """Return the arithmetic mean for a list, or None when unavailable."""
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return float(statistics.fmean(values))


def aggregate_seed_metrics(algorithm: str, experiment: str, seed_metrics: list[SeedMetrics]) -> AlgorithmMetrics | None:
    """Aggregate seed-level metrics into an experiment-level summary."""
    if not seed_metrics:
        return None

    avg_reward_last_n_values = [item.avg_reward_last_n for item in seed_metrics]
    avg_crash_last_n_values = [item.avg_crash_last_n for item in seed_metrics]
    avg_laps_last_n_values = [item.avg_laps_last_n for item in seed_metrics]
    reward_std_values = [item.reward_std for item in seed_metrics]
    max_reward_values = [item.max_reward for item in seed_metrics]
    convergence_values = [float(item.convergence_episode) for item in seed_metrics]
    final_reward_values = [item.final_reward for item in seed_metrics]
    episode_counts = [float(item.num_episodes) for item in seed_metrics]

    return AlgorithmMetrics(
        algorithm=algorithm,
        experiment=experiment,
        seeds=seed_metrics,
        avg_reward_last_n_mean=mean_or_none(avg_reward_last_n_values),
        avg_reward_last_n_variance=variance_or_none(avg_reward_last_n_values),
        avg_crash_last_n_mean=mean_or_none(avg_crash_last_n_values),
        avg_crash_last_n_variance=variance_or_none(avg_crash_last_n_values),
        avg_laps_last_n_mean=mean_or_none(avg_laps_last_n_values),
        avg_laps_last_n_variance=variance_or_none(avg_laps_last_n_values),
        reward_std_mean=mean_or_none(reward_std_values),
        reward_std_variance=variance_or_none(reward_std_values),
        max_reward_mean=mean_or_none(max_reward_values),
        max_reward_variance=variance_or_none(max_reward_values),
        convergence_episode_mean=mean_or_none(convergence_values),
        convergence_episode_variance=variance_or_none(convergence_values),
        final_reward_mean=mean_or_none(final_reward_values),
        final_reward_variance=variance_or_none(final_reward_values),
        num_episodes_mean=mean_or_none(episode_counts),
        num_episodes_variance=variance_or_none(episode_counts),
        status="complete",
    )


def discover_experiment_ids(logs_dir: Path) -> list[str]:
    """Discover experiment IDs shared across both algorithms."""
    experiments: set[str] = set()
    for algo in ALGORITHMS:
        algo_dir = logs_dir / algo
        if not algo_dir.exists():
            continue
        for child in algo_dir.iterdir():
            if child.is_dir():
                if (child / "training_log.jsonl").exists() or any(child.glob("seed_*/training_log.jsonl")):
                    experiments.add(child.name)
    return sorted(experiments)


def discover_seed_logs(logs_dir: Path, algorithm: str, experiment: str) -> list[Path]:
    """Return all training-log files for one algorithm/experiment pair."""
    exp_dir = logs_dir / algorithm / experiment
    if not exp_dir.exists():
        return []

    legacy_log = exp_dir / "training_log.jsonl"
    seed_logs = sorted(exp_dir.glob("seed_*/training_log.jsonl"))
    if seed_logs:
        return [path for path in seed_logs if path.is_file()]
    if legacy_log.exists():
        return [legacy_log]
    return []


def collect_algorithm_metrics(
    logs_dir: Path,
    algorithm: str,
    experiment: str,
    last_n: int = DEFAULT_LAST_N,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
) -> AlgorithmMetrics | None:
    """Collect and aggregate metrics for one algorithm within one experiment."""
    seed_logs = discover_seed_logs(logs_dir, algorithm, experiment)
    if not seed_logs:
        return None

    seed_metrics: list[SeedMetrics] = []
    for log_file in seed_logs:
        logs = load_jsonl_logs(log_file)
        metrics = compute_seed_metrics(
            logs,
            last_n=last_n,
            rolling_window=rolling_window,
            stability_window=stability_window,
        )
        if metrics is not None:
            seed_metrics.append(metrics)

    aggregated = aggregate_seed_metrics(algorithm, experiment, seed_metrics)
    if aggregated is None:
        return None
    return aggregated


def collect_experiment_reports(
    logs_dir: Path,
    experiment_ids: list[str] | None = None,
    last_n: int = DEFAULT_LAST_N,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
) -> list[ExperimentReport]:
    """Collect metrics for every experiment that has at least one log file."""
    if experiment_ids is None:
        experiment_ids = discover_experiment_ids(logs_dir)

    reports: list[ExperimentReport] = []
    for experiment in experiment_ids:
        td3_metrics = collect_algorithm_metrics(
            logs_dir,
            "td3",
            experiment,
            last_n=last_n,
            rolling_window=rolling_window,
            stability_window=stability_window,
        )
        ddpg_metrics = collect_algorithm_metrics(
            logs_dir,
            "ddpg",
            experiment,
            last_n=last_n,
            rolling_window=rolling_window,
            stability_window=stability_window,
        )

        if td3_metrics is None and ddpg_metrics is None:
            continue

        reports.append(ExperimentReport(experiment=experiment, td3=td3_metrics, ddpg=ddpg_metrics))

    return reports


def format_number(value: Any, digits: int = 2) -> str:
    """Render numeric values consistently for titles, captions, and analysis."""
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def format_metric_line(label: str, value: Any, unit: str = "") -> str:
    """Format a label/value pair for inclusion in the PDF."""
    suffix = f" {unit}" if unit else ""
    return f"{label}: {format_number(value)}{suffix}"


def normalize_text(text: str) -> str:
    """Collapse whitespace for PDF rendering."""
    return " ".join(str(text).split())


def wrap_paragraphs(text: str, width: int = 100) -> list[str]:
    """Wrap a block of text into PDF-friendly lines."""
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        lines.extend(fill(paragraph, width=width).splitlines())
    return lines or [""]


def extract_experiment_from_name(name: str) -> str | None:
    """Find a compact experiment identifier embedded in a file name."""
    match = re.search(r"(R\d+_N\d+)", name.upper())
    if match:
        return match.group(1)
    return None


def infer_metric_label(name: str) -> str:
    """Infer the metric focus from a plot file name."""
    lowered = name.lower()
    if "reward" in lowered:
        return "reward"
    if "crash" in lowered or "collision" in lowered:
        return "crash"
    if "lap" in lowered:
        return "laps"
    return "metric"


def scan_image_files(base_dir: Path) -> list[Path]:
    """Return all image files under a directory tree."""
    if not base_dir.exists():
        return []
    files = [path for path in base_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def resolve_results_sections(results_dir: Path) -> dict[str, Path]:
    """Resolve comparison and algorithm-specific plot roots across layouts."""
    candidates = {
        "comparison": [results_dir / "comparison", results_dir / "plots" / "comparison"],
        "grouped": [results_dir / "grouped", results_dir / "plots" / "grouped"],
        "aggregate": [results_dir / "aggregate", results_dir / "plots" / "aggregate"],
        "td3": [results_dir / "td3", results_dir / "plots" / "td3"],
        "ddpg": [results_dir / "ddpg", results_dir / "plots" / "ddpg"],
    }

    resolved: dict[str, Path] = {}
    for section, paths in candidates.items():
        for path in paths:
            if path.exists():
                resolved[section] = path
                break
    return resolved


def render_text_page(pdf: PdfPages, title: str, paragraphs: list[str], footer: str | None = None) -> None:
    """Render a text-only page into the PDF."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.95, title, fontsize=21, fontweight="bold", va="top", ha="left")

    y = 0.90
    for paragraph in paragraphs:
        if not paragraph:
            y -= 0.018
            continue
        for line in wrap_paragraphs(paragraph, width=94):
            fig.text(0.07, y, line, fontsize=11.5, va="top", ha="left")
            y -= 0.022
        y -= 0.010

    if footer:
        fig.text(0.07, 0.05, footer, fontsize=9, color="#555555", va="bottom", ha="left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def load_image(image_path: Path) -> np.ndarray | None:
    """Load an image for PDF rendering, returning None if unavailable."""
    if not image_path.exists():
        return None
    try:
        return plt.imread(str(image_path))
    except Exception:
        return None


def render_image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path | None,
    body_lines: list[str],
    footer: str | None = None,
) -> None:
    """Render a page with an image and explanatory text."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.96, title, fontsize=18, fontweight="bold", va="top", ha="left")

    image_ax = fig.add_axes((0.08, 0.38, 0.84, 0.48))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    image_ax.set_frame_on(True)

    if image_path is not None and image_path.exists():
        image = load_image(image_path)
        if image is not None:
            image_ax.imshow(image)
            image_ax.set_title(image_path.name, fontsize=10, loc="left", pad=6)
        else:
            image_ax.text(0.5, 0.5, f"Unable to load image:\n{image_path.name}", ha="center", va="center")
            image_ax.set_facecolor("#f7f7f7")
    else:
        image_ax.text(0.5, 0.5, "Plot missing", ha="center", va="center", fontsize=12)
        image_ax.set_facecolor("#f7f7f7")

    body = "\n".join(body_lines) if body_lines else "No analysis available."
    text_ax = fig.add_axes((0.08, 0.08, 0.84, 0.24))
    text_ax.axis("off")
    text_ax.text(0.0, 1.0, body, va="top", ha="left", fontsize=10.5)

    if footer:
        fig.text(0.07, 0.03, footer, fontsize=8.8, color="#555555", va="bottom", ha="left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_caption_page(
    pdf: PdfPages,
    title: str,
    image_path: Path | None,
    caption: str,
    footer: str | None = None,
) -> None:
    """Render a simple plot page with an image and caption."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(0.07, 0.96, title, fontsize=18, fontweight="bold", va="top", ha="left")

    image_ax = fig.add_axes((0.08, 0.20, 0.84, 0.68))
    image_ax.set_xticks([])
    image_ax.set_yticks([])
    image_ax.set_frame_on(True)

    if image_path is not None and image_path.exists():
        image = load_image(image_path)
        if image is not None:
            image_ax.imshow(image)
            image_ax.set_title(image_path.name, fontsize=10, loc="left", pad=6)
        else:
            image_ax.text(0.5, 0.5, f"Unable to load image:\n{image_path.name}", ha="center", va="center")
            image_ax.set_facecolor("#f7f7f7")
    else:
        image_ax.text(0.5, 0.5, "Plot missing", ha="center", va="center", fontsize=12)
        image_ax.set_facecolor("#f7f7f7")

    fig.text(0.08, 0.11, caption, fontsize=10.8, va="top", ha="left")
    if footer:
        fig.text(0.07, 0.03, footer, fontsize=8.8, color="#555555", va="bottom", ha="left")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_experiment_prompt(
    experiment: str,
    td3_metrics: dict[str, Any] | None,
    ddpg_metrics: dict[str, Any] | None,
    metric_focus: str | None = None,
) -> str:
    """Build the concise prompt template for the NVIDIA model."""
    focus_line = f"Metric focus: {metric_focus}\n\n" if metric_focus else ""
    return (
        "You are analyzing reinforcement learning results for a research report.\n\n"
        f"Experiment: {experiment}\n\n"
        f"{focus_line}"
        f"TD3 metrics:\n{json.dumps(td3_metrics, indent=2, sort_keys=True)}\n\n"
        f"DDPG metrics:\n{json.dumps(ddpg_metrics, indent=2, sort_keys=True)}\n\n"
        "Compare both algorithms and provide:\n"
        "1. Which algorithm performs better overall and why\n"
        "2. Convergence comparison\n"
        "3. Stability insights\n"
        "4. Relative crash-rate and lap-performance observations\n\n"
        "Keep the explanation concise, technical, and evidence-based.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
    )


def build_summary_prompt(experiment_reports: list[dict[str, Any]]) -> str:
    """Build the final summary prompt across all experiments."""
    return (
        "You are analyzing reinforcement learning results across multiple experiments for a research paper.\n\n"
        f"Experiment summaries:\n{json.dumps(experiment_reports, indent=2, sort_keys=True)}\n\n"
        "Provide a concise technical summary covering:\n"
        "1. Overall TD3 vs DDPG performance\n"
        "2. Stability trends across experiments\n"
        "3. Convergence insights\n"
        "4. Reward, crash-rate, and lap-trend observations\n\n"
        "Keep the explanation concise, technical, and evidence-based.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
    )


def build_noise_level_prompt(noise_label: str, noise_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for one noise level section."""
    return (
        "You are analyzing RL results for a research report.\n\n"
        f"Noise level: {noise_label}\n\n"
        f"Structured metrics:\n{json.dumps(noise_metrics, indent=2, sort_keys=True)}\n\n"
        "For this noise level, explain:\n"
        "1. Which reward system performs best\n"
        "2. Which system is most stable\n"
        "3. Observed reward, crash-rate, and lap trends\n\n"
        "Keep the explanation concise and evidence-based."
    )


def build_algorithm_comparison_prompt(comparison_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for TD3 vs DDPG comparison."""
    return (
        "You are comparing TD3 and DDPG in a research report.\n\n"
        f"Structured metrics:\n{json.dumps(comparison_metrics, indent=2, sort_keys=True)}\n\n"
        "Explain:\n"
        "1. Which algorithm performs better overall\n"
        "2. Convergence speed differences\n"
        "3. Stability differences\n"
        "4. Reward, crash-rate, and lap behavior\n\n"
        "Keep the explanation concise and evidence-based."
    )


def build_key_insights_prompt(insight_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for cross-noise key insights."""
    return (
        "You are extracting key insights from structured RL metrics.\n\n"
        f"Structured metrics:\n{json.dumps(insight_metrics, indent=2, sort_keys=True)}\n\n"
        "Provide:\n"
        "1. Best reward function across all noise levels\n"
        "2. Overall TD3 vs DDPG performance trend\n"
        "3. Convergence comparison\n\n"
        "Keep the explanation concise and evidence-based."
    )


def build_final_conclusion_prompt(conclusion_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for the final conclusion section."""
    return (
        "You are writing the conclusion for a research-style RL report.\n\n"
        f"Structured metrics:\n{json.dumps(conclusion_metrics, indent=2, sort_keys=True)}\n\n"
        "Summarize:\n"
        "1. Best algorithm overall\n"
        "2. Best reward system\n"
        "3. Effect of noise on performance\n"
        "4. Main trade-offs\n\n"
        "Keep the conclusion concise and evidence-based."
    )


def fallback_noise_analysis(noise_label: str, noise_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for one noise level."""
    td3 = noise_metrics.get("td3", {})
    ddpg = noise_metrics.get("ddpg", {})
    best_reward = noise_metrics.get("best_reward_system", "n/a")
    return (
        f"Noise {noise_label}: best reward system is {best_reward}. "
        f"TD3 reward mean {format_table_value(td3.get('avg_reward_last_n_mean'))}, "
        f"DDPG reward mean {format_table_value(ddpg.get('avg_reward_last_n_mean'))}. "
        "Stability is reflected by lower reward standard deviation and earlier convergence."
    )


def fallback_comparison_analysis(comparison_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for TD3 vs DDPG comparison."""
    td3 = comparison_metrics.get("td3", {})
    ddpg = comparison_metrics.get("ddpg", {})
    winner = comparison_metrics.get("winner", "n/a")
    return (
        f"Overall comparison favors {winner}. "
        f"TD3 reward mean {format_table_value(td3.get('avg_reward_last_n_mean'))}, "
        f"DDPG reward mean {format_table_value(ddpg.get('avg_reward_last_n_mean'))}. "
        "Earlier convergence and lower reward standard deviation indicate greater stability."
    )


def fallback_insights_analysis(insight_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for key insights across noise levels."""
    best_reward = insight_metrics.get("best_reward_system", "n/a")
    best_algo = insight_metrics.get("overall_algorithm_winner", "n/a")
    return (
        f"Best reward system across all noise levels: {best_reward}. "
        f"Overall algorithm trend favors {best_algo}. "
        "Noise generally reduces performance and increases variability."
    )


def fallback_conclusion_analysis(conclusion_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for the final conclusion."""
    return (
        f"Best algorithm overall: {conclusion_metrics.get('best_algorithm', 'n/a')}. "
        f"Best reward system: {conclusion_metrics.get('best_reward_system', 'n/a')}. "
        "Higher noise levels generally reduce reward and increase instability."
    )


class NvidiaLLMClient:
    """Small OpenAI-compatible client for NVIDIA AI API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_API_BASE,
        timeout: int = 90,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.api_key = api_key or ""
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())

    def generate(self, prompt: str) -> str:
        """Generate text from the model with retries and graceful fallback."""
        if not self.enabled:
            raise RuntimeError("NVIDIA API key is not configured.")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise technical analyst for RL experiments."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = request.Request(self.base_url, data=data, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                choices = parsed.get("choices", []) if isinstance(parsed, dict) else []
                if choices:
                    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                    content = message.get("content")
                    if content:
                        return str(content).strip()
                raise RuntimeError("Unexpected NVIDIA API response format.")
            except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 5.0))
                    continue
                break

        assert last_error is not None
        raise last_error


def fallback_analysis(experiment: str, td3_metrics: dict[str, Any] | None, ddpg_metrics: dict[str, Any] | None) -> str:
    """Generate a deterministic analysis when the LLM is unavailable."""
    if td3_metrics is None and ddpg_metrics is None:
        return "No complete metrics were available for this experiment."

    lines = [f"Experiment {experiment}:"]
    if td3_metrics is not None:
        lines.append(
            "TD3 metrics indicate "
            f"avg reward {format_number(td3_metrics.get('avg_reward_last_n_mean'))}, "
            f"crash rate {format_number(td3_metrics.get('avg_crash_last_n_mean'))}%, "
            f"laps {format_number(td3_metrics.get('avg_laps_last_n_mean'))}, "
            f"max reward {format_number(td3_metrics.get('max_reward_mean'))}, "
            f"convergence near episode {format_number(td3_metrics.get('convergence_episode_mean'))}."
        )
    if ddpg_metrics is not None:
        lines.append(
            "DDPG metrics indicate "
            f"avg reward {format_number(ddpg_metrics.get('avg_reward_last_n_mean'))}, "
            f"crash rate {format_number(ddpg_metrics.get('avg_crash_last_n_mean'))}%, "
            f"laps {format_number(ddpg_metrics.get('avg_laps_last_n_mean'))}, "
            f"max reward {format_number(ddpg_metrics.get('max_reward_mean'))}, "
            f"convergence near episode {format_number(ddpg_metrics.get('convergence_episode_mean'))}."
        )

    if td3_metrics is not None and ddpg_metrics is not None:
        td3_avg = safe_float(td3_metrics.get("avg_reward_last_n_mean"), 0.0)
        ddpg_avg = safe_float(ddpg_metrics.get("avg_reward_last_n_mean"), 0.0)
        td3_stability = safe_float(td3_metrics.get("reward_std_mean"), 0.0)
        ddpg_stability = safe_float(ddpg_metrics.get("reward_std_mean"), 0.0)
        if td3_avg > ddpg_avg:
            winner = "TD3"
        elif ddpg_avg > td3_avg:
            winner = "DDPG"
        else:
            winner = "neither"

        lines.append(
            f"Average-reward comparison favors {winner}. Reward stability is estimated from the seed-level reward standard deviation ({format_number(td3_stability)} vs {format_number(ddpg_stability)})."
        )

    return " ".join(lines)


def collect_analysis_text(
    client: NvidiaLLMClient,
    prompt: str,
    fallback_text: str,
) -> tuple[str, bool]:
    """Call the model if possible, otherwise return deterministic text."""
    if not client.enabled:
        return fallback_text, False

    try:
        text = client.generate(prompt)
        return text, True
    except Exception:
        return fallback_text, False


def build_report_styles() -> dict[str, ParagraphStyle]:
    """Create a compact style set for the research-style PDF."""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1f1f1f"),
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a4a4a"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#1f1f1f"),
            spaceBefore=6,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2f2f2f"),
            spaceBefore=4,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PlotHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=colors.HexColor("#1f1f1f"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportCaption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#505050"),
            spaceBefore=4,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ReportNote",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12,
            textColor=colors.HexColor("#444444"),
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCellLeft",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="AnalysisBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#232323"),
            spaceAfter=4,
        )
    )
    return styles


def add_page_number(canvas, doc):
    """Draw a page number footer on each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.4 * inch, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def format_percentage(value: Any, digits: int = 1) -> str:
    """Format numeric values as percentages for the summary table."""
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}%"


def format_table_value(value: Any, digits: int = 2) -> str:
    """Format a numeric table value with graceful fallback."""
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def score_algorithm(metrics: AlgorithmMetrics | None) -> float:
    """Score an algorithm using reward and stability."""
    if metrics is None:
        return float("-inf")

    reward = safe_float(metrics.avg_reward_last_n_mean, float("-inf"))
    stability = safe_float(metrics.reward_std_mean, float("inf"))
    if math.isinf(reward) or math.isinf(stability):
        return float("-inf")
    return reward - stability


def choose_winner(td3_metrics: AlgorithmMetrics | None, ddpg_metrics: AlgorithmMetrics | None) -> str:
    """Select a per-experiment winner using reward and stability."""
    if td3_metrics is None and ddpg_metrics is None:
        return "n/a"
    if td3_metrics is None:
        return "DDPG"
    if ddpg_metrics is None:
        return "TD3"

    td3_score = score_algorithm(td3_metrics)
    ddpg_score = score_algorithm(ddpg_metrics)
    if abs(td3_score - ddpg_score) <= 1e-9:
        td3_crash = safe_float(td3_metrics.avg_crash_last_n_mean, float("inf"))
        ddpg_crash = safe_float(ddpg_metrics.avg_crash_last_n_mean, float("inf"))
        if abs(td3_crash - ddpg_crash) <= 1e-9:
            td3_conv = safe_float(td3_metrics.convergence_episode_mean, float("inf"))
            ddpg_conv = safe_float(ddpg_metrics.convergence_episode_mean, float("inf"))
            if abs(td3_conv - ddpg_conv) <= 1e-9:
                return "Tie"
            return "TD3" if td3_conv < ddpg_conv else "DDPG"
        return "TD3" if td3_crash < ddpg_crash else "DDPG"
    return "TD3" if td3_score > ddpg_score else "DDPG"


def experiment_sort_key(experiment_id: str) -> tuple[int, int, str]:
    """Sort experiment identifiers in reward/noise order."""
    match = re.search(r"r(\d+)_n(\d+)", experiment_id.lower())
    if match:
        return int(match.group(1)), int(match.group(2)), experiment_id
    return 99, 99, experiment_id


def sorted_experiment_ids(experiment_ids: list[str]) -> list[str]:
    """Deduplicate and sort experiment identifiers."""
    return sorted({str(exp).strip() for exp in experiment_ids}, key=experiment_sort_key)


def find_existing_file(paths: list[Path]) -> Path | None:
    """Return the first existing path from a list of candidates."""
    for path in paths:
        if path.exists():
            return path
    return None


def locate_plot_file(results_dir: Path, relative_candidates: list[str]) -> Path | None:
    """Find a plot under results/ or results/plots/ fallback layouts."""
    candidates: list[Path] = []
    for relative_path in relative_candidates:
        candidates.append(results_dir / relative_path)
        candidates.append(results_dir / "plots" / relative_path)
    return find_existing_file(candidates)


def format_experiment_label(experiment_id: str) -> str:
    """Normalize an experiment identifier for grouping and comparisons."""
    return experiment_id.strip().upper().replace("-", "_").replace(" ", "")


def parse_experiment_components(experiment_id: str) -> tuple[str | None, str | None]:
    """Extract reward-system and noise-level labels from an experiment identifier."""
    match = re.fullmatch(r"(R[1-4])_(N[1-3])", format_experiment_label(experiment_id))
    if not match:
        return None, None
    return match.group(1), match.group(2)


def group_reports_by_reward_and_noise(reports: list[ExperimentReport]) -> dict[str, dict[str, dict[str, AlgorithmMetrics]]]:
    """Group experiment reports by noise level and reward system."""
    grouped: dict[str, dict[str, dict[str, AlgorithmMetrics]]] = {
        "N1": {},
        "N2": {},
        "N3": {},
    }

    for report in reports:
        reward_label, noise_label = parse_experiment_components(report.experiment)
        if reward_label is None or noise_label is None:
            continue

        grouped.setdefault(noise_label, {})[reward_label] = {
            "td3": report.td3,
            "ddpg": report.ddpg,
        }

    return grouped


def group_reports_by_reward(reports: list[ExperimentReport]) -> dict[str, list[ExperimentReport]]:
    """Group experiment reports by reward system across all noise levels."""
    grouped: dict[str, list[ExperimentReport]] = {reward: [] for reward in REWARD_LEVELS}
    for report in reports:
        reward_label, _ = parse_experiment_components(report.experiment)
        if reward_label is None:
            continue
        grouped.setdefault(reward_label, []).append(report)
    return grouped


def summarize_algorithm(metrics_list: list[AlgorithmMetrics | None]) -> dict[str, Any] | None:
    """Summarize one algorithm across a collection of experiments."""
    metrics = [item for item in metrics_list if item is not None]
    if not metrics:
        return None

    return {
        "experiment_count": len(metrics),
        "avg_reward_last_n_mean": mean_or_none([item.avg_reward_last_n_mean for item in metrics]),
        "avg_crash_last_n_mean": mean_or_none([item.avg_crash_last_n_mean for item in metrics]),
        "avg_laps_last_n_mean": mean_or_none([item.avg_laps_last_n_mean for item in metrics]),
        "reward_std_mean": mean_or_none([item.reward_std_mean for item in metrics]),
        "convergence_episode_mean": mean_or_none([item.convergence_episode_mean for item in metrics]),
    }


def summarize_reward_systems(reports: list[ExperimentReport]) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by reward system across all noise levels and algorithms."""
    grouped = group_reports_by_reward(reports)
    summary: dict[str, dict[str, Any]] = {}

    for reward_label, reward_reports in grouped.items():
        summary[reward_label] = {
            "td3": summarize_algorithm([report.td3 for report in reward_reports]),
            "ddpg": summarize_algorithm([report.ddpg for report in reward_reports]),
            "winner_count": {
                "TD3": sum(1 for report in reward_reports if choose_winner(report.td3, report.ddpg) == "TD3"),
                "DDPG": sum(1 for report in reward_reports if choose_winner(report.td3, report.ddpg) == "DDPG"),
            },
        }

    return summary


def build_scaled_image(image_path: Path, max_width: float, max_height: float):
    """Create a reportlab Image flowable scaled to fit inside the target box."""
    reader = ImageReader(str(image_path))
    image_width, image_height = reader.getSize()
    if image_width <= 0 or image_height <= 0:
        return None

    scale = min(max_width / float(image_width), max_height / float(image_height))
    scale = min(scale, 1.0)
    return Image(str(image_path), width=image_width * scale, height=image_height * scale)


def algorithm_metrics_lines(metrics: AlgorithmMetrics | None) -> list[str]:
    """Create a human-readable metrics summary for LLM prompts and narrative text."""
    if metrics is None:
        return ["No metrics available."]

    return [
        f"Avg reward (last N): {format_table_value(metrics.avg_reward_last_n_mean)}",
        f"Crash rate (last N): {format_percentage(metrics.avg_crash_last_n_mean)}",
        f"Laps per episode (last N): {format_table_value(metrics.avg_laps_last_n_mean)}",
        f"Reward std (stability): {format_table_value(metrics.reward_std_mean)}",
        f"Convergence episode: {format_table_value(metrics.convergence_episode_mean, digits=0)}",
    ]


def build_summary_table_rows(reports: list[ExperimentReport], styles) -> tuple[list[list[Paragraph]], list[str]]:
    """Build the research-style summary table and collect skipped experiments."""
    rows: list[list[Paragraph]] = [
        [
            Paragraph("Experiment", styles["TableCell"]),
            Paragraph("TD3 Reward", styles["TableCell"]),
            Paragraph("DDPG Reward", styles["TableCell"]),
            Paragraph("TD3 Crash", styles["TableCell"]),
            Paragraph("DDPG Crash", styles["TableCell"]),
            Paragraph("TD3 Laps", styles["TableCell"]),
            Paragraph("DDPG Laps", styles["TableCell"]),
            Paragraph("Winner", styles["TableCell"]),
        ]
    ]
    skipped: list[str] = []

    for report in reports:
        if report.td3 is None or report.ddpg is None:
            skipped.append(report.experiment)
            continue

        winner = choose_winner(report.td3, report.ddpg)
        rows.append(
            [
                Paragraph(report.experiment, styles["TableCellLeft"]),
                Paragraph(format_table_value(report.td3.avg_reward_last_n_mean), styles["TableCell"]),
                Paragraph(format_table_value(report.ddpg.avg_reward_last_n_mean), styles["TableCell"]),
                Paragraph(format_percentage(report.td3.avg_crash_last_n_mean), styles["TableCell"]),
                Paragraph(format_percentage(report.ddpg.avg_crash_last_n_mean), styles["TableCell"]),
                Paragraph(format_table_value(report.td3.avg_laps_last_n_mean), styles["TableCell"]),
                Paragraph(format_table_value(report.ddpg.avg_laps_last_n_mean), styles["TableCell"]),
                Paragraph(f"<b>{winner}</b>", styles["TableCell"]),
            ]
        )

    return rows, skipped


def build_summary_table(reports: list[ExperimentReport], styles) -> tuple[Table, list[str]]:
    """Create the summary metrics table for the report."""
    rows, skipped = build_summary_table_rows(reports, styles)
    table = Table(
        rows,
        colWidths=[1.05 * inch, 0.82 * inch, 0.82 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.78 * inch, 0.65 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEADING", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#8fa1c1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table, skipped


def build_plot_story_block(
    title: str,
    image_path: Path | None,
    caption: str,
    styles,
    analysis_lines: list[str] | None = None,
    image_height: float = 4.5 * inch,
):
    """Create a plot page block with optional analysis text below the figure."""
    block: list[Any] = [Paragraph(title, styles["PlotHeading"])]

    if image_path is None or not image_path.exists():
        block.append(Paragraph("Plot missing from the expected results directory.", styles["ReportNote"]))
    else:
        image = build_scaled_image(image_path, max_width=6.9 * inch, max_height=image_height)
        if image is None:
            block.append(Paragraph(f"Unable to render image: {image_path.name}", styles["ReportNote"]))
        else:
            block.append(image)

    block.append(Spacer(1, 0.12 * inch))
    block.append(Paragraph(f"<b>Caption.</b> {xml_escape(caption)}", styles["ReportCaption"]))
    if analysis_lines:
        analysis_text = "<br/>".join(xml_escape(line) for line in analysis_lines)
        block.append(Paragraph(analysis_text, styles["AnalysisBody"]))
    block.append(PageBreak())
    return block


def build_experiment_plot_block(
    algo: str,
    experiment_id: str,
    plot_paths: dict[str, Path | None],
    styles,
):
    """Create a compact individual-plot page for one experiment and one algorithm."""
    block: list[Any] = [
        Paragraph(f"{algo.upper()} Individual Plots - {experiment_id}", styles["SubSectionHeading"]),
        Paragraph(
            "Each figure shows the seed-aggregated mean with moving-average smoothing and standard-deviation shading.",
            styles["ReportNote"],
        ),
    ]

    for metric_key, metric_title in [
        ("reward", "Reward vs Episodes"),
        ("crash", "Crash Rate vs Episodes"),
        ("laps", "Laps vs Episodes"),
    ]:
        image_path = plot_paths.get(metric_key)
        if image_path is None or not image_path.exists():
            block.append(Paragraph(f"{metric_title}: plot missing.", styles["ReportNote"]))
            continue

        block.append(Paragraph(metric_title, styles["ReportCaption"]))
        image = build_scaled_image(image_path, max_width=6.85 * inch, max_height=1.8 * inch)
        if image is None:
            block.append(Paragraph(f"Unable to render image: {image_path.name}", styles["ReportNote"]))
        else:
            block.append(image)
        block.append(Spacer(1, 0.08 * inch))

    block.append(PageBreak())
    return block


def build_noise_section_block(
    noise_label: str,
    grouped_noise_reports: dict[str, dict[str, AlgorithmMetrics | None]],
    results_dir: Path,
    styles,
    analysis_text: str,
) -> list[Any]:
    """Create a compact section for one fixed noise level."""
    td3_rows: list[list[Any]] = [
        [
            Paragraph("Metric", styles["TableCell"]),
            Paragraph("TD3", styles["TableCell"]),
            Paragraph("DDPG", styles["TableCell"]),
        ]
    ]

    def image_or_missing(path: Path | None):
        if path is None or not path.exists():
            return Paragraph("Plot missing", styles["ReportNote"])
        image = build_scaled_image(path, max_width=3.15 * inch, max_height=2.15 * inch)
        if image is None:
            return Paragraph(f"Unable to render image: {path.name}", styles["ReportNote"])
        return image

    for metric_key, metric_label in [
        ("reward", "Reward"),
        ("crash", "Crash Rate"),
        ("laps", "Laps"),
    ]:
        td3_image = locate_plot_file(results_dir, [f"grouped/td3_{noise_label.lower()}_{metric_key}.png"])
        ddpg_image = locate_plot_file(results_dir, [f"grouped/ddpg_{noise_label.lower()}_{metric_key}.png"])

        td3_rows.append([
            Paragraph(metric_label, styles["TableCellLeft"]),
            image_or_missing(td3_image),
            image_or_missing(ddpg_image),
        ])

    table = Table(
        td3_rows,
        colWidths=[0.8 * inch, 3.3 * inch, 3.3 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8fa1c1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafe")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    block: list[Any] = [
        Paragraph(f"Noise Level {noise_label}", styles["SectionHeading"]),
        Paragraph(
            "Grouped experiment curves compare R1-R4 within a fixed noise setting. Each plot shows the seed-aggregated mean with light uncertainty shading.",
            styles["ReportNote"],
        ),
        table,
        Spacer(1, 0.12 * inch),
        Paragraph("Summary", styles["SubSectionHeading"]),
        Paragraph(xml_escape(analysis_text), styles["AnalysisBody"]),
        PageBreak(),
    ]
    return block


def build_overall_metric_view(metrics_list: list[AlgorithmMetrics | None], algorithm: str) -> dict[str, Any] | None:
    """Aggregate one algorithm's summary metrics across all experiments."""
    metrics = [item for item in metrics_list if item is not None]
    if not metrics:
        return None

    return {
        "algorithm": algorithm,
        "status": "complete",
        "experiment_count": len(metrics),
        "avg_reward_last_n_mean": mean_or_none([item.avg_reward_last_n_mean for item in metrics]),
        "avg_crash_last_n_mean": mean_or_none([item.avg_crash_last_n_mean for item in metrics]),
        "avg_laps_last_n_mean": mean_or_none([item.avg_laps_last_n_mean for item in metrics]),
        "reward_std_mean": mean_or_none([item.reward_std_mean for item in metrics]),
        "convergence_episode_mean": mean_or_none([item.convergence_episode_mean for item in metrics]),
    }


def _score_reward_system_summary(summary: dict[str, Any] | None) -> float:
    """Score a reward-system summary using reward and stability."""
    if summary is None:
        return float("-inf")

    reward = safe_float(summary.get("avg_reward_last_n_mean"), float("-inf"))
    stability = safe_float(summary.get("reward_std_mean"), float("inf"))
    if math.isinf(reward) or math.isinf(stability):
        return float("-inf")
    return reward - stability


def _reward_system_summary(reports: list[ExperimentReport]) -> dict[str, Any] | None:
    """Summarize one reward system across all noise levels."""
    td3_metrics = [report.td3 for report in reports if report.td3 is not None]
    ddpg_metrics = [report.ddpg for report in reports if report.ddpg is not None]
    if not td3_metrics and not ddpg_metrics:
        return None

    return {
        "td3": summarize_algorithm(td3_metrics),
        "ddpg": summarize_algorithm(ddpg_metrics),
        "winner_count": {
            "TD3": sum(1 for report in reports if choose_winner(report.td3, report.ddpg) == "TD3"),
            "DDPG": sum(1 for report in reports if choose_winner(report.td3, report.ddpg) == "DDPG"),
        },
    }


def build_noise_level_metrics(noise_label: str, noise_reports: dict[str, dict[str, AlgorithmMetrics | None]]) -> dict[str, Any]:
    """Build structured metrics for one fixed noise level."""
    reward_systems: dict[str, Any] = {}
    td3_summary_metrics = [pair.get("td3") for pair in noise_reports.values() if pair.get("td3") is not None]
    ddpg_summary_metrics = [pair.get("ddpg") for pair in noise_reports.values() if pair.get("ddpg") is not None]
    candidate_summaries: list[tuple[str, dict[str, Any] | None]] = []

    for reward_label in REWARD_LEVELS:
        reward_pair = noise_reports.get(reward_label, {})
        td3_metrics = reward_pair.get("td3")
        ddpg_metrics = reward_pair.get("ddpg")
        reward_systems[reward_label] = {
            "td3": td3_metrics.to_prompt_dict() if td3_metrics else None,
            "ddpg": ddpg_metrics.to_prompt_dict() if ddpg_metrics else None,
            "winner": choose_winner(td3_metrics, ddpg_metrics),
        }

        td3_score = _score_reward_system_summary(reward_systems[reward_label]["td3"])
        ddpg_score = _score_reward_system_summary(reward_systems[reward_label]["ddpg"])
        candidate_score = max(td3_score, ddpg_score)
        candidate_summaries.append((reward_label, reward_systems[reward_label]))

    best_reward_system = None
    best_score = float("-inf")
    for reward_label, summary in candidate_summaries:
        combined_summary = {
            "avg_reward_last_n_mean": mean_or_none(
                [summary[algo].get("avg_reward_last_n_mean") for algo in ("td3", "ddpg") if summary.get(algo)]
            ),
            "reward_std_mean": mean_or_none(
                [summary[algo].get("reward_std_mean") for algo in ("td3", "ddpg") if summary.get(algo)]
            ),
        }
        score = _score_reward_system_summary(combined_summary)
        if score > best_score:
            best_score = score
            best_reward_system = reward_label

    return {
        "noise_level": noise_label,
        "td3": summarize_algorithm(td3_summary_metrics),
        "ddpg": summarize_algorithm(ddpg_summary_metrics),
        "reward_systems": reward_systems,
        "best_reward_system": best_reward_system,
    }


def build_comparison_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the TD3 vs DDPG comparison section."""
    ordered_reports = [report for report in reports if report.td3 is not None and report.ddpg is not None]
    td3_summary = summarize_algorithm([report.td3 for report in ordered_reports])
    ddpg_summary = summarize_algorithm([report.ddpg for report in ordered_reports])
    td3_score = _score_reward_system_summary(td3_summary)
    ddpg_score = _score_reward_system_summary(ddpg_summary)
    winner = "TD3" if td3_score > ddpg_score else "DDPG" if ddpg_score > td3_score else "Tie"

    return {
        "td3": td3_summary,
        "ddpg": ddpg_summary,
        "winner": winner,
    }


def build_key_insight_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the cross-noise insights section."""
    reward_summaries = {
        reward_label: _reward_system_summary(reward_reports)
        for reward_label, reward_reports in group_reports_by_reward(reports).items()
    }

    best_reward_system = None
    best_score = float("-inf")
    for reward_label, summary in reward_summaries.items():
        score = _score_reward_system_summary(summary["td3"] if summary else None)
        if score > best_score:
            best_score = score
            best_reward_system = reward_label

    comparison_metrics = build_comparison_metrics(reports)
    return {
        "reward_systems": reward_summaries,
        "best_reward_system": best_reward_system,
        "overall_algorithm_winner": comparison_metrics.get("winner", "n/a"),
        "comparison": comparison_metrics,
    }


def build_conclusion_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the final conclusion section."""
    comparison_metrics = build_comparison_metrics(reports)
    reward_metrics = build_key_insight_metrics(reports)
    return {
        "best_algorithm": comparison_metrics.get("winner", "n/a"),
        "best_reward_system": reward_metrics.get("best_reward_system", "n/a"),
        "comparison": comparison_metrics,
        "insights": reward_metrics,
    }


def build_report(
    reports: list[ExperimentReport],
    results_dir: Path,
    output_file: Path,
    llm_client: NvidiaLLMClient,
) -> None:
    """Render the full structured PDF report."""
    styles = build_report_styles()
    complete_reports = sorted(
        [report for report in reports if report.td3 is not None and report.ddpg is not None],
        key=lambda item: experiment_sort_key(item.experiment),
    )
    report_index = {report.experiment: report for report in complete_reports}
    grouped_by_noise = group_reports_by_reward_and_noise(complete_reports)

    analysis_cache: dict[tuple[str, str], tuple[str, bool]] = {}

    def cached_analysis(cache_key: tuple[str, str], prompt: str, fallback_text: str) -> tuple[str, bool]:
        if cache_key in analysis_cache:
            return analysis_cache[cache_key]
        analysis_cache[cache_key] = collect_analysis_text(llm_client, prompt, fallback_text)
        return analysis_cache[cache_key]

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary_table, skipped_experiments = build_summary_table(complete_reports, styles)

    noise_analysis: dict[str, tuple[str, bool]] = {}
    for noise_label in NOISE_LEVELS:
        noise_metrics = build_noise_level_metrics(noise_label, grouped_by_noise.get(noise_label, {}))
        prompt = build_noise_level_prompt(noise_label, noise_metrics)
        fallback_text = fallback_noise_analysis(noise_label, noise_metrics)
        noise_analysis[noise_label] = cached_analysis(("noise", noise_label), prompt, fallback_text)

    comparison_metrics = build_comparison_metrics(complete_reports)
    comparison_prompt = build_algorithm_comparison_prompt(comparison_metrics)
    comparison_fallback = fallback_comparison_analysis(comparison_metrics)
    comparison_analysis, comparison_used_llm = cached_analysis(("comparison", "td3_ddpg"), comparison_prompt, comparison_fallback)

    insight_metrics = build_key_insight_metrics(complete_reports)
    insight_prompt = build_key_insights_prompt(insight_metrics)
    insight_fallback = fallback_insights_analysis(insight_metrics)
    insight_analysis, insight_used_llm = cached_analysis(("insight", "overall"), insight_prompt, insight_fallback)

    conclusion_metrics = build_conclusion_metrics(complete_reports)
    conclusion_prompt = build_final_conclusion_prompt(conclusion_metrics)
    conclusion_fallback = fallback_conclusion_analysis(conclusion_metrics)
    conclusion_analysis, conclusion_used_llm = cached_analysis(("conclusion", "final"), conclusion_prompt, conclusion_fallback)

    story: list[Any] = []
    story.extend(
        [
            Spacer(1, 1.12 * inch),
            Paragraph("TD3 vs DDPG Performance Analysis in Autonomous Driving Environment", styles["ReportTitle"]),
            Paragraph(
                "Structured research-style report built from seed-aggregated logs, grouped noise-level plots, algorithm comparisons, and summary text.",
                styles["ReportSubtitle"],
            ),
            Spacer(1, 0.28 * inch),
            Table(
                [
                    [Paragraph("Project Title", styles["TableCellLeft"]), Paragraph("TD3 vs DDPG Performance Analysis in Autonomous Driving Environment", styles["TableCellLeft"])],
                    [Paragraph("Description", styles["TableCellLeft"]), Paragraph("Comparison of reward systems across noise levels and algorithm behavior in a driving environment.", styles["TableCellLeft"])],
                    [Paragraph("Generated", styles["TableCellLeft"]), Paragraph(generated_at, styles["TableCellLeft"])],
                ],
                colWidths=[1.35 * inch, 5.25 * inch],
                hAlign="LEFT",
            ),
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Summary Metrics", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Average reward, crash rate, and laps are computed from the last N episodes for each seed, then averaged across seeds. The winner is selected using reward and stability.",
            styles["ReportNote"],
        )
    )
    story.append(summary_table)
    if skipped_experiments:
        story.append(Spacer(1, 0.10 * inch))
        story.append(Paragraph(f"Skipped incomplete experiments: {', '.join(skipped_experiments)}.", styles["ReportNote"]))
    story.append(PageBreak())

    story.append(Paragraph("Analysis by Noise Level", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "Each subsection fixes one noise level and compares R1-R4 for TD3 and DDPG using the grouped plots generated from seed-aggregated curves.",
            styles["ReportNote"],
        )
    )
    for noise_label in NOISE_LEVELS:
        noise_reports = grouped_by_noise.get(noise_label, {})
        if not noise_reports:
            story.append(Paragraph(f"Noise level {noise_label}: no grouped experiments were available.", styles["ReportNote"]))
            story.append(PageBreak())
            continue

        analysis_text, used_llm = noise_analysis[noise_label]
        story.extend(
            build_noise_section_block(
                noise_label=noise_label,
                grouped_noise_reports=noise_reports,
                results_dir=results_dir,
                styles=styles,
                analysis_text=analysis_text,
            )
        )

    story.append(Paragraph("Algorithm Comparison", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The next pages compare TD3 and DDPG directly using the existing comparison plots, focusing on convergence speed, stability, and final performance.",
            styles["ReportNote"],
        )
    )
    comparison_files = {
        "reward": locate_plot_file(results_dir, ["comparison/td3_vs_ddpg_reward.png"]),
        "crash": locate_plot_file(results_dir, ["comparison/td3_vs_ddpg_crash.png"]),
        "laps": locate_plot_file(results_dir, ["comparison/td3_vs_ddpg_laps.png"]),
    }
    for metric_key, metric_title, metric_label in [
        ("reward", "TD3 vs DDPG Reward Comparison", "Reward vs Episodes"),
        ("crash", "TD3 vs DDPG Crash Rate Comparison", "Crash Rate vs Episodes"),
        ("laps", "TD3 vs DDPG Laps Comparison", "Laps vs Episodes"),
    ]:
        story.extend(
            build_plot_story_block(
                title=metric_title,
                image_path=comparison_files[metric_key],
                caption=f"{metric_label} across all experiments.",
                styles=styles,
                analysis_lines=wrap_paragraphs(comparison_analysis, width=92),
                image_height=4.0 * inch,
            )
        )

    story.append(Paragraph("Key Aggregate Insights", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The following section summarizes the cross-noise trends and identifies the best-performing reward function and overall algorithm trend.",
            styles["ReportNote"],
        )
    )
    story.append(Paragraph(xml_escape(insight_analysis), styles["AnalysisBody"]))
    story.append(PageBreak())

    story.append(Paragraph("Final Conclusion", styles["SectionHeading"]))
    conclusion_paragraphs = wrap_paragraphs(conclusion_analysis, width=94)
    for paragraph in conclusion_paragraphs:
        story.append(Paragraph(xml_escape(paragraph), styles["AnalysisBody"]))

    doc = SimpleDocTemplate(
        str(output_file),
        pagesize=letter,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.65 * inch,
        title="TD3 vs DDPG Performance Analysis in Autonomous Driving Environment",
        author="GitHub Copilot",
        subject="Research-style TD3/DDPG performance report grouped by noise levels",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def main() -> None:
    """CLI entry point."""
    script_dir = Path(__file__).resolve().parent
    load_env_file(script_dir / ".env")
    load_env_file(Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="Generate a TD3 vs DDPG PDF report")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="Base logs directory")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base results directory containing comparison/td3/ddpg plots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output PDF file path",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        default=None,
        help="Optional experiment ID to include. Can be repeated.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=DEFAULT_LAST_N,
        help="Number of trailing episodes used for average reward",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=DEFAULT_ROLLING_WINDOW,
        help="Rolling window for convergence estimation",
    )
    parser.add_argument(
        "--stability-window",
        type=int,
        default=DEFAULT_STABILITY_WINDOW,
        help="Stability window for convergence estimation",
    )
    parser.add_argument(
        "--nvidia-api-key-env",
        type=str,
        default="NVIDIA_API_KEY",
        help="Environment variable containing the NVIDIA API key",
    )
    parser.add_argument(
        "--nvidia-model",
        type=str,
        default=DEFAULT_MODEL,
        help="NVIDIA model name to call through the API",
    )
    parser.add_argument(
        "--nvidia-api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help="OpenAI-compatible NVIDIA API endpoint",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum API retries for the NVIDIA call",
    )
    args = parser.parse_args()

    output_file = args.output
    if output_file.suffix.lower() != ".pdf":
        output_file = output_file.with_suffix(".pdf")

    experiment_ids = [str(item).strip() for item in args.experiments] if args.experiments else None
    reports = collect_experiment_reports(
        args.logs_dir,
        experiment_ids=experiment_ids,
        last_n=max(1, int(args.last_n)),
        rolling_window=max(1, int(args.rolling_window)),
        stability_window=max(1, int(args.stability_window)),
    )

    api_key = os.getenv(args.nvidia_api_key_env, "")
    llm_client = NvidiaLLMClient(
        api_key=api_key,
        model=args.nvidia_model,
        base_url=args.nvidia_api_base,
        max_retries=max(1, int(args.max_retries)),
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    build_report(reports, args.results_dir, output_file, llm_client)

    print(f"[report] Saved PDF report to: {output_file}")
    if not llm_client.enabled:
        print("[report][warn] NVIDIA_API_KEY is not set; generated analysis used local fallback text.")


if __name__ == "__main__":
    main()