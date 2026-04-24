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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_OUTPUT_FILE = Path("td3_ddpg_report.pdf")
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_API_BASE = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_LAST_N = 100
DEFAULT_ROLLING_WINDOW = 20
DEFAULT_STABILITY_WINDOW = 10
DEFAULT_MAX_RETRIES = 3

ALGORITHMS = ("td3", "ddpg")
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
    max_reward: float
    convergence_episode: int
    final_reward: float
    reward_std: float


@dataclass
class AlgorithmMetrics:
    """Aggregated metrics for one algorithm within one experiment."""

    algorithm: str
    experiment: str
    seeds: list[SeedMetrics] = field(default_factory=list)
    avg_reward_last_n_mean: float | None = None
    avg_reward_last_n_variance: float | None = None
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
    if not rewards:
        return None

    avg_reward_last_n = float(np.mean(rewards[-min(last_n, len(rewards)) :]))
    max_reward = float(np.max(rewards))
    convergence_episode = int(
        estimate_convergence_episode(
            rewards,
            rolling_window=rolling_window,
            stability_window=stability_window,
        )
    )
    reward_std = float(np.std(rewards))
    final_reward = float(rewards[-1])
    seed_value = ordered_logs[-1].get("seed", ordered_logs[0].get("seed", "unknown"))

    return SeedMetrics(
        seed=str(seed_value),
        num_episodes=len(rewards),
        avg_reward_last_n=avg_reward_last_n,
        max_reward=max_reward,
        convergence_episode=convergence_episode,
        final_reward=final_reward,
        reward_std=reward_std,
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
        "You are analyzing reinforcement learning results.\n\n"
        f"Experiment: {experiment}\n\n"
        f"{focus_line}"
        f"TD3:\n{json.dumps(td3_metrics, indent=2, sort_keys=True)}\n\n"
        f"DDPG:\n{json.dumps(ddpg_metrics, indent=2, sort_keys=True)}\n\n"
        "Compare both algorithms and provide:\n"
        "1. Which performs better and why\n"
        "2. Convergence comparison\n"
        "3. Stability insights\n\n"
        "Keep the explanation concise and technical.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
    )


def build_summary_prompt(experiment_reports: list[dict[str, Any]]) -> str:
    """Build the final summary prompt across all experiments."""
    return (
        "You are analyzing reinforcement learning results across multiple experiments.\n\n"
        f"Experiment summaries:\n{json.dumps(experiment_reports, indent=2, sort_keys=True)}\n\n"
        "Provide a concise technical summary covering:\n"
        "1. Overall performance comparison\n"
        "2. Stability trends\n"
        "3. Convergence insights\n\n"
        "Keep the explanation concise and technical.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
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
            f"max reward {format_number(td3_metrics.get('max_reward_mean'))}, "
            f"convergence near episode {format_number(td3_metrics.get('convergence_episode_mean'))}."
        )
    if ddpg_metrics is not None:
        lines.append(
            "DDPG metrics indicate "
            f"avg reward {format_number(ddpg_metrics.get('avg_reward_last_n_mean'))}, "
            f"max reward {format_number(ddpg_metrics.get('max_reward_mean'))}, "
            f"convergence near episode {format_number(ddpg_metrics.get('convergence_episode_mean'))}."
        )

    if td3_metrics is not None and ddpg_metrics is not None:
        td3_avg = safe_float(td3_metrics.get("avg_reward_last_n_mean"), 0.0)
        ddpg_avg = safe_float(ddpg_metrics.get("avg_reward_last_n_mean"), 0.0)
        if td3_avg > ddpg_avg:
            winner = "TD3"
        elif ddpg_avg > td3_avg:
            winner = "DDPG"
        else:
            winner = "neither"

        lines.append(
            f"Average-reward comparison favors {winner}. Stability is estimated from the variance and convergence episode spread across seeds."
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


def build_report(
    reports: list[ExperimentReport],
    results_dir: Path,
    output_file: Path,
    llm_client: NvidiaLLMClient,
) -> None:
    """Render the full PDF report."""
    sections = resolve_results_sections(results_dir)
    comparison_images = scan_image_files(sections.get("comparison", results_dir / "comparison"))
    td3_images = scan_image_files(sections.get("td3", results_dir / "td3"))
    ddpg_images = scan_image_files(sections.get("ddpg", results_dir / "ddpg"))

    report_index = {report.experiment: report for report in reports}
    analyses: dict[tuple[str, str], tuple[str, bool]] = {}

    def get_analysis_for(experiment: str, metric_focus: str) -> tuple[str, bool]:
        key = (experiment, metric_focus)
        if key in analyses:
            return analyses[key]

        report = report_index.get(experiment)
        td3_metrics = report.td3.to_prompt_dict() if report and report.td3 else None
        ddpg_metrics = report.ddpg.to_prompt_dict() if report and report.ddpg else None
        prompt = build_experiment_prompt(experiment, td3_metrics, ddpg_metrics, metric_focus=metric_focus)
        fallback_text = fallback_analysis(experiment, td3_metrics, ddpg_metrics)
        analysis = collect_analysis_text(llm_client, prompt, fallback_text)
        analyses[key] = analysis
        return analysis

    summary_prompt = build_summary_prompt(
        [
            {
                "experiment": report.experiment,
                "td3": report.td3.to_prompt_dict() if report.td3 else None,
                "ddpg": report.ddpg.to_prompt_dict() if report.ddpg else None,
            }
            for report in reports
        ]
    )
    summary_fallback = (
        "Overall, the report compares TD3 and DDPG across all discovered experiments using the same metrics. "
        f"TD3 experiments available: {sum(1 for report in reports if report.td3)}. "
        f"DDPG experiments available: {sum(1 for report in reports if report.ddpg)}."
    )
    summary_text, summary_used_llm = collect_analysis_text(llm_client, summary_prompt, summary_fallback)

    with PdfPages(output_file) as pdf:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        render_text_page(
            pdf,
            title="TD3 vs DDPG Report",
            paragraphs=[
                "Project: Reinforcement learning comparison for a custom car-racing environment.",
                f"Generated: {generated_at}",
                "This report is built from training logs only. Plot images are embedded when available, while the analysis text is generated from structured metrics rather than image inspection.",
            ],
            footer=f"Output file: {output_file.name}",
        )

        render_text_page(
            pdf,
            title="Introduction",
            paragraphs=[
                "TD3 and DDPG are deterministic policy-gradient algorithms for continuous control.",
                "TD3 improves on DDPG by using twin critics, delayed policy updates, and target policy smoothing to reduce overestimation bias and training instability.",
                "This experiment suite is intended to compare reward growth, convergence speed, and run-to-run stability across multiple reward/noise configurations.",
            ],
            footer="All analysis below is based on metrics computed from logs/training_log.jsonl.",
        )

        if comparison_images:
            for image_path in comparison_images:
                experiment_id = extract_experiment_from_name(image_path.stem) or "overall"
                metric_focus = infer_metric_label(image_path.stem)
                analysis_text, used_llm = get_analysis_for(experiment_id, metric_focus)
                footer = (
                    f"Source: {image_path.name} | Analysis source: {'NVIDIA AI API' if used_llm else 'local fallback'}"
                )
                render_image_page(
                    pdf,
                    title=f"Comparison Plot - {normalize_text(image_path.stem)}",
                    image_path=image_path,
                    body_lines=wrap_paragraphs(analysis_text, width=94),
                    footer=footer,
                )
        else:
            render_text_page(
                pdf,
                title="Comparison Plots",
                paragraphs=[
                    "No comparison plots were found in the expected results directories.",
                    "Checked locations include results/comparison and results/plots/comparison.",
                ],
                footer="Missing plots are skipped without stopping report generation.",
            )

        if td3_images:
            for image_path in td3_images:
                caption = f"TD3 plot: {normalize_text(image_path.stem)}"
                render_caption_page(
                    pdf,
                    title="TD3 Individual Plot",
                    image_path=image_path,
                    caption=caption,
                    footer=f"Source: {image_path.name}",
                )
        else:
            render_text_page(
                pdf,
                title="TD3 Individual Plots",
                paragraphs=[
                    "No TD3 plots were found in the expected results directories.",
                    "Checked locations include results/td3 and results/plots/td3.",
                ],
                footer="Missing plots are skipped without stopping report generation.",
            )

        if ddpg_images:
            for image_path in ddpg_images:
                caption = f"DDPG plot: {normalize_text(image_path.stem)}"
                render_caption_page(
                    pdf,
                    title="DDPG Individual Plot",
                    image_path=image_path,
                    caption=caption,
                    footer=f"Source: {image_path.name}",
                )
        else:
            render_text_page(
                pdf,
                title="DDPG Individual Plots",
                paragraphs=[
                    "No DDPG plots were found in the expected results directories.",
                    "Checked locations include results/ddpg and results/plots/ddpg.",
                ],
                footer="Missing plots are skipped without stopping report generation.",
            )

        render_text_page(
            pdf,
            title="Summary",
            paragraphs=[
                summary_text,
                "The tables below summarise the discovered experiments and the seeds included in each aggregate.",
            ],
            footer=f"Analysis source: {'NVIDIA AI API' if summary_used_llm else 'local fallback'}",
        )

        if reports:
            summary_lines = ["Experiment overview:"]
            for report in reports:
                td3_seeds = len(report.td3.seeds) if report.td3 else 0
                ddpg_seeds = len(report.ddpg.seeds) if report.ddpg else 0
                summary_lines.append(
                    f"- {report.experiment}: TD3 seeds={td3_seeds}, DDPG seeds={ddpg_seeds}"
                )

            render_text_page(
                pdf,
                title="Experiment Coverage",
                paragraphs=summary_lines,
                footer="Incomplete experiments are omitted from aggregate calculations.",
            )


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