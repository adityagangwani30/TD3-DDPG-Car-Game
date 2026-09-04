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
import csv
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
from typing import Any, Mapping, Sequence
from urllib import error, request
from xml.sax.saxutils import escape as xml_escape

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


DEFAULT_RESULTS_DIR = Path("results_v2")
DEFAULT_LOGS_DIR = Path("logs_v2")
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
    """Deterministic evaluation and training progression metrics for one seed run."""

    seed: str
    num_episodes: int  # 20 deterministic eval episodes
    avg_reward: float
    reward_std: float  # within-seed sample SD (ddof=1)
    reward_sem: float  # within-seed SEM
    crash_rate: float  # 0.0 to 1.0
    lap_completion_rate: float  # 0.0 to 1.0
    total_laps_completed: int
    mean_laps_per_episode: float
    laps_std: float
    distance_mean: float
    distance_std: float
    distance_sem: float
    avg_length: float
    length_std: float
    length_sem: float
    avg_speed: float
    speed_std: float
    best_lap_time: float | None
    avg_lap_time: float | None
    checkpoint: str | None = None
    convergence_episode: int | None = None
    num_training_episodes: int | None = None
    max_training_reward: float | None = None
    final_training_reward: float | None = None

    # Backward-compatible property aliases
    @property
    def avg_reward_last_n(self) -> float:
        return self.avg_reward

    @property
    def avg_crash_last_n(self) -> float:
        return self.crash_rate * 100.0

    @property
    def avg_laps_last_n(self) -> float:
        return self.mean_laps_per_episode

    @property
    def crash_std(self) -> float:
        return 0.0

    @property
    def final_reward(self) -> float:
        return self.avg_reward

    @property
    def max_reward(self) -> float:
        return self.avg_reward


@dataclass
class AlgorithmMetrics:
    """Aggregated condition-level metrics across seeds for one algorithm."""

    algorithm: str
    experiment: str
    seeds: list[SeedMetrics] = field(default_factory=list)
    avg_reward_mean: float | None = None
    avg_reward_std: float | None = None  # sample SD across seeds (ddof=1)
    avg_reward_sem: float | None = None
    crash_rate_mean: float | None = None  # 0.0 to 1.0
    crash_rate_std: float | None = None  # sample SD across seeds (ddof=1)
    crash_rate_sem: float | None = None
    lap_completion_rate_mean: float | None = None
    lap_completion_rate_std: float | None = None
    lap_completion_rate_sem: float | None = None
    mean_laps_mean: float | None = None
    mean_laps_std: float | None = None
    total_laps_mean: float | None = None
    total_laps_std: float | None = None
    distance_mean: float | None = None
    distance_std: float | None = None
    distance_sem: float | None = None
    avg_length_mean: float | None = None
    avg_length_std: float | None = None
    avg_length_sem: float | None = None
    avg_speed_mean: float | None = None
    avg_speed_std: float | None = None
    best_lap_time_min: float | None = None
    best_lap_time_mean: float | None = None
    avg_lap_time_mean: float | None = None
    reward_std_mean: float | None = None  # mean of within-seed evaluation reward std
    distance_std_mean: float | None = None  # mean of within-seed evaluation distance std
    convergence_episode_mean: float | None = None
    convergence_episode_std: float | None = None
    status: str = "missing"

    # Properties for downstream prompt/table/LLM compatibility
    @property
    def avg_reward_last_n_mean(self) -> float | None:
        return self.avg_reward_mean

    @property
    def avg_reward_last_n_std(self) -> float | None:
        return self.avg_reward_std

    @property
    def avg_reward_last_n_variance(self) -> float | None:
        return (self.avg_reward_std ** 2) if self.avg_reward_std is not None else None

    @property
    def avg_crash_last_n_mean(self) -> float | None:
        return (self.crash_rate_mean * 100.0) if self.crash_rate_mean is not None else None

    @property
    def avg_crash_last_n_std(self) -> float | None:
        return (self.crash_rate_std * 100.0) if self.crash_rate_std is not None else None

    @property
    def avg_crash_last_n_variance(self) -> float | None:
        return ((self.crash_rate_std * 100.0) ** 2) if self.crash_rate_std is not None else None

    @property
    def avg_laps_last_n_mean(self) -> float | None:
        return self.mean_laps_mean

    @property
    def avg_laps_last_n_std(self) -> float | None:
        return self.mean_laps_std

    @property
    def avg_laps_last_n_variance(self) -> float | None:
        return (self.mean_laps_std ** 2) if self.mean_laps_std is not None else None

    @property
    def convergence_episode_variance(self) -> float | None:
        return (self.convergence_episode_std ** 2) if self.convergence_episode_std is not None else None

    @property
    def final_reward_mean(self) -> float | None:
        return self.avg_reward_mean

    @property
    def final_reward_std(self) -> float | None:
        return self.avg_reward_std

    @property
    def final_reward_variance(self) -> float | None:
        return (self.avg_reward_std ** 2) if self.avg_reward_std is not None else None

    @property
    def max_reward_mean(self) -> float | None:
        return self.avg_reward_mean

    @property
    def max_reward_std(self) -> float | None:
        return self.avg_reward_std

    @property
    def max_reward_variance(self) -> float | None:
        return (self.avg_reward_std ** 2) if self.avg_reward_std is not None else None

    @property
    def num_episodes_mean(self) -> float | None:
        return 20.0

    @property
    def num_episodes_std(self) -> float | None:
        return 0.0

    @property
    def num_episodes_variance(self) -> float | None:
        return 0.0

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a compact serialisable view for LLM prompts."""
        crash_pct_mean = (self.crash_rate_mean * 100.0) if self.crash_rate_mean is not None else None
        crash_pct_std = (self.crash_rate_std * 100.0) if self.crash_rate_std is not None else None
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "seed_count": len(self.seeds),
            "avg_reward_mean": self.avg_reward_mean,
            "avg_reward_std": self.avg_reward_std,
            "avg_reward_sem": self.avg_reward_sem,
            "avg_reward_mean_pm_std": format_mean_pm(self.avg_reward_mean, self.avg_reward_std),
            "crash_rate_mean_pct": crash_pct_mean,
            "crash_rate_std_pct": crash_pct_std,
            "crash_rate_mean_pm_std": format_mean_pm(crash_pct_mean, crash_pct_std, unit="%"),
            "lap_completion_rate_mean": self.lap_completion_rate_mean,
            "mean_laps_mean": self.mean_laps_mean,
            "mean_laps_std": self.mean_laps_std,
            "mean_laps_mean_pm_std": format_mean_pm(self.mean_laps_mean, self.mean_laps_std),
            "distance_mean": self.distance_mean,
            "distance_std": self.distance_std,
            "distance_sem": self.distance_sem,
            "avg_length_mean": self.avg_length_mean,
            "avg_length_std": self.avg_length_std,
            "avg_speed_mean": self.avg_speed_mean,
            "best_lap_time_min": self.best_lap_time_min,
            "reward_std_within_seed": self.reward_std_mean,
            "convergence_episode_mean": self.convergence_episode_mean,
            "convergence_episode_std": self.convergence_episode_std,
            "convergence_episode_mean_pm_std": format_mean_pm(self.convergence_episode_mean, self.convergence_episode_std, digits=0),
            # Aliases for prompt templates expecting last_n keys
            "avg_reward_last_n_mean": self.avg_reward_mean,
            "avg_reward_last_n_std": self.avg_reward_std,
            "avg_crash_last_n_mean": crash_pct_mean,
            "avg_crash_last_n_std": crash_pct_std,
            "avg_laps_last_n_mean": self.mean_laps_mean,
            "avg_laps_last_n_std": self.mean_laps_std,
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


def sample_variance_or_none(values: Sequence[float | None]) -> float | None:
    """Return the sample variance (ddof=1) for a list, or None when unavailable."""
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(np.var(cleaned, ddof=1))


def sample_std_or_none(values: Sequence[float | None]) -> float | None:
    """Return the sample standard deviation (ddof=1) for a list, or None when unavailable."""
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(np.std(cleaned, ddof=1))


def sem_or_none(values: Sequence[float | None]) -> float | None:
    """Return the standard error of the mean (sample_std / sqrt(n))."""
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(np.std(cleaned, ddof=1) / np.sqrt(len(cleaned)))


def variance_or_none(values: Sequence[float | None]) -> float | None:
    """Sample variance alias for backward compatibility."""
    return sample_variance_or_none(values)


def mean_or_none(values: Sequence[float | None]) -> float | None:
    """Return the arithmetic mean for a list, or None when unavailable."""
    cleaned = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def load_seed_metrics(
    seed_dir: Path,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    strict: bool = False,
) -> SeedMetrics | None:
    """Load deterministic evaluation metrics from evaluation_summary.json and training progression from training_log.jsonl."""
    eval_sum_path = seed_dir / "evaluation_summary.json"
    if not eval_sum_path.exists():
        if strict:
            raise ValueError(f"Strict validation failure: missing {eval_sum_path}")
        return None

    try:
        with eval_sum_path.open("r", encoding="utf-8") as f:
            eval_data = json.load(f)
    except Exception as ex:
        if strict:
            raise ValueError(f"Strict validation failure: corrupted {eval_sum_path}: {ex}")
        return None

    # Load training logs for training curve / convergence metrics ONLY
    training_log_path = seed_dir / "training_log.jsonl"
    convergence_episode = None
    num_training_episodes = None
    max_training_reward = None
    final_training_reward = None

    if training_log_path.exists():
        train_logs = load_jsonl_logs(training_log_path)
        if train_logs:
            ordered_train = sorted(train_logs, key=lambda item: safe_int(item.get("episode"), 0))
            train_rewards = [safe_float(row.get("reward_total"), 0.0) for row in ordered_train]
            if train_rewards:
                num_training_episodes = len(train_rewards)
                max_training_reward = float(np.max(train_rewards))
                final_training_reward = float(train_rewards[-1])
                convergence_episode = int(
                    estimate_convergence_episode(
                        train_rewards,
                        rolling_window=rolling_window,
                        stability_window=stability_window,
                    )
                )

    seed_name = seed_dir.name.replace("seed_", "")
    meta = eval_data.get("metadata", {})
    if "seed" in meta:
        seed_name = str(meta["seed"])

    return SeedMetrics(
        seed=seed_name,
        num_episodes=safe_int(eval_data.get("num_episodes"), 20),
        avg_reward=safe_float(eval_data.get("avg_reward")),
        reward_std=safe_float(eval_data.get("reward_std")),
        reward_sem=safe_float(eval_data.get("reward_sem")),
        crash_rate=safe_float(eval_data.get("crash_rate")),
        lap_completion_rate=safe_float(eval_data.get("lap_completion_rate")),
        total_laps_completed=safe_int(eval_data.get("total_laps_completed")),
        mean_laps_per_episode=safe_float(eval_data.get("mean_laps_per_episode")),
        laps_std=safe_float(eval_data.get("laps_std")),
        distance_mean=safe_float(eval_data.get("distance_mean")),
        distance_std=safe_float(eval_data.get("distance_std")),
        distance_sem=safe_float(eval_data.get("distance_sem")),
        avg_length=safe_float(eval_data.get("avg_length")),
        length_std=safe_float(eval_data.get("length_std")),
        length_sem=safe_float(eval_data.get("length_sem")),
        avg_speed=safe_float(eval_data.get("avg_speed")),
        speed_std=safe_float(eval_data.get("speed_std")),
        best_lap_time=safe_float(eval_data.get("best_lap_time")) if eval_data.get("best_lap_time") is not None else None,
        avg_lap_time=safe_float(eval_data.get("avg_lap_time")) if eval_data.get("avg_lap_time") is not None else None,
        checkpoint=meta.get("checkpoint"),
        convergence_episode=convergence_episode,
        num_training_episodes=num_training_episodes,
        max_training_reward=max_training_reward,
        final_training_reward=final_training_reward,
    )


def compute_seed_metrics(
    logs: list[dict[str, Any]],
    last_n: int = DEFAULT_LAST_N,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
) -> SeedMetrics | None:
    """Legacy compatibility helper."""
    return None


def aggregate_seed_metrics(algorithm: str, experiment: str, seed_metrics: list[SeedMetrics]) -> AlgorithmMetrics | None:
    """Aggregate seed-level metrics into condition-level statistics using sample SD (ddof=1)."""
    if not seed_metrics:
        return None

    avg_rewards = [s.avg_reward for s in seed_metrics]
    crash_rates = [s.crash_rate for s in seed_metrics]
    lap_rates = [s.lap_completion_rate for s in seed_metrics]
    mean_laps = [s.mean_laps_per_episode for s in seed_metrics]
    total_laps = [float(s.total_laps_completed) for s in seed_metrics]
    distances = [s.distance_mean for s in seed_metrics]
    lengths = [s.avg_length for s in seed_metrics]
    speeds = [s.avg_speed for s in seed_metrics]
    best_laps = [s.best_lap_time for s in seed_metrics if s.best_lap_time is not None]
    avg_laps = [s.avg_lap_time for s in seed_metrics if s.avg_lap_time is not None]
    reward_stds_within = [s.reward_std for s in seed_metrics]
    distance_stds_within = [s.distance_std for s in seed_metrics]
    convergences = [float(s.convergence_episode) for s in seed_metrics if s.convergence_episode is not None]

    return AlgorithmMetrics(
        algorithm=algorithm,
        experiment=experiment,
        seeds=seed_metrics,
        avg_reward_mean=mean_or_none(avg_rewards),
        avg_reward_std=sample_std_or_none(avg_rewards),
        avg_reward_sem=sem_or_none(avg_rewards),
        crash_rate_mean=mean_or_none(crash_rates),
        crash_rate_std=sample_std_or_none(crash_rates),
        crash_rate_sem=sem_or_none(crash_rates),
        lap_completion_rate_mean=mean_or_none(lap_rates),
        lap_completion_rate_std=sample_std_or_none(lap_rates),
        lap_completion_rate_sem=sem_or_none(lap_rates),
        mean_laps_mean=mean_or_none(mean_laps),
        mean_laps_std=sample_std_or_none(mean_laps),
        total_laps_mean=mean_or_none(total_laps),
        total_laps_std=sample_std_or_none(total_laps),
        distance_mean=mean_or_none(distances),
        distance_std=sample_std_or_none(distances),
        distance_sem=sem_or_none(distances),
        avg_length_mean=mean_or_none(lengths),
        avg_length_std=sample_std_or_none(lengths),
        avg_length_sem=sem_or_none(lengths),
        avg_speed_mean=mean_or_none(speeds),
        avg_speed_std=sample_std_or_none(speeds),
        best_lap_time_min=min(best_laps) if best_laps else None,
        best_lap_time_mean=mean_or_none(best_laps),
        avg_lap_time_mean=mean_or_none(avg_laps),
        reward_std_mean=mean_or_none(reward_stds_within),
        distance_std_mean=mean_or_none(distance_stds_within),
        convergence_episode_mean=mean_or_none(convergences),
        convergence_episode_std=sample_std_or_none(convergences),
        status="complete",
    )


def discover_experiment_ids(logs_dir: Path) -> list[str]:
    """Discover experiment IDs shared across algorithms (case-insensitive)."""
    experiments: set[str] = set()
    for algo in ALGORITHMS:
        for a in (algo.upper(), algo.lower(), algo):
            algo_dir = logs_dir / a
            if not algo_dir.exists():
                continue
            for child in algo_dir.iterdir():
                if child.is_dir():
                    has_eval_sum = (child / "evaluation_summary.json").exists() or any(child.glob("seed_*/evaluation_summary.json"))
                    has_train_log = (child / "training_log.jsonl").exists() or any(child.glob("seed_*/training_log.jsonl"))
                    if has_eval_sum or has_train_log:
                        experiments.add(child.name)
    return sorted(experiments, key=experiment_sort_key)


def discover_seed_dirs(logs_dir: Path, algorithm: str, experiment: str) -> list[Path]:
    """Return all seed directories for one algorithm/experiment pair (case-insensitive)."""
    seed_dirs: list[Path] = []
    for a in (algorithm.upper(), algorithm.lower(), algorithm):
        exp_dir = logs_dir / a / experiment
        if not exp_dir.exists():
            continue
        found = sorted(exp_dir.glob("seed_*"))
        for p in found:
            if p.is_dir() and p not in seed_dirs:
                seed_dirs.append(p)
        if not seed_dirs and ((exp_dir / "evaluation_summary.json").exists() or (exp_dir / "training_log.jsonl").exists()):
            seed_dirs.append(exp_dir)
    return sorted(seed_dirs)


def collect_algorithm_metrics(
    logs_dir: Path,
    algorithm: str,
    experiment: str,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    strict: bool = False,
    last_n: int = DEFAULT_LAST_N,
) -> AlgorithmMetrics | None:
    """Collect and aggregate evaluation metrics for one algorithm within one experiment."""
    seed_dirs = discover_seed_dirs(logs_dir, algorithm, experiment)
    if not seed_dirs:
        if strict:
            raise ValueError(f"Strict validation failure: no seed directories found for {algorithm}/{experiment}")
        return None

    seed_metrics: list[SeedMetrics] = []
    for sdir in seed_dirs:
        sm = load_seed_metrics(
            sdir,
            rolling_window=rolling_window,
            stability_window=stability_window,
            strict=strict,
        )
        if sm is not None:
            seed_metrics.append(sm)

    if not seed_metrics:
        return None

    return aggregate_seed_metrics(algorithm, experiment, seed_metrics)


def collect_experiment_reports(
    logs_dir: Path,
    experiment_ids: list[str] | None = None,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    strict: bool = False,
    last_n: int = DEFAULT_LAST_N,
) -> list[ExperimentReport]:
    """Collect evaluation metrics for every experiment."""
    if experiment_ids is None:
        experiment_ids = discover_experiment_ids(logs_dir)

    reports: list[ExperimentReport] = []
    for experiment in experiment_ids:
        td3_metrics = collect_algorithm_metrics(
            logs_dir,
            "td3",
            experiment,
            rolling_window=rolling_window,
            stability_window=stability_window,
            strict=strict,
        )
        ddpg_metrics = collect_algorithm_metrics(
            logs_dir,
            "ddpg",
            experiment,
            rolling_window=rolling_window,
            stability_window=stability_window,
            strict=strict,
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
        "Analyze the following reinforcement learning results.\n\n"
        f"Experiment: {experiment}\n\n"
        f"{focus_line}"
        f"TD3 metrics:\n{json.dumps(td3_metrics, indent=2, sort_keys=True)}\n\n"
        f"DDPG metrics:\n{json.dumps(ddpg_metrics, indent=2, sort_keys=True)}\n\n"
        "Compare TD3 and DDPG in terms of:\n"
        "1. Reward performance (mean ± standard deviation)\n"
        "2. Stability (variance across seeds)\n"
        "3. Crash rate (safety; mean ± standard deviation)\n"
        "4. Convergence speed and sample efficiency (lower convergence episode = faster learning)\n"
        "5. Whether higher reward comes at the cost of higher crash rate\n"
        "6. Whether differences are consistent across seeds based on variance\n"
        "7. Which settings balance reward and low crash rate\n\n"
        "Do NOT declare a single overall winner.\n"
        "Important:\n"
        "- Treat crash rate as a primary metric, not a secondary metric\n"
        "- High reward with high crash rate should be interpreted as unsafe behavior\n"
        "- Lower standard deviation indicates more stable learning\n"
        "- Explain which algorithm converges faster and what it implies about training efficiency\n"
        "Instead:\n"
        "- Highlight strengths of each algorithm\n"
        "- Explain trade-offs between performance and stability\n"
        "- Mention how noise levels affect behavior\n\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Keep the explanation concise and technical.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
    )


def build_summary_prompt(experiment_reports: list[dict[str, Any]]) -> str:
    """Build the final summary prompt across all experiments."""
    return (
        "Analyze the following reinforcement learning results across multiple experiments.\n\n"
        f"Experiment summaries:\n{json.dumps(experiment_reports, indent=2, sort_keys=True)}\n\n"
        "Compare TD3 and DDPG in terms of:\n"
        "1. Reward performance (mean ± standard deviation)\n"
        "2. Stability (variance across seeds)\n"
        "3. Crash rate (safety; mean ± standard deviation)\n"
        "4. Convergence speed and sample efficiency (lower convergence episode = faster learning)\n"
        "5. Whether higher reward comes at the cost of higher crash rate\n"
        "6. Whether differences are consistent across seeds based on variance\n"
        "7. Safety-performance balance under different noise levels\n\n"
        "Do NOT declare a single overall winner.\n"
        "Important:\n"
        "- Treat crash rate as a primary metric, not a secondary metric\n"
        "- High reward with high crash rate should be interpreted as unsafe behavior\n"
        "- Lower standard deviation indicates more stable learning\n"
        "- Provide clear, technical reasoning\n\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n\n"
        "Highlight trade-offs and noise-related behavior changes.\n"
        "Keep the explanation concise, technical, and evidence-based.\n"
        "Do not guess missing values. Base the answer only on the metrics provided."
    )


def build_noise_level_prompt(noise_label: str, noise_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for one noise level section."""
    return (
        "Analyze the following reinforcement learning results.\n\n"
        f"Noise level: {noise_label}\n\n"
        f"Structured metrics:\n{json.dumps(noise_metrics, indent=2, sort_keys=True)}\n\n"
        "For this noise level, compare TD3 and DDPG in terms of:\n"
        "1. Reward performance (mean ± standard deviation)\n"
        "2. Stability (variance across seeds)\n"
        "3. Crash rate (safety; mean ± standard deviation)\n"
        "4. Convergence speed and sample efficiency (lower convergence episode = faster learning)\n"
        "5. Whether higher reward comes at the cost of higher crash rate\n"
        "6. Whether differences are consistent across seeds based on variance\n"
        "7. Which settings balance reward and low crash rate\n\n"
        "Do NOT declare a single overall winner.\n"
        "Important:\n"
        "- Treat crash rate as a primary metric, not a secondary metric\n"
        "- High reward with high crash rate should be interpreted as unsafe behavior\n"
        "- Lower standard deviation indicates more stable learning\n"
        "- Explain which algorithm converges faster and what it implies about training efficiency\n"
        "Instead, identify which reward systems show higher reward, which are more stable, and where trade-offs appear.\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Keep the explanation concise and evidence-based."
    )


def build_algorithm_comparison_prompt(comparison_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for TD3 vs DDPG comparison."""
    return (
        "Analyze the following reinforcement learning results.\n\n"
        f"Structured metrics:\n{json.dumps(comparison_metrics, indent=2, sort_keys=True)}\n\n"
        "Compare TD3 and DDPG in terms of:\n"
        "1. Reward performance (mean ± standard deviation)\n"
        "2. Stability (variance across seeds)\n"
        "3. Crash rate (safety; mean ± standard deviation)\n"
        "4. Convergence speed and sample efficiency (lower convergence episode = faster learning)\n"
        "5. Whether higher reward comes at the cost of higher crash rate\n"
        "6. Whether differences are consistent across seeds based on variance\n"
        "7. Which configurations balance reward and low crash rate\n\n"
        "Do NOT declare a single overall winner.\n"
        "Important:\n"
        "- High reward with high crash rate should be interpreted as unsafe behavior\n"
        "- Lower standard deviation indicates more stable learning\n"
        "- Explain which algorithm converges faster and what it implies about training efficiency\n"
        "Highlight strengths of each algorithm and explain trade-offs between reward and robustness.\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Keep the explanation concise and evidence-based."
    )


def build_key_insights_prompt(insight_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for cross-noise key insights."""
    return (
        "Analyze the following reinforcement learning results across noise levels.\n\n"
        f"Structured metrics:\n{json.dumps(insight_metrics, indent=2, sort_keys=True)}\n\n"
        "Provide:\n"
        "1. Reward-system patterns across noise levels (mean ± standard deviation)\n"
        "2. TD3 vs DDPG trend by metric (reward, crash, stability, convergence)\n"
        "3. Safety-aware trade-offs between performance and robustness\n"
        "4. Seed-consistency insights from variance\n"
        "5. Sample-efficiency insights from convergence episode\n\n"
        "Do NOT declare a single overall winner.\n"
        "Treat crash rate as a primary metric, not a secondary metric.\n"
        "Important: high reward with high crash rate should be interpreted as unsafe behavior.\n"
        "Lower standard deviation indicates more stable learning.\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Keep the explanation concise and evidence-based."
    )


def build_final_conclusion_prompt(conclusion_metrics: dict[str, Any]) -> str:
    """Build a structured prompt for the final conclusion section."""
    return (
        "You are writing the conclusion for a research-style RL report.\n\n"
        f"Structured metrics:\n{json.dumps(conclusion_metrics, indent=2, sort_keys=True)}\n\n"
        "Summarize:\n"
        "1. Trade-offs between reward performance and robustness\n"
        "2. When TD3 is preferable (stability and safety prioritized)\n"
        "3. When DDPG is preferable (reward performance prioritized)\n"
        "4. How crash-rate trends affect safety interpretation\n"
        "5. Effect of noise on behavior and reliability\n"
        "6. Convergence and sample-efficiency implications (lower convergence episode = faster learning)\n\n"
        "Do NOT declare a single overall winner.\n"
        "Avoid absolute statements such as 'best model'.\n"
        "Treat crash rate as a primary metric, not a secondary metric.\n"
        "Interpret stability correctly: lower standard deviation means more stable learning.\n"
        "End with:\n"
        "Key Takeaways:\n"
        "- Bullet 1\n"
        "- Bullet 2\n"
        "Keep the conclusion concise and evidence-based."
    )


def fallback_noise_analysis(noise_label: str, noise_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for one noise level."""
    td3 = noise_metrics.get("td3", {})
    ddpg = noise_metrics.get("ddpg", {})
    highest_reward = noise_metrics.get("highest_reward_system", "n/a")
    most_stable = noise_metrics.get("most_stable_reward_system", "n/a")
    return (
        f"Noise {noise_label}: higher reward tends to appear in {highest_reward}, while stronger stability tends to appear in {most_stable}. "
        f"TD3 reward {format_mean_pm(td3.get('avg_reward_last_n_mean'), td3.get('avg_reward_last_n_std'))}, "
        f"DDPG reward {format_mean_pm(ddpg.get('avg_reward_last_n_mean'), ddpg.get('avg_reward_last_n_std'))}; "
        f"TD3 crash {format_mean_pm(td3.get('avg_crash_last_n_mean'), td3.get('avg_crash_last_n_std'), unit='%')}, "
        f"DDPG crash {format_mean_pm(ddpg.get('avg_crash_last_n_mean'), ddpg.get('avg_crash_last_n_std'), unit='%')}.\n"
        "Key Takeaways:\n"
        "- Safety must be interpreted jointly with reward; high reward and high crash is unsafe.\n"
        "- Lower standard deviation indicates more stable learning under this noise level."
    )


def fallback_comparison_analysis(comparison_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for TD3 vs DDPG comparison."""
    td3 = comparison_metrics.get("td3", {})
    ddpg = comparison_metrics.get("ddpg", {})
    return (
        f"TD3 reward {format_mean_pm(td3.get('avg_reward_last_n_mean'), td3.get('avg_reward_last_n_std'))}, "
        f"DDPG reward {format_mean_pm(ddpg.get('avg_reward_last_n_mean'), ddpg.get('avg_reward_last_n_std'))}; "
        f"TD3 crash {format_mean_pm(td3.get('avg_crash_last_n_mean'), td3.get('avg_crash_last_n_std'), unit='%')}, "
        f"DDPG crash {format_mean_pm(ddpg.get('avg_crash_last_n_mean'), ddpg.get('avg_crash_last_n_std'), unit='%')}; "
        f"TD3 convergence {format_mean_pm(td3.get('convergence_episode_mean'), td3.get('convergence_episode_std'), digits=0)}, "
        f"DDPG convergence {format_mean_pm(ddpg.get('convergence_episode_mean'), ddpg.get('convergence_episode_std'), digits=0)}.\n"
        "Key Takeaways:\n"
        "- Lower convergence episode implies faster sample-efficient learning.\n"
        "- Reward must be interpreted with crash rate to avoid unsafe conclusions."
    )


def fallback_insights_analysis(insight_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for key insights across noise levels."""
    highest_reward = insight_metrics.get("highest_reward_system", "n/a")
    most_stable = insight_metrics.get("most_stable_reward_system", "n/a")
    lower_crash = insight_metrics.get("lower_crash_algorithm", "n/a")
    faster_convergence = insight_metrics.get("faster_converging_algorithm", "n/a")
    return (
        f"Highest reward-system trend across noise levels: {highest_reward}. "
        f"Most stable reward-system trend: {most_stable}. "
        f"Lower crash tendency appears in {lower_crash}, while faster convergence appears in {faster_convergence}. "
        "Noise generally reduces reward and increases variability, emphasizing a trade-off between raw performance and robustness.\n"
        "Key Takeaways:\n"
        "- Stability-sensitive deployments should prioritize low variance and low crash behavior.\n"
        "- Performance-oriented deployments should still verify safety under higher noise."
    )


def fallback_conclusion_analysis(conclusion_metrics: dict[str, Any]) -> str:
    """Deterministic fallback for the final conclusion."""
    return (
        "The results indicate a trade-off between performance and robustness. "
        f"Highest reward trend is associated with {conclusion_metrics.get('highest_reward_system', 'n/a')}, "
        f"while greater stability trend is associated with {conclusion_metrics.get('most_stable_reward_system', 'n/a')}. "
        "DDPG can achieve higher rewards and faster convergence in several settings, while TD3 can provide more stable learning and improved safety in certain conditions. "
        "Higher noise levels generally reduce reward and increase instability.\n"
        "Key Takeaways:\n"
        "- TD3 is preferable when stability and safety are prioritized.\n"
        "- DDPG is preferable when higher reward is prioritized and crash behavior remains acceptable."
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
        lines.append(
            "The metrics show a trade-off between reward and robustness: "
            f"TD3 reward {format_number(td3_avg)} vs DDPG reward {format_number(ddpg_avg)}, "
            f"with reward standard deviation {format_number(td3_stability)} vs {format_number(ddpg_stability)}. "
            "Lower standard deviation indicates more stable learning."
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


def build_report_styles() -> Any:
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


def std_from_variance(variance: Any) -> float | None:
    """Compute standard deviation from variance with numeric guards."""
    if variance is None:
        return None
    try:
        numeric = float(variance)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric) or numeric < 0.0:
        return None
    return float(math.sqrt(numeric))


def format_mean_pm(mean_value: Any, std_value: Any, digits: int = 2, unit: str = "") -> str:
    """Format a publication-style mean ± std string."""
    mean_text = format_table_value(mean_value, digits=digits)
    std_text = format_table_value(std_value, digits=digits)
    if mean_text == "n/a":
        return "n/a"
    if std_text == "n/a":
        return f"{mean_text}{unit}"
    return f"{mean_text} ± {std_text}{unit}"


def split_key_takeaways(analysis_text: str) -> tuple[str, list[str]]:
    """Split an analysis block into narrative body and takeaway bullets."""
    lines = [line.strip() for line in str(analysis_text).splitlines() if line.strip()]
    if not lines:
        return "", []

    marker_index = -1
    for idx, line in enumerate(lines):
        if line.lower().startswith("key takeaways"):
            marker_index = idx
            break

    if marker_index < 0:
        return "\n".join(lines), []

    body_lines = lines[:marker_index]
    bullets: list[str] = []
    for line in lines[marker_index + 1 :]:
        cleaned = line.lstrip("-*").strip()
        if cleaned:
            bullets.append(cleaned)
    return "\n".join(body_lines), bullets[:3]


def ensure_takeaways(analysis_text: str, fallback_takeaways: list[str]) -> tuple[str, list[str]]:
    """Guarantee at least two key takeaways for each analysis section."""
    body, bullets = split_key_takeaways(analysis_text)
    if len(bullets) >= 2:
        return body, bullets
    merged = bullets + [item for item in fallback_takeaways if item]
    deduped: list[str] = []
    for item in merged:
        if item not in deduped:
            deduped.append(item)
    return body, deduped[:3]


def build_noise_takeaways(noise_metrics: dict[str, Any]) -> list[str]:
    """Create concise noise-level takeaways from structured metrics."""
    td3 = noise_metrics.get("td3") or {}
    ddpg = noise_metrics.get("ddpg") or {}
    return [
        (
            "Higher reward trend: "
            f"TD3 {format_mean_pm(td3.get('avg_reward_last_n_mean'), td3.get('avg_reward_last_n_std'))}, "
            f"DDPG {format_mean_pm(ddpg.get('avg_reward_last_n_mean'), ddpg.get('avg_reward_last_n_std'))}."
        ),
        (
            "Safety and stability trend: "
            f"TD3 crash {format_mean_pm(td3.get('avg_crash_last_n_mean'), td3.get('avg_crash_last_n_std'), unit='%')}, "
            f"DDPG crash {format_mean_pm(ddpg.get('avg_crash_last_n_mean'), ddpg.get('avg_crash_last_n_std'), unit='%')}; "
            "lower crash and lower standard deviation indicate more robust behavior."
        ),
    ]


def build_comparison_takeaways(comparison_metrics: dict[str, Any]) -> list[str]:
    """Create concise TD3-vs-DDPG takeaways from aggregate metrics."""
    td3 = comparison_metrics.get("td3") or {}
    ddpg = comparison_metrics.get("ddpg") or {}
    return [
        (
            "Sample efficiency: lower convergence episode means faster learning; "
            f"TD3 {format_mean_pm(td3.get('convergence_episode_mean'), td3.get('convergence_episode_std'), digits=0)}, "
            f"DDPG {format_mean_pm(ddpg.get('convergence_episode_mean'), ddpg.get('convergence_episode_std'), digits=0)}."
        ),
        (
            "Reward-safety trade-off: "
            f"TD3 reward {format_mean_pm(td3.get('avg_reward_last_n_mean'), td3.get('avg_reward_last_n_std'))}, "
            f"DDPG reward {format_mean_pm(ddpg.get('avg_reward_last_n_mean'), ddpg.get('avg_reward_last_n_std'))}; "
            f"TD3 crash {format_mean_pm(td3.get('avg_crash_last_n_mean'), td3.get('avg_crash_last_n_std'), unit='%')}, "
            f"DDPG crash {format_mean_pm(ddpg.get('avg_crash_last_n_mean'), ddpg.get('avg_crash_last_n_std'), unit='%')}."
        ),
    ]


def pick_label_by_metric(items: dict[str, float | None], prefer: str = "higher") -> str | None:
    """Pick the label with the best available metric value without cross-metric scoring."""
    valid = [(label, float(value)) for label, value in items.items() if value is not None and not math.isnan(float(value))]
    if not valid:
        return None
    if prefer == "lower":
        return min(valid, key=lambda pair: pair[1])[0]
    return max(valid, key=lambda pair: pair[1])[0]


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


def group_reports_by_reward_and_noise(
    reports: list[ExperimentReport],
) -> dict[str, dict[str, dict[str, AlgorithmMetrics | None]]]:
    """Group experiment reports by noise level and reward system."""
    grouped: dict[str, dict[str, dict[str, AlgorithmMetrics | None]]] = {
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


def summarize_algorithm(metrics_list: Sequence[AlgorithmMetrics | None]) -> dict[str, Any] | None:
    """Summarize one algorithm across a collection of experiments."""
    metrics = [item for item in metrics_list if item is not None]
    if not metrics:
        return None

    reward_var = variance_or_none([item.avg_reward_last_n_mean for item in metrics])
    crash_var = variance_or_none([item.avg_crash_last_n_mean for item in metrics])
    laps_var = variance_or_none([item.avg_laps_last_n_mean for item in metrics])
    conv_var = variance_or_none([item.convergence_episode_mean for item in metrics])

    return {
        "experiment_count": len(metrics),
        "avg_reward_last_n_mean": mean_or_none([item.avg_reward_last_n_mean for item in metrics]),
        "avg_reward_last_n_variance": reward_var,
        "avg_reward_last_n_std": std_from_variance(reward_var),
        "avg_crash_last_n_mean": mean_or_none([item.avg_crash_last_n_mean for item in metrics]),
        "avg_crash_last_n_variance": crash_var,
        "avg_crash_last_n_std": std_from_variance(crash_var),
        "avg_laps_last_n_mean": mean_or_none([item.avg_laps_last_n_mean for item in metrics]),
        "avg_laps_last_n_variance": laps_var,
        "avg_laps_last_n_std": std_from_variance(laps_var),
        "reward_std_mean": mean_or_none([item.reward_std_mean for item in metrics]),
        "convergence_episode_mean": mean_or_none([item.convergence_episode_mean for item in metrics]),
        "convergence_episode_variance": conv_var,
        "convergence_episode_std": std_from_variance(conv_var),
    }


def summarize_reward_systems(reports: list[ExperimentReport]) -> dict[str, dict[str, Any]]:
    """Aggregate metrics by reward system across all noise levels and algorithms."""
    grouped = group_reports_by_reward(reports)
    summary: dict[str, dict[str, Any]] = {}

    for reward_label, reward_reports in grouped.items():
        summary[reward_label] = {
            "td3": summarize_algorithm([report.td3 for report in reward_reports]),
            "ddpg": summarize_algorithm([report.ddpg for report in reward_reports]),
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
        f"Deterministic Eval Reward: {format_mean_pm(metrics.avg_reward_mean, metrics.avg_reward_std)}",
        f"Crash Rate: {format_mean_pm(metrics.avg_crash_last_n_mean, metrics.avg_crash_last_n_std, unit='%')}",
        f"Laps per Episode: {format_mean_pm(metrics.mean_laps_mean, metrics.mean_laps_std)}",
        f"Within-seed Eval Stability: {format_table_value(metrics.reward_std_mean)}",
        f"Training Convergence Episode: {format_mean_pm(metrics.convergence_episode_mean, metrics.convergence_episode_std, digits=0)}",
    ]


def build_summary_table_rows(reports: list[ExperimentReport], styles) -> tuple[list[list[Paragraph]], list[str]]:
    """Build the research-style summary table from deterministic evaluation metrics across seeds."""
    rows: list[list[Paragraph]] = [
        [
            Paragraph("Experiment", styles["TableCell"]),
            Paragraph("TD3 Reward ± SD", styles["TableCell"]),
            Paragraph("DDPG Reward ± SD", styles["TableCell"]),
            Paragraph("TD3 Crash ± SD", styles["TableCell"]),
            Paragraph("DDPG Crash ± SD", styles["TableCell"]),
            Paragraph("TD3 Conv", styles["TableCell"]),
            Paragraph("DDPG Conv", styles["TableCell"]),
        ]
    ]
    skipped: list[str] = []

    for report in reports:
        if report.td3 is None or report.ddpg is None:
            skipped.append(report.experiment)
            continue

        rows.append(
            [
                Paragraph(report.experiment, styles["TableCellLeft"]),
                Paragraph(
                    format_mean_pm(report.td3.avg_reward_mean, report.td3.avg_reward_std),
                    styles["TableCell"],
                ),
                Paragraph(
                    format_mean_pm(report.ddpg.avg_reward_mean, report.ddpg.avg_reward_std),
                    styles["TableCell"],
                ),
                Paragraph(
                    format_mean_pm(
                        report.td3.avg_crash_last_n_mean,
                        report.td3.avg_crash_last_n_std,
                        unit="%",
                    ),
                    styles["TableCell"],
                ),
                Paragraph(
                    format_mean_pm(
                        report.ddpg.avg_crash_last_n_mean,
                        report.ddpg.avg_crash_last_n_std,
                        unit="%",
                    ),
                    styles["TableCell"],
                ),
                Paragraph(
                    format_mean_pm(
                        report.td3.convergence_episode_mean,
                        report.td3.convergence_episode_std,
                        digits=0,
                    ),
                    styles["TableCell"],
                ),
                Paragraph(
                    format_mean_pm(
                        report.ddpg.convergence_episode_mean,
                        report.ddpg.convergence_episode_std,
                        digits=0,
                    ),
                    styles["TableCell"],
                ),
            ]
        )

    return rows, skipped


def build_summary_table(reports: list[ExperimentReport], styles) -> tuple[Table, list[str]]:
    """Create the summary metrics table for the report."""
    rows, skipped = build_summary_table_rows(reports, styles)
    table = Table(
        rows,
        colWidths=[1.1 * inch, 1.0 * inch, 1.0 * inch, 0.95 * inch, 0.95 * inch, 0.9 * inch, 0.9 * inch],
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
    takeaway_lines: list[str] | None = None,
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
    if takeaway_lines:
        block.append(Spacer(1, 0.05 * inch))
        block.append(Paragraph("Key Takeaways", styles["SubSectionHeading"]))
        for point in takeaway_lines[:3]:
            block.append(Paragraph(f"- {xml_escape(point)}", styles["AnalysisBody"]))
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
    grouped_noise_reports: Mapping[str, Mapping[str, AlgorithmMetrics | None]],
    results_dir: Path,
    styles,
    analysis_text: str,
    takeaway_lines: list[str] | None = None,
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
    ]
    if takeaway_lines:
        block.append(Spacer(1, 0.04 * inch))
        block.append(Paragraph("Key Takeaways", styles["SubSectionHeading"]))
        for point in takeaway_lines[:3]:
            block.append(Paragraph(f"- {xml_escape(point)}", styles["AnalysisBody"]))
    block.append(PageBreak())
    return block


def build_overall_metric_view(metrics_list: Sequence[AlgorithmMetrics | None], algorithm: str) -> dict[str, Any] | None:
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


def _reward_system_summary(reports: list[ExperimentReport]) -> dict[str, Any] | None:
    """Summarize one reward system across all noise levels."""
    td3_metrics = [report.td3 for report in reports if report.td3 is not None]
    ddpg_metrics = [report.ddpg for report in reports if report.ddpg is not None]
    if not td3_metrics and not ddpg_metrics:
        return None

    return {
        "td3": summarize_algorithm(td3_metrics),
        "ddpg": summarize_algorithm(ddpg_metrics),
    }


def build_noise_level_metrics(
    noise_label: str,
    noise_reports: Mapping[str, Mapping[str, AlgorithmMetrics | None]],
) -> dict[str, Any]:
    """Build structured metrics for one fixed noise level."""
    reward_systems: dict[str, Any] = {}
    td3_summary_metrics = [pair.get("td3") for pair in noise_reports.values() if pair.get("td3") is not None]
    ddpg_summary_metrics = [pair.get("ddpg") for pair in noise_reports.values() if pair.get("ddpg") is not None]
    reward_candidates: dict[str, float | None] = {}
    stability_candidates: dict[str, float | None] = {}

    for reward_label in REWARD_LEVELS:
        reward_pair = noise_reports.get(reward_label, {})
        td3_metrics = reward_pair.get("td3")
        ddpg_metrics = reward_pair.get("ddpg")
        reward_systems[reward_label] = {
            "td3": td3_metrics.to_prompt_dict() if td3_metrics else None,
            "ddpg": ddpg_metrics.to_prompt_dict() if ddpg_metrics else None,
        }

        reward_values = []
        stability_values = []
        if td3_metrics is not None:
            reward_values.append(td3_metrics.avg_reward_last_n_mean)
            stability_values.append(td3_metrics.reward_std_mean)
        if ddpg_metrics is not None:
            reward_values.append(ddpg_metrics.avg_reward_last_n_mean)
            stability_values.append(ddpg_metrics.reward_std_mean)

        reward_candidates[reward_label] = mean_or_none(reward_values)
        stability_candidates[reward_label] = mean_or_none(stability_values)

    highest_reward_system = pick_label_by_metric(reward_candidates, prefer="higher")
    most_stable_reward_system = pick_label_by_metric(stability_candidates, prefer="lower")

    return {
        "noise_level": noise_label,
        "td3": summarize_algorithm(td3_summary_metrics),
        "ddpg": summarize_algorithm(ddpg_summary_metrics),
        "reward_systems": reward_systems,
        "highest_reward_system": highest_reward_system,
        "most_stable_reward_system": most_stable_reward_system,
    }


def build_comparison_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the TD3 vs DDPG comparison section."""
    ordered_reports = [report for report in reports if report.td3 is not None and report.ddpg is not None]
    td3_summary = summarize_algorithm([report.td3 for report in ordered_reports])
    ddpg_summary = summarize_algorithm([report.ddpg for report in ordered_reports])

    return {
        "td3": td3_summary,
        "ddpg": ddpg_summary,
        "reward_gap_td3_minus_ddpg": None
        if td3_summary is None or ddpg_summary is None
        else safe_float(td3_summary.get("avg_reward_last_n_mean"), 0.0)
        - safe_float(ddpg_summary.get("avg_reward_last_n_mean"), 0.0),
        "crash_gap_td3_minus_ddpg": None
        if td3_summary is None or ddpg_summary is None
        else safe_float(td3_summary.get("avg_crash_last_n_mean"), 0.0)
        - safe_float(ddpg_summary.get("avg_crash_last_n_mean"), 0.0),
        "stability_gap_td3_minus_ddpg": None
        if td3_summary is None or ddpg_summary is None
        else safe_float(td3_summary.get("reward_std_mean"), 0.0)
        - safe_float(ddpg_summary.get("reward_std_mean"), 0.0),
        "convergence_gap_td3_minus_ddpg": None
        if td3_summary is None or ddpg_summary is None
        else safe_float(td3_summary.get("convergence_episode_mean"), 0.0)
        - safe_float(ddpg_summary.get("convergence_episode_mean"), 0.0),
    }


def build_key_insight_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the cross-noise insights section."""
    reward_summaries = {
        reward_label: _reward_system_summary(reward_reports)
        for reward_label, reward_reports in group_reports_by_reward(reports).items()
    }

    highest_reward_system = None
    best_reward = float("-inf")
    most_stable_reward_system = None
    best_stability = float("inf")

    for reward_label, summary in reward_summaries.items():
        if summary is None:
            continue
        reward_value = mean_or_none(
            [
                summary[algo].get("avg_reward_last_n_mean")
                for algo in ("td3", "ddpg")
                if summary.get(algo) is not None
            ]
        )
        stability_value = mean_or_none(
            [
                summary[algo].get("reward_std_mean")
                for algo in ("td3", "ddpg")
                if summary.get(algo) is not None
            ]
        )

        if reward_value is not None and reward_value > best_reward:
            best_reward = reward_value
            highest_reward_system = reward_label
        if stability_value is not None and stability_value < best_stability:
            best_stability = stability_value
            most_stable_reward_system = reward_label

    comparison_metrics = build_comparison_metrics(reports)

    td3_summary = comparison_metrics.get("td3") or {}
    ddpg_summary = comparison_metrics.get("ddpg") or {}
    lower_crash_algorithm = pick_label_by_metric(
        {
            "TD3": td3_summary.get("avg_crash_last_n_mean"),
            "DDPG": ddpg_summary.get("avg_crash_last_n_mean"),
        },
        prefer="lower",
    )
    faster_converging_algorithm = pick_label_by_metric(
        {
            "TD3": td3_summary.get("convergence_episode_mean"),
            "DDPG": ddpg_summary.get("convergence_episode_mean"),
        },
        prefer="lower",
    )

    return {
        "reward_systems": reward_summaries,
        "highest_reward_system": highest_reward_system,
        "most_stable_reward_system": most_stable_reward_system,
        "lower_crash_algorithm": lower_crash_algorithm,
        "faster_converging_algorithm": faster_converging_algorithm,
        "comparison": comparison_metrics,
    }


def build_conclusion_metrics(reports: list[ExperimentReport]) -> dict[str, Any]:
    """Build structured metrics for the final conclusion section."""
    comparison_metrics = build_comparison_metrics(reports)
    reward_metrics = build_key_insight_metrics(reports)
    return {
        "highest_reward_system": reward_metrics.get("highest_reward_system", "n/a"),
        "most_stable_reward_system": reward_metrics.get("most_stable_reward_system", "n/a"),
        "lower_crash_algorithm": reward_metrics.get("lower_crash_algorithm", "n/a"),
        "faster_converging_algorithm": reward_metrics.get("faster_converging_algorithm", "n/a"),
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
    noise_metrics_by_label: dict[str, dict[str, Any]] = {}
    for noise_label in NOISE_LEVELS:
        noise_metrics = build_noise_level_metrics(noise_label, grouped_by_noise.get(noise_label, {}))
        noise_metrics_by_label[noise_label] = noise_metrics
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
                "Structured research-style report built from deterministic evaluation summaries (20 episodes, 600-step horizon, exploration noise OFF), seed-aggregated metrics, grouped noise-level plots, algorithm comparisons, and summary text.",
                styles["ReportSubtitle"],
            ),
            Spacer(1, 0.28 * inch),
            Table(
                [
                    [Paragraph("Project Title", styles["TableCellLeft"]), Paragraph("TD3 vs DDPG Performance Analysis in Autonomous Driving Environment", styles["TableCellLeft"])],
                    [Paragraph("Description", styles["TableCellLeft"]), Paragraph("Comparison of reward systems across noise levels and algorithm behavior under deterministic evaluation.", styles["TableCellLeft"])],
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
            "Evaluation metrics are reported as condition mean ± sample standard deviation (ddof=1) across independent training seeds. Crash rate represents deterministic collision frequency during evaluation, and convergence episode indicates sample-efficient training progression.",
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
        noise_body, noise_takeaways = ensure_takeaways(
            analysis_text,
            build_noise_takeaways(noise_metrics_by_label.get(noise_label, {})),
        )
        story.extend(
            build_noise_section_block(
                noise_label=noise_label,
                grouped_noise_reports=noise_reports,
                results_dir=results_dir,
                styles=styles,
                analysis_text=noise_body,
                takeaway_lines=noise_takeaways,
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
    comp_body, comp_takeaways = ensure_takeaways(comparison_analysis, build_comparison_takeaways(comparison_metrics))
    for idx, (metric_key, metric_title, metric_label) in enumerate([
        ("reward", "TD3 vs DDPG Reward Comparison", "Reward vs Episodes"),
        ("crash", "TD3 vs DDPG Crash Rate Comparison", "Crash Rate vs Episodes"),
        ("laps", "TD3 vs DDPG Laps Comparison", "Laps vs Episodes"),
    ]):
        story.extend(
            build_plot_story_block(
                title=metric_title,
                image_path=comparison_files[metric_key],
                caption=f"{metric_label} across all experiments with uncertainty represented as mean ± standard deviation.",
                styles=styles,
                analysis_lines=wrap_paragraphs(comp_body, width=92) if idx == 0 else None,
                takeaway_lines=comp_takeaways if idx == 0 else None,
                image_height=4.0 * inch,
            )
        )

    story.append(Paragraph("Key Aggregate Insights", styles["SectionHeading"]))
    story.append(
        Paragraph(
            "The following section summarizes cross-noise trade-offs with explicit safety, stability, and sample-efficiency interpretation.",
            styles["ReportNote"],
        )
    )
    insight_body, insight_takeaways = ensure_takeaways(
        insight_analysis,
        [
            "Noise increases variability and can worsen safety, so robustness must be checked alongside reward.",
            "Convergence and crash-rate trends should guide algorithm choice by deployment priorities.",
        ],
    )
    story.append(Paragraph(xml_escape(insight_body), styles["AnalysisBody"]))
    story.append(Spacer(1, 0.04 * inch))
    story.append(Paragraph("Key Takeaways", styles["SubSectionHeading"]))
    for point in insight_takeaways[:3]:
        story.append(Paragraph(f"- {xml_escape(point)}", styles["AnalysisBody"]))
    story.append(PageBreak())

    story.append(Paragraph("Final Conclusion", styles["SectionHeading"]))
    conclusion_body, conclusion_takeaways = ensure_takeaways(
        conclusion_analysis,
        [
            "Prioritize TD3 when stability and safety are primary constraints.",
            "Prioritize DDPG when maximizing reward is primary and crash-rate limits remain acceptable.",
        ],
    )
    conclusion_paragraphs = wrap_paragraphs(conclusion_body, width=94)
    for paragraph in conclusion_paragraphs:
        story.append(Paragraph(xml_escape(paragraph), styles["AnalysisBody"]))
    story.append(Spacer(1, 0.04 * inch))
    story.append(Paragraph("Key Takeaways", styles["SubSectionHeading"]))
    for point in conclusion_takeaways[:3]:
        story.append(Paragraph(f"- {xml_escape(point)}", styles["AnalysisBody"]))

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


def export_results_tables(reports: list[ExperimentReport], results_dir: Path) -> dict[str, str]:
    """Export machine-readable run-level, condition-level, and comparison CSV/JSON files."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run-level evaluation results
    run_rows: list[dict[str, Any]] = []
    for report in reports:
        for algo_name, algo_metrics in [("TD3", report.td3), ("DDPG", report.ddpg)]:
            if algo_metrics is None:
                continue
            for seed_metric in algo_metrics.seeds:
                reward_level, noise_level = parse_experiment_components(report.experiment)
                run_rows.append({
                    "algorithm": algo_name,
                    "experiment": report.experiment,
                    "condition": report.experiment,
                    "reward_level": reward_level or "",
                    "noise_level": noise_level or "",
                    "seed": seed_metric.seed,
                    "eval_episodes": seed_metric.num_episodes,
                    "avg_reward": seed_metric.avg_reward,
                    "reward_std": seed_metric.reward_std,
                    "reward_sem": seed_metric.reward_sem,
                    "crash_rate": seed_metric.crash_rate,
                    "crash_rate_pct": seed_metric.crash_rate * 100.0,
                    "lap_completion_rate": seed_metric.lap_completion_rate,
                    "total_laps_completed": seed_metric.total_laps_completed,
                    "mean_laps_per_episode": seed_metric.mean_laps_per_episode,
                    "laps_std": seed_metric.laps_std,
                    "distance_mean": seed_metric.distance_mean,
                    "distance_std": seed_metric.distance_std,
                    "distance_sem": seed_metric.distance_sem,
                    "avg_length": seed_metric.avg_length,
                    "length_std": seed_metric.length_std,
                    "length_sem": seed_metric.length_sem,
                    "avg_speed": seed_metric.avg_speed,
                    "speed_std": seed_metric.speed_std,
                    "best_lap_time": seed_metric.best_lap_time,
                    "avg_lap_time": seed_metric.avg_lap_time,
                    "convergence_episode": seed_metric.convergence_episode,
                    "num_training_episodes": seed_metric.num_training_episodes,
                    "checkpoint": seed_metric.checkpoint,
                })

    run_csv_path = results_dir / "run_level_eval_results.csv"
    run_json_path = results_dir / "run_level_eval_results.json"
    if run_rows:
        with run_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
        with run_json_path.open("w", encoding="utf-8") as f:
            json.dump(run_rows, f, indent=2)

    # 2. Condition-level results across seeds
    cond_rows: list[dict[str, Any]] = []
    for report in reports:
        for algo_name, algo_metrics in [("TD3", report.td3), ("DDPG", report.ddpg)]:
            if algo_metrics is None:
                continue
            reward_level, noise_level = parse_experiment_components(report.experiment)
            cond_rows.append({
                "algorithm": algo_name,
                "experiment": report.experiment,
                "condition": report.experiment,
                "reward_level": reward_level or "",
                "noise_level": noise_level or "",
                "seed_count": len(algo_metrics.seeds),
                "avg_reward_mean": algo_metrics.avg_reward_mean,
                "avg_reward_std": algo_metrics.avg_reward_std,
                "avg_reward_sem": algo_metrics.avg_reward_sem,
                "crash_rate_mean": algo_metrics.crash_rate_mean,
                "crash_rate_std": algo_metrics.crash_rate_std,
                "crash_rate_sem": algo_metrics.crash_rate_sem,
                "crash_rate_mean_pct": (algo_metrics.crash_rate_mean * 100.0) if algo_metrics.crash_rate_mean is not None else None,
                "lap_completion_rate_mean": algo_metrics.lap_completion_rate_mean,
                "lap_completion_rate_std": algo_metrics.lap_completion_rate_std,
                "lap_completion_rate_sem": algo_metrics.lap_completion_rate_sem,
                "mean_laps_mean": algo_metrics.mean_laps_mean,
                "mean_laps_std": algo_metrics.mean_laps_std,
                "total_laps_mean": algo_metrics.total_laps_mean,
                "total_laps_std": algo_metrics.total_laps_std,
                "distance_mean": algo_metrics.distance_mean,
                "distance_std": algo_metrics.distance_std,
                "distance_sem": algo_metrics.distance_sem,
                "avg_length_mean": algo_metrics.avg_length_mean,
                "avg_length_std": algo_metrics.avg_length_std,
                "avg_length_sem": algo_metrics.avg_length_sem,
                "avg_speed_mean": algo_metrics.avg_speed_mean,
                "avg_speed_std": algo_metrics.avg_speed_std,
                "best_lap_time_min": algo_metrics.best_lap_time_min,
                "reward_std_within_seed_mean": algo_metrics.reward_std_mean,
                "convergence_episode_mean": algo_metrics.convergence_episode_mean,
                "convergence_episode_std": algo_metrics.convergence_episode_std,
            })

    cond_csv_path = results_dir / "condition_level_eval_results.csv"
    cond_json_path = results_dir / "condition_level_eval_results.json"
    if cond_rows:
        with cond_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cond_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cond_rows)
        with cond_json_path.open("w", encoding="utf-8") as f:
            json.dump(cond_rows, f, indent=2)

    # 3. TD3 vs DDPG comparison
    comp_rows: list[dict[str, Any]] = []
    for report in reports:
        if report.td3 is None or report.ddpg is None:
            continue
        reward_level, noise_level = parse_experiment_components(report.experiment)

        td3_rew = report.td3.avg_reward_mean
        ddpg_rew = report.ddpg.avg_reward_mean
        rew_diff = (td3_rew - ddpg_rew) if (td3_rew is not None and ddpg_rew is not None) else None

        td3_crash = report.td3.crash_rate_mean
        ddpg_crash = report.ddpg.crash_rate_mean
        crash_diff = (td3_crash - ddpg_crash) if (td3_crash is not None and ddpg_crash is not None) else None

        td3_lap = report.td3.lap_completion_rate_mean
        ddpg_lap = report.ddpg.lap_completion_rate_mean
        lap_diff = (td3_lap - ddpg_lap) if (td3_lap is not None and ddpg_lap is not None) else None

        td3_dist = report.td3.distance_mean
        ddpg_dist = report.ddpg.distance_mean
        dist_diff = (td3_dist - ddpg_dist) if (td3_dist is not None and ddpg_dist is not None) else None

        td3_conv = report.td3.convergence_episode_mean
        ddpg_conv = report.ddpg.convergence_episode_mean
        conv_diff = (td3_conv - ddpg_conv) if (td3_conv is not None and ddpg_conv is not None) else None

        comp_rows.append({
            "experiment": report.experiment,
            "condition": report.experiment,
            "reward_level": reward_level or "",
            "noise_level": noise_level or "",
            "td3_reward_mean": td3_rew,
            "td3_reward_std": report.td3.avg_reward_std,
            "ddpg_reward_mean": ddpg_rew,
            "ddpg_reward_std": report.ddpg.avg_reward_std,
            "reward_diff_td3_minus_ddpg": rew_diff,
            "td3_crash_rate_mean": td3_crash,
            "td3_crash_rate_std": report.td3.crash_rate_std,
            "ddpg_crash_rate_mean": ddpg_crash,
            "ddpg_crash_rate_std": report.ddpg.crash_rate_std,
            "crash_diff_td3_minus_ddpg": crash_diff,
            "td3_lap_rate_mean": td3_lap,
            "td3_lap_rate_std": report.td3.lap_completion_rate_std,
            "ddpg_lap_rate_mean": ddpg_lap,
            "ddpg_lap_rate_std": report.ddpg.lap_completion_rate_std,
            "lap_rate_diff_td3_minus_ddpg": lap_diff,
            "td3_distance_mean": td3_dist,
            "td3_distance_std": report.td3.distance_std,
            "ddpg_distance_mean": ddpg_dist,
            "ddpg_distance_std": report.ddpg.distance_std,
            "distance_diff_td3_minus_ddpg": dist_diff,
            "td3_convergence_mean": td3_conv,
            "ddpg_convergence_mean": ddpg_conv,
            "convergence_diff_td3_minus_ddpg": conv_diff,
        })

    comp_csv_path = results_dir / "td3_vs_ddpg_comparison_results.csv"
    comp_json_path = results_dir / "td3_vs_ddpg_comparison_results.json"
    if comp_rows:
        with comp_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comp_rows)
        with comp_json_path.open("w", encoding="utf-8") as f:
            json.dump(comp_rows, f, indent=2)

    return {
        "run_csv": str(run_csv_path),
        "run_json": str(run_json_path),
        "cond_csv": str(cond_csv_path),
        "cond_json": str(cond_json_path),
        "comp_csv": str(comp_csv_path),
        "comp_json": str(comp_json_path),
    }


def validate_strict_evaluation_data(logs_dir: Path) -> None:
    """Validate that all 72 camera-ready deterministic evaluation runs exist and are valid."""
    missing_or_invalid: list[str] = []
    for algo in ("TD3", "DDPG"):
        for r in REWARD_LEVELS:
            for n in NOISE_LEVELS:
                cond = f"{r}_{n}"
                for s in (0, 42, 123):
                    run_id = f"{algo}/{cond}/seed_{s}"
                    # Check uppercase and lowercase paths
                    eval_path = None
                    for a in (algo, algo.lower()):
                        candidate = logs_dir / a / cond / f"seed_{s}" / "evaluation_summary.json"
                        if candidate.exists():
                            eval_path = candidate
                            break
                    if not eval_path:
                        missing_or_invalid.append(f"{run_id}: evaluation_summary.json missing")
                        continue

                    try:
                        with eval_path.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                        num_ep = data.get("num_episodes")
                        if num_ep != 20:
                            missing_or_invalid.append(f"{run_id}: num_episodes is {num_ep}, expected 20")
                        meta = data.get("metadata", {})
                        max_steps = meta.get("max_steps_per_episode")
                        if max_steps != 600:
                            missing_or_invalid.append(f"{run_id}: max_steps_per_episode is {max_steps}, expected 600")
                    except Exception as ex:
                        missing_or_invalid.append(f"{run_id}: corrupted JSON ({ex})")

    if missing_or_invalid:
        err_report = "\n".join(f"  - {item}" for item in missing_or_invalid)
        raise ValueError(f"Strict validation failed for {len(missing_or_invalid)}/72 runs:\n{err_report}")


def main() -> None:
    """CLI entry point."""
    script_dir = Path(__file__).resolve().parent
    load_env_file(script_dir / ".env")
    load_env_file(Path.cwd() / ".env")

    parser = argparse.ArgumentParser(description="Generate a TD3 vs DDPG PDF report")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="Base logs directory (default: logs_v2)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Base results directory (default: results_v2)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Output PDF file path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict validation across all 72 evaluation runs (fail on any missing run)",
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
        help="Number of trailing episodes used for average reward (legacy)",
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

    if args.strict:
        print("[report][strict] Validating camera-ready evaluation dataset across all 72 expected runs...")
        validate_strict_evaluation_data(args.logs_dir)
        print("[report][strict] All 72 evaluation summaries verified!")

    output_file = args.output
    if output_file.suffix.lower() != ".pdf":
        output_file = output_file.with_suffix(".pdf")

    experiment_ids = [str(item).strip() for item in args.experiments] if args.experiments else None
    reports = collect_experiment_reports(
        args.logs_dir,
        experiment_ids=experiment_ids,
        rolling_window=max(1, int(args.rolling_window)),
        stability_window=max(1, int(args.stability_window)),
        strict=args.strict,
        last_n=max(1, int(args.last_n)),
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    exported_files = export_results_tables(reports, args.results_dir)
    print(f"[report] Exported machine-readable results:")
    for k, v in exported_files.items():
        print(f"  - {k}: {v}")

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