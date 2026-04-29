"""generate_report.py - Multi-algorithm comparison report generator.

Reads per-episode JSONL training logs from results/logs/{algo}/{experiment}/seed_*/
and produces a single multi-page PDF comparing all algorithms found.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "reward_total",
    "collisions",
    "laps_completed",
    "length",
    "speed_mean",
    "speed_max",
    "steering_smooth",
]

HIGHER_IS_BETTER = {
    "reward_total": True,
    "laps_completed": True,
    "length": True,
    "speed_mean": True,
    "speed_max": True,
    "collisions": False,
    "steering_smooth": False,
}


def discover_algorithms(results_dir: Path) -> list[str]:
    """Return algorithm directory names under results_dir/logs/ that contain seed logs."""
    logs_root = results_dir / "logs"
    if not logs_root.is_dir():
        return []

    algos = []
    for algo_dir in sorted(logs_root.iterdir()):
        if not algo_dir.is_dir():
            continue
        has_data = any(algo_dir.glob("*/seed_*/training_log.jsonl"))
        if has_data:
            algos.append(algo_dir.name)
    return algos


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _extract_series(rows: list[dict]) -> dict[str, np.ndarray]:
    """Extract per-episode metric arrays from a single seed's log rows."""
    if not rows:
        return {}
    series = {m: np.array([float(r.get(m, np.nan)) for r in rows], dtype=float) for m in METRICS}
    series["episode"] = np.array([int(r.get("episode", i + 1)) for i, r in enumerate(rows)], dtype=int)
    return series


def _aggregate_seeds(seed_series: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Mean / std across variable-length seed series, NaN-padded to longest."""
    if not seed_series:
        return {}

    out: dict[str, np.ndarray] = {}
    for metric in METRICS:
        arrays = [s[metric] for s in seed_series if metric in s and len(s[metric]) > 0]
        if not arrays:
            continue
        max_len = max(len(a) for a in arrays)
        stacked = np.full((len(arrays), max_len), np.nan, dtype=float)
        for i, a in enumerate(arrays):
            stacked[i, : len(a)] = a
        out[metric] = np.nanmean(stacked, axis=0)
        out[f"{metric}_std"] = np.nanstd(stacked, axis=0)
    return out


def load_algorithm_results(results_dir: Path, algo_name: str) -> dict:
    """Load and aggregate all seed logs for a single algorithm.

    Aggregation: pool all seeds across all experiments (R*_N*), then take the
    NaN-aware mean/std at each episode index. Legacy non-seeded logs are skipped.

    Returns a dict with one numpy array per metric (mean curve) plus
    `<metric>_std`, `episode_count`, `n_seeds`, `n_experiments`, `reward_modes`,
    `sensor_noise_levels`, and `training_time_s` (always None — not in source).
    """
    algo_dir = results_dir / "logs" / algo_name
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"No log directory for {algo_name!r}: {algo_dir}")

    seed_logs = sorted(algo_dir.glob("*/seed_*/training_log.jsonl"))
    if not seed_logs:
        raise FileNotFoundError(f"No seed_*/training_log.jsonl files under {algo_dir}")

    seed_series: list[dict[str, np.ndarray]] = []
    experiments = set()
    reward_modes = set()
    noise_levels = set()
    total_rows = 0

    for log_file in seed_logs:
        rows = _load_jsonl(log_file)
        if not rows:
            continue
        total_rows += len(rows)
        experiments.add(log_file.parent.parent.name)
        first = rows[0]
        if first.get("reward_mode") is not None:
            reward_modes.add(str(first["reward_mode"]))
        if first.get("sensor_noise_std") is not None:
            noise_levels.add(float(first["sensor_noise_std"]))
        series = _extract_series(rows)
        if series:
            seed_series.append(series)

    if not seed_series:
        raise ValueError(f"All seed logs empty for {algo_name!r}")

    aggregated = _aggregate_seeds(seed_series)
    primary_len = len(aggregated.get("reward_total", []))

    return {
        **aggregated,
        "training_time_s": None,
        "n_seeds": len(seed_series),
        "n_experiments": len(experiments),
        "experiments": sorted(experiments),
        "reward_modes": sorted(reward_modes),
        "sensor_noise_levels": sorted(noise_levels),
        "episode_count": primary_len,
        "total_rows": total_rows,
    }


def _smoke_print(results: dict[str, dict]) -> None:
    print("=" * 70)
    print("Loaded algorithm results")
    print("=" * 70)
    for algo, data in results.items():
        if "error" in data:
            print(f"\n[{algo}] DATA UNAVAILABLE: {data['error']}")
            continue
        print(f"\n[{algo}]")
        print(f"  experiments         : {data['n_experiments']} -> {data['experiments']}")
        print(f"  reward_modes        : {data['reward_modes']}")
        print(f"  sensor_noise_levels : {data['sensor_noise_levels']}")
        print(f"  seed runs aggregated: {data['n_seeds']}")
        print(f"  total rows          : {data['total_rows']}")
        print(f"  aggregated episodes : {data['episode_count']}")
        print(f"  training_time_s     : {data['training_time_s']}")
        print(f"  metric curves:")
        for m in METRICS:
            arr = data.get(m)
            if arr is None or len(arr) == 0:
                print(f"    {m:18s} <missing>")
                continue
            print(f"    {m:18s} shape={arr.shape}  mean={np.nanmean(arr):.4f}  last={arr[-1]:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-algorithm comparison report generator.")
    p.add_argument("--results-dir", type=Path, default=Path("results"),
                   help="Folder containing logs/{algo}/...  (default: results)")
    p.add_argument("--agents", type=str, default=None,
                   help="Comma-separated algorithm names. Omit to auto-detect.")
    p.add_argument("--output", type=Path, default=Path("results/report.pdf"),
                   help="Output PDF path (default: results/report.pdf)")
    p.add_argument("--smooth-window", type=int, default=50,
                   help="Rolling mean window for line plots (default: 50)")
    p.add_argument("--last-n-eps", type=int, default=100,
                   help="Averaging window for the summary table (default: 100)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.agents:
        agents = [a.strip() for a in args.agents.split(",") if a.strip()]
        available = set(discover_algorithms(args.results_dir))
        missing = [a for a in agents if a not in available]
        for a in missing:
            print(f"WARNING: requested algorithm {a!r} has no logs under {args.results_dir}/logs/", file=sys.stderr)
    else:
        agents = discover_algorithms(args.results_dir)
        if not agents:
            print(f"ERROR: no algorithms with seed logs found under {args.results_dir}/logs/", file=sys.stderr)
            return 1
        print(f"Auto-detected algorithms: {agents}")

    results: dict[str, dict] = {}
    for algo in agents:
        try:
            results[algo] = load_algorithm_results(args.results_dir, algo)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"WARNING: failed to load {algo!r}: {e}", file=sys.stderr)
            results[algo] = {"error": str(e)}

    _smoke_print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
