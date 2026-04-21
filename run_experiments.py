"""
run_experiments.py - Sequential experiment runner for TD3 or DDPG training.

Runs all experiment combinations from config.EXPERIMENTS for one algorithm:
    - Logs:   logs/{algo}/R{reward_idx}_N{noise_idx}/training_log.jsonl
    - Models: models/{algo}/R{reward_idx}_N{noise_idx}/
"""

import argparse
import os
import json
from pathlib import Path

import pygame
import torch

from config import (
    EXPERIMENT_BASE_SEED,
    EXPERIMENTS,
    EXPERIMENT_REWARD_MODES,
    EXPERIMENT_SENSOR_NOISE_LEVELS,
    LOGS_DIR,
    MODEL_DIR,
)
from environment import CarRacingEnv
from train import train_with_config
from utils import init_pygame, set_global_seed


ALGORITHMS = ["td3", "ddpg"]
assert len(EXPERIMENTS) == 12, "Expected 12 base experiments"


def _experiment_tag(reward_mode: str, sensor_noise_std: float) -> str:
    """Build a compact deterministic experiment tag like R1_N2."""
    reward_idx = EXPERIMENT_REWARD_MODES.index(reward_mode) + 1
    noise_idx = EXPERIMENT_SENSOR_NOISE_LEVELS.index(sensor_noise_std) + 1
    return f"R{reward_idx}_N{noise_idx}"


def _experiment_ids_for_algo(algo: str) -> list[tuple[str, dict]]:
    """Return the ordered experiment grid for a given algorithm."""
    if algo not in ALGORITHMS:
        raise ValueError("Unsupported algorithm")
    return list(EXPERIMENTS.items())


def _latest_checkpoint(exp_model_dir: str, algo: str) -> str | None:
    """Return the latest checkpoint in an experiment directory if one exists."""
    model_dir = Path(exp_model_dir)
    if not model_dir.exists():
        return None

    candidates = []
    patterns = [
        f"{algo}_ep*.pth",
        f"*_{algo}_ep*.pth",
        f"{algo}_best.pth",
        f"{algo}_best_avg100.pth",
        f"*_{algo}_best.pth",
        f"*_{algo}_best_avg100.pth",
    ]
    for pattern in patterns:
        for path in model_dir.glob(pattern):
            stem = path.stem
            if "_ep" in stem:
                try:
                    episode = int(stem.split("_ep")[-1])
                except ValueError:
                    episode = -1
                candidates.append((episode, path))
            else:
                candidates.append((-1, path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return str(candidates[0][1])


def _read_last_logged_episode(exp_log_dir: str) -> int:
    """Return the highest valid episode index in the training log for an experiment."""
    log_file = Path(exp_log_dir) / "training_log.jsonl"
    if not log_file.exists():
        return 0

    last_episode = 0
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode = int(payload.get("episode", 0) or 0)
                if episode > last_episode:
                    last_episode = episode
    except OSError:
        return 0

    return last_episode


def _is_experiment_complete(
    exp_log_dir: str,
    exp_model_dir: str,
    algo: str,
    max_episodes: int | None,
) -> bool:
    """Return True when completion is confirmed by logs or best-model existence."""
    if max_episodes is not None:
        last_ep = _read_last_logged_episode(exp_log_dir)
        if last_ep >= int(max_episodes):
            return True

    best_model = Path(exp_model_dir) / f"{algo}_best.pth"
    if best_model.exists():
        return True

    wildcard_best = list(Path(exp_model_dir).glob(f"*_{algo}_best.pth"))
    if wildcard_best:
        return True

    return False


def run_all_experiments(
    algo: str,
    max_experiments: int | None = None,
    headless: bool = False,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    start_index: int = 0,
    resume: bool = False,
):
    """Run all configured experiments sequentially with isolated outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_grid = _experiment_ids_for_algo(algo)

    scheduled_runs = []
    for experiment_name, cfg in experiment_grid:
        reward_mode = cfg["reward_mode"]
        sensor_noise_std = cfg["sensor_noise_std"]
        tag = _experiment_tag(reward_mode, sensor_noise_std)
        experiment_id = f"{algo}_{tag}"
        scheduled_runs.append((algo, experiment_id, experiment_name, cfg))

    scheduled_runs = scheduled_runs[start_index:]
    if max_experiments is not None:
        scheduled_runs = scheduled_runs[: max(0, max_experiments)]

    if not scheduled_runs:
        print("[experiments] No experiments selected. Nothing to run.")
        return

    print("=" * 72)
    print(f"Running {len(scheduled_runs)} {algo.upper()} experiments sequentially on device: {device}")
    print("=" * 72)

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    batch_total = 4
    batch_number = 1
    if max_experiments:
        if algo == "td3":
            batch_number = 1 if start_index < max_experiments else 2
        else:
            batch_number = 3 if start_index < max_experiments else 4

    for index, (algo, experiment_id, experiment_name, cfg) in enumerate(scheduled_runs, start=1):
        reward_mode = cfg["reward_mode"]
        sensor_noise_std = cfg["sensor_noise_std"]
        seed = EXPERIMENT_BASE_SEED + start_index + index - 1
        tag = _experiment_tag(reward_mode, sensor_noise_std)

        exp_log_dir = os.path.join(LOGS_DIR, algo, tag)
        exp_model_dir = os.path.join(MODEL_DIR, algo, tag)
        os.makedirs(exp_log_dir, exist_ok=True)
        os.makedirs(exp_model_dir, exist_ok=True)

        last_logged_episode = _read_last_logged_episode(exp_log_dir)

        if resume and _is_experiment_complete(exp_log_dir, exp_model_dir, algo, max_episodes):
            print(
                f"[SKIP] [{algo.upper()}][Batch {batch_number}/{batch_total}] -> {tag} "
                f"(Exp {index}/{len(scheduled_runs)}) | logged episodes: {last_logged_episode}"
            )
            skipped_count += 1
            continue

        checkpoint_path = _latest_checkpoint(exp_model_dir, algo) if resume else None

        print("\n" + "-" * 72)
        print(
            f"[{algo.upper()}][Batch {batch_number}/{batch_total}] -> {tag} "
            f"(Exp {index}/{len(scheduled_runs)})"
        )
        print(f"  reward_mode      : {reward_mode}")
        print(f"  sensor_noise_std : {sensor_noise_std}")
        print(f"  seed             : {seed}")
        print(f"  logs             : {exp_log_dir}")
        print(f"  models           : {exp_model_dir}")
        if resume:
            print(f"  last_logged_ep   : {last_logged_episode}")
        if checkpoint_path:
            print(f"  resume_checkpoint: {checkpoint_path}")
        elif resume and last_logged_episode > 0:
            print("  [warn] Resume requested but no checkpoint found; training may restart from fresh weights.")
        print("-" * 72)

        set_global_seed(seed)
        init_pygame(headless=headless)
        env = CarRacingEnv(
            enable_metrics=True,
            reward_mode=reward_mode,
            sensor_noise_std=sensor_noise_std,
            metrics_log_dir=exp_log_dir,
            experiment_name=experiment_id,
            seed=seed,
            headless=headless,
        )

        try:
            train_with_config(
                env,
                algo=algo,
                device=device,
                model_dir=exp_model_dir,
                run_label=f"{algo.upper()} Batch {index}/{len(scheduled_runs)}",
                checkpoint_path=checkpoint_path,
                require_checkpoint=False,
                experiment_name=experiment_id,
                seed=seed,
                max_episodes=max_episodes,
                max_steps_per_episode=max_steps,
            )
            completed_count += 1
        except (KeyboardInterrupt, SystemExit):
            print("\n[experiments] Interrupted by user. Stopping remaining runs.")
            env.close()
            pygame.quit()
            break
        except Exception as exc:
            failed_count += 1
            print(
                f"[ERROR] [{algo.upper()}][Batch {batch_number}/{batch_total}] {tag}: {exc}"
            )
            print("[experiments][warn] Continuing to next experiment.")
            env.close()
        finally:
            pygame.quit()

    print("\n[experiments] All scheduled experiment runs finished.")
    print(
        f"[experiments] Summary | completed: {completed_count} | skipped: {skipped_count} | failed: {failed_count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TD3 or DDPG experiments sequentially")
    parser.add_argument(
        "--algo",
        choices=["td3", "ddpg"],
        default=None,
        help="Algorithm to run",
    )
    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Run only the first N experiments (for validation/debug)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this experiment index (0-based) for batched runs",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless pygame mode",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional override for training episodes per experiment",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for max steps per episode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint and skip completed experiments",
    )
    cli_args = parser.parse_args()
    if cli_args.algo is None:
        raise ValueError("You must specify --algo {td3, ddpg}")
    run_all_experiments(
        algo=cli_args.algo,
        max_experiments=cli_args.max_experiments,
        headless=cli_args.headless,
        max_episodes=cli_args.max_episodes,
        max_steps=cli_args.max_steps,
        start_index=cli_args.start_index,
        resume=cli_args.resume,
    )
