"""
run_experiments.py - Sequential experiment runner for TD3 car training.

Runs all experiment combinations from config.EXPERIMENTS and keeps outputs isolated:
  - Logs:   logs/R{reward_idx}_N{noise_idx}/training_log.jsonl
  - Models: models/R{reward_idx}_N{noise_idx}/
"""

import argparse
import os

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
from td3_agent import TD3Agent
from train import train_with_config
from utils import init_pygame, set_global_seed


def _experiment_tag(reward_mode: str, sensor_noise_std: float) -> str:
    """Build a compact deterministic experiment tag like R1_N2."""
    reward_idx = EXPERIMENT_REWARD_MODES.index(reward_mode) + 1
    noise_idx = EXPERIMENT_SENSOR_NOISE_LEVELS.index(sensor_noise_std) + 1
    return f"R{reward_idx}_N{noise_idx}"


def run_all_experiments(
    max_experiments: int | None = None,
    headless: bool = False,
    max_episodes: int | None = None,
    max_steps: int | None = None,
    start_index: int = 0,
):
    """Run all configured experiments sequentially with isolated outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = list(EXPERIMENTS.items())
    experiments = experiments[start_index:]
    if max_experiments is not None:
        experiments = experiments[: max(0, max_experiments)]

    if not experiments:
        print("[experiments] No experiments selected. Nothing to run.")
        return

    print("=" * 72)
    print(f"Total experiments available: {len(EXPERIMENTS)}")
    print(f"Running {len(experiments)} experiments sequentially on device: {device}")
    print("=" * 72)

    for index, (experiment_name, cfg) in enumerate(experiments, start=1):
        reward_mode = cfg["reward_mode"]
        sensor_noise_std = cfg["sensor_noise_std"]
        seed = EXPERIMENT_BASE_SEED + index - 1
        tag = _experiment_tag(reward_mode, sensor_noise_std)

        exp_log_dir = os.path.join(LOGS_DIR, tag)
        exp_model_dir = os.path.join(MODEL_DIR, tag)
        os.makedirs(exp_log_dir, exist_ok=True)
        os.makedirs(exp_model_dir, exist_ok=True)

        print("\n" + "-" * 72)
        print(f"Experiment {index}/{len(experiments)}: {experiment_name} ({tag})")
        print(f"  reward_mode      : {reward_mode}")
        print(f"  sensor_noise_std : {sensor_noise_std}")
        print(f"  seed             : {seed}")
        print(f"  logs             : {exp_log_dir}")
        print(f"  models           : {exp_model_dir}")
        print("-" * 72)

        set_global_seed(seed)
        init_pygame(headless=headless)
        env = CarRacingEnv(
            enable_metrics=True,
            reward_mode=reward_mode,
            sensor_noise_std=sensor_noise_std,
            metrics_log_dir=exp_log_dir,
            experiment_name=experiment_name,
            seed=seed,
            headless=headless,
        )

        agent = TD3Agent(device=device)

        try:
            train_with_config(
                env,
                agent,
                model_dir=exp_model_dir,
                run_label=f"{tag} {index}/{len(experiments)}",
                experiment_name=experiment_name,
                seed=seed,
                max_episodes=max_episodes,
                max_steps_per_episode=max_steps,
            )
        except (KeyboardInterrupt, SystemExit):
            print("\n[experiments] Interrupted by user. Stopping remaining runs.")
            env.close()
            pygame.quit()
            break
        finally:
            pygame.quit()

    print("\n[experiments] All scheduled experiment runs finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TD3 experiments sequentially")
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
    cli_args = parser.parse_args()
    run_all_experiments(
        max_experiments=cli_args.max_experiments,
        headless=cli_args.headless,
        max_episodes=cli_args.max_episodes,
        max_steps=cli_args.max_steps,
        start_index=cli_args.start_index,
    )
