"""
run_experiments.py - Sequential experiment runner for TD3 car training.

Runs all experiment combinations from config.EXPERIMENTS and keeps outputs isolated:
  - Logs:   logs/R{reward_idx}_N{noise_idx}/training_log.jsonl
  - Models: models/R{reward_idx}_N{noise_idx}/
"""

import os

import pygame
import torch

from config import (
    EXPERIMENTS,
    EXPERIMENT_REWARD_MODES,
    EXPERIMENT_SENSOR_NOISE_LEVELS,
    LOGS_DIR,
    MODEL_DIR,
)
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train_with_config


def _experiment_tag(reward_mode: str, sensor_noise_std: float) -> str:
    """Build a compact deterministic experiment tag like R1_N2."""
    reward_idx = EXPERIMENT_REWARD_MODES.index(reward_mode) + 1
    noise_idx = EXPERIMENT_SENSOR_NOISE_LEVELS.index(sensor_noise_std) + 1
    return f"R{reward_idx}_N{noise_idx}"


def run_all_experiments():
    """Run all configured experiments sequentially with isolated outputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = list(EXPERIMENTS.items())

    print("=" * 72)
    print(f"Running {len(experiments)} experiments sequentially on device: {device}")
    print("=" * 72)

    for index, (experiment_name, cfg) in enumerate(experiments, start=1):
        reward_mode = cfg["reward_mode"]
        sensor_noise_std = cfg["sensor_noise_std"]
        tag = _experiment_tag(reward_mode, sensor_noise_std)

        exp_log_dir = os.path.join(LOGS_DIR, tag)
        exp_model_dir = os.path.join(MODEL_DIR, tag)
        os.makedirs(exp_log_dir, exist_ok=True)
        os.makedirs(exp_model_dir, exist_ok=True)

        print("\n" + "-" * 72)
        print(f"Experiment {index}/{len(experiments)}: {experiment_name} ({tag})")
        print(f"  reward_mode      : {reward_mode}")
        print(f"  sensor_noise_std : {sensor_noise_std}")
        print(f"  logs             : {exp_log_dir}")
        print(f"  models           : {exp_model_dir}")
        print("-" * 72)

        pygame.init()
        env = CarRacingEnv(
            enable_metrics=True,
            reward_mode=reward_mode,
            sensor_noise_std=sensor_noise_std,
            metrics_log_dir=exp_log_dir,
        )
        # Explicit runtime updates keep the runner robust if environment defaults change.
        env.reward_mode = reward_mode
        env.set_sensor_noise(sensor_noise_std)

        agent = TD3Agent(device=device)

        try:
            train_with_config(
                env,
                agent,
                model_dir=exp_model_dir,
                run_label=f"{tag} {index}/{len(experiments)}",
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
    run_all_experiments()
