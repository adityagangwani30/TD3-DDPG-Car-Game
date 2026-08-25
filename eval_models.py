import argparse
import os
import re
from pathlib import Path

import pygame
import torch

from config import DEFAULT_SEED, EVAL_EPISODES, MODEL_DIR
from ddpg_agent import DDPGAgent
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import evaluate
from utils import init_pygame, set_global_seed


def _infer_algo_from_path(path: Path) -> str:
    """Infer whether a checkpoint belongs to TD3 or DDPG."""
    path_str = str(path).lower()
    if "ddpg" in path_str and "td3" not in path.name.lower():
        return "ddpg"
    return "td3"


def _infer_condition_from_path(path: Path) -> tuple[str, float]:
    """Infer (reward_mode, sensor_noise_std) from path like R2_N3."""
    path_str = str(path)
    match = re.search(r"R([1-4])_N([1-3])", path_str)
    if match:
        r_idx = int(match.group(1))
        n_idx = int(match.group(2))
        reward_modes = ["basic", "shaped", "modified", "tuned"]
        noise_levels = [0.0, 0.02, 0.05]
        return reward_modes[r_idx - 1], noise_levels[n_idx - 1]
    return "shaped", 0.02


def main():
    """Compare multiple trained models with deterministic evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate and compare TD3 / DDPG models")
    parser.add_argument(
        "--model",
        type=str,
        default="*.pth",
        help="Model file, directory, or glob pattern to evaluate (default: *.pth in models/)",
    )
    parser.add_argument(
        "--algo",
        choices=["auto", "td3", "ddpg"],
        default="auto",
        help="Algorithm architecture to instantiate (default: auto-detect)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help=f"Number of deterministic evaluation episodes per model (default: {EVAL_EPISODES})",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render evaluation visually",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless pygame mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=MODEL_DIR,
        help=f"Base model directory to search (default: {MODEL_DIR})",
    )
    args = parser.parse_args()

    mode_str = "HEADLESS (off-screen)" if args.headless else "GUI (interactive)"
    set_global_seed(args.seed)
    init_pygame(headless=args.headless)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[eval] Visualization : {mode_str}")
    print(f"[eval] Using device  : {device}")
    print(f"[eval] Seed          : {args.seed}")
    print(f"[eval] Episodes/model: {args.episodes}\n")

    base_dir = Path(args.models_dir)
    model_files: list[Path] = []
    if os.path.isfile(args.model):
        model_files = [Path(args.model)]
    elif (base_dir / args.model).is_file():
        model_files = [base_dir / args.model]
    else:
        # Search recursively for matching patterns
        pattern = args.model
        model_files = sorted(base_dir.rglob(pattern))

    model_files = [p for p in model_files if p.suffix == ".pth"]

    if not model_files:
        print(f"No models found matching: '{args.model}' in {base_dir}")
        return

    print(f"Found {len(model_files)} model(s) to evaluate:")
    for mf in model_files:
        print(f"  - {mf}")

    results = {}

    for model_file in model_files:
        algo = args.algo if args.algo != "auto" else _infer_algo_from_path(model_file)
        reward_mode, sensor_noise_std = _infer_condition_from_path(model_file)

        print(f"\n{'='*70}")
        print(f"Evaluating: {model_file.name} [{algo.upper()}] (Reward: {reward_mode}, Noise: {sensor_noise_std})")
        print(f"{'='*70}")

        env = CarRacingEnv(
            enable_metrics=False,
            reward_mode=reward_mode,
            sensor_noise_std=sensor_noise_std,
            seed=args.seed,
            headless=args.headless,
        )
        agent = TD3Agent(device=device) if algo == "td3" else DDPGAgent(device=device)

        try:
            result = evaluate(
                env,
                agent,
                num_episodes=args.episodes,
                render=args.render,
                checkpoint_path=str(model_file),
            )
            results[str(model_file)] = result
            env.close()
        except Exception as e:
            print(f"Error evaluating {model_file.name}: {e}")
            env.close()
            continue

    if results:
        print(f"\n{'='*95}")
        print("DETERMINISTIC EVALUATION SUMMARY COMPARISON")
        print(f"{'='*95}")
        print(
            f"{'Model Path':<45} {'Reward (Mean+/-Std)':<22} {'Crash %':<10} {'Laps %':<10} {'Distance':<10}"
        )
        print("-" * 95)

        for path_str, result in sorted(results.items()):
            rel_name = Path(path_str).name
            if len(rel_name) > 42:
                rel_name = rel_name[:39] + "..."
            reward_str = f"{result['avg_reward']:6.2f} +/- {result['reward_std']:5.2f}"
            print(
                f"{rel_name:<45} {reward_str:<22} "
                f"{result['crash_rate']:>7.1%}   {result['lap_completion_rate']:>7.1%}   "
                f"{result['distance_mean']:>8.1f} px"
            )
        print("=" * 95)

    pygame.quit()


if __name__ == "__main__":
    main()

