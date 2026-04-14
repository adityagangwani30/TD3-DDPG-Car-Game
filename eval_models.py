"""
eval_models.py - Compare and evaluate multiple trained models.

Allows easy testing of different checkpoints and model performance comparison.
"""

import argparse
import os
from pathlib import Path

import pygame
import torch

from config import MODEL_DIR
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import evaluate
from utils import init_pygame


def main():
    """Compare multiple trained models."""
    parser = argparse.ArgumentParser(description="Evaluate and compare TD3 models")
    parser.add_argument(
        "--model",
        type=str,
        default="td3_best.pth",
        help="Model file or pattern to evaluate (default: td3_best.pth)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes per model (default: 10)"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Render evaluation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless pygame mode",
    )
    args = parser.parse_args()

    init_pygame(headless=args.headless)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Using device: {device}\n")

    # Find model files
    if "*" in args.model or "?" in args.model:
        # Pattern matching
        pattern = args.model
        model_files = list(Path(MODEL_DIR).glob(pattern))
    else:
        # Single file
        model_path = os.path.join(MODEL_DIR, args.model)
        model_files = [Path(model_path)] if os.path.exists(model_path) else []

    if not model_files:
        print(f"No models found matching: {args.model}")
        print(f"Available models in {MODEL_DIR}:")
        if os.path.exists(MODEL_DIR):
            for f in os.listdir(MODEL_DIR):
                if f.endswith(".pth"):
                    print(f"  - {f}")
        return

    # Evaluate each model
    results = {}
    
    for model_file in sorted(model_files):
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_file.name}")
        print(f"{'='*60}")
        
        env = CarRacingEnv(enable_metrics=False)
        agent = TD3Agent(device=device)
        
        try:
            result = evaluate(
                env,
                agent,
                num_episodes=args.episodes,
                render=args.render,
                checkpoint_path=str(model_file)
            )
            results[model_file.name] = result
            env.close()
        except Exception as e:
            print(f"Error evaluating {model_file.name}: {e}")
            env.close()
            continue

    # Summary comparison
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY COMPARISON")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Avg Reward':>12} {'Crash Rate':>12} {'Avg Laps':>10}")
        print("-" * 60)
        
        for name, result in sorted(results.items()):
            print(
                f"{name:<30} {result['avg_reward']:>12.2f} "
                f"{result['crash_rate']:>11.1%} {result['avg_laps']:>10.1f}"
            )

    pygame.quit()


if __name__ == "__main__":
    main()
