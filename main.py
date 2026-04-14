"""
main.py - Entry point for the TD3 self-driving car simulation.

Usage:
    python main.py [--mode train|eval] [--checkpoint path/to/model.pth]
    
Examples:
    python main.py                                    # Train the agent
    python main.py --mode eval                        # Evaluate using latest checkpoint
    python main.py --mode eval --checkpoint models/td3_ep500.pth  # Evaluate specific model
"""

import argparse
import os
import pygame
import torch

from config import DEFAULT_SEED
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train, evaluate
from utils import init_pygame, set_global_seed


def _resolve_checkpoint(requested: str | None) -> str | None:
    """Resolve checkpoint path with sensible defaults for eval/demo runs."""
    if requested:
        if os.path.exists(requested):
            return requested
        print(f"[main] Checkpoint not found: {requested}")
        return None

    candidates = [
        os.path.join("models", "td3_best_avg100.pth"),
        os.path.join("models", "td3_best.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    """Initialize Pygame, create the environment, and run training or evaluation."""
    parser = argparse.ArgumentParser(description="TD3 Self-Driving Car - Training/Eval/Demo")
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "demo"], 
        default="train",
        help="Run training, evaluation, or demo playback (default: train)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint file for evaluation or continued training"
    )
    parser.add_argument(
        "--eval-episodes", 
        type=int, 
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        default=True,
        help="Render evaluation episodes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless pygame mode (useful for servers/Colab)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional override for training episodes (validation/debug)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for max steps per training episode (validation/debug)",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    init_pygame(headless=args.headless)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    env = CarRacingEnv(experiment_name="default", seed=args.seed)
    agent = TD3Agent(device=device)

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    if checkpoint_path and args.mode in {"eval", "demo"}:
        agent.load(checkpoint_path)
        print(f"[main] Loaded checkpoint: {checkpoint_path}")
    elif args.mode in {"eval", "demo"}:
        print("[main] No checkpoint found. Running evaluation/demo with current agent weights.")

    try:
        if args.mode == "train":
            print("[main] Starting training...")
            train(
                env,
                agent,
                experiment_name="default",
                seed=args.seed,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps,
            )
        elif args.mode == "eval":
            print(f"[main] Starting evaluation ({args.eval_episodes} episodes)...")
            evaluate(env, agent, num_episodes=args.eval_episodes, render=args.render)
        else:  # demo mode
            demo_episodes = max(1, min(args.eval_episodes, 5))
            print(f"[main] Starting demo ({demo_episodes} episodes, render enabled)...")
            evaluate(env, agent, num_episodes=demo_episodes, render=True)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down gracefully.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
