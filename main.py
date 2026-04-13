"""
main.py - Entry point for the TD3 self-driving car simulation.

Usage:
    python main.py [--mode train|eval|demo] [--checkpoint path/to/model.pth]

Examples:
    python main.py                                    # Train the agent
    python main.py --mode eval                        # Evaluate using latest checkpoint
    python main.py --mode demo                        # Run a quick demo
    python main.py --mode eval --checkpoint models/td3_ep500.pth  # Evaluate specific model
"""

import argparse
import os
from pathlib import Path

if os.environ.get("COLAB_RELEASE_TAG") or not os.environ.get("DISPLAY"):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch

from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train, evaluate


def _find_default_checkpoint() -> str | None:
    """Return the best available checkpoint, if one exists."""
    candidates = [
        Path("models/td3_best.pth"),
        Path("models/td3_best_avg100.pth"),
    ]
    candidates.extend(sorted(Path("models").glob("td3_ep*.pth"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def main():
    """Initialize Pygame, create the environment, and run training or evaluation."""
    parser = argparse.ArgumentParser(
        description="TD3 Self-Driving Car - Training and Evaluation"
    )
    parser.add_argument(
        "--mode", 
        choices=["train", "eval", "demo"], 
        default="train",
        help="Run training or evaluation (default: train)"
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
        "--demo-episodes",
        type=int,
        default=2,
        help="Number of episodes to run in demo mode (default: 2)",
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="Render evaluation episodes"
    )
    args = parser.parse_args()

    pygame.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    env = CarRacingEnv()
    agent = TD3Agent(device=device)

    checkpoint_path = args.checkpoint or _find_default_checkpoint()
    if checkpoint_path:
        print(f"[main] Using checkpoint: {checkpoint_path}")
        agent.load(checkpoint_path)
    elif args.mode in {"eval", "demo"}:
        print("[main] No checkpoint found. Running with the current policy weights.")

    try:
        if args.mode == "train":
            print("[main] Starting training...")
            train(env, agent)
        else:  # eval mode
            if args.mode == "demo":
                print(f"[main] Starting quick demo ({args.demo_episodes} episodes)...")
                evaluate(env, agent, num_episodes=args.demo_episodes, render=False)
            else:
                print(f"[main] Starting evaluation ({args.eval_episodes} episodes)...")
                evaluate(env, agent, num_episodes=args.eval_episodes, render=args.render)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down gracefully.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
