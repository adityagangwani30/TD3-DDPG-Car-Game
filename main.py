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
import pygame
import torch

from config import DEFAULT_SEED
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train, evaluate
from utils import set_global_seed


def main():
    """Initialize Pygame, create the environment, and run training or evaluation."""
    parser = argparse.ArgumentParser(
        description="TD3 Self-Driving Car - Training and Evaluation"
    )
    parser.add_argument(
        "--mode", 
        choices=["train", "eval"], 
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
    args = parser.parse_args()

    set_global_seed(args.seed)
    pygame.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    env = CarRacingEnv(experiment_name="default", seed=args.seed)
    agent = TD3Agent(device=device)

    if args.checkpoint:
        agent.load(args.checkpoint)
        print(f"[main] Loaded checkpoint: {args.checkpoint}")

    try:
        if args.mode == "train":
            print("[main] Starting training...")
            train(env, agent, experiment_name="default", seed=args.seed)
        else:  # eval mode
            print(f"[main] Starting evaluation ({args.eval_episodes} episodes)...")
            evaluate(env, agent, num_episodes=args.eval_episodes, render=args.render)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down gracefully.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
