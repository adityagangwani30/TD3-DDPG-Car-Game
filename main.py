"""
main.py - Entry point for the TD3 self-driving car simulation.

Usage:
    python main.py
"""

import pygame

from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train


def main():
    """Initialise Pygame, create the environment, and start training."""
    pygame.init()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    env = CarRacingEnv()
    agent = TD3Agent(device=device)

    print("[main] Starting training...")
    try:
        train(env, agent)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
