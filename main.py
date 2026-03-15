"""
main.py – Entry point for the TD3 self-driving car simulation.

Usage:
    python main.py

Initialises Pygame, generates any missing assets, creates the environment
and TD3 agent, then starts the training loop.
"""

import pygame

from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train


def main():
    # Initialise Pygame subsystems
    pygame.init()

    # Detect GPU availability
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] Using device: {device}")

    # Create environment and agent
    env = CarRacingEnv()
    agent = TD3Agent(device=device)

    print("[main] Starting training…")
    try:
        train(env, agent)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted – shutting down.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
