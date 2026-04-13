"""
main.py - Entry point for the TD3 self-driving car simulation.

Usage:
    python main.py [--mode train|eval|demo] [--checkpoint path/to/model.pth] [--resume]

Examples:
    python main.py                                    # Train the agent
    python main.py --mode eval                        # Evaluate using latest checkpoint
    python main.py --mode demo                        # Run a quick demo
    python main.py --mode eval --checkpoint models/td3_ep500.pth  # Evaluate specific model
"""

import argparse
import os
import sys
from pathlib import Path

HEADLESS = bool(os.environ.get("COLAB_RELEASE_TAG")) or (
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
)

if HEADLESS:
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


def _find_latest_training_checkpoint() -> str | None:
    """Return the most recent periodic training checkpoint, if one exists."""
    checkpoints = []
    for candidate in Path("models").glob("td3_ep*.pth"):
        stem = candidate.stem
        try:
            episode_number = int(stem.replace("td3_ep", ""))
        except ValueError:
            continue
        checkpoints.append((episode_number, candidate))

    if checkpoints:
        checkpoints.sort(key=lambda item: item[0], reverse=True)
        return str(checkpoints[0][1])

    best_candidates = [Path("models/td3_best_avg100.pth"), Path("models/td3_best.pth")]
    for candidate in best_candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _find_resume_candidates() -> list[str]:
    """Return resume candidates ordered from most to least preferred."""
    candidates: list[Path] = []

    periodic = []
    for candidate in Path("models").glob("td3_ep*.pth"):
        stem = candidate.stem
        try:
            episode_number = int(stem.replace("td3_ep", ""))
        except ValueError:
            continue
        periodic.append((episode_number, candidate))

    periodic.sort(key=lambda item: item[0], reverse=True)
    candidates.extend(path for _, path in periodic)

    for fallback in [Path("models/td3_best_avg100.pth"), Path("models/td3_best.pth")]:
        if fallback.exists() and fallback not in candidates:
            candidates.append(fallback)

    return [str(path) for path in candidates]


def _try_load_checkpoint(agent: TD3Agent, checkpoint_path: str, required: bool) -> bool:
    """Load a checkpoint and optionally fail if it is incompatible."""
    try:
        agent.load(checkpoint_path)
        return True
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        message = f"[main] Could not load checkpoint '{checkpoint_path}': {exc}"
        if required:
            raise RuntimeError(message) from exc
        print(message)
        print("[main] Continuing without checkpoint.")
        return False


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
        "--resume",
        action="store_true",
        help="Resume training from the latest saved checkpoint",
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

    checkpoint_path = None
    loaded_for_training = False
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.mode == "train" and args.resume:
        resume_candidates = _find_resume_candidates()
        for candidate in resume_candidates:
            print(f"[main] Trying resume checkpoint: {candidate}")
            if _try_load_checkpoint(agent, candidate, required=False):
                checkpoint_path = candidate
                loaded_for_training = True
                break
    elif args.mode in {"eval", "demo"}:
        checkpoint_path = _find_default_checkpoint()

    if checkpoint_path and not loaded_for_training:
        print(f"[main] Using checkpoint: {checkpoint_path}")
        _try_load_checkpoint(agent, checkpoint_path, required=bool(args.checkpoint or args.resume))
        loaded_for_training = True
    elif args.mode in {"eval", "demo"}:
        print("[main] No checkpoint found. Running with the current policy weights.")
    else:
        print("[main] Starting training from scratch with freshly initialized weights.")

    if args.mode == "train" and loaded_for_training:
        print("[main] Resuming training from the selected checkpoint.")
    if args.mode == "train" and args.resume and not loaded_for_training:
        print("[main] Resume requested, but no compatible checkpoint was found. Starting from scratch.")

    try:
        if args.mode == "train":
            print("[main] Starting training...")
            train(env, agent)
        else:  # eval mode
            if args.mode == "demo":
                print(f"[main] Starting quick demo ({args.demo_episodes} episodes)...")
                preview_path = str(Path("logs") / "demo_preview.png")
                evaluate(
                    env,
                    agent,
                    num_episodes=args.demo_episodes,
                    render=True,
                    preview_path=preview_path,
                )
                print(f"[main] Demo preview saved to: {preview_path}")
            else:
                print(f"[main] Starting evaluation ({args.eval_episodes} episodes)...")
                evaluate(env, agent, num_episodes=args.eval_episodes, render=args.render)
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down gracefully.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
