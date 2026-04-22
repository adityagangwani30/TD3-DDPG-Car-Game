"""
main.py - Entry point for deterministic policy-gradient car training.

Usage:
    python main.py --algo {td3,ddpg} [--mode train|eval|demo] [--checkpoint path/to/model.pth] [--resume]

Examples:
    python main.py --algo td3                         # Train the TD3 agent
    python main.py --algo ddpg --mode eval            # Evaluate DDPG using latest checkpoint
    python main.py --algo td3 --mode demo             # Run a quick TD3 demo
    python main.py --algo ddpg --mode eval --checkpoint models/ddpg/ddpg_ep500.pth
"""

import argparse
import os
from pathlib import Path

from utils import detect_headless_environment

# Detect headless environment BEFORE pygame import
HEADLESS = detect_headless_environment()

if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame
import torch

from config import DEFAULT_SEED, MAX_EPISODES, MAX_STEPS_PER_EPISODE
from ddpg_agent import DDPGAgent
from environment import CarRacingEnv
from td3_agent import TD3Agent
from train import train, evaluate
from utils import init_pygame, set_global_seed

def _find_default_checkpoint(algo: str) -> str | None:
    """Return the best available checkpoint, if one exists."""
    model_dir = Path("models") / algo
    candidates = [
        model_dir / f"{algo}_best.pth",
        model_dir / f"{algo}_best_avg100.pth",
    ]
    candidates.extend(sorted(model_dir.glob(f"{algo}_ep*.pth"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _find_resume_candidates(algo: str) -> list[str]:
    """Return resume candidates ordered from most to least preferred."""
    model_dir = Path("models") / algo
    candidates: list[Path] = []

    periodic = []
    for candidate in model_dir.glob(f"{algo}_ep*.pth"):
        stem = candidate.stem
        try:
            episode_number = int(stem.replace(f"{algo}_ep", ""))
        except ValueError:
            continue
        periodic.append((episode_number, candidate))

    periodic.sort(key=lambda item: item[0], reverse=True)
    candidates.extend(path for _, path in periodic)

    for fallback in [model_dir / f"{algo}_best_avg100.pth", model_dir / f"{algo}_best.pth"]:
        if fallback.exists() and fallback not in candidates:
            candidates.append(fallback)

    return [str(path) for path in candidates]


def _build_agent(algo: str, device: str):
    """Create an agent instance for the requested algorithm."""
    if algo == "td3":
        return TD3Agent(device=device)
    if algo == "ddpg":
        return DDPGAgent(device=device)
    raise ValueError("Unsupported algorithm")


def _try_load_checkpoint(agent, checkpoint_path: str, required: bool) -> bool:
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
    parser = argparse.ArgumentParser(description="Deterministic Policy Gradient Car - Training/Eval/Demo")
    parser.add_argument(
        "--algo",
        choices=["td3", "ddpg"],
        default=None,
        help="Algorithm to run",
    )
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
        default=MAX_EPISODES,
        help=f"Maximum training episodes (default: {MAX_EPISODES})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS_PER_EPISODE,
        help=f"Maximum steps per training episode (default: {MAX_STEPS_PER_EPISODE})",
    )
    args = parser.parse_args()

    if args.algo is None:
        raise ValueError("You must specify --algo {td3, ddpg}")

    # Use auto-detected headless mode unless explicitly overridden
    effective_headless = args.headless or HEADLESS
    
    set_global_seed(args.seed)
    init_pygame(headless=effective_headless)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print initialization information
    mode_str = "HEADLESS (off-screen rendering)" if effective_headless else "GUI (interactive window)"
    print(f"\n{'='*70}")
    print(f"[main] Visualization: {mode_str}")
    print(f"[main] Algorithm: {args.algo}")
    print(f"[main] Device: {device}")
    print(f"{'='*70}\n")

    algo_logs_dir = str(Path("logs") / args.algo)
    algo_model_dir = str(Path("models") / args.algo)

    env = CarRacingEnv(
        experiment_name="default",
        metrics_log_dir=algo_logs_dir,
        seed=args.seed,
        headless=effective_headless
    )
    agent = _build_agent(args.algo, device=device)

    checkpoint_path = None
    loaded_for_training = False
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.mode == "train" and args.resume:
        resume_candidates = _find_resume_candidates(args.algo)
        for candidate in resume_candidates:
            print(f"[main] Trying resume checkpoint: {candidate}")
            if _try_load_checkpoint(agent, candidate, required=False):
                checkpoint_path = candidate
                loaded_for_training = True
                break
    elif args.mode in {"eval", "demo"}:
        checkpoint_path = _find_default_checkpoint(args.algo)

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
            train(
                env,
                algo=args.algo,
                device=device,
                model_dir=algo_model_dir,
                checkpoint_path=checkpoint_path if loaded_for_training else None,
                require_checkpoint=bool(args.checkpoint),
                experiment_name="default",
                seed=args.seed,
                max_episodes=args.max_episodes,
                max_steps_per_episode=args.max_steps,
            )
        elif args.mode == "eval":
            print(f"[main] Starting evaluation ({args.eval_episodes} episodes)...")
            evaluate(env, agent, num_episodes=args.eval_episodes, render=args.render)
        else:  # demo mode
            demo_episodes = max(1, min(args.demo_episodes, 5))
            print(f"[main] Starting demo ({demo_episodes} episodes, render enabled)...")
            preview_path = str(Path("logs") / "demo_preview.png")
            evaluate(
                env,
                agent,
                num_episodes=demo_episodes,
                render=True,
                preview_path=preview_path,
            )
            print(f"[main] Demo preview saved to: {preview_path}")
    except (KeyboardInterrupt, SystemExit):
        print("\n[main] Interrupted - shutting down gracefully.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
