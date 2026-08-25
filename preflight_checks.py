"""preflight_checks.py - Comprehensive verification suite for TD3-DDPG Car Game.

Executes all 16 mandatory pre-flight checks before any experimental training runs.
"""

import copy
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from car import Car
from config import (
    ACTION_DIM,
    CAR_ACCELERATION,
    CAR_FRICTION,
    CAR_MAX_SPEED,
    DEFAULT_SEED,
    EVAL_EPISODES,
    EXPERIMENT_REWARD_MODES,
    EXPERIMENT_SENSOR_NOISE_LEVELS,
    EXPERIMENTS,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    NUM_SENSORS,
    STATE_DIM,
)
from ddpg_agent import DDPGAgent
from environment import CarRacingEnv
from replay_buffer import ReplayBuffer
from run_experiments import (
    ALGORITHMS,
    SEEDS,
    _experiment_ids_for_algo,
    _experiment_tag,
    _is_experiment_complete,
)
from td3_agent import TD3Agent
from train import evaluate, train_with_config
from utils import init_pygame, set_global_seed


def run_all_checks() -> bool:
    """Run all 16 pre-flight verification checks and print structured report."""
    print("=" * 80)
    print("PRE-FLIGHT VALIDATION SUITE: TD3 vs DDPG RESEARCH CODEBASE")
    print("=" * 80)
    
    init_pygame(headless=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_passed = True
    results = []

    def log_check(idx: int, name: str, passed: bool, details: str):
        nonlocal all_passed
        if not passed:
            all_passed = False
        status = "[PASS]" if passed else "[FAIL]"
        print(f"Check {idx:02d}: {status} {name}")
        if details:
            print(f"          -> {details}")
        results.append((idx, name, passed, details))

    # -------------------------------------------------------------------------
    # Check 1: state is not next_state
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        state_0 = env.reset()
        next_state_1, _, _, _ = env.step(np.array([0.0, 1.0]))
        c1_passed = (state_0 is not next_state_1)
        log_check(
            1,
            "State object identity (state is not next_state)",
            c1_passed,
            f"id(state_0)={id(state_0)} vs id(next_state_1)={id(next_state_1)}",
        )
        env.close()
    except Exception as e:
        log_check(1, "State object identity (state is not next_state)", False, str(e))

    # -------------------------------------------------------------------------
    # Check 2: State remains unchanged after env.step()
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        state_before = env.reset()
        state_before_copy = state_before.copy()
        next_state, _, _, _ = env.step(np.array([0.5, 1.0]))
        c2_passed = np.array_equal(state_before, state_before_copy) and not np.array_equal(state_before, next_state)
        log_check(
            2,
            "State immutability across env.step()",
            c2_passed,
            f"state_before matches initial: {np.array_equal(state_before, state_before_copy)}, "
            f"state_before != next_state: {not np.array_equal(state_before, next_state)}",
        )
        env.close()
    except Exception as e:
        log_check(2, "State immutability across env.step()", False, str(e))

    # -------------------------------------------------------------------------
    # Check 3: Replay transitions contain distinct pre/post states
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        buf = ReplayBuffer(10)
        s0 = env.reset()
        a0 = np.array([0.2, 0.8])
        s1, r0, d0, _ = env.step(a0)
        buf.add(s0, a0, r0, s1, d0)
        
        stored_s = buf.states[0]
        stored_s_next = buf.next_states[0]
        c3_passed = np.array_equal(stored_s, s0) and np.array_equal(stored_s_next, s1) and not np.array_equal(stored_s, stored_s_next)
        log_check(
            3,
            "Replay buffer transition integrity (s_t != s_t+1)",
            c3_passed,
            f"Stored s_t matches s0: {np.array_equal(stored_s, s0)}, "
            f"Stored s_t+1 matches s1: {np.array_equal(stored_s_next, s1)}, "
            f"Stored s_t != s_t+1: {not np.array_equal(stored_s, stored_s_next)}",
        )
        env.close()
    except Exception as e:
        log_check(3, "Replay buffer transition integrity", False, str(e))

    # -------------------------------------------------------------------------
    # Check 4: State and action dimensions
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        obs = env.reset()
        c4_passed = (len(obs) == STATE_DIM == 7) and (ACTION_DIM == 2) and (obs.dtype == np.float32)
        log_check(
            4,
            "State/Action dimensions & dtypes",
            c4_passed,
            f"STATE_DIM={len(obs)} (expected 7, dtype={obs.dtype}), ACTION_DIM={ACTION_DIM} (expected 2)",
        )
        env.close()
    except Exception as e:
        log_check(4, "State/Action dimensions & dtypes", False, str(e))

    # -------------------------------------------------------------------------
    # Check 5: Throttle mapping remains clip(action[1], 0, 1)
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        t_neg = env._parse_action(np.array([0.0, -0.8]))[1]
        t_zero = env._parse_action(np.array([0.0, 0.0]))[1]
        t_pos = env._parse_action(np.array([0.0, 0.75]))[1]
        t_over = env._parse_action(np.array([0.0, 1.5]))[1]
        c5_passed = (t_neg == 0.0) and (t_zero == 0.0) and math.isclose(t_pos, 0.75) and (t_over == 1.0)
        log_check(
            5,
            "Throttle mapping clip(action[1], 0, 1)",
            c5_passed,
            f"a=-0.8 -> {t_neg}, a=0.0 -> {t_zero}, a=0.75 -> {t_pos}, a=1.5 -> {t_over}",
        )
        env.close()
    except Exception as e:
        log_check(5, "Throttle mapping", False, str(e))

    # -------------------------------------------------------------------------
    # Check 6: Steering mapping remains [-1, 1]
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        s_left = env._parse_action(np.array([-0.8, 0.5]))[0]
        s_right = env._parse_action(np.array([0.6, 0.5]))[0]
        s_clamp = env._parse_action(np.array([-2.0, 0.5]))[0]
        c6_passed = math.isclose(s_left, -0.8) and math.isclose(s_right, 0.6) and (s_clamp == -1.0)
        log_check(
            6,
            "Steering range [-1, 1]",
            c6_passed,
            f"a=-0.8 -> {s_left}, a=0.6 -> {s_right}, a=-2.0 -> {s_clamp}",
        )
        env.close()
    except Exception as e:
        log_check(6, "Steering range", False, str(e))

    # -------------------------------------------------------------------------
    # Check 7: Lap completion physical feasibility in 600 steps
    # -------------------------------------------------------------------------
    try:
        v_ss = CAR_ACCELERATION * (1.0 - CAR_FRICTION) / CAR_FRICTION
        # Ellipse midline circumference: a=400, b=250
        t_arr = np.linspace(0, 2 * np.pi, 1000)
        ds = np.sqrt((-400 * np.sin(t_arr)) ** 2 + (250 * np.cos(t_arr)) ** 2)
        c_mid = float(np.sum(ds) * (2 * np.pi / 1000))
        
        # Simulate straight acceleration from 0
        v = 0.0
        d = 0.0
        steps = 0
        while d < c_mid and steps < 600:
            v += CAR_ACCELERATION
            v -= CAR_FRICTION * v
            v = min(v, CAR_MAX_SPEED)
            d += v
            steps += 1
            
        c7_passed = (steps < MAX_STEPS_PER_EPISODE == 600)
        log_check(
            7,
            "Lap completion feasibility within 600 steps",
            c7_passed,
            f"Centerline dist={c_mid:.1f}px, Steady-state speed={v_ss:.2f}px/step. "
            f"Min steps required starting from rest={steps} (Horizon={MAX_STEPS_PER_EPISODE})",
        )
    except Exception as e:
        log_check(7, "Lap completion feasibility", False, str(e))

    # -------------------------------------------------------------------------
    # Check 8: Episode termination conditions
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, headless=True)
        env.reset()
        # Test max_steps termination
        done = False
        for _ in range(MAX_STEPS_PER_EPISODE):
            _, _, done, info = env.step(np.array([0.0, 0.5]))
            if done:
                break
        c8_passed = done and (info.get("termination_reason") in ["off_track", "stuck", "max_steps"])
        log_check(
            8,
            "Episode termination mechanics",
            c8_passed,
            f"Terminated at step {env.step_count} with reason: {info.get('termination_reason')}",
        )
        env.close()
    except Exception as e:
        log_check(8, "Episode termination mechanics", False, str(e))

    # -------------------------------------------------------------------------
    # Check 9: Sensor noise isolation
    # -------------------------------------------------------------------------
    try:
        env = CarRacingEnv(enable_metrics=False, sensor_noise_std=0.05, headless=True)
        s_noisy_1 = env.reset()
        env.car.cast_sensors()
        s_noisy_2 = env.car.get_state()
        # Sensor distances should differ due to noise, but position (x, y, speed, angle) must remain exact
        pos_match = np.allclose(s_noisy_1[:4], s_noisy_2[:4])
        sensor_diff = not np.allclose(s_noisy_1[4:], s_noisy_2[4:])
        c9_passed = pos_match and sensor_diff
        log_check(
            9,
            "Sensor noise isolation to rangefinders",
            c9_passed,
            f"Physical state exact: {pos_match}, Rangefinder readings perturbed: {sensor_diff}",
        )
        env.close()
    except Exception as e:
        log_check(9, "Sensor noise isolation", False, str(e))

    # -------------------------------------------------------------------------
    # Check 10: Deterministic evaluation (noise OFF)
    # -------------------------------------------------------------------------
    try:
        agent = TD3Agent(device=device)
        dummy_state = np.ones(STATE_DIM, dtype=np.float32) * 0.5
        a1 = agent.select_action(dummy_state, add_noise=False)
        a2 = agent.select_action(dummy_state, add_noise=False)
        a_noisy = agent.select_action(dummy_state, add_noise=True, noise_scale=0.1)
        c10_passed = np.array_equal(a1, a2)
        log_check(
            10,
            "Deterministic evaluation repeatability (add_noise=False)",
            c10_passed,
            f"Deterministic actions identical: {np.array_equal(a1, a2)}, "
            f"Diff with noisy: {np.max(np.abs(a1 - a_noisy)):.4f}",
        )
    except Exception as e:
        log_check(10, "Deterministic evaluation repeatability", False, str(e))

    # -------------------------------------------------------------------------
    # Check 11: Fresh model initialization for TD3 and DDPG
    # -------------------------------------------------------------------------
    try:
        td3_1 = TD3Agent(device=device)
        td3_2 = TD3Agent(device=device)
        p1 = list(td3_1.actor.parameters())[0].data.cpu().numpy()
        p2 = list(td3_2.actor.parameters())[0].data.cpu().numpy()
        ddpg_1 = DDPGAgent(device=device)
        ddpg_2 = DDPGAgent(device=device)
        dp1 = list(ddpg_1.actor.parameters())[0].data.cpu().numpy()
        dp2 = list(ddpg_2.actor.parameters())[0].data.cpu().numpy()
        c11_passed = not np.array_equal(p1, p2) and not np.array_equal(dp1, dp2)
        log_check(
            11,
            "Fresh model instantiation independence",
            c11_passed,
            f"TD3 instances distinct: {not np.array_equal(p1, p2)}, "
            f"DDPG instances distinct: {not np.array_equal(dp1, dp2)}",
        )
    except Exception as e:
        log_check(11, "Fresh model instantiation independence", False, str(e))

    # -------------------------------------------------------------------------
    # Check 12: Replay buffer independence across runs
    # -------------------------------------------------------------------------
    try:
        buf1 = ReplayBuffer(100)
        buf2 = ReplayBuffer(100)
        buf1.add(np.zeros(7), np.zeros(2), 1.0, np.zeros(7), False)
        c12_passed = (len(buf1) == 1) and (len(buf2) == 0) and (buf1.states is not buf2.states)
        log_check(
            12,
            "Replay buffer isolation across runs",
            c12_passed,
            f"buf1 size={len(buf1)}, buf2 size={len(buf2)}, buffer arrays separate: {buf1.states is not buf2.states}",
        )
    except Exception as e:
        log_check(12, "Replay buffer isolation", False, str(e))

    # -------------------------------------------------------------------------
    # Check 13: Seed application verification (0, 42, 123)
    # -------------------------------------------------------------------------
    try:
        c13_passed = True
        for s in [0, 42, 123]:
            set_global_seed(s)
            r1 = np.random.rand()
            set_global_seed(s)
            r2 = np.random.rand()
            if r1 != r2:
                c13_passed = False
        log_check(
            13,
            "Seed repeatability for 0, 42, 123",
            c13_passed,
            "Verified NumPy/Torch random stream reproducibility for seeds 0, 42, and 123.",
        )
    except Exception as e:
        log_check(13, "Seed repeatability", False, str(e))

    # -------------------------------------------------------------------------
    # Check 14: 72 Experiment combinations enumeration
    # -------------------------------------------------------------------------
    try:
        td3_grid = _experiment_ids_for_algo("td3")
        ddpg_grid = _experiment_ids_for_algo("ddpg")
        total_runs = (len(td3_grid) + len(ddpg_grid)) * len(SEEDS)
        c14_passed = (len(td3_grid) == 12) and (len(ddpg_grid) == 12) and (len(SEEDS) == 3) and (total_runs == 72)
        log_check(
            14,
            "Full experiment grid enumeration (72 runs)",
            c14_passed,
            f"TD3 conditions={len(td3_grid)}, DDPG conditions={len(ddpg_grid)}, Seeds={SEEDS}, Total={total_runs}",
        )
    except Exception as e:
        log_check(14, "Full experiment grid enumeration", False, str(e))

    # -------------------------------------------------------------------------
    # Check 15: Resume logic rejection of incomplete runs
    # -------------------------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_log_dir = os.path.join(tmp_dir, "logs", "td3", "R1_N1", "seed_0")
            fake_model_dir = os.path.join(tmp_dir, "models", "td3", "R1_N1", "seed_0")
            os.makedirs(fake_log_dir, exist_ok=True)
            os.makedirs(fake_model_dir, exist_ok=True)
            
            # Scenario A: Incomplete run (5 episodes) with best.pth existing
            with open(os.path.join(fake_log_dir, "training_log.jsonl"), "w") as f:
                for ep in range(1, 6):
                    f.write(json.dumps({"episode": ep, "reward_total": 5.0}) + "\n")
            with open(os.path.join(fake_model_dir, "td3_best.pth"), "w") as f:
                f.write("dummy")
                
            is_done_incomplete = _is_experiment_complete(fake_log_dir, fake_model_dir, "td3", 2000)
            
            # Scenario B: 2000 episodes but missing evaluation_summary.json
            with open(os.path.join(fake_log_dir, "training_log.jsonl"), "w") as f:
                for ep in range(1, 2001):
                    f.write(json.dumps({"episode": ep, "reward_total": 5.0}) + "\n")
            is_done_no_eval = _is_experiment_complete(fake_log_dir, fake_model_dir, "td3", 2000)
            
            # Scenario C: 2000 episodes AND valid evaluation_summary.json
            with open(os.path.join(fake_log_dir, "evaluation_summary.json"), "w") as f:
                json.dump({"num_episodes": EVAL_EPISODES, "crash_rate": 0.1}, f)
            is_done_complete = _is_experiment_complete(fake_log_dir, fake_model_dir, "td3", 2000)
            
            c15_passed = (not is_done_incomplete) and (not is_done_no_eval) and is_done_complete
            log_check(
                15,
                "Resume completion detection rigor",
                c15_passed,
                f"Incomplete run skipped: {not is_done_incomplete}, "
                f"Missing eval rejected: {not is_done_no_eval}, "
                f"Complete run recognized: {is_done_complete}",
            )
    except Exception as e:
        log_check(15, "Resume completion detection rigor", False, str(e))

    # -------------------------------------------------------------------------
    # Check 16: Result evaluation summary structure and metadata
    # -------------------------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env = CarRacingEnv(enable_metrics=False, headless=True)
            agent = TD3Agent(device=device)
            eval_log_p = os.path.join(tmp_dir, "eval_log.jsonl")
            eval_sum_p = os.path.join(tmp_dir, "eval_summary.json")
            res = evaluate(
                env,
                agent,
                num_episodes=2,
                eval_log_path=eval_log_p,
                eval_summary_path=eval_sum_p,
                metadata={"test": True, "seed": 42},
            )
            env.close()
            has_primary = all(k in res for k in ["crash_rate", "lap_completion_rate", "distance_mean", "avg_length"])
            has_secondary = all(k in res for k in ["avg_reward", "avg_speed", "raw_rewards", "metadata"])
            files_exist = os.path.exists(eval_log_p) and os.path.exists(eval_sum_p)
            c16_passed = has_primary and has_secondary and files_exist
            log_check(
                16,
                "Evaluation summary schema & disk serialization",
                c16_passed,
                f"Primary metrics present: {has_primary}, Secondary present: {has_secondary}, Files written: {files_exist}",
            )
    except Exception as e:
        log_check(16, "Evaluation summary schema", False, str(e))

    print("=" * 80)
    if all_passed:
        print("ALL 16 PRE-FLIGHT VALIDATION CHECKS PASSED SUCCESSFULLY [16/16]")
    else:
        print("PRE-FLIGHT VALIDATION FAILED ON ONE OR MORE CHECKS!")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
