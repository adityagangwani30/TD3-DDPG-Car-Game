# Reward System

This document explains the four reward modes (R1–R4) used in this project and the broader concepts of reward design in reinforcement learning.

---

## Overview

The reward function defines *what* the agent is incentivized to learn. This project uses four progressively shaped reward modes to study how reward design affects agent behavior, stability, and safety.

All reward modes share a common crash penalty:

- **On termination (crash or stuck):** reward = `REWARD_CRASH` = **−5.0**

The modes differ in what reward the agent receives at each non-terminal step.

---

## R1 — Basic Reward (`basic`)

```python
reward = REWARD_ALIVE  # +0.05 per step
```

**What it encourages:** Pure survival. The agent is rewarded simply for staying on the track.

**Why it was added:** As a baseline — the simplest possible reward. It tests whether the agent can learn useful behavior from minimal incentives alone.

**Expected benefit:** The agent should learn to avoid crashes, since crashing ends the episode and cuts off future survival rewards.

**Possible limitation:** Without any speed or lap bonuses, the agent may learn to drive slowly in circles or sit near the track center — maximizing steps without actually racing. This is a form of **reward hacking**: the agent optimizes the reward signal without performing the intended task.

**Key observation:** R1 can show misleadingly high reward values because the agent collects +0.05 for every step it survives. A slow, cautious agent that never completes laps but avoids crashes can accumulate a high total reward — while an agent that drives fast, completes laps, but occasionally crashes may show lower total reward. This is why reward alone is not a sufficient metric.

---

## R2 — Shaped Reward (`shaped`)

```python
reward = 0.05                                     # alive bonus
reward += 0.15  (if speed > 0.15)                 # speed bonus
reward += 15.0  (if lap completed)                # lap completion bonus
reward -= 0.05 * (|steering|²)                    # steering penalty
```

**What it encourages:** Moving at speed, completing laps, and steering smoothly.

**Why it was added:** To provide standard shaped incentives that guide the agent toward racing behavior, beyond just survival.

**Expected benefit:** The agent should learn to drive at moderate speed and complete laps, with smooth steering reducing oscillation.

**Possible limitation:** The steering penalty may discourage aggressive cornering, which is sometimes necessary on the oval track.

---

## R3 — Modified Reward (`modified`)

```python
reward = 0.05                                     # alive bonus
reward += 0.18  (if speed > 0.15)                 # higher speed bonus
reward += 16.0  (if lap completed)                # higher lap bonus
reward -= 0.04 * (|steering|²)                    # reduced steering penalty
reward += 0.06 * (speed / max_speed)              # gentle speed scaling
reward += 0.03  (if speed > 0.15 and |steer| < 0.2)  # stability bonus
reward -= 0.02  (if speed <= 0.15)                # anti-idling penalty
```

**What it encourages:** Faster driving with stability bonuses. The reduced steering penalty allows slightly more aggressive cornering, while speed scaling rewards continuous acceleration and the stability bonus rewards straight, fast driving.

**Why it was added:** To explore whether mid-level reward tuning — enhancing the shaped reward with additional stability incentives — improves performance over the standard shaped reward (R2).

**Expected benefit:** Better overall driving quality through a balance of speed, stability, and lap completion.

**Possible limitation:** The multiple additive terms create a more complex reward landscape. The anti-idling penalty and stability bonus introduce opposing pressures that could interact unpredictably with different noise levels.

**Key observation:** In recent experimental runs, R3 combined with moderate noise (N2) has shown strong performance — the combination of informative reward shaping and mild noise regularization appears to produce effective policies.

---

## R4 — Tuned Reward (`tuned`)

```python
reward = 0.08                                     # higher alive bonus
reward += 0.25  (if speed > 0.15)                 # much higher speed bonus
reward += 18.0  (if lap completed)                # much higher lap bonus
reward -= 0.03 * (|steering|²)                    # lowest steering penalty
reward += 0.10 * (speed / max_speed)              # strong speed scaling
reward += 0.05  (if speed > 0.15 and |steer| < 0.2)  # higher stability bonus
reward -= 0.04  (if speed <= 0.15)                # stronger anti-idling penalty
```

**What it encourages:** Maximum speed, aggressive lap completion, strong anti-idling behavior.

**Why it was added:** To test whether aggressive shaping — pushing all reward components to their maximum — produces the best policies or whether over-shaping introduces problems.

**Expected benefit:** Fastest learning and highest lap completion rates due to strong incentives.

**Possible limitation:** Aggressive shaping can cause:
- **Reward saturation:** The reward signal becomes dominated by speed and lap bonuses, making fine behavioral differences hard to distinguish
- **Risky behavior:** High speed bonuses incentivize driving near maximum speed even in tight corners, increasing crash risk
- **Over-optimization:** The agent may exploit the reward function in ways that do not transfer to slightly different conditions

**Key observation:** R4 is a strong performer but is not always the best configuration. Despite having the most aggressive shaping, it does not consistently outperform R3 across all metrics and noise levels.

---

## Reward Design Concepts

### Reward Shaping

Reward shaping adds intermediate signals to guide the agent toward desired behavior without changing the optimal policy. In this project, R2–R4 progressively add more shaping:

- R1: No shaping (survival only)
- R2: Standard shaping (speed + lap + smooth steering)
- R3: Enhanced shaping (stability bonuses, anti-idling)
- R4: Aggressive shaping (maximum incentives)

### Reward Hacking

Reward hacking occurs when the agent finds a way to maximize the reward signal without performing the intended task. R1 is particularly susceptible: an agent that drives slowly in safe areas collects +0.05 per step indefinitely, accumulating high reward without ever completing a lap.

### Reward Saturation

When reward magnitudes are too large, the gradient signal becomes dominated by a few components. R4's large lap bonus (+18.0) and speed bonus (+0.25) can saturate the learning signal, making it harder for the agent to learn subtle behavioral adjustments.

---

## Why Reward Alone Is Insufficient (and Cross-Reward Averaging Is Invalid)

This project demonstrates a critical methodological requirement: **raw reward values must NEVER be averaged across different reward configurations (R1–R4).**

Because R1 through R4 define different reward scales (e.g. alive bonus scaling, velocity scaling, and lap completion bonuses of 0 vs 15 vs 18), an average across reward functions is mathematically and scientifically meaningless.

To properly evaluate and compare agent performance:
1. **Compare algorithms (TD3 vs DDPG) strictly within the exact same reward formulation.**
2. **Examine physical, scale-independent metrics across reward configurations:**

| Metric | Type | What It Reveals |
|--------|:---:|-----------------|
| Crash Rate | Physical / Safety | Fraction of episodes terminating in off-track collision |
| Lap Completion Rate | Task Success | Fraction of episodes completing $\ge 1$ full centerline lap |
| Track Distance | Progress | Total Euclidean distance traversed (centerline lap = 2069 px) |
| Survival Length | Reliability | Steps survived per episode (up to 600) |
| Episodic Reward | Scale-Dependent | How well the agent optimizes that specific objective function |

Examining these metrics together reveals the **Reward Aggressiveness vs. Safety Trade-Off**: whether higher reward shaping intensity pushes the agent toward higher speed at the cost of elevated crash risk and reduced overall lap completion.

---

## Summary Table

| Mode | Alive | Speed Bonus | Lap Bonus | Steering Penalty | Extras |
|------|-------|-------------|-----------|------------------|--------|
| R1 | +0.05 | — | — | — | — |
| R2 | +0.05 | +0.15 | +15.0 | −0.05·s² | — |
| R3 | +0.05 | +0.18 | +16.0 | −0.04·s² | Speed scaling, stability bonus, anti-idle |
| R4 | +0.08 | +0.25 | +18.0 | −0.03·s² | Strong speed scaling, strong stability bonus, strong anti-idle |

All modes: crash penalty = **−5.0**

---

## Further Reading

- [observation_noise.md](observation_noise.md) — How noise interacts with reward modes
- [results_interpretation.md](results_interpretation.md) — How to evaluate performance across metrics
- [experiment_design.md](experiment_design.md) — How reward modes are combined with noise levels
- **Implementation:** [`environment.py`](../environment.py) — `_compute_reward()`, `_compute_shaped_reward()`, `_compute_modified_reward()`, `_compute_tuned_reward()`
