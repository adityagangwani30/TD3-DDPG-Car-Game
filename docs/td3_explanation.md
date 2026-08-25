# TD3 — Twin Delayed Deep Deterministic Policy Gradient

This document explains how the **TD3 algorithm** works and how it is applied in this project to train a self-driving car agent in a 2D racing environment.

---

## Why TD3 for Continuous Control?

Reinforcement learning tasks like car navigation require **continuous actions** — the agent must output a precise steering angle and throttle value at every time step, not choose from a fixed list of discrete options.

TD3 (Twin Delayed Deep Deterministic Policy Gradient) is specifically designed for continuous action spaces. It combines:

- A **deterministic policy** that directly maps states to actions
- A **critic** that evaluates how good those actions are
- Three targeted improvements over DDPG to stabilize learning

In this project, the agent receives a 7-dimensional state (position, speed, heading, and sensor distances) and must produce a 2-dimensional continuous action (steering and throttle).

---

## Core Components

### Actor Network

The actor is a neural network that takes the current state as input and outputs an action.

In this project (`td3_agent.py`):

- **Input:** 7 state dimensions (position x, position y, speed, heading, 3 sensor distances)
- **Architecture:** Two hidden layers with 64 units each, using ReLU activations
- **Output:** 2 action dimensions (steering, throttle), passed through `tanh` to bound outputs to [-1, +1]

The actor is the agent's **policy** — it decides what the car should do at every moment.

### Twin Critic Networks

TD3 uses **two independent critic networks** (Q-networks) that share the same architecture but have separate parameters.

Each critic:

- **Input:** Concatenation of state (7 dims) and action (2 dims) = 9 dimensions
- **Architecture:** Two hidden layers with 64 units each, using ReLU activations
- **Output:** A single scalar Q-value estimating the expected future reward

Both critics are defined inside the `Critic` class in `td3_agent.py`. The class contains two parallel network paths (fc1–fc3 and fc4–fc6). During training, the **minimum** of the two Q-values is used for computing the target:

```python
target_q = torch.min(target_q1, target_q2)
```

This minimum operation is the key to reducing **overestimation bias** — a problem where a single critic tends to overestimate action values, leading to unstable training.

### Target Networks

TD3 maintains **target copies** of both the actor and the critic. These target networks are slowly updated versions of the main networks, created using `copy.deepcopy()` at initialization:

```python
self.actor_target = copy.deepcopy(self.actor)
self.critic_target = copy.deepcopy(self.critic)
```

Target networks provide stable Q-value targets during training. Without them, the critic would be chasing a moving target, leading to divergence.

They are updated via **Polyak averaging** (soft updates) with parameter τ = 0.005:

```python
target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
```

This means each update moves the target network 0.5% toward the current network — slow enough to maintain stability.

### Delayed Policy Updates

In TD3, the **actor (policy) is updated less frequently than the critic**. Specifically, the policy is updated once every 2 critic updates (controlled by `POLICY_DELAY = 2` in `config.py`):

```python
if self.total_it % POLICY_DELAY == 0:
    actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
    # ... update actor ...
    # ... soft-update target networks ...
```

The rationale: if the critic is still learning and its Q-estimates are noisy, updating the policy based on those noisy estimates can amplify errors. By waiting for the critic to become more accurate before updating the policy, training is more stable.

### Target Policy Smoothing

When computing target Q-values, TD3 adds **clipped noise** to the target actions:

```python
noise = (torch.randn_like(action) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
```

In this project:

- `POLICY_NOISE = 0.2` — standard deviation of the smoothing noise
- `NOISE_CLIP = 0.5` — maximum magnitude of the noise

This prevents the critic from exploiting narrow peaks in the Q-function. By smoothing the target policy, the critic must learn Q-values that are robust to small action perturbations.

### Replay Buffer

The agent stores past experiences in a **circular replay buffer** with capacity 200,000 transitions (`replay_buffer.py`). Each transition contains:

- State (7 dims)
- Action (2 dims)
- Reward (scalar)
- Next state (7 dims)
- Done flag (boolean)

During training, mini-batches of 256 transitions are sampled uniformly at random. This breaks temporal correlations and allows the agent to learn from diverse past experiences.

Training begins only after the buffer contains at least 5,000 transitions (`TRAINING_START = 5000`), ensuring sufficient data diversity before the first update.

### Exploration Noise

During training, the agent adds **Gaussian exploration noise** to its actions:

```python
noise = np.random.normal(0, noise_scale, size=ACTION_DIM)
action = action + noise
```

- Initial noise scale: 0.1 (`EXPLORATION_NOISE`)
- Noise decays by a factor of 0.9999 each episode (`EXPLORATION_NOISE_DECAY`)
- Minimum noise floor: 0.01

This exploration noise is different from the target policy smoothing noise. Exploration noise encourages the agent to try new actions during data collection, while target smoothing noise stabilizes the critic's learning targets.

---

## Why TD3 Suits This Car Navigation Task

1. **Continuous control is essential.** The car needs precise steering and throttle values — not just "turn left" or "turn right." TD3's deterministic policy handles this naturally.

2. **Stability matters.** The 2D racing environment has sparse positive feedback (lap completion bonuses) and frequent negative feedback (crashes). Twin critics and delayed updates prevent the agent from overreacting to noisy reward signals.

3. **Robustness under noise.** This project tests agents under varying sensor noise levels (N1–N3). TD3's error-dampening mechanisms (twin critics, target smoothing) make it more resilient to noisy observations compared to DDPG.

4. **Gradient clipping.** Both actor and critic gradients are clipped to a maximum norm of 1.0 (`GRADIENT_CLIP_MAX_NORM`), preventing catastrophic weight updates that could destabilize training in a fast-paced physics environment.

---

## TD3 Training Flow in This Project

```
1. Agent observes state from environment (7 dims)
2. Actor outputs action + exploration noise (2 dims)
3. Environment executes action, returns next_state, reward, done
4. Transition stored in replay buffer
5. If buffer has ≥ 5,000 samples:
   a. Sample mini-batch of 256
   b. Compute target Q using twin critic targets + smoothing noise
   c. Update both critics via MSE loss
   d. Every 2 iterations: update actor, soft-update all target networks
6. Decay exploration noise (0.9999 per step, floor 0.01)
7. Repeat for up to 600 steps per episode, 2,000 episodes total
```

---

## Key Hyperparameters (from `config.py`)

| Parameter | Value | Role |
|-----------|-------|------|
| Actor learning rate | 3 × 10⁻⁴ | Speed of policy updates |
| Critic learning rate | 3 × 10⁻⁴ | Speed of Q-function updates |
| Discount factor (γ) | 0.99 | How much future rewards matter |
| Soft update rate (τ) | 0.005 | Target network update speed |
| Policy delay | 2 | Critic updates per policy update |
| Policy noise (σ) | 0.2 | Target smoothing noise std |
| Noise clip | 0.5 | Target smoothing noise bound |
| Exploration noise | 0.1 (decaying) | Action exploration noise |
| Hidden layers | 2 × 64 units | Network capacity |
| Gradient clip norm | 1.0 | Prevents exploding gradients |

---

## Further Reading

- **TD3 paper:** [Addressing Function Approximation Error in Actor-Critic Methods (Fujimoto et al., 2018)](https://arxiv.org/abs/1802.09477)
- **DDPG background:** See [ddpg_background.md](ddpg_background.md) for context on the algorithm TD3 improves upon
- **Implementation:** See [`td3_agent.py`](../td3_agent.py) for the full source code
