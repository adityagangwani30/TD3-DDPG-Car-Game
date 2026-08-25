# Reproducibility

This document discusses reproducibility in this reinforcement learning project — why results vary, what causes randomness, and how to improve confidence in findings.

---

## Why Reinforcement Learning Results Vary

RL results are inherently stochastic. Even with the same code, hyperparameters, and random seed, results may differ across runs due to hardware and software non-determinism. Between different seeds, variation is expected and can be substantial.

This is a known property of deep RL, not a bug. It means that:
- Single-run conclusions are unreliable
- Results must be interpreted as trends across multiple seeds
- Error bars and standard deviations are essential

---

## Sources of Randomness

### 1. Neural Network Initialization

Both the actor and critic networks are initialized with random weights (PyTorch default initialization). Different weight initializations lead to different optimization trajectories and potentially different final policies.

### 2. Exploration Noise

During training, Gaussian noise is added to the agent's actions for exploration:
```python
noise = np.random.normal(0, noise_scale, size=ACTION_DIM)
```

Different noise samples lead to different experiences being collected, which changes the contents of the replay buffer and therefore the training signal.

### 3. Replay Buffer Sampling

Mini-batches are sampled uniformly at random from the replay buffer:
```python
indices = np.random.randint(0, self.size, size=batch_size)
```

Different mini-batches expose the networks to different combinations of past experiences, affecting gradient updates.

### 4. Environment Interaction

Sensor noise (when N2 or N3 is used) adds random perturbations to observations at every time step. This changes the agent's perception of the environment and therefore its behavior and collected experiences.

### 5. Hardware Non-Determinism

Even with deterministic mode enabled (`torch.backends.cudnn.deterministic = True`), some operations may produce slightly different results across:
- CPU vs GPU execution
- Different GPU architectures
- Different CUDA versions
- Floating-point operation ordering

---

## How Seeds Are Used

This project sets random seeds across multiple libraries in `utils.py`:

```python
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

The experiment runner uses three seeds: **0, 42, and 123**.

Seeds control:
- Neural network weight initialization
- Exploration noise generation
- Replay buffer sampling order
- Sensor noise generation

---

## Current Limitation

The current study trains **3 seeds per configuration**. While this is better than a single run, 3 seeds provide only a rough estimate of the mean and a noisy estimate of variance.

With 3 seeds:
- The sample mean may not closely approximate the true mean
- The standard deviation estimate has high uncertainty
- Outlier seeds can substantially shift the reported statistics

---

## Recommended Future Improvements

### Multi-Seed Experiments

For stronger statistical validity, future work should use:

- **5–10 seeds per configuration** — provides more reliable mean estimates
- **Mean and standard deviation reporting** — across all seeds for every metric
- **Statistical tests** — e.g., paired t-tests or Wilcoxon signed-rank tests to determine whether differences between configurations are statistically significant

### Additional Practices

- **Report confidence intervals** instead of or in addition to standard deviations
- **Use box plots or violin plots** to show the full distribution across seeds
- **Track computational cost** to enable cost-performance trade-off analysis
- **Log hardware specifications** to enable cross-platform comparison

---

## Reproducibility Statement

The following statement is suitable for inclusion in a README or research paper:

> *Due to the stochastic nature of reinforcement learning, exact values may vary across runs, but overall trends should be interpreted across reward-noise configurations.*

---

## Technical Code Corrections for Camera-Ready Reproducibility

During the pre-camera-ready audit, four critical technical corrections were implemented to guarantee rigorous reproducibility:

### 1. State Buffer Copying (`car.py`)
- **Issue in Legacy Implementation:** `Car.get_state()` returned an internal NumPy buffer without an explicit array copy (`return self._state_buffer`). When the environment stepped, mutable in-place array updates caused transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ stored in the `ReplayBuffer` to share array pointers, corrupting historical state observations.
- **Correction:** `Car.get_state()` now returns `self._state_buffer.copy()`, ensuring uncoupled state snapshots throughout training and replay sampling.

### 2. Physical Lap Horizon Feasibility (`config.py`)
- **Issue in Legacy Implementation:** The previous 300-step limit truncated episodes before a car starting from standstill could physically finish a full 2068.8-pixel centerline lap (which requires a theoretical minimum of 382 steps at full throttle).
- **Correction:** The episode horizon was set to `MAX_STEPS_PER_EPISODE = 600`.

### 3. Separation of Training vs. Deterministic Evaluation (`train.py`)
- **Protocol:** Exploration noise ($\sigma = 0.1 \times 0.9999^{\text{step}}$) remains active during training. Following training, the best checkpoint is evaluated deterministically for **20 independent episodes** with `add_noise=False`.
- Headline safety and performance metrics (Crash Rate, Lap Completion Rate, Track Distance) are derived exclusively from deterministic evaluation runs.

### 4. Robust Completion & Resume Logic (`run_experiments.py`)
- Resuming interrupted batches requires both 2,000 completed training episodes **AND** a valid 20-episode `evaluation_summary.json`. Missing evaluations are automatically executed on resume.

---

## 16-Point Pre-Flight Validation Suite

To ensure no regressions, [`preflight_checks.py`](../preflight_checks.py) validates the entire pipeline prior to long runs:

1. **State Object Identity:** `state is not next_state`
2. **State Immutability:** Pre-step state array values are unaltered by `env.step()`
3. **Replay Transition Integrity:** Transitions in `ReplayBuffer` maintain distinct pre/post arrays
4. **State/Action Dimensionality:** State $\in \mathbb{R}^7$, Action $\in \mathbb{R}^2$
5. **Throttle Mapping:** Environment clips $a_1 \in [0, 1]$ correctly
6. **Steering Range:** Steering bounds remain within $[-1, 1]$
7. **Lap Feasibility:** Centerline lap is physically navigable within 600 steps
8. **Episode Termination:** Off-track crashes and max steps terminate cleanly
9. **Sensor Noise Isolation:** Sensor noise is active only in observations, not underlying vehicle physics
10. **Deterministic Evaluation:** Noise is strictly disabled during evaluation
11. **Fresh Model Initialization:** Consecutive runs initialize independent network weights
12. **Replay Buffer Isolation:** Buffers are instantiated freshly per seed
13. **Seed Repeatability:** Identical seeds reproduce identical trajectories
14. **72-Run Factorial Grid:** Exact enumeration of 72 unique $(algo, R_i, N_j, s_k)$ combinations
15. **Resume Logic Rigor:** Detects complete vs incomplete logs accurately
16. **Evaluation Serialization:** Verifies `evaluation_summary.json` schema and statistics

---

## What Is Reproducible

| Aspect | How |
|--------|-----|
| Experiment Grid | 72 runs ($2 \text{ algos} \times 4 \text{ rewards} \times 3 \text{ noises} \times 3 \text{ seeds}$) |
| Canonical Codebase | Single shared implementation with configurable path arguments |
| Isolated Output Directories | `logs_v2/`, `models_v2/`, `results_v2/` (legacy `logs/`, `models/` preserved) |
| Hyperparameters | Configured in `config.py` and frozen in `EXPERIMENT_V2.md` |
| Seeds | Fixed at 0, 42, 123 ($n=3$ independent training replicates) |
| Environment | Deterministic physics given same control inputs and noise seed |

---

## Further Reading

- [EXPERIMENT_V2.md](../EXPERIMENT_V2.md) — Authoritative camera-ready protocol specification
- [experiment_design.md](experiment_design.md) — Factorial study structure
- [results_interpretation.md](results_interpretation.md) — Interpreting deterministic evaluation metrics
- **Implementation:** [`utils.py`](../utils.py), [`preflight_checks.py`](../preflight_checks.py), [`config.py`](../config.py)

