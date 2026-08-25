# EXPERIMENT V2: Camera-Ready Experimental Protocol & Specification

## 1. Overview & Objective
This document defines the exact, frozen experimental protocol for generating the camera-ready results for the accepted research paper:
**"TD3 vs. DDPG: Continuous Control Robustness Under Observation Noise and Reward Shaping"**.

All historical bugs identified during the technical audit (NumPy state buffer aliasing, 300-step episode horizon truncation, ambiguous resume checks) have been resolved. The experimental methodology, reward formulations, and core hyperparameter grid remain strictly preserved.

---

## 2. Experimental Grid Matrix (72 Training Runs)

The evaluation matrix consists of a fully factorial design:
$$\text{2 Algorithms} \times \text{4 Reward Formulations} \times \text{3 Sensor-Noise Levels} \times \text{3 Random Seeds} = \mathbf{72\text{ Training Runs}}$$

### Factors:
1. **Algorithms (2):**
   - `TD3` (Twin Delayed Deep Deterministic Policy Gradient)
   - `DDPG` (Deep Deterministic Policy Gradient)
2. **Reward Configurations (4):**
   - `R1` — `basic`: $R = +0.05 \cdot v - 5.0 \cdot \mathbb{I}_{\text{crash}} + 10.0 \cdot \mathbb{I}_{\text{lap}}$
   - `R2` — `shaped`: $R = +0.10 \cdot v + 0.05 \cdot \mathbb{I}_{\text{alive}} - 0.02 \cdot |\delta| - 5.0 \cdot \mathbb{I}_{\text{crash}} + 10.0 \cdot \mathbb{I}_{\text{lap}}$
   - `R3` — `modified`: $R = +0.20 \cdot v + 0.10 \cdot \mathbb{I}_{\text{alive}} - 0.05 \cdot |\delta| - 5.0 \cdot \mathbb{I}_{\text{crash}} + 10.0 \cdot \mathbb{I}_{\text{lap}}$
   - `R4` — `tuned`: $R = +0.30 \cdot v + 0.15 \cdot \mathbb{I}_{\text{alive}} - 0.05 \cdot |\delta| - 5.0 \cdot \mathbb{I}_{\text{crash}} + 10.0 \cdot \mathbb{I}_{\text{lap}}$
3. **Observation Sensor-Noise Levels (3):**
   - `N1` — $\sigma_{\text{sensor}} = 0.00$ (Deterministic rangefinders)
   - `N2` — $\sigma_{\text{sensor}} = 0.02$ (Low Gaussian noise on 3 distance sensors)
   - `N3` — $\sigma_{\text{sensor}} = 0.05$ (High Gaussian noise on 3 distance sensors)
4. **Random Seeds (3):**
   - Seed `0`
   - Seed `42`
   - Seed `123`

---

## 3. Training & Evaluation Protocol

| Parameter | Value | Details |
|---|:---:|---|
| **Training Episodes per Run** | `2,000` | Full training curve per seed |
| **Max Steps per Episode ($T_{\max}$)** | `600` | Increased from 300 to allow physically feasible lap completion (min required: 382 steps) |
| **Exploration Noise ($\sigma_0$)** | `0.1` | Gaussian action perturbation |
| **Exploration Noise Decay** | `0.9999` | Per-step geometric decay with $0.01$ floor |
| **Deterministic Evaluation Episodes** | `20` | Run immediately upon training completion using the best checkpoint |
| **Evaluation Exploration Noise** | `OFF` (`add_noise=False`) | Strictly deterministic greedy policy |
| **Total Evaluation Episodes** | `1,440` | $72\text{ models} \times 20\text{ episodes}$ |
| **State Dimension** | `7` | $(x_{\text{norm}}, y_{\text{norm}}, v_{\text{norm}}, \theta_{\text{norm}}, d_0, d_1, d_2)$ |
| **Action Dimension** | `2` | Steering $\in [-1, 1]$, Throttle $\in [0, 1]$ |
| **Throttle Mapping** | $\text{clip}(a_1, 0, 1)$ | Intentionally retained as documented |
| **Replay Buffer Capacity** | `200,000` | Fresh instance per run |
| **Batch Size** | `256` | Uniform random mini-batch sampling |
| **Warmup Steps** | `5,000` | Random action collection before gradient updates |

---

## 4. Directory & File Isolation

To maintain strict scientific reproducibility and preserve the audit trail:

- **Legacy Results (PRESERVED / READ-ONLY):**
  - `logs/`: Original 72 training logs from the accepted paper (300-step horizon, aliased transitions). Kept for reference.
  - `models/`: Original checkpoints.
- **Corrected Camera-Ready Results (V2):**
  - `logs_v2/{algo}/R{1-4}_N{1-3}/seed_{0,42,123}/`:
    - `training_log.jsonl`: Complete 2,000-episode training log.
    - `evaluation_log.jsonl`: 20 deterministic evaluation episode trajectories.
    - `evaluation_summary.json`: Aggregated deterministic evaluation metrics (mean, std, SEM) and full metadata.
  - `models_v2/{algo}/R{1-4}_N{1-3}/seed_{0,42,123}/`:
    - `{algo}_best.pth`: Best individual episode checkpoint.
    - `{algo}_best_avg100.pth`: Best 100-episode rolling average checkpoint.
    - `{algo}_ep{N}.pth`: Periodic checkpoints every 100 episodes.

---

## 5. Result Aggregation & Reporting Rules

1. **Seed Aggregation:** For each exact condition $(\text{Algorithm}, \text{Reward}, \text{Noise})$, compute sample mean and standard deviation across seeds $(0, 42, 123)$:
   $$\bar{X} = \frac{1}{3}\sum_{s \in \{0, 42, 123\}} X_s, \quad s_X = \sqrt{\frac{1}{2}\sum_{s}(X_s - \bar{X})^2}$$
2. **Controlled Comparisons:** Comparisons between TD3 and DDPG are performed strictly **within the same reward and noise condition**.
3. **No Cross-Reward Raw Averaging:** Raw episodic rewards are never averaged across different reward configurations ($R_1 \dots R_4$) due to non-commensurate reward scales ($0.05 \to 0.45\text{ velocity weight}$).
4. **Headline Metrics:** Safety and performance metrics (Crash Rate, Lap Completion Rate, Track Distance Traveled, Survival Length) are derived exclusively from the 20-episode deterministic evaluation pipeline.

---

---

## 7. Canonical Shared Code Architecture

The repository maintains a **single canonical implementation** for execution, evaluation, metrics aggregation, and plotting. The same scripts operate on both legacy and camera-ready datasets via path parameters:

- **Batch Experiment Runner:** `python run_experiments.py --algo {td3,ddpg} --logs-dir logs_v2 --models-dir models_v2 --headless --resume`
- **Result & Table Aggregator:** `python generate_td3_ddpg_report.py --logs-dir logs_v2 --results-dir results_v2 --strict`
- **Publication Figures Engine:** `python plot_metrics.py --logs-dir logs_v2 --results-dir results_v2 --compare-algos`
- **Pre-Flight Sanity Suite:** `python preflight_checks.py`

No parallel `_v2.py` analysis scripts exist; version separation is maintained strictly at the data and filesystem level (`logs_v2/`, `models_v2/`, `results_v2/`).

---

## 8. Current Experiment Execution Status

> [!IMPORTANT]
> **CURRENT STATUS: PHASE 2 (72-RUN TRAINING EXECUTION) IS NOT STARTED.**  
> Code corrections, deterministic evaluation pipelines, canonical aggregation scripts, and 16-point pre-flight checks are fully verified and waiting for explicit execution authorization. No camera-ready training runs have been executed yet.
