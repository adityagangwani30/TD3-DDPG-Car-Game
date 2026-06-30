# FINAL PAPER UPDATE CHECKLIST

## Overview
This checklist contains every remaining results-related update required for the IEEE research paper manuscript ([conference_101719.tex](file:///d:/Mini%20Project/td3-car-game/conference_101719.tex)) before final submission. 

* **Phase 1 Update:** (Methodology, equations, and implementation details) has already been completed.
* **Phase 2 Update:** (Results, Tables, Figures, Discussion, and Conclusion) is covered in this checklist.

This document reorganizes the analysis findings from [TO_BE_UPDATED_RESULTS.md](file:///d:/Mini%20Project/td3-car-game/TO_BE_UPDATED_RESULTS.md) into a clean, easy-to-follow guide to ensure the paper aligns with actual experiment outputs (seeds 0, 42, 123 over 2000 episodes and 300 steps).

---

## 🔴 Mandatory Updates

The team **must** perform the following updates to ensure the paper remains factually accurate:

1. **Numerical Values in Text:** Update the overall averages for rewards, crash rates, and convergence speeds in Section V.
   * **Location:** Section V.B (Learning Performance), Section V.D (Convergence), Section V.E (Crash Rates), Section V.C.3 (High Noise)
   * **Current Value:** Draft values from Phase 1 / placeholder files.
   * **New Value:** Real computed averages (see [Numerical Update Checklist](#numerical-update-checklist)).
   * **Reason:** Align text with correct recomputed statistics (excluding `seed_1`).

2. **Performance Comparison Table:** Replace the incorrect rows in TABLE II.
   * **Location:** TABLE II (PERFORMANCE COMPARISON OF TD3 AND DDPG)
   * **Current Value:** Old rows for Average Reward, Convergence Rate, and Crash Rate.
   * **New Value:** Complete corrected rows (see [Results Tables Checklist](#results-tables-checklist)).
   * **Reason:** Reflect the exact averages of the 72 training runs.

3. **Figure Discussion and Descriptions:** Rewrite the narrative surrounding Figures 1, 2, 3, and 4 to match the actual plots.
   * **Location:** Discussions of Fig. 1 (Section V.B & V.D), Fig. 2 (Section V.E), Fig. 3 (Section V.G), and Fig. 4 (Section V.J).
   * **Current Value:** Outdated analysis detailing high lap completion and incorrect reward levels.
   * **New Value:** Safety-performance trade-off descriptions and flat lap curves (see [Figure Discussion Checklist](#figure-discussion-checklist)).
   * **Reason:** The regenerated figures show that lap completion was near zero under the 300-step limit.

4. **Rankings and Best/Worst Configurations:** Update the configuration comparisons.
   * **Location:** Results Section V.C.1 and V.C.3.
   * **Current Value:** Implicit or incorrect rankings of reward structures.
   * **New Value:** TD3 `R4_N2` (best TD3 final reward), DDPG `R4_N3` (overall best reward), TD3 `R3_N2` (safest configuration) (see [Rankings Checklist](#rankings-checklist)).
   * **Reason:** Corrected rankings according to final seed-aggregated data.

5. **Discussion Sentences:** Replace the 9 specific results-related sentences identified as incorrect.
   * **Location:** Section V (Results and Discussion) text.
   * **Current Value:** 9 sentences referencing old placeholder values.
   * **New Value:** Exact corrected sentences (see [Discussion Checklist](#discussion-checklist)).
   * **Reason:** Remove all incorrect factual claims.

---

## 🟡 Optional Updates

The following tables and summaries are **optional** improvements. They should only be added if the manuscript has enough page space. Adding these tables may exceed the IEEE page limits:

1. **Complete TD3 Results Table:** Detailed breakdown of all 12 TD3 configurations across the 3 seeds.
   * *Status:* Optional (see Table 1 in [TO_BE_UPDATED_RESULTS.md](file:///d:/Mini%20Project/td3-car-game/TO_BE_UPDATED_RESULTS.md#2-td3-results-table)).
2. **Complete DDPG Results Table:** Detailed breakdown of all 12 DDPG configurations across the 3 seeds.
   * *Status:* Optional (see Table 2 in [TO_BE_UPDATED_RESULTS.md](file:///d:/Mini%20Project/td3-car-game/TO_BE_UPDATED_RESULTS.md#3-ddpg-results-table)).
3. **Cross-Algorithm Comparative Summary Table:** Replaces or supplements Table II with a compact summary of averages, peaks, and weakest points.
   * *Status:* Optional (see Table 4 in [TO_BE_UPDATED_RESULTS.md](file:///d:/Mini%20Project/td3-car-game/TO_BE_UPDATED_RESULTS.md#4-cross-algorithm-comparative-summary-table)).

---

## ❌ Do NOT Change

To preserve the integrity of the peer-reviewed sections, the following parts of the paper **must remain untouched**:

* **Abstract** (already corrected and aligned)
* **Introduction**
* **Literature Review** / **Related Work**
* **Equations** (Markov formulation, physics equations, and update rules)
* **Methodology** (bicycle dynamics model, raycasting sensor configurations)
* **Hyperparameters** (Table I settings like learning rates, capacities, delays)
* **Citations** / **References**
* **Figures themselves** (plots have been regenerated; do not run additional scripts)
* **Writing style** / **Formatting**

---

## Numerical Update Checklist

Verify and update the following numerical values in the text of [conference_101719.tex](file:///d:/Mini%20Project/td3-car-game/conference_101719.tex):

| Section | Current | Correct | Reason |
|---|---|---|---|
| **V.B (Learning Performance)** | DDPG: `45.80 ± 11.34`<br>TD3: `39.79 ± 8.32` | **Final Reward Averages:**<br>DDPG: `17.74 ± 15.58`<br>TD3: `14.79 ± 14.39`<br><br>**Last-100 Reward Averages:**<br>DDPG: `17.07 ± 15.43`<br>TD3: `13.23 ± 11.49` | Recomputed using seeds 0, 42, and 123 over 2000 episodes and 300 steps. |
| **V.C.3 (High Noise (N3))** | DDPG shows `70.99 ± 38.47`<br>TD3 shows `54.95 ± 29.95` | **Last-100 Reward Averages (R4_N3):**<br>DDPG R4_N3: `51.31 ± 20.58`<br>TD3 R4_N3: `38.32 ± 9.45`<br><br>**Final Reward Averages (R4_N3):**<br>DDPG R4_N3: `57.44 ± 32.15`<br>TD3 R4_N3: `25.94 ± 14.31` | Corrected using actual performance statistics for the high-noise R4 configuration. |
| **V.D (Convergence)** | TD3: `episode 61`<br>DDPG: `episode 58` | TD3: `episode 75` (average `75.14`)<br>DDPG: `episode 71` (average `70.72`) | Corrected overall average convergence episodes. |
| **V.E (Crash Rates)** | TD3: `67.31 ± 14.97%`<br>DDPG: `71.42 ± 6.80%` | TD3: `86.75% ± 20.53%` (collisions: `0.868 ± 0.205`)<br>DDPG: `91.08% ± 9.32%` (collisions: `0.911 ± 0.093`) | Corrected overall average crash rate statistics. |

---

## Results Tables Checklist

Update TABLE II in [conference_101719.tex](file:///d:/Mini%20Project/td3-car-game/conference_101719.tex) with the recomputed values:

| Table | Replace | Reason |
|---|---|---|
| **TABLE II (Performance Comparison)** | **Old Rows:**<br>`Average Reward 45.80±11.34 39.79±8.32`<br>`Convergence Rate ~58 episodes ~61 episodes`<br>`Crash Rate 71.42±6.80% 67.31±14.97%`<br><br>**New Rows (Final Reward Averages):**<br>`Average Final Reward 17.74 ± 15.58 14.79 ± 14.39`<br>`Convergence Rate ~71 episodes ~75 episodes`<br>`Crash Rate 91.08% ± 9.32% 86.75% ± 20.53%`<br><br>**New Rows (Last-100 Reward Averages):**<br>`Average Last-100 Reward 17.07 ± 15.43 13.23 ± 11.49`<br>`Convergence Rate ~71 episodes ~75 episodes`<br>`Crash Rate 91.08% ± 9.32% 86.75% ± 20.53%` | Incorporates the correct averages computed strictly from seeds 0, 42, and 123. |

---

## Figure Discussion Checklist

Update the narrative surrounding figures to accurately describe the behavior shown in the regenerated plots:

| Figure | Old discussion | New discussion | Reason |
|---|---|---|---|
| **Fig. 1 (Reward curves)** | Focuses on DDPG achieving `45.80±11.34` and TD3 achieving `39.79±8.32` and claims convergence around episode 58 and 61. | Describe that both algorithms experience an early drop due to exploration before climbing steadily. DDPG reaches a higher reward ceiling in shaped settings (Final: `17.74±15.58`, Last-100: `17.07±15.43`), while TD3 is more conservative but exhibits lower overall reward variance (Final: `14.79±14.39`, Last-100: `13.23±11.49`). Convergence occurs around episode 71 for DDPG and episode 75 for TD3. | Regenerated figure represents training over 2000 episodes and 300 steps. |
| **Fig. 2 (Crash rates)** | Focuses on average crash rates of DDPG (`71.42±6.80%`) and TD3 (`67.31±14.97%`). | Explain that crash rates remain high for both algorithms throughout the entire training process. The curves do not converge to a zero-risk regime. TD3 averages to a crash rate of `86.75% ± 20.53%` (collisions: `0.868`), which is slightly safer than DDPG's crash rate of `91.08% ± 9.32%` (collisions: `0.911`). | Corrected statistics show crash rates are significantly higher than previously drafted. |
| **Fig. 3 (Trade-off plane)** | Mentions that points toward the upper-left represent favorable combinations. | Explain that there is no single algorithm that dominates the high-reward, low-crash region. DDPG `R4_N3` achieves the highest reward (`57.44` final reward) but has a `78.7%` crash rate. TD3 `R3_N2` is the safest configuration, reducing the crash rate to `30.7%` (with seeds 123 and 42 achieving 0% crash rate in the final 100 episodes) but yields a lower final reward (`9.67`). | Reflects the fundamental safety-performance trade-off. |
| **Fig. 4 (Laps completed)** | States that TD3 maintains more consistent lap counts after training, while DDPG exhibits greater variability. | Explain that the lap completion curves remain flat and close to zero for both algorithms. Because the vehicle rarely survived long enough to complete a lap under the 300-step limit, the overall lap completion average is `0.002` for TD3 and `0.006` for DDPG. Reward improvements are driven by forward progress and speed before collision, rather than full lap completions. | In the actual seeds, lap completion was extremely rare. The previous claim of consistent lap completion is incorrect. |

---

## Rankings Checklist

Verify and update the following ranking statements in the manuscript:

* **Best TD3 Configuration:**
  * *Correct Statement:* By mean final reward, the best TD3 configuration is `R4_N2` (50.56 ± 44.48). By mean last-100 reward, the best TD3 configuration is `R4_N3` (38.32 ± 9.45). Under clean conditions (N1), the best TD3 configuration is `R2_N1` (23.83 ± 13.87).
  * *Evidence:* Logs show TD3 `R4_N2` final reward = 50.56, `R4_N3` last-100 reward = 38.32, and `R2_N1` final reward = 23.83.
* **Best DDPG Configuration:**
  * *Correct Statement:* The best DDPG configuration is `R4_N3` (mean final reward: 57.44 ± 32.15; mean last-100 reward: 51.31 ± 20.58).
  * *Evidence:* Logs show DDPG `R4_N3` final reward = 57.44, last-100 reward = 51.31.
* **Overall Best Configuration:**
  * *Correct Statement:* The overall best configuration by reward optimization is DDPG `R4_N3` (mean final reward of 57.44 and mean last-100 reward of 51.31).
  * *Evidence:* DDPG `R4_N3` outperforms all other 23 configurations on both final and last-100 reward.
* **Safest Configuration:**
  * *Correct Statement:* The safest configuration overall is TD3 `R3_N2` with a last-100 crash rate of 30.7% (collisions: 0.307 ± 0.434).
  * *Evidence:* TD3 `R3_N2` has the lowest collision count, with seed 123 and seed 42 achieving a 0.00% crash rate in the final 100 episodes.
* **Worst Configuration:**
  * *Correct Statement:* The worst configuration is TD3 `R1_N3` (mean final reward: -2.75 ± 0.40; mean last-100 reward: -2.66 ± 0.45), representing complete failure to learn.
  * *Evidence:* TD3 `R1_N3` has the lowest final reward (-2.75) and a 100.0% crash rate.
* **Fastest Convergence:**
  * *Correct Statement:* DDPG converges faster on average (70.72 episodes) compared to TD3 (75.14 episodes). The fastest individual convergence occurs in TD3 `R1_N1` (32.33 episodes) and TD3 `R4_N1` (33.00 episodes), though R1 fails to learn driving.
  * *Evidence:* Mean convergence episodes are 70.72 (DDPG) and 75.14 (TD3).
* **Highest Reward:**
  * *Correct Statement:* The highest reward is achieved by DDPG `R4_N3` (mean final reward: 57.44; best observed single-episode reward: 117.99).
  * *Evidence:* Logs show DDPG `R4_N3` achieves a mean final reward of 57.44 and a maximum single-episode reward of 117.99.
* **Lowest Crash Rate:**
  * *Correct Statement:* The lowest crash rate is achieved by TD3 `R3_N2` (30.7% crash rate). Under DDPG, the lowest crash rate is achieved by `R2_N1` (76.0% crash rate).
  * *Evidence:* Logs show TD3 `R3_N2` averages 0.307 collisions, and DDPG `R2_N1` averages 0.760 collisions.

---

## Discussion Checklist

Update the following discussion sentences in Section V text to ensure factual alignment with the logs:

1. **Current:**
   > "DDPG achieves higher average rewards (45.80±11.34) compared to TD3 (39.79±8.32)."
   **↓**
   **Replace With:**
   > "DDPG achieves higher average final rewards (17.74±15.58) and average last-100 rewards (17.07±15.43) compared to TD3, which yields an average final reward of 14.79±14.39 and average last-100 reward of 13.23±11.49."
   **↓**
   **Reason:** Recomputed using averages from the three seeds.

2. **Current:**
   > "As DDPG has significantly higher variance, it exhibits a tendency for greater variation in per-episode performance. Higher average rewards and higher instability result as a direct indication of the performance–consistency trade-off."
   **↓**
   **Replace With:**
   > "DDPG exhibits higher variance in performance across configurations (standard deviation of 15.58 on final reward vs. 14.39 for TD3). Higher average rewards and higher instability result as a direct indication of the performance–consistency trade-off."
   **↓**
   **Reason:** Updated to reflect the actual variation across the 12 configurations.

3. **Current:**
   > "Low Noise (N1): Both algorithms find it flexible with minimal observation noise. R3 produces the best rewards in this case."
   **↓**
   **Replace With:**
   > "Low Noise (N1): Both algorithms learn successfully with minimal observation noise, with DDPG R3_N1 yielding the highest reward under DDPG (mean final reward: 28.02), while TD3 R2_N1 achieves the best final reward under TD3 (mean final reward: 23.83)."
   **↓**
   **Reason:** Updated to align with actual rankings under clean conditions.

4. **Current:**
   > "At very high noise levels, both algorithms start to struggle. R4 yields the highest rewards in this setting. DDPG shows 70.99±38.47 while TD3 shows 54.95±29.95."
   **↓**
   **Replace With:**
   > "At very high noise levels, both algorithms benefit from shaped rewards, with the R4 configuration yielding the highest performance. Specifically, DDPG R4_N3 achieves a mean last-100 reward of 51.31±20.58 (mean final reward: 57.44±32.15), while TD3 R4_N3 yields a mean last-100 reward of 38.32±9.45 (mean final reward: 25.94±14.31)."
   **↓**
   **Reason:** Corrected using actual data for R4_N3 under seeds 0, 42, and 123.

5. **Current:**
   > "TD3 stabilizes around episode 61, while DDPG stabilizes around episode 58."
   **↓**
   **Replace With:**
   > "TD3 stabilizes around episode 75 on average, while DDPG stabilizes around episode 71."
   **↓**
   **Reason:** Corrected using actual overall average convergence episodes (75.14 for TD3 and 70.72 for DDPG).

6. **Current:**
   > "Fig. 1 shows the average reward of TD3 and DDPG over the training episodes. Initially, both methods drop due to exploration, and then they stabilize and eventually converge to similar performance levels. TD3 appears slightly more stable with lower variability, while DDPG shows more variations."
   **↓**
   **Replace With:**
   > "Fig. 1 shows the average reward of TD3 and DDPG over the training episodes. Initially, both methods drop due to exploration, and then they stabilize and learn steadily. DDPG generally converges to higher late-stage rewards in shaped configurations (averaging 17.07 last-100 reward), whereas TD3 is more conservative but exhibits lower overall reward variance."
   **↓**
   **Reason:** Updated to align with the actual learning curves showing DDPG's reward advantage and TD3's lower variance.

7. **Current:**
   > "The crash rate data confirms that TD3 averages to 67.31±14.97% while DDPG averages to 71.42±6.80%, as shown in Fig. 2."
   **↓**
   **Replace With:**
   > "The crash rate data confirms that TD3 averages to 86.75%±20.53% while DDPG averages to 91.08%±9.32%, as shown in Fig. 2."
   **↓**
   **Reason:** Corrected using the actual overall average crash rates across all experiments.

8. **Current:**
   > "Fig. 4 depicts lap completion rates over all conditions. TD3 maintains more consistent lap counts after training, while DDPG exhibits greater variability, which is consistent with its reward instability."
   **↓**
   **Replace With:**
   > "Fig. 4 depicts lap completion rates over all conditions. Both TD3 and DDPG curves remain flat and close to zero, reflecting the fact that the agent rarely completed full laps under the 300-step limit due to off-track collisions."
   **↓**
   **Reason:** Updated to reflect that lap completion was extremely rare in the actual experiments, and the curves are near zero for both.

9. **Current:**
   > "DDPG reaches higher reward peaks but with more instability and more crashes. A high cumulative reward does not mean that the agent is driving well. It may be due to the agent looking at each metric on its own. Strong behavior in intelligent systems arises from balancing multiple objectives, and not from maximizing any single one."
   **↓**
   **Replace With:**
   > "DDPG reaches higher reward peaks (especially in R4_N3 with 57.44 mean final reward) but with more crashes (91.1% overall average). A high cumulative reward does not mean that the agent is driving safely; for example, DDPG R4_N2 achieves good reward but retains a 99.3% crash rate. Strong behavior in intelligent systems arises from balancing multiple objectives, and not from maximizing any single one."
   **↓**
   **Reason:** Added concrete examples from actual experiment outputs.

---

## Conclusion Checklist

Verify the conclusion paragraph in Section VI:

✅ No conclusion changes required.

*Note: The qualitative conclusion remains factually accurate with the recomputed data. TD3 remains preferable for safety-critical settings where stability is prioritized, and DDPG can be preferred when peak performance is prioritized and safety is monitored separately.*

---

## Statistics Verification

Verify that the updated results represent the following experimental scope:

* **Seeds:** 0, 42, 123 (seed 1 excluded).
* **Episodes:** 2000 episodes per training run.
* **Max Steps:** 300 steps per episode.
* **Scope:** 24 configuration–algorithm combinations.
* **Runs:** 72 independent training runs.

---

## Final Manual Checklist

The team should tick off the following items while editing the LaTeX manuscript:

- [ ] Update average reward, crash rate, and convergence speed in TABLE II.
- [ ] Correct the overall average reward numbers in Section V.B (Learning Performance) text.
- [ ] Correct the N3 R4_N3 reward values in Section V.C.3 (High Noise) text.
- [ ] Correct the average convergence episodes in Section V.D (Convergence) text.
- [ ] Correct the average crash rate percentages in Section V.E (Stability and Safety) text.
- [ ] Rewrite the Fig. 1 discussion (Section V.B & V.D) to describe the actual 2000-episode trajectories.
- [ ] Rewrite the Fig. 2 discussion (Section V.E) to explain the recomputed crash statistics.
- [ ] Update the Fig. 3 discussion (Section V.G) to highlight the R3_N2 and R4_N3 trade-off values.
- [ ] Rewrite the Fig. 4 discussion (Section V.J) to reflect near-zero lap completion.
- [ ] Update rankings (best/worst configurations) in Section V.C.1 and V.C.3.
- [ ] Correct the 9 results-related sentences in the text of Section V.
- [ ] Double-check Table I (implementation settings) is left untouched.
- [ ] Double-check Section VI (Conclusion) requires no modifications.
- [ ] Compile LaTeX source code `conference_101719.tex` using `pdflatex` to generate the final paper PDF.
- [ ] Verify that all references and citations in the paper remain consistent.
