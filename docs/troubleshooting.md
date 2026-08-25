# Troubleshooting

This document covers common issues and their solutions when working with this project.

---

## Pygame Window Not Opening

**Possible cause:** Pygame cannot find a display server (common on headless servers, SSH sessions, or containers).

**Suggested fix:**
```bash
# Run in headless mode
python main.py --algo td3 --mode train --headless
```

The `--headless` flag sets `SDL_VIDEODRIVER=dummy` and `SDL_AUDIODRIVER=dummy`, allowing Pygame to run without a display. Training and logging work identically in headless mode.

---

## Running in Headless Mode

**Possible cause:** You are on a remote server, cloud VM, or Docker container without a GUI.

**Suggested fix:** The project auto-detects headless environments via `detect_headless_environment()` in `utils.py`. This checks for:
- Google Colab (`COLAB_RELEASE_TAG`)
- Kaggle (`KAGGLE_DATA_MOUNT_DIR`)
- Missing `DISPLAY` variable on Linux
- SSH sessions

If auto-detection fails, force headless mode with the `--headless` flag.

---

## Google Colab Display Limitations

**Possible cause:** Colab runs in a headless Linux container. Pygame windows cannot be displayed inline.

**Suggested fix:**
- Use the provided Colab notebooks (`colab_demo_td3.ipynb`, `colab_demo_ddpg.ipynb`, `colab_demo_both.ipynb`) — they handle headless configuration automatically
- Training runs in headless mode; plots are saved as PNG files and can be viewed inline
- For preview frames, use `env.save_frame(output_path)` to capture screenshots to disk

---

## Missing Dependencies

**Possible cause:** Required Python packages are not installed.

**Suggested fix:**
```bash
pip install -r requirements.txt
```

Required packages:
- `torch` — PyTorch for neural networks
- `pygame` — Environment rendering and physics
- `numpy` — Numerical computations
- `matplotlib` — Plot generation
- `reportlab` — PDF report generation

If you encounter version conflicts, create a fresh virtual environment:
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

---

## Plots Not Generated

**Possible cause:** No training logs exist, or the log directory structure does not match what `plot_metrics.py` expects.

**Suggested fix:**

1. Verify logs exist in the target logs directory:
   ```bash
   # Check if log files are present in logs_v2 (or legacy logs)
   dir logs_v2\td3\R1_N1\seed_0\training_log.jsonl    # Windows
   ls logs_v2/td3/R1_N1/seed_0/training_log.jsonl    # Linux/Mac
   ```

2. Run the plot command with the configured logs and results directories:
   ```bash
   python plot_metrics.py --logs-dir logs_v2 --results-dir results_v2 --compare-algos
   ```

3. Run the automated pre-flight sanity check to verify your environment setup:
   ```bash
   python preflight_checks.py
   ```

---

## ZIP File Not Downloading in Colab

**Possible cause:** The results directory was not properly zipped, or the file path is incorrect.

**Suggested fix:**
- Ensure the results directory exists and contains files before zipping
- Use the download cell provided in the Colab notebooks
- If the notebook does not include a download cell, use:
  ```python
  import shutil
  shutil.make_archive('results', 'zip', '.', 'results')
  from google.colab import files
  files.download('results.zip')
  ```

---

## Results Folder Structure Mismatch

**Possible cause:** Experiments were run with different settings or partially completed, creating an inconsistent directory structure.

**Suggested fix:**
- Ensure experiments are run with `run_experiments.py`, which enforces the expected directory structure:
  ```
  logs/{algo}/{experiment_tag}/seed_{seed}/training_log.jsonl
  models/{algo}/{experiment_tag}/seed_{seed}/{algo}_best.pth
  ```
- Use `--resume` to continue interrupted experiments without recreating directories
- If the structure is corrupted, check which experiments completed by inspecting the log files

---

## Model Not Loading

**Possible cause:** The checkpoint file is missing, corrupted, or was saved with a different network architecture.

**Suggested fix:**

1. Verify the checkpoint exists:
   ```bash
   ls models/td3/td3_best.pth          # Linux/Mac
   dir models\td3\td3_best.pth         # Windows
   ```

2. If the file is missing, check alternative checkpoint names:
   - `{algo}_best.pth`
   - `{algo}_best_avg100.pth`
   - `{algo}_ep{N}.pth` (periodic saves)

3. If the checkpoint was saved with a different architecture (e.g., different hidden layer sizes), it will fail to load. Ensure `config.py` matches the settings used during training.

4. Check for the experiment-prefixed version:
   ```
   models/{algo}/{tag}/seed_{seed}/{experiment_id}_{algo}_best.pth
   ```

---

## Training Taking Too Long

**Possible cause:** Training 2,000 episodes with rendering enabled is slow, especially without a GPU.

**Suggested fix:**

1. **Use headless mode** — rendering is the biggest bottleneck:
   ```bash
   python main.py --algo td3 --mode train --headless
   ```

2. **Reduce episode count** for testing:
   ```bash
   python main.py --algo td3 --mode train --max-episodes 200 --headless
   ```

3. **Use GPU** if available — the code automatically detects CUDA:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Use Colab** for free GPU access — the notebooks are pre-configured for cloud training.

5. **Run experiments in batches** using `--start-index` and `--max-experiments`:
   ```bash
   python run_experiments.py --algo td3 --start-index 0 --max-experiments 6 --headless
   python run_experiments.py --algo td3 --start-index 6 --max-experiments 6 --headless
   ```

---

## Track or Car Image Not Appearing

**Possible cause:** The `assets/` directory is missing or the generated images are corrupted.

**Suggested fix:** The track and car images are programmatically generated at startup by `ensure_assets_exist()` in `utils.py`. If they are missing:

1. Delete the `assets/` directory (if it exists with corrupted files)
2. Run any training or demo command — assets will be regenerated automatically

---

## Agent Not Learning (Reward Stays Flat)

**Possible cause:** Several factors can prevent learning:

1. **Not enough training steps:** The replay buffer needs 5,000 transitions before training begins. The agent explores randomly until then.
2. **Too few episodes:** Some configurations need 500+ episodes to show improvement.
3. **Basic reward mode (R1):** R1 provides minimal learning signal. The agent may learn to survive without racing. Try R2 or R3 for more informative rewards.
4. **High noise (N3):** Heavy sensor noise can prevent learning entirely in some configurations.

**Suggested fix:** Check the JSONL log file for the `replay_buffer_size` field to confirm training has started, and monitor `exploration_noise` to ensure it is decaying.

---

## Further Reading

- [training_and_evaluation.md](training_and_evaluation.md) — Detailed commands and options
- [project_architecture.md](project_architecture.md) — Understanding the code structure
- [reproducibility.md](reproducibility.md) — Why results may differ between runs
