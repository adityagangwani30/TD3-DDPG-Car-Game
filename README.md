# TD3 Self-Driving Car
# TD3 Self-Driving Car - Enhanced Edition

A reinforcement learning project where a car learns to drive around an oval track using TD3 (Twin Delayed DDPG), PyTorch, and Pygame.

## 🚀 What's New?

This enhanced version includes major improvements:
- ✅ **Improved Reward Function** - Lap completion bonuses, reduced crash penalty, smoother steering incentives
- ✅ **Speed-Dependent Steering** - Realistic turning behavior (turn less at high speed)
- ✅ **Sensor Noise** - More realistic and robust sensor readings
- ✅ **Exploration Noise Decay** - Automatic reduction of exploration over episodes
- ✅ **Gradient Clipping** - Stable training with bounded gradients
- ✅ **Comprehensive Metrics** - Real-time tracking of training progress
- ✅ **Evaluation Mode** - Test trained agents without exploration noise
- ✅ **Model Comparison** - Evaluate multiple checkpoints
- ✅ **Visualization** - Plot training metrics and progress
- ✅ **Difficulty Settings** - Easy/normal/hard modes for future expansion
- ✅ **Reduced Complexity** - Simplified sensor count (5→3), network size (128→64)

## 📋 Project Structure

```text
td3-car-game/
├── main.py                  # Entry point (train or eval)
├── config.py                # All hyperparameters and settings
├── car.py                   # Car physics, sensors, raycasting
├── environment.py           # RL environment, rewards, rendering
├── lap_timer.py             # Lap timing and finish line detection
├── td3_agent.py             # TD3 agent, actor, critic networks
├── replay_buffer.py         # Experience replay buffer
├── train.py                 # Training loop and evaluation function
├── metrics_tracker.py       # Comprehensive metrics tracking
├── plot_metrics.py          # Visualization of training progress
├── eval_models.py           # Model comparison and evaluation
├── utils.py                 # Asset generation and helpers
├── requirements.txt         # Python dependencies
└── assets/, models/, logs/  # Generated during runtime
```

## 🎮 Observation & Action Spaces

**State (9 dimensions)**
- Position: (x, y) normalized to [0, 1]
- Speed: normalized to [0, 1] (max = 8.0)
- Angle: normalized to [0, 1]
- **3 sensors**: front-left, front, front-right with noise

**Action (2 continuous)**
- Steering: [-1, 1] with speed-dependent turning
- Throttle: [0, 1] (forward motion only)

## 💾 Installation

```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Train
```bash
python main.py

## Evaluate
```bash
python main.py --mode eval --checkpoint models/td3_best.pth
```

## Compare Models
```bash
python eval_models.py --episodes 20
```

## Plot Metrics
```bash
python plot_metrics.py
```

## 🔧 Configuration

Edit `config.py` to tune:
- Training parameters (episodes, batch size, learning rates)
- Physics (speed, friction, steering)
- Rewards (bonuses and penalties)
- Sensors (noise, angles, count)
- Rendering (enable/disable for speed)

## 📊 Training Progress

Metrics logged to `logs/training_log.jsonl` include:
- Episode reward and moving average
- Crashes, laps completed, lap times
- Speed statistics and steering smoothness
- Network losses and buffer utilization

Monitor training with:
```bash
python plot_metrics.py  # Generate plots
tail -f logs/training_log.jsonl  # Watch live
```

## 🐛 Troubleshooting

**Slow training**: Set `RENDER_DURING_TRAINING = False` in config

**Agent doesn't learn**: Check `TRAINING_START`, `BATCH_SIZE`, and reward function

**Frequent crashes**: Reduce `CAR_MAX_SPEED` or increase steering penalty

## 📚 Key Concepts

- **TD3**: Twin Delayed DDPG for off-policy continuous control
- **Replay Buffer**: Store and sample past experiences
- **Target Networks**: Separate networks for stable learning
- **Exploration Decay**: Gradually reduce randomness during training

## 🎬 Expected Progress

| Episode | Avg Reward | Laps | Crashes | Status |
|---------|-----------|------|---------|--------|
| 100     | ~0        | 0    | 90%     | Random |
| 500     | 20-50     | 1-2  | 50%     | Learning |
| 1500    | 100-130   | 8-10 | 20%     | Good |
| 2500+   | 150+      | 12+  | <10%    | Excellent |

## 📝 License

MIT - Use freely for learning!

## Notes

- The TD3 core logic is standard and intentionally kept separate from the environment logic.
- Assets are generated automatically if they are missing.
- Best and last lap times persist across episode resets for display.
