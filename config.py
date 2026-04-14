"""
config.py - Central configuration for the TD3 self-driving car project.

All hyperparameters, screen dimensions, physics constants, file paths,
and TD3 settings are defined here for easy tuning.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
TRACK_IMAGE_PATH = os.path.join(ASSETS_DIR, "track.png")
CAR_IMAGE_PATH = os.path.join(ASSETS_DIR, "car.png")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# Display / Pygame
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
WINDOW_TITLE = "TD3 Self-Driving Car"
RENDER_DURING_TRAINING = True  # Show visuals during local training runs

# ---------------------------------------------------------------------------
# Track geometry
# ---------------------------------------------------------------------------
TRACK_CENTER_X = SCREEN_WIDTH // 2
TRACK_CENTER_Y = SCREEN_HEIGHT // 2
TRACK_OUTER_RADIUS_X = 480
TRACK_OUTER_RADIUS_Y = 320
TRACK_INNER_RADIUS_X = 320
TRACK_INNER_RADIUS_Y = 180
FINISH_LINE_WIDTH = 6

# ---------------------------------------------------------------------------
# Track colours
# ---------------------------------------------------------------------------
TRACK_ROAD_COLOR = (60, 60, 60)
TRACK_GRASS_COLOR = (34, 139, 34)
TRACK_BORDER_COLOR = (255, 255, 255)
HUD_BG_COLOR = (0, 0, 0, 160)
HUD_TEXT_COLOR = (255, 255, 255)
SENSOR_COLOR = (0, 255, 255)
SENSOR_ENDPOINT_COLOR = (255, 50, 50)
FINISH_LINE_COLOR = (245, 245, 245)
FINISH_LINE_STRIPE_COLOR = (30, 30, 30)

# ---------------------------------------------------------------------------
# Car physics
# ---------------------------------------------------------------------------
CAR_MAX_SPEED = 8.0
CAR_ACCELERATION = 0.3
CAR_FRICTION = 0.05
CAR_TURN_RATE = 4.0
CAR_TURN_SPEED_FACTOR = 0.5  # Reduce turning at high speed: turn_angle *= (1 - speed/max_speed * factor)
CAR_WIDTH = 20
CAR_HEIGHT = 40

# Starting pose
CAR_START_X = 600
CAR_START_Y = 620
CAR_START_ANGLE = 0.0

# ---------------------------------------------------------------------------
# Sensors (raycasting)
# ---------------------------------------------------------------------------
NUM_SENSORS = 3  # Reduced from 5 for simplicity
SENSOR_MAX_DIST = 200
SENSOR_ANGLES = [-45, 0, 45]  # Simplified: front-left, front, front-right
SENSOR_NOISE_STD = 0.02  # Add Gaussian noise to sensor readings
SENSOR_RAYCAST_STEP = 2  # Check every Nth pixel for speedup (approx 1% accuracy loss)

# ---------------------------------------------------------------------------
# Research experiments
# ---------------------------------------------------------------------------
EXPERIMENT_REWARD_MODES = ("basic", "shaped", "modified")
EXPERIMENT_SENSOR_NOISE_LEVELS = (0.0, 0.02, 0.05)

# Cartesian-product experiment grid used for reproducible research runs.
EXPERIMENTS = {
    f"{reward_mode}_noise_{noise_std:.2f}": {
        "reward_mode": reward_mode,
        "sensor_noise_std": noise_std,
    }
    for reward_mode in EXPERIMENT_REWARD_MODES
    for noise_std in EXPERIMENT_SENSOR_NOISE_LEVELS
}

# ---------------------------------------------------------------------------
# State / Action dimensions
# ---------------------------------------------------------------------------
# State: (x, y, speed, angle) + sensor_distances
STATE_DIM = 4 + NUM_SENSORS
ACTION_DIM = 2

# ---------------------------------------------------------------------------
# Reward shaping (improved to avoid reward hacking)
# ---------------------------------------------------------------------------
REWARD_ALIVE = 0.05  # Small per-step bonus (reduced from 0.1)
REWARD_SPEED_BONUS = 0.15  # Bonus for moving (new)
REWARD_LAP_COMPLETION = 15.0  # Big bonus for completing a lap (new)
REWARD_STEERING_PENALTY = 0.05  # Smoother movements
REWARD_CRASH = -5.0  # Smaller penalty to encourage exploration

# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------
STUCK_SPEED_THRESHOLD = 0.15
STUCK_STEP_LIMIT = 180
MIN_LAP_STEPS = 120

# ---------------------------------------------------------------------------
# TD3 hyper-parameters
# ---------------------------------------------------------------------------
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99
TAU = 0.005
POLICY_DELAY = 2
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORATION_NOISE = 0.1
EXPLORATION_NOISE_DECAY = 0.9999  # Decay exploration noise each episode

# Network architecture
HIDDEN_DIM_1 = 64  # Reduced from 128
HIDDEN_DIM_2 = 64  # Reduced from 128

# Gradient clipping
GRADIENT_CLIP_MAX_NORM = 1.0

# Reward normalization
NORMALIZE_REWARDS = True
REWARD_NORM_UPDATE_FREQ = 100  # Update normalization stats every N episodes

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
BUFFER_CAPACITY = 200_000
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEFAULT_SEED = 42
EXPERIMENT_BASE_SEED = 1000
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 2000
TRAINING_START = 5000
SAVE_MODEL_EVERY = 100
RENDER_EVERY_EPISODES = 25

# ---------------------------------------------------------------------------
# Difficulty Levels (for future expansion)
# ---------------------------------------------------------------------------
DIFFICULTY = "normal"  # "easy", "normal", "hard"

DIFFICULTY_SETTINGS = {
    "easy": {
        "track_width": 100,  # Wider track
        "sensor_noise": 0.01,
        "friction": 0.02,
        "car_max_speed": 6.0,
    },
    "normal": {
        "track_width": 50,  # Standard
        "sensor_noise": 0.02,
        "friction": 0.05,
        "car_max_speed": 8.0,
    },
    "hard": {
        "track_width": 30,  # Narrow track
        "sensor_noise": 0.05,
        "friction": 0.10,
        "car_max_speed": 10.0,
    },
}
