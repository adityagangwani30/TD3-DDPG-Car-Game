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

# ---------------------------------------------------------------------------
# Display / Pygame
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
WINDOW_TITLE = "TD3 Self-Driving Car"

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

# ---------------------------------------------------------------------------
# Car physics
# ---------------------------------------------------------------------------
CAR_MAX_SPEED = 8.0
CAR_ACCELERATION = 0.3
CAR_FRICTION = 0.05
CAR_TURN_RATE = 4.0
CAR_WIDTH = 20
CAR_HEIGHT = 40

# Starting pose
CAR_START_X = 600
CAR_START_Y = 620
CAR_START_ANGLE = 0.0

# ---------------------------------------------------------------------------
# Sensors (raycasting)
# ---------------------------------------------------------------------------
NUM_SENSORS = 5
SENSOR_MAX_DIST = 200
SENSOR_ANGLES = [-90, -45, 0, 45, 90]
SENSOR_COLOR = (0, 255, 255)

# ---------------------------------------------------------------------------
# State / Action dimensions
# ---------------------------------------------------------------------------
STATE_DIM = 4 + NUM_SENSORS
ACTION_DIM = 2

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
REWARD_ALIVE = 0.1
REWARD_VELOCITY_SCALE = 1.0
REWARD_STEERING_PENALTY = 0.1
REWARD_CRASH = -10.0

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

HIDDEN_DIM_1 = 128
HIDDEN_DIM_2 = 128

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
BUFFER_CAPACITY = 200_000
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 2000
TRAINING_START = 5000
SAVE_MODEL_EVERY = 100
RENDER_EVERY_EPISODES = 25
