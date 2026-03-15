"""
config.py – Central configuration for the TD3 self-driving car project.

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
FPS = 60                       # frames per second cap
WINDOW_TITLE = "TD3 Self-Driving Car"

# ---------------------------------------------------------------------------
# Track colours (used for programmatic track generation & on-track detection)
# ---------------------------------------------------------------------------
TRACK_ROAD_COLOR = (60, 60, 60)        # dark asphalt
TRACK_GRASS_COLOR = (34, 139, 34)      # green grass
TRACK_BORDER_COLOR = (255, 255, 255)   # white lane markings

# Pixel brightness threshold: pixels darker than this are considered "road"
ROAD_BRIGHTNESS_THRESHOLD = 100

# ---------------------------------------------------------------------------
# Car physics
# ---------------------------------------------------------------------------
CAR_MAX_SPEED = 8.0
CAR_ACCELERATION = 0.3
CAR_FRICTION = 0.05
CAR_TURN_RATE = 4.0            # degrees per frame at full steering
CAR_WIDTH = 20
CAR_HEIGHT = 40

# Starting pose (will be placed on the track automatically if possible)
CAR_START_X = 600
CAR_START_Y = 620
CAR_START_ANGLE = 0.0          # degrees, 0 = pointing right

# ---------------------------------------------------------------------------
# Sensors (raycasting)
# ---------------------------------------------------------------------------
NUM_SENSORS = 5
SENSOR_MAX_DIST = 200          # pixels
SENSOR_ANGLES = [-90, -45, 0, 45, 90]  # relative to car heading
SENSOR_COLOR = (0, 255, 255)   # cyan rays drawn on screen

# ---------------------------------------------------------------------------
# State / Action dimensions
# ---------------------------------------------------------------------------
# State: [x, y, velocity, angle, sensor_0 … sensor_N-1]
STATE_DIM = 4 + NUM_SENSORS    # 9
ACTION_DIM = 2                 # [steering, throttle]

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
REWARD_ALIVE = 0.1             # small reward each step for staying on track
REWARD_VELOCITY_SCALE = 1.0    # scale factor for velocity-based reward
REWARD_STEERING_PENALTY = 0.1  # penalty multiplied by |steering|
REWARD_CRASH = -10.0           # large penalty for leaving the track

# ---------------------------------------------------------------------------
# TD3 hyper-parameters
# ---------------------------------------------------------------------------
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99                   # discount factor
TAU = 0.005                    # soft target update rate
POLICY_DELAY = 2               # update actor every N critic updates
POLICY_NOISE = 0.2             # noise added to target policy
NOISE_CLIP = 0.5               # clipping range for target policy noise
EXPLORATION_NOISE = 0.1        # noise added during action selection

HIDDEN_DIM_1 = 256             # first hidden layer size
HIDDEN_DIM_2 = 256             # second hidden layer size

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
BUFFER_CAPACITY = 1_000_000
BATCH_SIZE = 256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MAX_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 2000
TRAINING_START = BATCH_SIZE     # start training after this many transitions
SAVE_MODEL_EVERY = 100         # save model weights every N episodes
