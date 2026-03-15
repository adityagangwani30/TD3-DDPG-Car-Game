"""
car.py – Car physics and raycasting sensor model.

Implements a simple bicycle-model car with:
  • Position, velocity, acceleration, heading
  • Continuous steering and throttle inputs
  • N raycasting sensors returning distance to nearest off-track pixel
"""

import math
import numpy as np
import pygame

from config import (
    CAR_MAX_SPEED, CAR_ACCELERATION, CAR_FRICTION, CAR_TURN_RATE,
    CAR_WIDTH, CAR_HEIGHT,
    CAR_START_X, CAR_START_Y, CAR_START_ANGLE,
    SENSOR_ANGLES, SENSOR_MAX_DIST, SENSOR_COLOR, NUM_SENSORS,
    SCREEN_WIDTH, SCREEN_HEIGHT,
)


class Car:
    """Top-down car with simple physics and raycasting sensors."""

    def __init__(self, track_mask: np.ndarray, car_image: pygame.Surface):
        """
        Args:
            track_mask: Boolean array (W×H) where True = road pixel.
            car_image:  Pre-loaded car sprite surface.
        """
        self.track_mask = track_mask
        self.original_image = car_image
        self.image = car_image

        # Physics state
        self.x: float = CAR_START_X
        self.y: float = CAR_START_Y
        self.angle: float = CAR_START_ANGLE   # degrees, 0 = right
        self.speed: float = 0.0

        # Sensor readings (normalised 0→1)
        self.sensor_dists: list[float] = [1.0] * NUM_SENSORS
        # Raw endpoints for drawing sensor rays
        self.sensor_endpoints: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        """Reset the car to its starting pose and zero velocity."""
        self.x = CAR_START_X
        self.y = CAR_START_Y
        self.angle = CAR_START_ANGLE
        self.speed = 0.0
        self.sensor_dists = [1.0] * NUM_SENSORS
        self.sensor_endpoints = []

    # ------------------------------------------------------------------
    # Physics update
    # ------------------------------------------------------------------
    def update(self, steering: float, throttle: float):
        """Advance the car one simulation step.

        Args:
            steering: ∈ [-1, 1]  (−1 = hard left, +1 = hard right)
            throttle: ∈ [ 0, 1]  (0 = coast, 1 = full throttle)
        """
        # --- Heading update ---
        # Multiply normalised steering by the maximum turn rate (deg/frame)
        # and wrap the angle into [0, 360) to avoid accumulation issues.
        self.angle += steering * CAR_TURN_RATE
        self.angle %= 360.0

        # --- Speed update (simplified kinematic model) ---
        # Throttle adds to speed; proportional drag (Coulomb-style friction)
        # ensures speed decays naturally when throttle is released.
        # Speed is clamped to [0, CAR_MAX_SPEED] to prevent reversing.
        self.speed += throttle * CAR_ACCELERATION
        self.speed -= CAR_FRICTION * self.speed       # drag proportional to speed
        self.speed = max(0.0, min(self.speed, CAR_MAX_SPEED))

        # --- Position update (Euler integration) ---
        # Pygame's y-axis points *downward*, so we negate the sin component
        # so that angle=0 → move right, angle=90 → move up on screen.
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)   # negative because y grows downward

    # ------------------------------------------------------------------
    # Sensors (raycasting)
    # ------------------------------------------------------------------
    def cast_sensors(self):
        """Cast all sensor rays and store normalised distances."""
        self.sensor_dists = []
        self.sensor_endpoints = []
        for rel_angle in SENSOR_ANGLES:
            dist, endpoint = self._cast_ray(self.angle + rel_angle)
            self.sensor_dists.append(dist / SENSOR_MAX_DIST)   # normalise
            self.sensor_endpoints.append(endpoint)

    def _cast_ray(self, angle_deg: float):
        """Cast a single ray from the car centre and return
        (distance, endpoint) to the first off-track pixel.

        Uses a discrete ray-marching approach: step one pixel at a time along
        the ray direction and sample the Boolean road mask until we hit an
        off-track pixel or reach SENSOR_MAX_DIST.
        """
        rad = math.radians(angle_deg)
        dx = math.cos(rad)       # unit-step in x per pixel of ray length
        dy = -math.sin(rad)      # unit-step in y (negated for Pygame's y-down)

        mask_w, mask_h = self.track_mask.shape  # (W, H) – note surfarray column order

        # March along the ray one pixel at a time
        for d in range(1, SENSOR_MAX_DIST + 1):
            px = int(round(self.x + dx * d))
            py = int(round(self.y + dy * d))

            # Treat the screen boundary as an off-track wall
            if px < 0 or px >= mask_w or py < 0 or py >= mask_h:
                return d, (self.x + dx * d, self.y + dy * d)

            # mask[px, py] is True for road pixels; False triggers a hit
            if not self.track_mask[px, py]:
                return d, (px, py)

        # Ray reached max range without hitting any boundary → return full length
        end_x = self.x + dx * SENSOR_MAX_DIST
        end_y = self.y + dy * SENSOR_MAX_DIST
        return SENSOR_MAX_DIST, (end_x, end_y)

    # ------------------------------------------------------------------
    # State vector
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """Return the observation vector.

        Layout: [x_norm, y_norm, speed_norm, angle_norm, sensor_0 … sensor_N]
        """
        state = np.array([
            self.x / SCREEN_WIDTH,
            self.y / SCREEN_HEIGHT,
            self.speed / CAR_MAX_SPEED,
            self.angle / 360.0,
        ] + self.sensor_dists, dtype=np.float32)
        return state

    # ------------------------------------------------------------------
    # Collision check
    # ------------------------------------------------------------------
    def is_off_track(self) -> bool:
        """Return True if the car centre is off the road."""
        ix, iy = int(round(self.x)), int(round(self.y))
        w, h = self.track_mask.shape
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return True
        return not self.track_mask[ix, iy]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface):
        """Draw the car sprite (rotated) and sensor rays onto *surface*."""
        # Rotate sprite
        rotated = pygame.transform.rotate(self.original_image,
                                          self.angle - 90)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)

        # Draw sensor rays
        for endpoint in self.sensor_endpoints:
            pygame.draw.line(surface, SENSOR_COLOR,
                             (int(self.x), int(self.y)),
                             (int(endpoint[0]), int(endpoint[1])), 1)
            # Small circle at hit-point
            pygame.draw.circle(surface, (255, 50, 50),
                               (int(endpoint[0]), int(endpoint[1])), 3)
