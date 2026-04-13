"""
car.py - Car physics and raycasting sensor model.

Implements a simple bicycle-model car with:
  - Position, velocity, acceleration, heading
  - Continuous steering and throttle inputs (forward motion only)
  - N raycasting sensors returning distance to nearest off-track pixel
  - Speed-dependent steering and sensor noise
"""

import math

import numpy as np
import pygame

from config import (
    CAR_ACCELERATION,
    CAR_FRICTION,
    CAR_MAX_SPEED,
    CAR_START_ANGLE,
    CAR_START_X,
    CAR_START_Y,
    CAR_TURN_RATE,
    CAR_TURN_SPEED_FACTOR,
    NUM_SENSORS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SENSOR_ANGLES,
    SENSOR_COLOR,
    SENSOR_ENDPOINT_COLOR,
    SENSOR_MAX_DIST,
    SENSOR_NOISE_STD,
    SENSOR_RAYCAST_STEP,
)


class Car:
    """Top-down car with simple physics, speed-dependent steering, and raycasting sensors."""

    def __init__(self, track_mask: np.ndarray, car_image: pygame.Surface):
        self.track_mask = track_mask
        self.original_image = car_image

        self.x = CAR_START_X
        self.y = CAR_START_Y
        self.angle = CAR_START_ANGLE
        self.speed = 0.0

        self.sensor_dists: list[float] = [1.0] * NUM_SENSORS
        self.sensor_endpoints: list[tuple[float, float]] = []

    def reset(self):
        """Reset the car to its starting pose and zero velocity."""
        self.x = CAR_START_X
        self.y = CAR_START_Y
        self.angle = CAR_START_ANGLE
        self.speed = 0.0
        self.sensor_dists = [1.0] * NUM_SENSORS
        self.sensor_endpoints = []

    def update(self, steering: float, throttle: float):
        """Advance the car one simulation step."""
        # Speed-dependent steering: turn less at high speed (like real cars)
        speed_factor = 1.0 - (self.speed / CAR_MAX_SPEED) * CAR_TURN_SPEED_FACTOR
        adjusted_steering = steering * speed_factor
        self.angle = (self.angle + adjusted_steering * CAR_TURN_RATE) % 360.0

        # Forward motion only (throttle range [0, 1] means acceleration)
        self.speed += throttle * CAR_ACCELERATION
        self.speed -= CAR_FRICTION * self.speed
        self.speed = max(0.0, min(self.speed, CAR_MAX_SPEED))

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

    def cast_sensors(self):
        """Cast all sensor rays and store normalised distances with noise."""
        self.sensor_dists = []
        self.sensor_endpoints = []
        for rel_angle in SENSOR_ANGLES:
            dist, endpoint = self._cast_ray(self.angle + rel_angle)
            # Normalize distance
            normalized_dist = dist / SENSOR_MAX_DIST
            # Add sensor noise
            if SENSOR_NOISE_STD > 0:
                noise = np.random.normal(0, SENSOR_NOISE_STD)
                normalized_dist = np.clip(normalized_dist + noise, 0.0, 1.0)
            self.sensor_dists.append(normalized_dist)
            self.sensor_endpoints.append(endpoint)

    def _cast_ray(self, angle_deg: float):
        """Cast a single ray from the car centre with ray-step optimization."""
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = -math.sin(rad)

        mask_w, mask_h = self.track_mask.shape
        # Step through ray in increments for faster raycasting (minimal accuracy loss)
        for dist in range(SENSOR_RAYCAST_STEP, SENSOR_MAX_DIST + 1, SENSOR_RAYCAST_STEP):
            px = int(round(self.x + dx * dist))
            py = int(round(self.y + dy * dist))

            # Treat out-of-bounds as obstacles (prevents wrapping)
            if px < 0 or px >= mask_w or py < 0 or py >= mask_h:
                return dist, (self.x + dx * dist, self.y + dy * dist)

            # Check if we hit off-track area
            if not self.track_mask[px, py]:
                return dist, (px, py)

        # No collision found, return max distance
        end_x = self.x + dx * SENSOR_MAX_DIST
        end_y = self.y + dy * SENSOR_MAX_DIST
        return SENSOR_MAX_DIST, (end_x, end_y)

    def get_state(self) -> np.ndarray:
        """Return the observation vector."""
        return np.array(
            [
                self.x / SCREEN_WIDTH,
                self.y / SCREEN_HEIGHT,
                self.speed / CAR_MAX_SPEED,
                self.angle / 360.0,
            ]
            + self.sensor_dists,
            dtype=np.float32,
        )

    def is_off_track(self) -> bool:
        """Return True if the car centre leaves the road."""
        mask_w, mask_h = self.track_mask.shape
        ix, iy = int(round(self.x)), int(round(self.y))
        if ix < 0 or ix >= mask_w or iy < 0 or iy >= mask_h:
            return True
        return not self.track_mask[ix, iy]

    def draw(self, surface: pygame.Surface):
        """Draw the rotated car sprite and sensor rays."""
        rotated = pygame.transform.rotate(self.original_image, self.angle - 90)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)

        for endpoint in self.sensor_endpoints:
            pygame.draw.line(
                surface,
                SENSOR_COLOR,
                (int(self.x), int(self.y)),
                (int(endpoint[0]), int(endpoint[1])),
                1,
            )
            pygame.draw.circle(
                surface,
                SENSOR_ENDPOINT_COLOR,
                (int(endpoint[0]), int(endpoint[1])),
                3,
            )
