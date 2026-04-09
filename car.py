"""
car.py - Car physics and raycasting sensor model.

Implements a simple bicycle-model car with:
  - Position, velocity, acceleration, heading
  - Continuous steering and throttle inputs
  - N raycasting sensors returning distance to nearest off-track pixel
"""

import math

import numpy as np
import pygame

from config import (
    CAR_ACCELERATION,
    CAR_FRICTION,
    CAR_HEIGHT,
    CAR_MAX_SPEED,
    CAR_START_ANGLE,
    CAR_START_X,
    CAR_START_Y,
    CAR_TURN_RATE,
    CAR_WIDTH,
    NUM_SENSORS,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SENSOR_ANGLES,
    SENSOR_COLOR,
    SENSOR_MAX_DIST,
)


class Car:
    """Top-down car with simple physics and raycasting sensors."""

    def __init__(self, track_mask: np.ndarray, car_image: pygame.Surface):
        self.track_mask = track_mask
        self.original_image = car_image
        self.image = car_image

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
        self.angle = (self.angle + steering * CAR_TURN_RATE) % 360.0

        self.speed += throttle * CAR_ACCELERATION
        self.speed -= CAR_FRICTION * self.speed
        self.speed = max(0.0, min(self.speed, CAR_MAX_SPEED))

        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y -= self.speed * math.sin(rad)

    def cast_sensors(self):
        """Cast all sensor rays and store normalised distances."""
        self.sensor_dists = []
        self.sensor_endpoints = []
        for rel_angle in SENSOR_ANGLES:
            dist, endpoint = self._cast_ray(self.angle + rel_angle)
            self.sensor_dists.append(dist / SENSOR_MAX_DIST)
            self.sensor_endpoints.append(endpoint)

    def _cast_ray(self, angle_deg: float):
        """Cast a single ray from the car centre."""
        rad = math.radians(angle_deg)
        dx = math.cos(rad)
        dy = -math.sin(rad)

        mask_w, mask_h = self.track_mask.shape
        for dist in range(1, SENSOR_MAX_DIST + 1):
            px = int(round(self.x + dx * dist))
            py = int(round(self.y + dy * dist))

            if px < 0 or px >= mask_w or py < 0 or py >= mask_h:
                return dist, (self.x + dx * dist, self.y + dy * dist)

            if not self.track_mask[px, py]:
                return dist, (px, py)

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
                (255, 50, 50),
                (int(endpoint[0]), int(endpoint[1])),
                3,
            )
