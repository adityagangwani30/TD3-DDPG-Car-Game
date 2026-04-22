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

    # Precompute inverse constants to replace division with multiplication
    _INV_MAX_SPEED = 1.0 / CAR_MAX_SPEED
    _INV_SCREEN_WIDTH = 1.0 / SCREEN_WIDTH
    _INV_SCREEN_HEIGHT = 1.0 / SCREEN_HEIGHT
    _INV_360 = 1.0 / 360.0
    _INV_SENSOR_MAX_DIST = 1.0 / SENSOR_MAX_DIST

    # Precompute sensor ray distances as a range for vectorized casting
    _RAY_DISTANCES = np.arange(
        SENSOR_RAYCAST_STEP, SENSOR_MAX_DIST + 1, SENSOR_RAYCAST_STEP, dtype=np.float64
    )

    def __init__(
        self,
        track_mask: np.ndarray,
        car_image: pygame.Surface,
        sensor_noise_std: float = SENSOR_NOISE_STD,
    ):
        self.track_mask = track_mask
        self._mask_w, self._mask_h = track_mask.shape  # Cache mask dimensions
        self.original_image = car_image

        self.x = float(CAR_START_X)
        self.y = float(CAR_START_Y)
        self.angle = float(CAR_START_ANGLE)
        self.speed = 0.0
        self.sensor_noise_std = max(0.0, float(sensor_noise_std))

        self.sensor_dists: list[float] = [1.0] * NUM_SENSORS
        self.sensor_endpoints: list[tuple[float, float]] = []

        # Pre-allocate state buffer to avoid repeated array construction
        self._state_buffer = np.zeros(4 + NUM_SENSORS, dtype=np.float32)

    def reset(self):
        """Reset the car to its starting pose and zero velocity."""
        self.x = float(CAR_START_X)
        self.y = float(CAR_START_Y)
        self.angle = float(CAR_START_ANGLE)
        self.speed = 0.0
        self.sensor_dists = [1.0] * NUM_SENSORS
        self.sensor_endpoints = []

    def update(self, steering: float, throttle: float):
        """Advance the car one simulation step."""
        # Speed-dependent steering: turn less at high speed (like real cars)
        speed_factor = 1.0 - (self.speed * self._INV_MAX_SPEED) * CAR_TURN_SPEED_FACTOR
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
        """Cast all sensor rays and store normalised distances with noise.
        
        Optimized: precomputes sin/cos for all sensor angles once, then
        iterates rays with cached trig values.
        """
        self.sensor_dists = []
        self.sensor_endpoints = []

        # Precompute sin/cos for all sensor absolute angles at once
        car_x = self.x
        car_y = self.y
        base_angle = self.angle
        mask_w = self._mask_w
        mask_h = self._mask_h
        track_mask = self.track_mask
        inv_max = self._INV_SENSOR_MAX_DIST
        noise_std = self.sensor_noise_std
        has_noise = noise_std > 0

        for rel_angle in SENSOR_ANGLES:
            rad = math.radians(base_angle + rel_angle)
            dx = math.cos(rad)
            dy = -math.sin(rad)

            # Ray march with cached values
            hit_dist = SENSOR_MAX_DIST
            end_x = car_x + dx * SENSOR_MAX_DIST
            end_y = car_y + dy * SENSOR_MAX_DIST

            for dist in range(SENSOR_RAYCAST_STEP, SENSOR_MAX_DIST + 1, SENSOR_RAYCAST_STEP):
                px = int(car_x + dx * dist + 0.5)  # Faster than int(round(...))
                py = int(car_y + dy * dist + 0.5)

                # Treat out-of-bounds as obstacles (prevents wrapping)
                if px < 0 or px >= mask_w or py < 0 or py >= mask_h:
                    hit_dist = dist
                    end_x = car_x + dx * dist
                    end_y = car_y + dy * dist
                    break

                # Check if we hit off-track area
                if not track_mask[px, py]:
                    hit_dist = dist
                    end_x = float(px)
                    end_y = float(py)
                    break

            # Normalize distance
            normalized_dist = hit_dist * inv_max
            # Add sensor noise
            if has_noise:
                noise = np.random.normal(0, noise_std)
                normalized_dist += noise
                # Inline clip for scalars (faster than np.clip)
                if normalized_dist < 0.0:
                    normalized_dist = 0.0
                elif normalized_dist > 1.0:
                    normalized_dist = 1.0
            self.sensor_dists.append(normalized_dist)
            self.sensor_endpoints.append((end_x, end_y))

    def set_sensor_noise(self, sensor_noise_std: float):
        """Update Gaussian sensor-noise standard deviation at runtime."""
        self.sensor_noise_std = max(0.0, float(sensor_noise_std))

    def get_state(self) -> np.ndarray:
        """Return the observation vector using pre-allocated buffer."""
        buf = self._state_buffer
        buf[0] = self.x * self._INV_SCREEN_WIDTH
        buf[1] = self.y * self._INV_SCREEN_HEIGHT
        buf[2] = self.speed * self._INV_MAX_SPEED
        buf[3] = self.angle * self._INV_360
        # Copy sensor distances into buffer
        for i, d in enumerate(self.sensor_dists):
            buf[4 + i] = d
        return buf

    def is_off_track(self) -> bool:
        """Return True if the car centre leaves the road."""
        ix = int(self.x + 0.5)  # Faster than int(round(...))
        iy = int(self.y + 0.5)
        if ix < 0 or ix >= self._mask_w or iy < 0 or iy >= self._mask_h:
            return True
        return not self.track_mask[ix, iy]

    def draw(self, surface: pygame.Surface):
        """Draw the rotated car sprite and sensor rays."""
        rotated = pygame.transform.rotate(self.original_image, self.angle - 90)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(rotated, rect)

        car_ix = int(self.x)
        car_iy = int(self.y)
        for endpoint in self.sensor_endpoints:
            ep_x = int(endpoint[0])
            ep_y = int(endpoint[1])
            pygame.draw.line(
                surface,
                SENSOR_COLOR,
                (car_ix, car_iy),
                (ep_x, ep_y),
                1,
            )
            pygame.draw.circle(
                surface,
                SENSOR_ENDPOINT_COLOR,
                (ep_x, ep_y),
                3,
            )
