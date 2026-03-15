"""
environment.py – Gym-style RL environment for the self-driving car.

Wraps the Car model, track loading, reward computation, and rendering
into a clean reset() / step() / render() interface.
"""

import numpy as np
import pygame

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, WINDOW_TITLE,
    CAR_IMAGE_PATH, TRACK_IMAGE_PATH,
    REWARD_ALIVE, REWARD_VELOCITY_SCALE,
    REWARD_STEERING_PENALTY, REWARD_CRASH,
    CAR_MAX_SPEED,
)
from car import Car
from utils import load_track_mask, draw_text, ensure_assets_exist


class CarRacingEnv:
    """Custom 2-D racing environment with a Gym-like API."""

    def __init__(self):
        # Make sure assets are generated before loading
        ensure_assets_exist()

        # Pygame display
        self.screen: pygame.Surface = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT)
        )
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        # Load track image and build road mask
        self.track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
        self.track_mask = load_track_mask(self.track_surface)

        # Load car sprite
        car_img = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
        self.car = Car(self.track_mask, car_img)

        # Episode bookkeeping
        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0.0

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self.car.reset()
        self.car.cast_sensors()
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode += 1
        return self.car.get_state()

    def step(self, action: np.ndarray):
        """Execute one environment step.

        Args:
            action: np.array([steering, throttle])
                    steering ∈ [-1, 1], throttle ∈ [-1, 1]
                    (throttle is remapped to [0, 1] internally)

        Returns:
            next_state (np.ndarray), reward (float), done (bool), info (dict)
        """
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float((np.clip(action[1], -1.0, 1.0) + 1.0) / 2.0)  # → [0, 1]

        # Advance physics
        self.car.update(steering, throttle)
        self.car.cast_sensors()

        # Collision / termination check
        done = self.car.is_off_track()

        # ----- Reward computation -----
        if done:
            reward = REWARD_CRASH
        else:
            # Reward for staying alive
            reward = REWARD_ALIVE
            # Reward proportional to forward speed
            reward += REWARD_VELOCITY_SCALE * (self.car.speed / CAR_MAX_SPEED)
            # Penalise excessive steering
            reward -= REWARD_STEERING_PENALTY * abs(steering)

        self.step_count += 1
        self.episode_reward += reward

        info = {
            "step": self.step_count,
            "speed": self.car.speed,
        }

        return self.car.get_state(), reward, done, info

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self):
        """Draw the current frame (track + car + HUD) and flip the display."""
        # Handle Pygame events (allow window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Background – track image
        self.screen.blit(self.track_surface, (0, 0))

        # Car + sensors
        self.car.draw(self.screen)

        # HUD overlay
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_hud(self):
        """Render heads-up-display text onto the screen."""
        hud_x, hud_y = 12, 10
        line_h = 22

        # Semi-transparent background box
        hud_bg = pygame.Surface((240, 120), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 160))
        self.screen.blit(hud_bg, (hud_x - 6, hud_y - 4))

        draw_text(self.screen,
                  f"Episode:  {self.episode}", hud_x, hud_y, self.font)
        draw_text(self.screen,
                  f"Step:     {self.step_count}", hud_x, hud_y + line_h, self.font)
        draw_text(self.screen,
                  f"Reward:   {self.episode_reward:+.2f}",
                  hud_x, hud_y + 2 * line_h, self.font)
        draw_text(self.screen,
                  f"Speed:    {self.car.speed:.2f}",
                  hud_x, hud_y + 3 * line_h, self.font)
        draw_text(self.screen,
                  f"Angle:    {self.car.angle:.1f}°",
                  hud_x, hud_y + 4 * line_h, self.font)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def close(self):
        """Shut down Pygame."""
        pygame.quit()
