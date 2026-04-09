"""
environment.py - Gym-style RL environment for the self-driving car.

Keeps the environment logic intentionally simple:
  - episode ends when the car is off track or stuck
  - reward is alive bonus + speed bonus - steering penalty
  - lap timing is tracked only for display
"""

import numpy as np
import pygame

from car import Car
from config import (
    CAR_IMAGE_PATH,
    CAR_MAX_SPEED,
    FINISH_LINE_WIDTH,
    FPS,
    MIN_LAP_STEPS,
    REWARD_ALIVE,
    REWARD_CRASH,
    REWARD_STEERING_PENALTY,
    REWARD_VELOCITY_SCALE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STUCK_SPEED_THRESHOLD,
    STUCK_STEP_LIMIT,
    TRACK_CENTER_X,
    TRACK_CENTER_Y,
    TRACK_IMAGE_PATH,
    TRACK_INNER_RADIUS_Y,
    TRACK_OUTER_RADIUS_Y,
    WINDOW_TITLE,
)
from utils import draw_text, ensure_assets_exist, load_track_mask


class CarRacingEnv:
    """Custom 2-D racing environment with a small Gym-like API."""

    def __init__(self):
        ensure_assets_exist()

        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)

        self.track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
        self.track_mask = load_track_mask(self.track_surface)
        self.finish_line_start, self.finish_line_end = self._build_finish_line()

        car_img = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
        self.car = Car(self.track_mask, car_img)

        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.lap_start_step = 0
        self.stuck_steps = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self.car.reset()
        self.car.cast_sensors()

        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_start_step = 0
        self.stuck_steps = 0
        self.episode += 1

        return self.car.get_state()

    def step(self, action: np.ndarray):
        """Execute one environment step."""
        steering, throttle = self._parse_action(action)
        previous_position = (self.car.x, self.car.y)

        self.car.update(steering, throttle)
        self.car.cast_sensors()
        self.step_count += 1

        self._update_lap_timing(previous_position)

        off_track = self.car.is_off_track()
        stuck = self._update_stuck_counter()
        done = off_track or stuck

        reward = self._compute_reward(steering, done)
        self.episode_reward += reward

        termination_reason = None
        if done:
            termination_reason = "off_track" if off_track else "stuck"

        info = {"termination_reason": termination_reason}
        return self.car.get_state(), reward, done, info

    def _parse_action(self, action: np.ndarray) -> tuple[float, float]:
        """Clip the action to the supported steering and throttle ranges."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float((np.clip(action[1], -1.0, 1.0) + 1.0) / 2.0)
        return steering, throttle

    def _compute_reward(self, steering: float, done: bool) -> float:
        """Return the simplified step reward."""
        if done:
            return REWARD_CRASH
        reward = REWARD_ALIVE
        reward += REWARD_VELOCITY_SCALE * (self.car.speed / CAR_MAX_SPEED)
        reward -= REWARD_STEERING_PENALTY * abs(steering)
        return reward

    def _update_stuck_counter(self) -> bool:
        """Track how long the car has been almost stationary."""
        if self.car.speed <= STUCK_SPEED_THRESHOLD:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        return self.stuck_steps >= STUCK_STEP_LIMIT

    def _update_lap_timing(self, previous_position: tuple[float, float]):
        """Update lap timing if the car centre crosses the finish line."""
        if not self._crossed_finish_line(previous_position):
            return
        if self.step_count - self.lap_start_step < MIN_LAP_STEPS:
            return

        lap_steps = self.step_count - self.lap_start_step
        lap_time = lap_steps / FPS
        self.last_lap_time = lap_time
        if self.best_lap_time is None or lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
        self.lap_start_step = self.step_count

    def _crossed_finish_line(self, previous_position: tuple[float, float]) -> bool:
        """Return True when the car centre crosses the finish line left-to-right."""
        line_x = TRACK_CENTER_X
        min_y = self.finish_line_start[1]
        max_y = self.finish_line_end[1]
        prev_x, prev_y = previous_position
        curr_x, curr_y = self.car.x, self.car.y
        on_line_band = min_y <= prev_y <= max_y and min_y <= curr_y <= max_y
        crossed_line = prev_x < line_x <= curr_x
        return on_line_band and crossed_line

    def render(self, enabled: bool = True, limit_fps: bool = False):
        """Draw the current frame. Event handling always stays active."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        if not enabled:
            return

        self.screen.blit(self.track_surface, (0, 0))
        self._draw_finish_line()
        self.car.draw(self.screen)
        self._draw_hud()

        pygame.display.flip()
        if limit_fps:
            self.clock.tick(FPS)

    def _draw_hud(self):
        """Render a small HUD onto the screen."""
        hud_x, hud_y = 12, 10
        line_h = 22
        hud_lines = [
            f"Episode:   {self.episode}",
            f"Reward:    {self.episode_reward:+.2f}",
            f"Speed:     {self.car.speed:.2f}",
            f"Current:   {self._format_time(self.get_current_lap_time())}",
            f"Last Lap:  {self._format_time(self.last_lap_time)}",
            f"Best Lap:  {self._format_time(self.best_lap_time)}",
        ]

        hud_bg = pygame.Surface((250, line_h * len(hud_lines) + 8), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 160))
        self.screen.blit(hud_bg, (hud_x - 6, hud_y - 4))

        for index, line in enumerate(hud_lines):
            draw_text(self.screen, line, hud_x, hud_y + index * line_h, self.font)

    def get_current_lap_time(self) -> float:
        """Return the in-progress lap time in seconds."""
        return (self.step_count - self.lap_start_step) / FPS

    def _build_finish_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the visible finish line endpoints."""
        return (
            (TRACK_CENTER_X, TRACK_CENTER_Y + TRACK_INNER_RADIUS_Y),
            (TRACK_CENTER_X, TRACK_CENTER_Y + TRACK_OUTER_RADIUS_Y),
        )

    def _draw_finish_line(self):
        """Draw a simple finish line across the track."""
        pygame.draw.line(
            self.screen,
            (245, 245, 245),
            self.finish_line_start,
            self.finish_line_end,
            FINISH_LINE_WIDTH,
        )

        segments = 8
        total_height = self.finish_line_end[1] - self.finish_line_start[1]
        for index in range(0, segments, 2):
            start_y = self.finish_line_start[1] + total_height * index / segments
            end_y = self.finish_line_start[1] + total_height * (index + 1) / segments
            pygame.draw.line(
                self.screen,
                (30, 30, 30),
                (TRACK_CENTER_X, start_y),
                (TRACK_CENTER_X, end_y),
                max(1, FINISH_LINE_WIDTH // 2),
            )

    @staticmethod
    def _format_time(value: float | None) -> str:
        """Format a lap time for the HUD."""
        if value is None:
            return "--.--s"
        return f"{value:5.2f}s"

    def close(self):
        """Shut down Pygame."""
        pygame.quit()
