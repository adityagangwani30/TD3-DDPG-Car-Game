"""
environment.py - Gym-style RL environment for the self-driving car.

Wraps the car model, reward computation, lap tracking, and rendering
into a clean reset() / step() / render() interface.
"""

import math

import numpy as np
import pygame

from car import Car
from config import (
    CAR_IMAGE_PATH,
    CAR_MAX_SPEED,
    FINISH_LINE_WIDTH,
    FPS,
    REWARD_ALIVE,
    REWARD_CRASH,
    REWARD_LAP_COMPLETION,
    REWARD_PROGRESS_SCALE,
    REWARD_STEERING_PENALTY,
    REWARD_VELOCITY_SCALE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STUCK_SPEED_THRESHOLD,
    STUCK_STEP_LIMIT,
    TRACK_CENTER_X,
    TRACK_CENTER_Y,
    TRACK_IMAGE_PATH,
    TRACK_INNER_RADIUS_X,
    TRACK_INNER_RADIUS_Y,
    TRACK_OUTER_RADIUS_X,
    TRACK_OUTER_RADIUS_Y,
    TRACK_START_ANGLE_DEG,
    WINDOW_TITLE,
)
from utils import draw_text, ensure_assets_exist, load_track_mask


class CarRacingEnv:
    """Custom 2-D racing environment with a Gym-like API."""

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
        self.laps_completed = 0
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.lap_start_step = 0
        self.previous_track_progress = 0.0
        self.progress_since_lap_start = 0.0
        self.stuck_steps = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self.car.reset()
        self.car.cast_sensors()

        self.step_count = 0
        self.episode_reward = 0.0
        self.laps_completed = 0
        self.lap_start_step = 0
        self.previous_track_progress = self._compute_track_progress()
        self.progress_since_lap_start = 0.0
        self.stuck_steps = 0
        self.episode += 1

        return self.car.get_state()

    def step(self, action: np.ndarray):
        """Execute one environment step."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle = float((np.clip(action[1], -1.0, 1.0) + 1.0) / 2.0)

        self.car.update(steering, throttle)
        self.car.cast_sensors()
        self.step_count += 1

        current_progress = self._compute_track_progress()
        progress_delta = self._unwrap_progress_delta(
            self.previous_track_progress,
            current_progress,
        )
        self.progress_since_lap_start += progress_delta

        lap_completed = False
        lap_time = None
        while self.progress_since_lap_start >= 1.0:
            lap_completed = True
            self.progress_since_lap_start -= 1.0
            self.laps_completed += 1

            lap_steps = self.step_count - self.lap_start_step
            lap_time = lap_steps / FPS
            self.last_lap_time = lap_time
            if self.best_lap_time is None or lap_time < self.best_lap_time:
                self.best_lap_time = lap_time
            self.lap_start_step = self.step_count

        if self.car.speed <= STUCK_SPEED_THRESHOLD:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        off_track = self.car.is_off_track()
        stuck = self.stuck_steps >= STUCK_STEP_LIMIT
        done = off_track or stuck

        if done:
            reward = REWARD_CRASH
        else:
            reward = REWARD_ALIVE
            reward += REWARD_VELOCITY_SCALE * (self.car.speed / CAR_MAX_SPEED)
            reward -= REWARD_STEERING_PENALTY * abs(steering)
            reward += REWARD_PROGRESS_SCALE * progress_delta

        if lap_completed:
            reward += REWARD_LAP_COMPLETION

        self.episode_reward += reward
        self.previous_track_progress = current_progress

        info = {
            "step": self.step_count,
            "speed": self.car.speed,
            "lap_completed": lap_completed,
            "laps_completed": self.laps_completed,
            "lap_time": lap_time,
            "last_lap_time": self.last_lap_time,
            "best_lap_time": self.best_lap_time,
            "current_lap_time": self.get_current_lap_time(),
            "lap_progress": self.get_lap_progress(),
            "progress_delta": progress_delta,
            "off_track": off_track,
            "stuck": stuck,
            "stuck_steps": self.stuck_steps,
        }

        return self.car.get_state(), reward, done, info

    def render(self):
        """Draw the current frame and flip the display."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.blit(self.track_surface, (0, 0))
        self._draw_finish_line()
        self.car.draw(self.screen)
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(FPS)

    def _draw_hud(self):
        """Render heads-up-display text onto the screen."""
        hud_x, hud_y = 12, 10
        line_h = 22
        hud_lines = [
            f"Episode:  {self.episode}",
            f"Step:     {self.step_count}",
            f"Reward:   {self.episode_reward:+.2f}",
            f"Speed:    {self.car.speed:.2f}",
            f"Laps:     {self.laps_completed}",
            f"Progress: {self.get_lap_progress() * 100:5.1f}%",
            f"Lap Time: {self._format_time(self.get_current_lap_time())}",
            f"Last Lap: {self._format_time(self.last_lap_time)}",
            f"Best Lap: {self._format_time(self.best_lap_time)}",
        ]

        hud_bg = pygame.Surface((260, line_h * len(hud_lines) + 8), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 160))
        self.screen.blit(hud_bg, (hud_x - 6, hud_y - 4))

        for index, line in enumerate(hud_lines):
            draw_text(self.screen, line, hud_x, hud_y + index * line_h, self.font)

    def get_current_lap_time(self) -> float:
        """Return the in-progress lap time in seconds."""
        return (self.step_count - self.lap_start_step) / FPS

    def get_lap_progress(self) -> float:
        """Return the current lap progress clamped to the HUD-friendly range."""
        return max(0.0, min(self.progress_since_lap_start, 0.999))

    def _compute_track_progress(self) -> float:
        """Return clockwise progress around the oval in the range [0, 1)."""
        dx = self.car.x - TRACK_CENTER_X
        dy = TRACK_CENTER_Y - self.car.y
        raw_angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        clockwise_degrees = (raw_angle - TRACK_START_ANGLE_DEG) % 360.0
        return clockwise_degrees / 360.0

    @staticmethod
    def _unwrap_progress_delta(previous: float, current: float) -> float:
        """Return the signed shortest-path delta between wrapped progress values."""
        delta = current - previous
        if delta < -0.5:
            delta += 1.0
        elif delta > 0.5:
            delta -= 1.0
        return delta

    def _build_finish_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the inner and outer endpoints of the finish line."""
        return (
            self._ellipse_point(TRACK_INNER_RADIUS_X, TRACK_INNER_RADIUS_Y),
            self._ellipse_point(TRACK_OUTER_RADIUS_X, TRACK_OUTER_RADIUS_Y),
        )

    @staticmethod
    def _format_time(value: float | None) -> str:
        """Format a lap time for the HUD."""
        if value is None:
            return "--.--s"
        return f"{value:5.2f}s"

    @staticmethod
    def _lerp_point(
        start: tuple[float, float],
        end: tuple[float, float],
        amount: float,
    ) -> tuple[float, float]:
        """Linearly interpolate between two 2-D points."""
        return (
            start[0] + (end[0] - start[0]) * amount,
            start[1] + (end[1] - start[1]) * amount,
        )

    def _ellipse_point(self, radius_x: float, radius_y: float) -> tuple[float, float]:
        """Return a point on the oval track at the configured start angle."""
        angle_rad = math.radians(TRACK_START_ANGLE_DEG)
        return (
            TRACK_CENTER_X + radius_x * math.cos(angle_rad),
            TRACK_CENTER_Y - radius_y * math.sin(angle_rad),
        )

    def _draw_finish_line(self):
        """Draw a simple checkered finish line across the track width."""
        pygame.draw.line(
            self.screen,
            (245, 245, 245),
            self.finish_line_start,
            self.finish_line_end,
            FINISH_LINE_WIDTH,
        )

        segments = 8
        for index in range(0, segments, 2):
            seg_start = self._lerp_point(
                self.finish_line_start,
                self.finish_line_end,
                index / segments,
            )
            seg_end = self._lerp_point(
                self.finish_line_start,
                self.finish_line_end,
                (index + 1) / segments,
            )
            pygame.draw.line(
                self.screen,
                (30, 30, 30),
                seg_start,
                seg_end,
                max(1, FINISH_LINE_WIDTH // 2),
            )

    def close(self):
        """Shut down Pygame."""
        pygame.quit()
