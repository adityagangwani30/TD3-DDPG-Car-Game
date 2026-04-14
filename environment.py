"""
environment.py - Gym-style RL environment for the self-driving car.

Simplified environment logic:
  - Episode ends when the car is off track or stuck
  - Improved reward function with lap completion bonuses
  - Lap timing tracked using LapTimer class
  - Integrated metrics tracking
"""

import numpy as np
import pygame

from car import Car
from config import (
    CAR_MAX_SPEED,
    CAR_IMAGE_PATH,
    FINISH_LINE_WIDTH,
    FPS,
    HUD_BG_COLOR,
    HUD_TEXT_COLOR,
    RENDER_DURING_TRAINING,
    REWARD_ALIVE,
    REWARD_CRASH,
    REWARD_LAP_COMPLETION,
    REWARD_SPEED_BONUS,
    REWARD_STEERING_PENALTY,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    STUCK_SPEED_THRESHOLD,
    STUCK_STEP_LIMIT,
    TRACK_CENTER_X,
    TRACK_IMAGE_PATH,
    WINDOW_TITLE,
)
from lap_timer import LapTimer
from metrics_tracker import MetricsTracker
from utils import draw_text, ensure_assets_exist, load_track_mask


class CarRacingEnv:
    """Custom 2-D racing environment with a small Gym-like API."""

    def __init__(
        self,
        enable_metrics: bool = True,
        reward_mode: str = "shaped",
        sensor_noise_std: float | None = None,
    ):
        ensure_assets_exist()

        self.screen: pygame.Surface = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)

        self.track_surface = pygame.image.load(TRACK_IMAGE_PATH).convert()
        self.track_mask = load_track_mask(self.track_surface)

        car_img = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
        self.car = Car(self.track_mask, car_img)
        if sensor_noise_std is not None:
            self.set_sensor_noise(sensor_noise_std)

        allowed_reward_modes = {"basic", "shaped", "modified"}
        self.reward_mode = reward_mode if reward_mode in allowed_reward_modes else "shaped"

        self.lap_timer = LapTimer()
        self.metrics = MetricsTracker() if enable_metrics else None

        self.episode = 0
        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_steps = 0

    def reset(self) -> np.ndarray:
        """Reset the environment and return the initial observation."""
        self.car.reset()
        self.car.cast_sensors()

        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_steps = 0
        self.episode += 1

        self.lap_timer.reset()
        if self.metrics:
            self.metrics.reset_episode()

        return self.car.get_state()

    def step(self, action: np.ndarray):
        """Execute one environment step."""
        steering, throttle = self._parse_action(action)
        previous_position = (self.car.x, self.car.y)

        self.car.update(steering, throttle)
        self.car.cast_sensors()
        self.step_count += 1

        # Check for lap completion
        lap_completed = self.lap_timer.update(
            self.step_count, previous_position, (self.car.x, self.car.y)
        )

        off_track = self.car.is_off_track()
        stuck = self._update_stuck_counter()
        done = off_track or stuck

        reward = self._compute_reward(steering, done, lap_completed)
        self.episode_reward += reward

        termination_reason = None
        if done:
            termination_reason = "off_track" if off_track else "stuck"

        info = {
            "termination_reason": termination_reason,
            "lap_completed": lap_completed,
        }

        # Log metrics if tracking enabled
        if self.metrics:
            self.metrics.log_step(reward, self.car.speed, steering, action)
            if lap_completed:
                self.metrics.log_lap_completion(self.lap_timer.last_lap_time or 0.0)
            if done:
                self.metrics.log_termination(termination_reason or "unknown")

        return self.car.get_state(), reward, done, info

    def _parse_action(self, action: np.ndarray) -> tuple[float, float]:
        """Parse action to steering and throttle. Both in [-1, 1] and [0, 1] respectively."""
        steering = float(np.clip(action[0], -1.0, 1.0))
        # Throttle: clip to [0, 1] for forward motion only
        throttle = float(np.clip(action[1], 0.0, 1.0))
        return steering, throttle

    def _compute_reward(self, steering: float, done: bool, lap_completed: bool) -> float:
        """Compute reward according to the configured reward mode."""
        if done:
            return REWARD_CRASH

        if self.reward_mode == "basic":
            # Basic reward: only survival bonus and crash penalty.
            return REWARD_ALIVE

        if self.reward_mode == "modified":
            return self._compute_modified_reward(steering, lap_completed)

        return self._compute_shaped_reward(steering, lap_completed)

    def _compute_shaped_reward(self, steering: float, lap_completed: bool) -> float:
        """Original shaped reward currently used by training."""
        reward = REWARD_ALIVE

        if self.car.speed > STUCK_SPEED_THRESHOLD:
            reward += REWARD_SPEED_BONUS

        if lap_completed:
            reward += REWARD_LAP_COMPLETION

        reward -= REWARD_STEERING_PENALTY * (abs(steering) ** 2)

        return reward

    def _compute_modified_reward(self, steering: float, lap_completed: bool) -> float:
        """Slightly enhanced shaped reward for experiment runs."""
        reward = self._compute_shaped_reward(steering, lap_completed)

        # Gentle speed-scaling encourages smoother acceleration.
        reward += 0.05 * (self.car.speed / CAR_MAX_SPEED)

        # Encourage straight, stable driving when moving.
        if self.car.speed > STUCK_SPEED_THRESHOLD and abs(steering) < 0.2:
            reward += 0.02

        # Small anti-idling penalty to reduce stationary exploitation.
        if self.car.speed <= STUCK_SPEED_THRESHOLD:
            reward -= 0.02

        return reward

    def set_sensor_noise(self, noise_std: float):
        """Update the car sensor noise level during runtime."""
        self.car.set_sensor_noise(noise_std)

    def _update_stuck_counter(self) -> bool:
        """Track how long the car has been almost stationary."""
        if self.car.speed <= STUCK_SPEED_THRESHOLD:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        return self.stuck_steps >= STUCK_STEP_LIMIT

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

    def _draw_finish_line(self):
        """Draw the finish line across the track."""
        start = self.lap_timer.finish_line_start
        end = self.lap_timer.finish_line_end
        
        pygame.draw.line(
            self.screen,
            HUD_TEXT_COLOR,
            start,
            end,
            FINISH_LINE_WIDTH,
        )

        # Draw checkered pattern
        segments = 8
        total_height = end[1] - start[1]
        for i in range(0, segments, 2):
            start_y = start[1] + total_height * i / segments
            end_y = start[1] + total_height * (i + 1) / segments
            pygame.draw.line(
                self.screen,
                (30, 30, 30),
                (TRACK_CENTER_X, start_y),
                (TRACK_CENTER_X, end_y),
                max(1, FINISH_LINE_WIDTH // 2),
            )

    def _draw_hud(self):
        """Render a simple, clean HUD onto the screen."""
        hud_x, hud_y = 12, 10
        line_h = 20

        current_lap_time = self.lap_timer.get_current_lap_time()
        hud_lines = [
            f"Ep {self.episode:5d} | Reward {self.episode_reward:+7.1f}",
            f"Step {self.step_count:4d} | Speed {self.car.speed:5.2f}",
            f"Lap: {current_lap_time:6.2f}s (Best: {LapTimer.format_time(self.lap_timer.best_lap_time)})",
            f"Crashes: {self.metrics.collisions if self.metrics else 0:3d} | Laps: {self.lap_timer.laps_completed:2d}",
        ]

        hud_bg = pygame.Surface((320, line_h * len(hud_lines) + 8), pygame.SRCALPHA)
        hud_bg.fill(HUD_BG_COLOR)
        self.screen.blit(hud_bg, (hud_x - 6, hud_y - 4))

        for index, line in enumerate(hud_lines):
            draw_text(self.screen, line, hud_x, hud_y + index * line_h, self.font, HUD_TEXT_COLOR)

    def close(self):
        """Shut down Pygame."""
        pygame.quit()
