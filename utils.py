"""
utils.py - Utility helpers for the TD3 self-driving car project.

Includes track loading, simple asset generation, and text helpers.
"""

import os

import numpy as np
import pygame

from config import (
    ASSETS_DIR,
    CAR_HEIGHT,
    CAR_IMAGE_PATH,
    CAR_WIDTH,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    TRACK_BORDER_COLOR,
    TRACK_CENTER_X,
    TRACK_CENTER_Y,
    TRACK_GRASS_COLOR,
    TRACK_IMAGE_PATH,
    TRACK_INNER_RADIUS_X,
    TRACK_INNER_RADIUS_Y,
    TRACK_OUTER_RADIUS_X,
    TRACK_OUTER_RADIUS_Y,
    TRACK_ROAD_COLOR,
)


def generate_track_image():
    """Programmatically create a racing-track PNG and save it to assets/."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(TRACK_GRASS_COLOR)

    _draw_ellipse(
        surface,
        TRACK_ROAD_COLOR,
        TRACK_CENTER_X,
        TRACK_CENTER_Y,
        TRACK_OUTER_RADIUS_X,
        TRACK_OUTER_RADIUS_Y,
    )
    _draw_ellipse(
        surface,
        TRACK_GRASS_COLOR,
        TRACK_CENTER_X,
        TRACK_CENTER_Y,
        TRACK_INNER_RADIUS_X,
        TRACK_INNER_RADIUS_Y,
    )

    pygame.draw.ellipse(
        surface,
        TRACK_BORDER_COLOR,
        pygame.Rect(
            TRACK_CENTER_X - TRACK_OUTER_RADIUS_X + 8,
            TRACK_CENTER_Y - TRACK_OUTER_RADIUS_Y + 8,
            2 * (TRACK_OUTER_RADIUS_X - 8),
            2 * (TRACK_OUTER_RADIUS_Y - 8),
        ),
        width=3,
    )
    pygame.draw.ellipse(
        surface,
        TRACK_BORDER_COLOR,
        pygame.Rect(
            TRACK_CENTER_X - TRACK_INNER_RADIUS_X - 8,
            TRACK_CENTER_Y - TRACK_INNER_RADIUS_Y - 8,
            2 * (TRACK_INNER_RADIUS_X + 8),
            2 * (TRACK_INNER_RADIUS_Y + 8),
        ),
        width=3,
    )

    pygame.image.save(surface, TRACK_IMAGE_PATH)
    print(f"[utils] Track image saved -> {TRACK_IMAGE_PATH}")


def _draw_ellipse(surface, color, cx, cy, rx, ry):
    """Draw a filled ellipse using pygame.draw.ellipse."""
    rect = pygame.Rect(cx - rx, cy - ry, 2 * rx, 2 * ry)
    pygame.draw.ellipse(surface, color, rect)


def generate_car_image():
    """Create a small top-down car sprite and save it to assets/."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    surface = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)

    pygame.draw.rect(
        surface,
        (220, 30, 30),
        (2, 4, CAR_WIDTH - 4, CAR_HEIGHT - 8),
        border_radius=4,
    )
    pygame.draw.rect(
        surface,
        (140, 200, 255),
        (4, 6, CAR_WIDTH - 8, 10),
        border_radius=2,
    )
    pygame.draw.rect(surface, (255, 200, 0), (3, CAR_HEIGHT - 10, 5, 4))
    pygame.draw.rect(surface, (255, 200, 0), (CAR_WIDTH - 8, CAR_HEIGHT - 10, 5, 4))

    pygame.image.save(surface, CAR_IMAGE_PATH)
    print(f"[utils] Car sprite saved  -> {CAR_IMAGE_PATH}")


def load_track_mask(track_surface: pygame.Surface) -> np.ndarray:
    """Convert a track surface into a boolean mask (True = road)."""
    arr = pygame.surfarray.array3d(track_surface)
    road = np.array(TRACK_ROAD_COLOR, dtype=arr.dtype)
    return np.all(arr == road, axis=2)


def draw_text(surface, text, x, y, font, color=(255, 255, 255)):
    """Render text onto a surface at (x, y)."""
    rendered = font.render(str(text), True, color)
    surface.blit(rendered, (x, y))


def ensure_assets_exist():
    """Generate track and car images if they do not already exist."""
    if not os.path.exists(TRACK_IMAGE_PATH):
        generate_track_image()
    if not os.path.exists(CAR_IMAGE_PATH):
        generate_car_image()
