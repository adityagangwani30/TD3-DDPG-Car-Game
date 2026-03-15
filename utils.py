"""
utils.py – Utility helpers for the TD3 self-driving car project.

Includes track-mask loading, on-track detection, asset generation,
and Pygame text-rendering helpers.
"""

import os
import math
import numpy as np
import pygame
from config import (
    ASSETS_DIR, TRACK_IMAGE_PATH, CAR_IMAGE_PATH,
    SCREEN_WIDTH, SCREEN_HEIGHT,
    TRACK_ROAD_COLOR, TRACK_GRASS_COLOR, TRACK_BORDER_COLOR,
    ROAD_BRIGHTNESS_THRESHOLD,
    CAR_WIDTH, CAR_HEIGHT,
)


# ======================================================================
# Track generation
# ======================================================================
def generate_track_image():
    """Programmatically create a racing-track PNG and save it to assets/.

    The track is an oval loop with lane markings on a grass background.
    """
    os.makedirs(ASSETS_DIR, exist_ok=True)
    surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    surface.fill(TRACK_GRASS_COLOR)

    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

    # Outer and inner ellipse radii
    outer_rx, outer_ry = 480, 320
    inner_rx, inner_ry = 320, 180

    # Draw road as thick ellipse ring
    # 1. draw filled outer ellipse (road colour)
    _draw_ellipse(surface, TRACK_ROAD_COLOR, cx, cy, outer_rx, outer_ry)
    # 2. draw filled inner ellipse (grass colour → cuts hole)
    _draw_ellipse(surface, TRACK_GRASS_COLOR, cx, cy, inner_rx, inner_ry)

    # Lane-edge markings (dashed white lines along outer and inner edges)
    mark_outer_rx, mark_outer_ry = outer_rx - 8, outer_ry - 8
    mark_inner_rx, mark_inner_ry = inner_rx + 8, inner_ry + 8
    _draw_dashed_ellipse(surface, TRACK_BORDER_COLOR, cx, cy,
                         mark_outer_rx, mark_outer_ry, dash_len=20, gap=15, width=3)
    _draw_dashed_ellipse(surface, TRACK_BORDER_COLOR, cx, cy,
                         mark_inner_rx, mark_inner_ry, dash_len=20, gap=15, width=3)

    # Centre-line dashed marking
    mid_rx = (outer_rx + inner_rx) // 2
    mid_ry = (outer_ry + inner_ry) // 2
    _draw_dashed_ellipse(surface, (200, 200, 200), cx, cy,
                         mid_rx, mid_ry, dash_len=15, gap=20, width=1)

    pygame.image.save(surface, TRACK_IMAGE_PATH)
    print(f"[utils] Track image saved → {TRACK_IMAGE_PATH}")


def _draw_ellipse(surface, color, cx, cy, rx, ry):
    """Draw a filled ellipse using pygame.draw.ellipse."""
    rect = pygame.Rect(cx - rx, cy - ry, 2 * rx, 2 * ry)
    pygame.draw.ellipse(surface, color, rect)


def _draw_dashed_ellipse(surface, color, cx, cy, rx, ry,
                          dash_len=18, gap=12, width=2):
    """Draw a dashed ellipse outline."""
    num_points = 360
    draw = True
    accum = 0
    prev = None
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        if prev is not None:
            seg_len = math.hypot(x - prev[0], y - prev[1])
            accum += seg_len
            if draw:
                pygame.draw.line(surface, color, prev, (x, y), width)
            threshold = dash_len if draw else gap
            if accum >= threshold:
                accum = 0
                draw = not draw
        prev = (x, y)


# ======================================================================
# Car sprite generation
# ======================================================================
def generate_car_image():
    """Create a small top-down car sprite and save it to assets/."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    w, h = CAR_WIDTH, CAR_HEIGHT
    surface = pygame.Surface((w, h), pygame.SRCALPHA)

    # Body
    body_color = (220, 30, 30)
    pygame.draw.rect(surface, body_color, (2, 4, w - 4, h - 8),
                     border_radius=4)
    # Windshield
    pygame.draw.rect(surface, (140, 200, 255), (4, 6, w - 8, 10),
                     border_radius=2)
    # Rear lights
    pygame.draw.rect(surface, (255, 200, 0), (3, h - 10, 5, 4))
    pygame.draw.rect(surface, (255, 200, 0), (w - 8, h - 10, 5, 4))

    pygame.image.save(surface, CAR_IMAGE_PATH)
    print(f"[utils] Car sprite saved  → {CAR_IMAGE_PATH}")


# ======================================================================
# Track mask helpers
# ======================================================================
def load_track_mask(track_surface: pygame.Surface) -> np.ndarray:
    """Convert a track surface into a boolean mask (True = road).

    A pixel is considered 'road' if its RGB colour is close to the
    known TRACK_ROAD_COLOR (Euclidean distance in RGB space).
    """
    arr = pygame.surfarray.array3d(track_surface).astype(np.float32)  # (W, H, 3)
    road = np.array(TRACK_ROAD_COLOR, dtype=np.float32)
    dist = np.sqrt(np.sum((arr - road) ** 2, axis=2))        # (W, H)
    mask = dist < 60.0                                        # threshold in colour space
    return mask  # indexed as mask[x, y]


def is_on_track(mask: np.ndarray, x: float, y: float) -> bool:
    """Return True if the pixel at (x, y) is on the road."""
    ix, iy = int(round(x)), int(round(y))
    h, w = mask.shape[0], mask.shape[1]   # mask is (W, H) from surfarray
    if ix < 0 or ix >= h or iy < 0 or iy >= w:
        return False
    return bool(mask[ix, iy])


# ======================================================================
# Pygame text helpers
# ======================================================================
def draw_text(surface, text, x, y, font, color=(255, 255, 255)):
    """Render text onto a surface at (x, y)."""
    rendered = font.render(str(text), True, color)
    surface.blit(rendered, (x, y))


def ensure_assets_exist():
    """Generate track and car images if they don't already exist."""
    if not os.path.exists(TRACK_IMAGE_PATH):
        generate_track_image()
    if not os.path.exists(CAR_IMAGE_PATH):
        generate_car_image()
