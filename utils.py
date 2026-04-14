"""
utils.py - Utility helpers for the TD3 self-driving car project.

Includes track loading, simple asset generation, and text helpers.
"""

import os
import random
import sys

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


def detect_headless_environment() -> bool:
    """
    Detect if we are in a headless environment (no display available).
    
    Returns True if running in:
    - Google Colab
    - SSH session without X11 forwarding
    - Container/cloud without display
    - System with no DISPLAY variable
    
    Returns False for normal desktop systems.
    """
    # Check for Google Colab
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    
    # Check for Kaggle notebooks
    if os.environ.get("KAGGLE_DATA_MOUNT_DIR"):
        return True
    
    # On Windows, assume GUI is available (no DISPLAY variable anyway)
    if sys.platform.startswith("win"):
        return False
    
    # On macOS, assume GUI is available unless explicitly disabled
    if sys.platform.startswith("darwin"):
        if not os.environ.get("DISPLAY"):
            # Check if running in SSH session
            return os.environ.get("SSH_CONNECTION") is not None
        return False
    
    # On Linux, check for DISPLAY variable or SSH session
    if sys.platform.startswith("linux"):
        # No DISPLAY and either SSH or container
        if not os.environ.get("DISPLAY"):
            is_ssh = os.environ.get("SSH_CONNECTION") is not None
            is_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
            return is_ssh or is_container
        return False
    
    # Default: assume GUI for unknown platforms
    return False


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
    # Simple car sprite (removed extra details for clarity)

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


def set_global_seed(seed: int):
    """Set deterministic random seeds across Python, NumPy, and PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[utils] Global seed set -> {seed}")


def init_pygame(headless: bool = False):
    """
    Initialize pygame with appropriate settings for the environment.
    
    Args:
        headless: If True, use dummy video driver for off-screen rendering.
                 If False, attempt to create a display window.
    """
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        print("[utils] Pygame initialized in HEADLESS mode (off-screen rendering)")
    else:
        # Try to use default display
        # Remove dummy driver if it was set, to allow GUI
        if os.environ.get("SDL_VIDEODRIVER") == "dummy":
            del os.environ["SDL_VIDEODRIVER"]
        print("[utils] Pygame initialized in GUI mode (interactive display)")
    
    try:
        pygame.init()
    except pygame.error as err:
        if not headless:
            # Try fallback to dummy driver
            print(f"[utils] GUI initialization failed: {err}")
            print("[utils] Attempting headless fallback...")
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            try:
                pygame.init()
                print("[utils] Pygame display fallback -> dummy")
                return
            except pygame.error as err2:
                raise RuntimeError(f"Failed to initialize pygame in both modes: {err2}") from err2
        else:
            raise RuntimeError(f"Failed to initialize pygame in headless mode: {err}") from err
