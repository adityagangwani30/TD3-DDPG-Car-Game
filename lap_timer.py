"""
lap_timer.py - Lap timing and finish line detection logic.

Handles lap timing, finish line crossing detection, and lap statistics.
"""

from config import (
    FPS,
    MIN_LAP_STEPS,
    TRACK_CENTER_X,
    TRACK_CENTER_Y,
    TRACK_INNER_RADIUS_Y,
    TRACK_OUTER_RADIUS_Y,
)


class LapTimer:
    """Tracks lap times and detects finish line crossings."""

    def __init__(self):
        self.lap_start_step = 0
        self.step_count = 0
        self.last_lap_time: float | None = None
        self.best_lap_time: float | None = None
        self.laps_completed = 0
        self.finish_line_start, self.finish_line_end = self._build_finish_line()

    def reset(self):
        """Reset lap timer for a new episode."""
        self.lap_start_step = 0
        self.step_count = 0
        self.laps_completed = 0
        self.last_lap_time = None

    def update(self, step_count: int, previous_position: tuple[float, float], 
               current_position: tuple[float, float]) -> bool:
        """
        Update lap timing and detect finish line crossing.
        
        Returns:
            True if a lap was completed, False otherwise.
        """
        self.step_count = step_count
        
        if not self._crossed_finish_line(previous_position, current_position):
            return False
        
        # Check if minimum steps have passed (avoid crossing at start)
        if self.step_count - self.lap_start_step < MIN_LAP_STEPS:
            return False
        
        # Calculate lap time
        lap_steps = self.step_count - self.lap_start_step
        lap_time = lap_steps / FPS
        self.last_lap_time = lap_time
        
        # Update best lap time
        if self.best_lap_time is None or lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
        
        self.laps_completed += 1
        self.lap_start_step = self.step_count
        return True

    def get_current_lap_time(self) -> float:
        """Return the in-progress lap time in seconds."""
        return (self.step_count - self.lap_start_step) / FPS

    def _crossed_finish_line(self, previous_position: tuple[float, float], 
                             current_position: tuple[float, float]) -> bool:
        """Return True when the car centre crosses the finish line left-to-right."""
        line_x = TRACK_CENTER_X
        min_y = self.finish_line_start[1]
        max_y = self.finish_line_end[1]
        prev_x, prev_y = previous_position
        curr_x, curr_y = current_position
        
        on_line_band = min_y <= prev_y <= max_y and min_y <= curr_y <= max_y
        crossed_line = prev_x < line_x <= curr_x
        return on_line_band and crossed_line

    def _build_finish_line(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the finish line endpoints."""
        return (
            (TRACK_CENTER_X, TRACK_CENTER_Y + TRACK_INNER_RADIUS_Y),
            (TRACK_CENTER_X, TRACK_CENTER_Y + TRACK_OUTER_RADIUS_Y),
        )

    @staticmethod
    def format_time(value: float | None) -> str:
        """Format a lap time for display."""
        if value is None:
            return "--.--s"
        return f"{value:5.2f}s"
