"""Screen classification with hysteresis and dwell duration filtering."""

import time


class ScreenClassifier:
    """Classifies continuous horizontal gaze score into discrete screen indices with hysteresis.

    Screen indices:
      0 = Left screen
      1 = Center screen
      2 = Right screen
    """

    def __init__(
        self,
        left_enter: float = 0.60,
        left_exit: float = 0.55,
        right_enter: float = 0.40,
        right_exit: float = 0.45,
        initial_screen: int = 1,
    ):
        if left_exit > left_enter:
            raise ValueError("left_exit cannot be greater than left_enter")
        if right_exit < right_enter:
            raise ValueError("right_exit cannot be less than right_enter")

        self.left_enter = left_enter
        self.left_exit = left_exit
        self.right_enter = right_enter
        self.right_exit = right_exit
        self.current_screen = initial_screen

    def classify(self, score: float) -> int:
        """Evaluate continuous score against current state and hysteresis boundaries."""
        if self.current_screen == 0:  # Currently Left
            if score < self.left_exit:
                self.current_screen = 2 if score <= self.right_enter else 1
        elif self.current_screen == 2:  # Currently Right
            if score > self.right_exit:
                self.current_screen = 0 if score >= self.left_enter else 1
        else:  # Currently Center (1)
            if score >= self.left_enter:
                self.current_screen = 0
            elif score <= self.right_enter:
                self.current_screen = 2

        return self.current_screen


class DwellFilter:
    """Debounces candidate screen switches by requiring gaze to dwell for a minimum duration."""

    def __init__(self, dwell_threshold_sec: float = 0.25, initial_focused_screen: int | None = None):
        self.dwell_threshold_sec = dwell_threshold_sec
        self.current_focused_screen = initial_focused_screen
        self.active_target: int | None = None
        self.target_start_time: float = 0.0

    def update(self, detected_screen: int, timestamp: float | None = None) -> int | None:
        """Process detected screen candidate at given timestamp.

        Returns newly committed screen index when dwell threshold is reached, or None.
        """
        now = time.time() if timestamp is None else timestamp

        if detected_screen == self.current_focused_screen:
            self.active_target = None
            return None

        if detected_screen != self.active_target:
            self.active_target = detected_screen
            self.target_start_time = now
            return None

        if (now - self.target_start_time) >= self.dwell_threshold_sec:
            committed_screen = self.active_target
            self.current_focused_screen = committed_screen
            self.active_target = None
            return committed_screen

        return None
