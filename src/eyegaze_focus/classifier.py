"""Screen classification with hysteresis and dwell duration filtering."""

import time


class ScreenClassifier:
    """Classifies continuous horizontal gaze angle in degrees into discrete screen indices with hysteresis.

    Screen indices:
      0 = Left screen (gaze_angle >= +center_span_deg)
      1 = Center screen (gaze_angle between [-center_span_deg, +center_span_deg])
      2 = Right screen (gaze_angle <= -center_span_deg)
    """

    def __init__(
        self,
        center_span_deg: float = 25.0,
        hysteresis_deg: float = 5.0,
        initial_screen: int = 1,
    ):
        if center_span_deg <= 0.0:
            raise ValueError("center_span_deg must be positive")
        if hysteresis_deg < 0.0 or hysteresis_deg >= center_span_deg:
            raise ValueError("hysteresis_deg must be non-negative and less than center_span_deg")

        self.center_span_deg = center_span_deg
        self.hysteresis_deg = hysteresis_deg
        self.current_screen = initial_screen

    @property
    def left_enter(self) -> float:
        return self.center_span_deg

    @property
    def left_exit(self) -> float:
        return self.center_span_deg - self.hysteresis_deg

    @property
    def right_enter(self) -> float:
        return -self.center_span_deg

    @property
    def right_exit(self) -> float:
        return -(self.center_span_deg - self.hysteresis_deg)

    def classify(self, angle_deg: float) -> int:
        """Evaluate continuous gaze angle against current state and hysteresis boundaries."""
        if self.current_screen == 0:  # Currently Left
            if angle_deg < self.left_exit:
                self.current_screen = 2 if angle_deg <= self.right_enter else 1
        elif self.current_screen == 2:  # Currently Right
            if angle_deg > self.right_exit:
                self.current_screen = 0 if angle_deg >= self.left_enter else 1
        else:  # Currently Center (1)
            if angle_deg >= self.left_enter:
                self.current_screen = 0
            elif angle_deg <= self.right_enter:
                self.current_screen = 2

        return self.current_screen


class DwellFilter:
    """Debounces candidate screen switches by requiring gaze to dwell for a minimum duration."""

    def __init__(self, dwell_threshold_sec: float = 0.15, initial_focused_screen: int | None = None):
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
