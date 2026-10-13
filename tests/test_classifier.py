"""Unit tests for ScreenClassifier and DwellFilter."""

import pytest

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier


def test_screen_classifier_initialization_validation():
    with pytest.raises(ValueError):
        ScreenClassifier(left_enter=0.50, left_exit=0.60)
    with pytest.raises(ValueError):
        ScreenClassifier(right_enter=0.45, right_exit=0.40)


def test_screen_classifier_hysteresis_left():
    # left_enter=0.60, left_exit=0.55
    clf = ScreenClassifier(initial_screen=1)

    # Center to Left
    assert clf.classify(0.58) == 1  # Not past left_enter yet
    assert clf.classify(0.61) == 0  # Reached left_enter -> transitions to 0

    # Remains Left despite minor dip down to 0.56 (above left_exit 0.55)
    assert clf.classify(0.56) == 0

    # Crosses below left_exit -> transitions back to Center
    assert clf.classify(0.54) == 1


def test_screen_classifier_hysteresis_right():
    # right_enter=0.40, right_exit=0.45
    clf = ScreenClassifier(initial_screen=1)

    # Center to Right
    assert clf.classify(0.42) == 1  # Not below right_enter yet
    assert clf.classify(0.39) == 2  # Reached right_enter -> transitions to 2

    # Remains Right despite slight rise to 0.44 (below right_exit 0.45)
    assert clf.classify(0.44) == 2

    # Crosses above right_exit -> transitions back to Center
    assert clf.classify(0.46) == 1


def test_dwell_filter_behavior():
    filter_ = DwellFilter(dwell_threshold_sec=0.25, initial_focused_screen=1)

    # Target remains current focused screen -> no event
    assert filter_.update(1, timestamp=10.0) is None

    # Change to screen 0 at t=10.1s
    assert filter_.update(0, timestamp=10.1) is None

    # After 0.20s (t=10.3s) -> still below 0.25s threshold
    assert filter_.update(0, timestamp=10.3) is None

    # At t=10.36s (delta = 0.26s >= 0.25s) -> triggers focus change
    committed = filter_.update(0, timestamp=10.36)
    assert committed == 0

    # Subsequent frames looking at screen 0 return None (already focused)
    assert filter_.update(0, timestamp=10.40) is None


def test_dwell_filter_jitter_reset():
    filter_ = DwellFilter(dwell_threshold_sec=0.25, initial_focused_screen=1)

    # Saccade to screen 0 at t=10.0
    assert filter_.update(0, timestamp=10.0) is None
    assert filter_.update(0, timestamp=10.15) is None

    # Gaze flinches back to center (screen 1) at t=10.18
    assert filter_.update(1, timestamp=10.18) is None

    # Looking back at screen 0 at t=10.20 resets start time
    assert filter_.update(0, timestamp=10.20) is None

    # At t=10.40 (delta from 10.20 is 0.20s < 0.25s) -> not triggered yet
    assert filter_.update(0, timestamp=10.40) is None

    # At t=10.46 (delta from 10.20 is 0.26s >= 0.25s) -> triggers
    assert filter_.update(0, timestamp=10.46) == 0
