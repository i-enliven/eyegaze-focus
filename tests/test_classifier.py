"""Unit tests for ScreenClassifier and DwellFilter."""

import pytest

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier


def test_screen_classifier_initialization_validation():
    with pytest.raises(ValueError):
        ScreenClassifier(center_span_deg=-10.0)
    with pytest.raises(ValueError):
        ScreenClassifier(center_span_deg=25.0, hysteresis_deg=25.0)


def test_screen_classifier_hysteresis_left():
    # center_span_deg=25.0, hysteresis_deg=5.0
    # Left enter: >= 25.0, Left exit: < 20.0
    clf = ScreenClassifier(center_span_deg=25.0, hysteresis_deg=5.0, initial_screen=1)

    # Within center: [-25°, +25°]
    assert clf.classify(15.0) == 1
    assert clf.classify(24.0) == 1

    # Crosses +25.0° -> transitions to Left (0)
    assert clf.classify(25.5) == 0

    # Remains Left despite dipping slightly down to 22.0° (above left_exit 20.0°)
    assert clf.classify(22.0) == 0

    # Crosses below left_exit (20.0°) -> transitions back to Center (1)
    assert clf.classify(19.0) == 1


def test_screen_classifier_hysteresis_right():
    # center_span_deg=25.0, hysteresis_deg=5.0
    # Right enter: <= -25.0, Right exit: > -20.0
    clf = ScreenClassifier(center_span_deg=25.0, hysteresis_deg=5.0, initial_screen=1)

    # Within center
    assert clf.classify(-15.0) == 1
    assert clf.classify(-24.0) == 1

    # Crosses -25.0° -> transitions to Right (2)
    assert clf.classify(-25.5) == 2

    # Remains Right despite rising to -22.0° (below right_exit -20.0°)
    assert clf.classify(-22.0) == 2

    # Crosses above right_exit (-20.0°) -> transitions back to Center (1)
    assert clf.classify(-19.0) == 1


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
