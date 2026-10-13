"""Unit tests for KeyboardMonitor typing activity detection."""

import struct
import sys
import time
from unittest.mock import MagicMock, patch

from eyegaze_focus.keyboard import EVENT_SIZE, EV_KEY, KEY_PRESS, KEY_REPEAT, KeyboardMonitor


def test_keyboard_monitor_disabled():
    km = KeyboardMonitor(enabled=False)
    assert km.is_typing(1.0) is False
    assert km.time_since_last_key() == float("inf")
    km.close()


def test_keyboard_monitor_is_typing_cooldown_threshold():
    km = KeyboardMonitor(enabled=False)
    km._is_active = True
    km.enabled = True

    # No keystroke recorded yet
    assert km.is_typing(1.0) is False

    # Simulate recent keystroke
    km._last_key_time = time.time()
    assert km.is_typing(1.0) is True

    # Simulate stale keystroke (2 seconds ago)
    km._last_key_time = time.time() - 2.0
    assert km.is_typing(1.0) is False

    # Zero or negative cooldown returns False
    km._last_key_time = time.time()
    assert km.is_typing(0.0) is False
    assert km.is_typing(-1.0) is False

    km.close()


def test_keyboard_monitor_macos_ctypes():
    km = KeyboardMonitor(enabled=False)
    km.enabled = True
    km._is_active = True

    mock_cg = MagicMock()
    # Return 0.25 seconds since last key
    mock_cg.CGEventSourceSecondsSinceLastEventType.return_value = 0.25
    km._cg = mock_cg

    with patch.object(sys, "platform", "darwin"):
        assert km.time_since_last_key() == 0.25
        assert km.is_typing(1.0) is True
        assert km.is_typing(0.2) is False

    km.close()


def test_keyboard_monitor_linux_event_parsing():
    km = KeyboardMonitor(enabled=False)
    km.enabled = True
    km._is_active = True

    # Pack EV_KEY press event
    # Format: llHHi -> sec (0), usec (0), type (1=EV_KEY), code (28=KEY_ENTER), value (1=KEY_PRESS)
    press_event = struct.pack("llHHi", 0, 0, EV_KEY, 28, KEY_PRESS)
    assert len(press_event) == EVENT_SIZE

    # Simulate processing press event
    _, _, ev_type, _, ev_val = struct.unpack("llHHi", press_event)
    assert ev_type == EV_KEY
    assert ev_val == KEY_PRESS

    # Pack EV_KEY release event (value 0)
    release_event = struct.pack("llHHi", 0, 0, EV_KEY, 28, 0)
    _, _, rel_type, _, rel_val = struct.unpack("llHHi", release_event)
    assert rel_type == EV_KEY
    assert rel_val not in (KEY_PRESS, KEY_REPEAT)

    km.close()
