"""Unit tests for Sway, i3, macOS Yabai, and Osascript backends."""

import json
from unittest.mock import ANY, MagicMock, patch

from eyegaze_focus.ipc import IpcClient, OsascriptBackend, SwayI3Backend, YabaiBackend


def test_sway_i3_output_sorting():
    fake_outputs = [
        {"name": "DP-2", "active": True, "rect": {"x": 1920, "y": 0, "width": 1920, "height": 1080}},
        {"name": "HDMI-A-1", "active": True, "rect": {"x": 3840, "y": 0, "width": 1920, "height": 1080}},
        {"name": "eDP-1", "active": True, "rect": {"x": 0, "y": 0, "width": 1920, "height": 1080}},
        {"name": "VIRTUAL-1", "active": False, "rect": {"x": 0, "y": 0, "width": 1920, "height": 1080}},
    ]
    mock_res = MagicMock(stdout=json.dumps(fake_outputs))

    with patch("shutil.which", return_value="/usr/bin/swaymsg"):
        with patch("subprocess.run", return_value=mock_res):
            backend = SwayI3Backend("swaymsg")
            outputs = backend.refresh_outputs()
            assert outputs == ["eDP-1", "DP-2", "HDMI-A-1"]

            success = backend.focus_display("DP-2", dry_run=True)
            assert success is True


def test_yabai_backend_sorting_and_focus():
    fake_displays = [
        {"id": 2, "index": 2, "frame": {"x": 1920.0, "y": 0.0, "w": 2560.0, "h": 1440.0}},
        {"id": 1, "index": 1, "frame": {"x": 0.0, "y": 0.0, "w": 1920.0, "h": 1080.0}},
    ]
    mock_res = MagicMock(stdout=json.dumps(fake_displays))

    with patch("shutil.which", return_value="/usr/local/bin/yabai"):
        with patch("subprocess.run", return_value=mock_res) as mock_run:
            backend = YabaiBackend()
            outputs = backend.refresh_outputs()
            assert outputs == ["Display-1", "Display-2"]

            # Focus Display-1 (x=0)
            success = backend.focus_display("Display-1", dry_run=False)
            assert success is True
            mock_run.assert_called_with(
                ["yabai", "-m", "display", "--focus", "1"],
                stdout=ANY,
                stderr=ANY,
                check=True,
            )


def test_yabai_fallback_to_osascript_when_missing():
    mock_fallback = MagicMock()
    mock_fallback.name = "osascript"
    mock_fallback.refresh_outputs.return_value = ["Mac-Display-1"]

    with patch("shutil.which", return_value=None):
        backend = YabaiBackend(fallback=mock_fallback)
        outputs = backend.refresh_outputs()
        assert outputs == ["Mac-Display-1"]
        mock_fallback.refresh_outputs.assert_called_once()


def test_osascript_backend_sorting_and_focus():
    fake_screens = [
        {"name": "Secondary", "x": 1920.0, "y": 0.0, "w": 1920.0, "h": 1080.0},
        {"name": "Primary", "x": 0.0, "y": 0.0, "w": 1920.0, "h": 1080.0},
    ]
    mock_res = MagicMock(stdout=json.dumps(fake_screens))

    with patch("shutil.which", return_value="/usr/bin/osascript"):
        with patch("subprocess.run", return_value=mock_res) as mock_run:
            backend = OsascriptBackend()
            outputs = backend.refresh_outputs()
            assert outputs == ["Primary", "Secondary"]

            success = backend.focus_display("Primary", dry_run=False)
            assert success is True
            args = mock_run.call_args[0][0]
            assert "osascript" in args[0]
            assert "$.CGWarpMouseCursorPosition" in args[4]
            assert "960.0" in args[4]


def test_ipc_client_auto_selection():
    # macOS with yabai
    with patch("sys.platform", "darwin"):
        with patch("shutil.which", side_effect=lambda x: "/usr/local/bin/" + x if x == "yabai" else None):
            with patch.object(YabaiBackend, "refresh_outputs", return_value=["D1", "D2"]):
                client = IpcClient(backend="auto")
                assert isinstance(client.backend, YabaiBackend)

    # macOS without yabai (osascript fallback)
    with patch("sys.platform", "darwin"):
        with patch("shutil.which", return_value=None):
            with patch.object(OsascriptBackend, "refresh_outputs", return_value=["D1"]):
                client = IpcClient(backend="auto")
                assert isinstance(client.backend, OsascriptBackend)

    # Explicit backend yabai
    with patch.object(YabaiBackend, "refresh_outputs", return_value=["D1"]):
        client = IpcClient(backend="yabai")
        assert isinstance(client.backend, YabaiBackend)


def test_ipc_screen_mapping():
    with patch("shutil.which", return_value=None):
        client = IpcClient(dry_run=True)
        client._outputs = ["LEFT-DISP", "CENTER-DISP", "RIGHT-DISP"]

        assert client.get_output_for_screen(0) == "LEFT-DISP"
        assert client.get_output_for_screen(1) == "CENTER-DISP"
        assert client.get_output_for_screen(2) == "RIGHT-DISP"
        assert client.focus_screen(0) is True
