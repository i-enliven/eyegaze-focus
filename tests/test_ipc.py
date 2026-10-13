"""Unit tests for IpcClient output querying and screen mapping."""

import json
from unittest.mock import MagicMock, patch

from eyegaze_focus.ipc import IpcClient


def test_ipc_output_sorting():
    # Mock swaymsg get_outputs response with out-of-order x coordinates
    fake_outputs = [
        {"name": "DP-2", "active": True, "rect": {"x": 1920, "y": 0, "width": 1920, "height": 1080}},
        {"name": "HDMI-A-1", "active": True, "rect": {"x": 3840, "y": 0, "width": 1920, "height": 1080}},
        {"name": "eDP-1", "active": True, "rect": {"x": 0, "y": 0, "width": 1920, "height": 1080}},
        {"name": "VIRTUAL-1", "active": False, "rect": {"x": 0, "y": 0, "width": 1920, "height": 1080}},
    ]

    mock_res = MagicMock()
    mock_res.stdout = json.dumps(fake_outputs)

    with patch("shutil.which", return_value="/usr/bin/swaymsg"):
        with patch("subprocess.run", return_value=mock_res):
            client = IpcClient()
            outputs = client.outputs
            assert outputs == ["eDP-1", "DP-2", "HDMI-A-1"]


def test_ipc_screen_mapping_3_monitors():
    with patch("shutil.which", return_value=None):
        client = IpcClient(dry_run=True)
        client._outputs = ["LEFT-DISP", "CENTER-DISP", "RIGHT-DISP"]

        assert client.get_output_for_screen(0) == "LEFT-DISP"
        assert client.get_output_for_screen(1) == "CENTER-DISP"
        assert client.get_output_for_screen(2) == "RIGHT-DISP"


def test_ipc_screen_mapping_2_monitors():
    with patch("shutil.which", return_value=None):
        client = IpcClient(dry_run=True)
        client._outputs = ["LEFT-DISP", "RIGHT-DISP"]

        assert client.get_output_for_screen(0) == "LEFT-DISP"
        assert client.get_output_for_screen(1) == "RIGHT-DISP"
        assert client.get_output_for_screen(2) == "RIGHT-DISP"


def test_ipc_focus_screen_dry_run():
    with patch("shutil.which", return_value=None):
        client = IpcClient(dry_run=True)
        client._outputs = ["MON-1", "MON-2"]
        success = client.focus_screen(0)
        assert success is True
