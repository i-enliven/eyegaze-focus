"""Unit tests for configuration file loading and saving."""

from pathlib import Path
from unittest.mock import patch

from eyegaze_focus.config import get_default_config_path, load_config, save_config
from eyegaze_focus.daemon import parse_args


def test_default_config_path_resolution():
    with patch.dict("os.environ", {"XDG_CONFIG_HOME": "/tmp/custom_xdg"}):
        p = get_default_config_path()
        assert p == Path("/tmp/custom_xdg/eyegaze-focus/config.toml")

    with patch.dict("os.environ", {}, clear=True):
        p = get_default_config_path()
        assert p.name == "config.toml"
        assert p.parent.name == "eyegaze-focus"


def test_save_and_load_config_roundtrip(tmp_path):
    config_file = tmp_path / "test_config.toml"
    test_values = {
        "camera": 1,
        "fps": 25.0,
        "width": 1280,
        "height": 720,
        "dwell": 0.20,
        "ear_threshold": 0.22,
        "pitch_down_ignore": 30.0,
        "disable_eye_tracking": True,
        "typing_cooldown": 1.5,
        "center_span": 28.0,
        "hysteresis": 6.0,
        "iris_weight": 22.0,
        "center_yaw_offset": 3.5,
        "backend": "yabai",
    }

    saved_path = save_config(test_values, config_file)
    assert saved_path == config_file
    assert config_file.is_file()

    loaded = load_config(config_file)
    assert loaded["camera"] == 1
    assert loaded["fps"] == 25.0
    assert loaded["width"] == 1280
    assert loaded["dwell"] == 0.20
    assert loaded["pitch_down_ignore"] == 30.0
    assert loaded["disable_eye_tracking"] is True
    assert loaded["typing_cooldown"] == 1.5
    assert loaded["center_span"] == 28.0
    assert loaded["center_yaw_offset"] == 3.5
    assert loaded["backend"] == "yabai"


def test_load_config_head_only_alias(tmp_path):
    config_file = tmp_path / "alias_config.toml"
    config_file.write_text("""
[tracking]
head_only = true
""")
    loaded = load_config(config_file)
    assert loaded.get("disable_eye_tracking") is True


def test_parse_args_head_only_flags(tmp_path):
    empty_config = tmp_path / "empty.toml"
    empty_config.write_text("")

    args1 = parse_args(["--config", str(empty_config)])
    assert args1.disable_eye_tracking is False
    assert args1.typing_cooldown == 1.0

    args2 = parse_args(["--config", str(empty_config), "--disable-eye-tracking", "--typing-cooldown", "2.0"])
    assert args2.disable_eye_tracking is True
    assert args2.typing_cooldown == 2.0

    args3 = parse_args(["--config", str(empty_config), "--head-only"])
    assert args3.disable_eye_tracking is True

    # Test override when config has disable_eye_tracking = true
    head_only_config = tmp_path / "head_only.toml"
    head_only_config.write_text("""[tracking]
disable_eye_tracking = true
typing_cooldown = 0.5
""")

    args4 = parse_args(["--config", str(head_only_config)])
    assert args4.disable_eye_tracking is True
    assert args4.typing_cooldown == 0.5

    args5 = parse_args(["--config", str(head_only_config), "--enable-eye-tracking"])
    assert args5.disable_eye_tracking is False


def test_load_nonexistent_config_returns_empty(tmp_path):
    missing_file = tmp_path / "does_not_exist.toml"
    assert load_config(missing_file) == {}
