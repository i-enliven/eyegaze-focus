# eyegaze-focus

Lightweight gaze-driven screen focus daemon for **Sway**, **i3**, and **macOS** (**yabai** with native **osascript** fallback).

Instead of performing high-frequency, jittery 2D cursor emulation, `eyegaze-focus` uses coarse discrete classification: **Left (0)**, **Center (1)**, or **Right (2)** mapped directly to window manager IPC calls:
* **Sway / i3**: `swaymsg focus output <NAME>` / `i3-msg focus output <NAME>`
* **macOS (yabai)**: `yabai -m display --focus <INDEX>`
* **macOS (native fallback)**: `osascript` (JXA / CoreGraphics mouse-warp focus)

---

## Architecture & Pipeline

```text
[Webcam (640x480 @ 30fps, zero-buffer)]
                  │
                  ▼
[MediaPipe FaceMesh (refine_landmarks=True)]
     │                    │             │
     ▼                    ▼             ▼
[Blink Gate]        [Head Pitch]   [Head Yaw]
 (EAR < 0.20)      (Keyboard Look) (User-Centric: +Left, -Right)
     │                    │             │
     │       [Pitch >= 25°? Freeze]     │
     ▼                    │             ▼
[Bilateral Iris Ratio]    │     [Combined Gaze Angle (°)]
     │                    │             │
     └────────────────────┴─────────────┤
                                        ▼
                        [Dwell & Hysteresis State Machine]
                         (Center: [-25°, +25°], Dwell: 150ms)
                                        │
                                        ▼
                        [IPC / Display Focus Dispatcher]
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
                  [Sway / i3]        [yabai]          [osascript]
                  (Linux IPC)      (macOS TWM)     (macOS Fallback)
```

### Key Technical Characteristics
1. **Low-Latency Camera Pipeline**:
   * Camera circular buffer queue is pinned to 1 frame (`CAP_PROP_BUFFERSIZE=1`), eliminating stale driver queues and cutting buffer delay from ~130ms to 0ms.
   * Paced directly by hardware camera VSYNC at 30 FPS.
2. **Physical Gaze Angle Fusion**: Combines head yaw in degrees with bilateral ocular deflection ($\text{Gaze Angle} = \text{Yaw}^\circ + 20^\circ \cdot (\text{Iris} - 0.50)$).
3. **Keyboard Glance Gating (`pitch_down_ignore`)**: Automatically pauses gaze transitions whenever your head tilts downward (e.g. past $+25^\circ$) to look at your keyboard or desk, preventing accidental focus shifts while typing.
4. **User-Centric Orientation**:
   * Turning to user's **Left** produces positive degrees ($+^\circ$), targeting Left monitor (`Screen 0`).
   * Looking straight at **Center** produces $\approx 0^\circ$, targeting Center monitor (`Screen 1`).
   * Turning to user's **Right** produces negative degrees ($-^\circ$), targeting Right monitor (`Screen 2`).
5. **Wide Center Deadband**: Center screen spans a full physical angular range (default: **$[-25^\circ, +25^\circ]$**), providing a natural resting zone without accidental focus shifts while reading across the main screen.
6. **Fast Dwell Debounce**: 150ms dwell requirement (4–5 consecutive confirmation frames at 30 FPS) provides instantaneous transitions while ignoring casual micro-saccades.
7. **Hysteresis Stability**: Configurable hysteresis boundary (default $5^\circ$):
   * Left Screen: enters at $\ge +25^\circ$, releases at $< +20^\circ$.
   * Right Screen: enters at $\le -25^\circ$, releases at $> -20^\circ$.
8. **Blink Gating**: Computes Eye Aspect Ratio (EAR). Frames where eyes are closed are skipped to prevent eyelid seam collapses from triggering focus switches.
9. **Automatic Multi-Monitor Geometry Discovery**:
   * **Linux**: Queries `swaymsg -t get_outputs` (or `i3-msg`) and sorts active displays by `rect.x`.
   * **macOS (yabai)**: Queries `yabai -m query --displays` and sorts displays by `frame.x`.
   * **macOS (osascript)**: Queries `$.NSScreen.screens` via native JXA and sorts by screen `origin.x`.

---

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/i-enliven/eyegaze-focus.git
cd eyegaze-focus
uv sync
```

---

## Configuration Guide

`eyegaze-focus` supports a human-editable TOML configuration file.

### Default Configuration Location
* **Linux / macOS**: `~/.config/eyegaze-focus/config.toml` (respects `$XDG_CONFIG_HOME`)

### Precedence Hierarchy
Settings are resolved in three tiers:
$$\text{CLI Flags} > \text{config.toml} > \text{Hardcoded Defaults}$$

### Generating a Default Configuration File
Run with `--save-config` to generate a formatted, commented `config.toml`:

```bash
uv run eyegaze-focus --save-config
```

You can also pass custom values while saving:
```bash
uv run eyegaze-focus --center-span 28.0 --pitch-down-ignore 30.0 --save-config
```

### Example `config.toml` Reference

```toml
# eyegaze-focus configuration file
# Location: ~/.config/eyegaze-focus/config.toml

[tracking]
# Video capture device index (0 = default webcam)
camera = 0

# Capture and inference frame rate
fps = 30.0

# Capture resolution
width = 640
height = 480

# Dwell duration required before switching focus (seconds)
dwell = 0.15

# Eye Aspect Ratio threshold (frames below this are skipped as blinks)
ear_threshold = 0.20

# Ignore gaze focus switches when head tilts downward past this angle (looking at keyboard)
pitch_down_ignore = 25.0

# Disable eye/iris tracking and determine focus strictly from head orientation
disable_eye_tracking = false

[geometry]
# Half-width of center screen deadband in degrees (center spans [-25.0°, +25.0°])
center_span = 25.0

# Hysteresis margin in degrees before releasing focus back to center
hysteresis = 5.0

# Maximum ocular deflection contribution in degrees
iris_weight = 20.0

# Baseline head yaw offset in degrees to compensate for camera positioning
center_yaw_offset = 0.0

[backend]
# Window manager display backend: "auto", "sway", "i3", "yabai", or "osascript"
name = "auto"
```

---

## Head-Only Tracking Mode

If you wear thick glasses with reflections, experience lighting conditions where iris landmarks are noisy, or prefer intentional head turns to trigger screen switches, enable head-only tracking:

```bash
uv run eyegaze-focus --head-only
```

Or persist it in `~/.config/eyegaze-focus/config.toml`:
```toml
[tracking]
disable_eye_tracking = true
```

In head-only mode:
* **Zero blink dropouts**: MediaPipe EAR blink gating is skipped, keeping head tracking continuous during blinks or squinting.
* **Lower latency**: Iris landmark extraction is bypassed.
* **Simple mental model**: Focus is calculated strictly from calibrated head yaw: $\text{Angle}^\circ = \text{HeadYaw}^\circ - \text{CenterYawOffset}^\circ$.

---

## Calibration & Tuning Workflow

1. **Start Calibration Mode**:
   ```bash
   uv run eyegaze-focus --calibrate
   ```
   *(Suppresses all window manager IPC calls so your screen focus will not shift while inspecting values).*

2. **Calibrate Center Display**:
   Look naturally across your center monitor from left edge to right edge. Observe `Gaze Angle` on the terminal status line:
   ```text
   EAR: 0.28 | Yaw:  +6.2° | Pitch: +12.0° | Iris: 0.51 | Gaze:  +6.4° -> CENTER (eDP-1)
   ```
   * If your gaze angle reaches $\pm 28^\circ$ on wide screens, increase `center_span` to `30.0`:
     ```bash
     uv run eyegaze-focus --center-span 30.0 --save-config
     ```
   * If your camera is mounted off-center and sitting straight shows e.g. `Yaw: +4.0°`, set `center_yaw_offset`:
     ```bash
     uv run eyegaze-focus --center-yaw-offset 4.0 --save-config
     ```

3. **Calibrate Keyboard Ignore Threshold**:
   Look down at your keyboard as you would while typing. Note the `Pitch` reading:
   ```text
   EAR: 0.28 | Pitch: +28.5° | [KEYBOARD LOOK (>= 25°) - FOCUS PAUSED]
   ```
   * If keyboard look triggers prematurely while looking at the bottom of your screen, raise `pitch_down_ignore` (e.g. `30.0` or `35.0`).
   * If typing still occasionally shifts focus, lower it (e.g. `20.0` or `22.0`).

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--config` | `~/.config/.../config.toml` | Custom path to TOML configuration file |
| `--save-config` | `False` | Write active settings to config file and exit |
| `--pitch-down-ignore` | `25.0` | Downward pitch in degrees to ignore gaze (keyboard look) |
| `--disable-eye-tracking`, `--head-only` | `False` | Disable iris gaze tracking and follow head orientation only |
| `--enable-eye-tracking` | `True` | Enable iris gaze tracking (overrides `config.toml`) |
| `--center-span` | `25.0` | Half-width of center screen deadband in degrees (`[-25°, +25°]`) |
| `--hysteresis` | `5.0` | Hysteresis margin in degrees before releasing screen focus |
| `--center-yaw-offset` | `0.0` | Baseline offset in degrees subtracted from raw head yaw |
| `--dwell` | `0.15` | Dwell duration in seconds required before switching |
| `--fps` | `30.0` | Capture and inference frame rate |
| `--iris-weight` | `20.0` | Maximum ocular deflection contribution in degrees |
| `--camera` | `0` | V4L2 / AVFoundation webcam device index |
| `--backend` | `auto` | Display backend (`auto`, `sway`, `i3`, `yabai`, `osascript`) |
| `--calibrate` | `False` | Run interactive terminal monitor without firing IPC |
| `--dry-run` | `False` | Log IPC commands to stdout without executing them |
| `-v, --verbose` | `False` | Enable debug logging |

---

## Window Manager Integration

### Sway / i3 (Linux)

Add the daemon to your Sway config (`~/.config/sway/config`) or i3 config:

```text
exec_always --no-startup-id /path/to/eyegaze-focus/.venv/bin/eyegaze-focus
```

### yabai / macOS

Add to your `~/.yabairc`:

```bash
# Start eyegaze-focus daemon in background
/path/to/eyegaze-focus/.venv/bin/eyegaze-focus &
```

> **macOS Permissions**:
> On first run, macOS will prompt for Camera access for your terminal emulator. Grant permission in **System Settings > Privacy & Security > Camera**.

---

## Testing

Run unit tests via `pytest`:

```bash
uv run pytest -v
```

---

## License

MIT
