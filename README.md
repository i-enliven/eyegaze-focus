# eyegaze-focus

Lightweight gaze-driven screen focus daemon for **Sway**, **i3**, and **macOS** (**yabai** with native **osascript** fallback).

Instead of performing high-frequency, jittery 2D cursor emulation, `eyegaze-focus` uses coarse discrete classification: **Left (0)**, **Center (1)**, or **Right (2)** mapped directly to window manager IPC calls:
* **Sway / i3**: `swaymsg focus output <NAME>` / `i3-msg focus output <NAME>`
* **macOS (yabai)**: `yabai -m display --focus <INDEX>`
* **macOS (native fallback)**: `osascript` (JXA / CoreGraphics mouse-warp focus)

---

## Architecture & Pipeline

```text
[Webcam (640x480 @ 15fps)]
           │
           ▼
[MediaPipe FaceMesh (refine_landmarks=True)]
     │                                │
     ▼                                ▼
[Blink Gate (EAR < 0.20)]      [Head Pose (solvePnP Yaw)]
     │ (Pass if open)                 │
     ▼                                ▼
[Bilateral Iris Ratio]         [User-Centric Yaw (+Left, -Right)]
     │                                │
     └───────────────┬────────────────┘
                     ▼
         [Combined Gaze Angle (°)]
                     │
                     ▼
      [Dwell & Hysteresis State Machine]
       (Center: [-25°, +25°] by default)
                     │
                     ▼
      [IPC / Display Focus Dispatcher]
   ┌─────────────────┼──────────────────┐
   ▼                 ▼                  ▼
[Sway / i3]       [yabai]          [osascript]
(Linux IPC)     (macOS TWM)     (macOS Fallback)
```

### Key Technical Characteristics
1. **Physical Gaze Angle Fusion**: Combines head yaw in degrees with bilateral ocular deflection ($\text{Gaze Angle} = \text{Yaw}^\circ + 20^\circ \cdot (\text{Iris} - 0.50)$).
2. **User-Centric Orientation**:
   - Turning to user's **Left** produces positive degrees ($+^\circ$), targeting Left monitor (`Screen 0`).
   - Looking straight at **Center** produces $\approx 0^\circ$, targeting Center monitor (`Screen 1`).
   - Turning to user's **Right** produces negative degrees ($-^\circ$), targeting Right monitor (`Screen 2`).
3. **Wide Center Deadband**: Center screen spans a full physical angular range (default: **$[-25^\circ, +25^\circ]$**), providing a natural resting zone without accidental focus shifts while reading or looking across the main screen.
4. **Hysteresis Stability**: Configurable hysteresis boundary (default $5^\circ$):
   - Left Screen: enters at $\ge +25^\circ$, releases at $< +20^\circ$.
   - Right Screen: enters at $\le -25^\circ$, releases at $> -20^\circ$.
5. **Blink Gating**: Computes Eye Aspect Ratio (EAR). Frames where eyes are closed are skipped to prevent eyelid seam collapses from triggering focus switches.
6. **Dwell Debounce**: Requires sustained gaze (default: 250ms) before committing focus switches to ignore casual saccades.
7. **Automatic Multi-Monitor Geometry Discovery**:
   - **Linux**: Automatically queries `swaymsg -t get_outputs` (or `i3-msg`) and sorts active displays by `rect.x`.
   - **macOS (yabai)**: Queries `yabai -m query --displays` and sorts displays by `frame.x`.
   - **macOS (osascript)**: Queries `$.NSScreen.screens` via native JXA and sorts by screen `origin.x`.

---

## Installation

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/i-enliven/eyegaze-focus.git
cd eyegaze-focus
uv sync
```

---

## Calibration

Run in calibration mode to inspect live metrics:

```bash
uv run eyegaze-focus --calibrate
```

Sample output:
```text
EAR: 0.28 | Head:  +8.2° | Iris: 0.51 | Gaze Angle:  +8.4° -> CENTER (eDP-1)
```

If your camera is offset horizontally relative to your center screen (e.g. mounted to the side), note the resting Yaw shown when looking directly at your center monitor, and supply it via `--center-yaw-offset`:

```bash
uv run eyegaze-focus --calibrate --center-yaw-offset 5.0
```

To adjust the angular width of your center display:

```bash
uv run eyegaze-focus --center-span 30.0 --hysteresis 6.0
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--backend` | `auto` | Display backend (`auto`, `sway`, `i3`, `yabai`, `osascript`) |
| `--calibrate` | `False` | Run interactive terminal monitor without firing IPC |
| `--dry-run` | `False` | Log IPC commands to stdout without executing them |
| `--camera` | `0` | V4L2 / AVFoundation webcam device index |
| `--fps` | `15.0` | Capture and inference frame rate |
| `--dwell` | `0.25` | Dwell duration in seconds required before switching |
| `--center-span` | `25.0` | Half-width of center screen deadband in degrees (`[-25°, +25°]`) |
| `--hysteresis` | `5.0` | Hysteresis margin in degrees before releasing screen focus |
| `--iris-weight` | `20.0` | Maximum ocular deflection contribution in degrees |
| `--center-yaw-offset` | `0.0` | Baseline offset in degrees subtracted from raw head yaw |
| `--ear-threshold` | `0.20` | Minimum EAR below which frame is discarded as a blink |
| `--width` | `640` | Video capture width |
| `--height` | `480` | Video capture height |
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

> **Note on macOS Permissions**:
> On first run, macOS will prompt for Camera access for your terminal emulator (Terminal, iTerm2, Kitty). Grant permission in **System Settings > Privacy & Security > Camera**.

---

## Testing

Run unit tests via `pytest`:

```bash
uv run pytest -v
```

---

## License

MIT
