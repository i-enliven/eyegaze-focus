# eyegaze-focus

Lightweight gaze-driven screen focus daemon for **Sway** and **i3** tiling window managers.

Instead of performing high-frequency, jittery 2D cursor emulation, `eyegaze-focus` uses coarse discrete classification: **Left (0)**, **Center (1)**, or **Right (2)** mapped directly to window manager IPC calls (`swaymsg focus output <NAME>` or `i3-msg focus output <NAME>`).

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
     ▼                                │
[Bilateral Iris Ratio]                │
     │                                │
     └───────────────┬────────────────┘
                     ▼
       [Composite Horizontal Score]
                     │
                     ▼
      [Dwell & Hysteresis State Machine]
                     │
                     ▼
       [swaymsg / i3-msg focus output]
```

### Key Technical Characteristics
1. **Composite Gaze Fusion**: Fuses bilateral eye iris ratios with 3D Head Pose yaw estimation (`cv2.solvePnP` using canonical 3D facial anchors). This prevents misclassification caused by natural head turns where irises re-center in eye sockets.
2. **Bilateral Tracking**: Simultaneously tracks both eyes to avoid epicanthic occlusion errors when looking to lateral extremes.
3. **Blink Gating**: Computes Eye Aspect Ratio (EAR). Frames where eyes are closed are skipped to prevent eyelid seam landmark collapse from corrupting the dwell state machine.
4. **Hysteresis Boundaries**: Dual entry and exit thresholds prevent oscillation along screen borders:
   - Left Screen: enters at `score >= 0.60`, exits at `score < 0.55`.
   - Right Screen: enters at `score <= 0.40`, exits at `score > 0.45`.
5. **Dwell Debounce**: Requires sustained gaze (default: 250ms) before committing focus switches to ignore casual saccades.
6. **Automatic Output Geometry Discovery**: Automatically queries `swaymsg -t get_outputs` (or `i3-msg`), filters active monitors, and sorts them horizontally by `rect.x` coordinate.

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

Run in calibration mode to inspect live metrics and tune thresholds for your physical monitor layout and webcam angle:

```bash
uv run eyegaze-focus --calibrate
```

Sample output:
```text
EAR: 0.28 | Yaw: +16.2° | Iris: 0.61 | Score: 0.77 -> LEFT   (DP-1)
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--calibrate` | `False` | Run interactive terminal monitor without firing IPC |
| `--dry-run` | `False` | Log IPC commands to stdout without executing them |
| `--camera` | `0` | V4L2 webcam device index (`/dev/video0`) |
| `--fps` | `15.0` | Capture and inference frame rate |
| `--dwell` | `0.25` | Dwell duration in seconds required before switching |
| `--ear-threshold` | `0.20` | Minimum EAR below which frame is discarded as a blink |
| `--yaw-weight` | `0.015` | Multiplier for head yaw in composite gaze score |
| `--width` | `640` | Video capture width |
| `--height` | `480` | Video capture height |
| `-v, --verbose` | `False` | Enable debug logging |

---

## Sway / i3 Configuration

Add the daemon to your Sway config (`~/.config/sway/config`) or i3 config:

```text
exec_always --no-startup-id /path/to/eyegaze-focus/.venv/bin/eyegaze-focus
```

---

## Testing

Run unit tests via `pytest`:

```bash
uv run pytest -v
```

---

## License

MIT
