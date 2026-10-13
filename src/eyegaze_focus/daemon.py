"""Main daemon and CLI entrypoint for eyegaze-focus."""

import argparse
import logging
from pathlib import Path
import signal
import sys
import time

import cv2
import mediapipe as mp

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier
from eyegaze_focus.config import get_default_config_path, load_config, save_config
from eyegaze_focus.geometry import (
    calculate_ear,
    compute_gaze_angle_deg,
    estimate_head_pose,
    get_bilateral_iris_ratio,
)
from eyegaze_focus.ipc import IpcClient

logger = logging.getLogger("eyegaze_focus")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # First pass to discover if --config was specified
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=Path, default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_path = pre_args.config or get_default_config_path()
    cfg = load_config(config_path)

    parser = argparse.ArgumentParser(
        description="Gaze-driven discrete display focus daemon for Sway, i3, and macOS.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_path,
        help=f"Path to TOML configuration file (default: {config_path})",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save current configuration to the config file and exit",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=cfg.get("camera", 0),
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=cfg.get("fps", 30.0),
        help="Target processing FPS (default: 30.0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=cfg.get("width", 640),
        help="Capture width (default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=cfg.get("height", 480),
        help="Capture height (default: 480)",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=cfg.get("dwell", 0.15),
        help="Sustained dwell threshold in seconds (default: 0.15)",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=cfg.get("ear_threshold", 0.20),
        help="Eye Aspect Ratio threshold below which eye is considered closed (default: 0.20)",
    )
    parser.add_argument(
        "--center-span",
        type=float,
        default=cfg.get("center_span", 25.0),
        help="Half-width of center screen deadband in degrees (default: 25.0 -> [-25°, +25°])",
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=cfg.get("hysteresis", 5.0),
        help="Hysteresis margin in degrees before releasing screen focus (default: 5.0)",
    )
    parser.add_argument(
        "--iris-weight",
        type=float,
        default=cfg.get("iris_weight", 20.0),
        help="Maximum ocular deflection contribution in degrees (default: 20.0)",
    )
    parser.add_argument(
        "--center-yaw-offset",
        type=float,
        default=cfg.get("center_yaw_offset", 0.0),
        help="Offset in degrees to subtract from raw head yaw for center baseline (default: 0.0)",
    )
    parser.add_argument(
        "--pitch-down-ignore",
        type=float,
        default=cfg.get("pitch_down_ignore", 25.0),
        help="Downward head pitch threshold in degrees to ignore gaze (looking at keyboard) (default: 25.0)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sway", "i3", "yabai", "osascript"],
        default=cfg.get("backend", "auto"),
        help="Window manager display focus backend (default: auto)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run in interactive calibration mode without dispatching IPC commands",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log IPC commands without executing them",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args(argv)


def run_daemon(args: argparse.Namespace) -> int:
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.save_config:
        values = {
            "camera": args.camera,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "dwell": args.dwell,
            "ear_threshold": args.ear_threshold,
            "pitch_down_ignore": args.pitch_down_ignore,
            "center_span": args.center_span,
            "hysteresis": args.hysteresis,
            "iris_weight": args.iris_weight,
            "center_yaw_offset": args.center_yaw_offset,
            "backend": args.backend,
        }
        saved_path = save_config(values, args.config)
        logger.info("Saved configuration to %s", saved_path)
        print(f"Configuration successfully saved to {saved_path}")
        return 0

    ipc_client = IpcClient(backend=args.backend, dry_run=args.dry_run or args.calibrate)
    logger.info("Using backend: %s | Detected outputs: %s", ipc_client.backend.name, ipc_client.outputs)

    classifier = ScreenClassifier(center_span_deg=args.center_span, hysteresis_deg=args.hysteresis)
    dwell_filter = DwellFilter(dwell_threshold_sec=args.dwell)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        logger.error("Unable to open camera index %d", args.camera)
        return 1

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    running = True

    def _sig_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    frame_interval = 1.0 / max(1.0, args.fps)
    screen_names = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}

    logger.info(
        "Starting low-latency gaze tracking loop (Center: [-%s°, +%s°], Dwell: %.2fs, Target FPS: %.1f)",
        args.center_span,
        args.center_span,
        args.dwell,
        args.fps,
    )
    if args.calibrate:
        print(f"\n--- CALIBRATION MODE (Press Ctrl+C to exit) | Config: {args.config} ---")
        print("Format: [EAR] [Yaw / Pitch] [Iris Ratio] [Gaze Angle] -> Screen (Output)\n")

    try:
        while running and cap.isOpened():
            start_loop = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                _, _, avg_ear = calculate_ear(landmarks, w, h)

                if avg_ear >= args.ear_threshold:
                    head_yaw, head_pitch = estimate_head_pose(landmarks, w, h)

                    # Check if user is looking down at keyboard
                    if head_pitch >= args.pitch_down_ignore:
                        if args.calibrate:
                            sys.stdout.write(
                                f"\rEAR: {avg_ear:.2f} | Pitch: {head_pitch:+5.1f}° | "
                                f"[KEYBOARD LOOK (>= {args.pitch_down_ignore:.0f}°) - FOCUS PAUSED]{' ' * 20}"
                            )
                            sys.stdout.flush()
                        # Do not update dwell filter; freeze current focus
                        continue

                    iris_ratio = get_bilateral_iris_ratio(landmarks, w, h)
                    gaze_angle = compute_gaze_angle_deg(
                        iris_ratio=iris_ratio,
                        head_yaw_deg=head_yaw,
                        iris_weight_deg=args.iris_weight,
                        yaw_offset_deg=args.center_yaw_offset,
                    )
                    detected_screen = classifier.classify(gaze_angle)
                    output_name = ipc_client.get_output_for_screen(detected_screen)

                    if args.calibrate:
                        effective_yaw = head_yaw - args.center_yaw_offset
                        sys.stdout.write(
                            f"\rEAR: {avg_ear:.2f} | Yaw: {effective_yaw:+5.1f}° | Pitch: {head_pitch:+5.1f}° | "
                            f"Iris: {iris_ratio:.2f} | Gaze: {gaze_angle:+5.1f}° -> "
                            f"{screen_names.get(detected_screen, '?'):<6} ({output_name}){' ' * 10}"
                        )
                        sys.stdout.flush()
                    else:
                        committed = dwell_filter.update(detected_screen, timestamp=start_loop)
                        if committed is not None:
                            logger.info(
                                "Focusing screen %d (%s) on output %s",
                                committed,
                                screen_names.get(committed, "?"),
                                output_name,
                            )
                            ipc_client.focus_screen(committed)
                else:
                    if args.calibrate:
                        sys.stdout.write(f"\rEAR: {avg_ear:.2f} | [BLINK DETECTED - SKIPPED]{' ' * 35}")
                        sys.stdout.flush()
            else:
                if args.calibrate:
                    sys.stdout.write(f"\r[NO FACE DETECTED]{' ' * 50}")
                    sys.stdout.flush()

            elapsed = time.time() - start_loop
            sleep_time = max(0.0, frame_interval - elapsed)
            if sleep_time > 0.002:
                time.sleep(sleep_time)

    finally:
        if args.calibrate:
            print("\nExiting calibration mode.")
        cap.release()
        face_mesh.close()
        logger.info("eyegaze-focus daemon stopped.")

    return 0


def main() -> None:
    args = parse_args()
    sys.exit(run_daemon(args))


if __name__ == "__main__":
    main()
