"""Main daemon and CLI entrypoint for eyegaze-focus."""

import argparse
import logging
import signal
import sys
import time

import cv2
import mediapipe as mp

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier
from eyegaze_focus.geometry import (
    calculate_ear,
    compute_gaze_score,
    estimate_head_yaw,
    get_bilateral_iris_ratio,
)
from eyegaze_focus.ipc import IpcClient

logger = logging.getLogger("eyegaze_focus")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaze-driven discrete display focus daemon for Sway and i3.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--fps", type=float, default=15.0, help="Target processing FPS (default: 15.0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (default: 480)")
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.25,
        help="Sustained dwell threshold in seconds (default: 0.25)",
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.20,
        help="Eye Aspect Ratio threshold below which eye is considered closed (default: 0.20)",
    )
    parser.add_argument(
        "--yaw-weight",
        type=float,
        default=0.015,
        help="Weight multiplier for head yaw in composite gaze score (default: 0.015)",
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

    ipc_client = IpcClient(dry_run=args.dry_run or args.calibrate)
    logger.info("Detected outputs: %s", ipc_client.outputs)

    classifier = ScreenClassifier()
    dwell_filter = DwellFilter(dwell_threshold_sec=args.dwell)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

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

    logger.info("Starting gaze tracking loop (FPS: %.1f, Dwell: %.2fs)", args.fps, args.dwell)
    if args.calibrate:
        print("\n--- CALIBRATION MODE (Press Ctrl+C to exit) ---")
        print("Format: [EAR] [Yaw Deg] [Iris Ratio] [Gaze Score] -> Screen (Output)\n")

    try:
        while running and cap.isOpened():
            start_loop = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                _, _, avg_ear = calculate_ear(landmarks, w, h)

                if avg_ear >= args.ear_threshold:
                    iris_ratio = get_bilateral_iris_ratio(landmarks, w, h)
                    head_yaw = estimate_head_yaw(landmarks, w, h)
                    score = compute_gaze_score(iris_ratio, head_yaw, yaw_weight=args.yaw_weight)
                    detected_screen = classifier.classify(score)
                    output_name = ipc_client.get_output_for_screen(detected_screen)

                    if args.calibrate:
                        sys.stdout.write(
                            f"\rEAR: {avg_ear:.2f} | Yaw: {head_yaw:+5.1f}° | Iris: {iris_ratio:.2f} | "
                            f"Score: {score:.2f} -> {screen_names.get(detected_screen, '?'):<6} "
                            f"({output_name})"
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
