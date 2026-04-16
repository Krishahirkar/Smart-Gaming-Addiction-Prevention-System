import argparse
import time

import cv2

from Perception.eye_fatigue import EyeFatigueMonitor
from Perception.mediapipe_tracker import create_mediapipe_tracker, get_mediapipe_error
from Perception.rage_detection import DEFAULT_DATA_PATH, append_training_sample
from main import (
    ALERT_COLOR,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FONT,
    GOOD_COLOR,
    MUTED_TEXT,
    PANEL_ACCENT,
    TEXT_COLOR,
    WARNING_COLOR,
    build_rage_features,
    draw_panel,
    draw_panel_text,
    draw_progress_bar,
    face_center,
    head_motion,
)


WINDOW_NAME = "Rage Training Data Collector"


def draw_collector_ui(frame, phase, label, seconds_left, saved_count, detector_name, progress):
    height, width = frame.shape[:2]
    panel_width = min(720, width - 48)
    panel_height = 250
    x = 24
    y = 72
    color = ALERT_COLOR if label == 1 else GOOD_COLOR

    draw_panel(frame, x, y, panel_width, panel_height, border_color=color)
    draw_panel_text(frame, "RAGE MODEL DATA COLLECTION", x + 22, y + 38, PANEL_ACCENT, 0.72, 2)
    draw_panel_text(frame, phase.upper(), x + 22, y + 105, color, 1.35, 3)
    draw_panel_text(frame, f"{seconds_left:0.1f}s left", x + 22, y + 150, TEXT_COLOR, 0.75, 2)
    draw_panel_text(frame, f"Saved samples: {saved_count}", x + 260, y + 150, MUTED_TEXT, 0.65, 1)
    draw_panel_text(frame, f"Detector: {detector_name}", x + 22, y + 190, MUTED_TEXT, 0.55, 1)
    draw_panel_text(frame, "Press Q to stop early", x + 22, y + 222, WARNING_COLOR, 0.55, 1)
    draw_progress_bar(frame, x + 22, y + panel_height - 24, panel_width - 44, 12, progress, color)

    instruction = "Stay relaxed and natural" if label == 0 else "Act frustrated: tense face, quick head movement, open mouth"
    cv2.putText(
        frame,
        instruction,
        (24, height - 28),
        FONT,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def collect(args):
    tracker = create_mediapipe_tracker()
    if tracker is None:
        raise SystemExit(f"MediaPipe is required for good rage data: {get_mediapipe_error()}")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        tracker.close()
        raise SystemExit(f"Could not open webcam index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    phases = [
        ("calm round", 0, args.calm_seconds),
        ("rage/frustration round", 1, args.rage_seconds),
        ("calm recovery round", 0, args.calm_seconds),
    ]
    fatigue_monitor = EyeFatigueMonitor()
    previous_center = None
    saved_count = 0
    start_time = time.time()
    next_sample_time = start_time
    phase_started_at = start_time
    phase_index = 0

    try:
        while phase_index < len(phases):
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            phase_name, label, phase_seconds = phases[phase_index]
            elapsed = now - phase_started_at
            if elapsed >= phase_seconds:
                phase_index += 1
                phase_started_at = now
                continue

            tracking = tracker.process(frame)
            faces = [tracking.face_box] if tracking.face_box is not None else []
            current_center = face_center(faces)
            motion = head_motion(previous_center, current_center)
            previous_center = current_center

            fatigue = fatigue_monitor.update(
                now,
                len(faces) > 0,
                tracking.eye_openness,
                tracking.blink_detected,
                now - start_time,
            )
            features = build_rage_features(
                tracking.eye_openness,
                tracking.mouth_openness,
                motion,
                fatigue,
                tracking.posture,
                now - start_time,
            )

            if faces and now >= next_sample_time:
                append_training_sample(args.output, features, label)
                saved_count += 1
                next_sample_time = now + (1 / args.samples_per_second)

            progress = elapsed / phase_seconds
            seconds_left = max(0, phase_seconds - elapsed)
            draw_collector_ui(
                frame,
                phase_name,
                label,
                seconds_left,
                saved_count,
                tracker.name,
                progress,
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved_count} samples to {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Guided rage/calm data collection.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--calm-seconds", type=int, default=20)
    parser.add_argument("--rage-seconds", type=int, default=20)
    parser.add_argument("--samples-per-second", type=float, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
