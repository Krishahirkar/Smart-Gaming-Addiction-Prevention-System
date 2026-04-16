import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from perception import detect_face
from Perception.blink_detection import detect_blink
from Perception.eye_fatigue import EyeFatigueMonitor
from Perception.mediapipe_tracker import create_mediapipe_tracker, get_mediapipe_error
from Perception.posture_detection import detect_posture
from Perception.rage_detection import DEFAULT_DATA_PATH, RageDetector, append_training_sample
from Perception.rage_detection import MODEL_PATH as RAGE_MODEL_PATH


CAMERA_INDEX = 0
WINDOW_NAME = "Smart Gaming Addiction Prevention System"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_WARMUP_FRAMES = 20

EYE_CLOSED_THRESHOLD = 0.21
BLINK_ALERT_INTERVAL_SECONDS = 20
BREAK_REMINDER_SECONDS = 20 * 60
POSTURE_ALERTS = {"LEAN LEFT", "LEAN RIGHT", "TOO CLOSE", "SIT UP"}
FOCUS_ALERTS = {"LOOK LEFT", "LOOK RIGHT"}

TEXT_COLOR = (245, 248, 250)
MUTED_TEXT = (190, 198, 205)
GOOD_COLOR = (70, 220, 120)
WARNING_COLOR = (0, 210, 255)
ALERT_COLOR = (45, 65, 245)
FACE_BOX_COLOR = (255, 170, 50)
PANEL_COLOR = (18, 20, 24)
PANEL_BORDER = (84, 96, 112)
PANEL_ACCENT = (0, 180, 255)
SHADOW_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class SessionState:
    session_start: Optional[float] = None
    total_play_time: float = 0
    blink_count: int = 0
    last_blink_time: float = 0
    previous_face_center: Optional[tuple] = None

    def start_if_needed(self, now):
        if self.session_start is None:
            self.session_start = now
            self.last_blink_time = now

    def pause_if_needed(self, now):
        if self.session_start is not None:
            self.total_play_time += now - self.session_start
            self.session_start = None

    def display_time(self, now):
        if self.session_start is None:
            return self.total_play_time

        return self.total_play_time + (now - self.session_start)


def format_duration(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"


def draw_text(frame, text, y, color=TEXT_COLOR, scale=0.7, thickness=2):
    cv2.putText(
        frame,
        text,
        (20, y),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel(frame, x, y, width, height, alpha=0.78, border_color=PANEL_BORDER):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL_COLOR, -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), border_color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def text_size(text, scale=0.62, thickness=1):
    size, _ = cv2.getTextSize(text, FONT, scale, thickness)
    return size


def trim_text_to_width(text, max_width, scale=0.62, thickness=1):
    if text_size(text, scale, thickness)[0] <= max_width:
        return text

    ellipsis = "..."
    trimmed = text
    while trimmed and text_size(trimmed + ellipsis, scale, thickness)[0] > max_width:
        trimmed = trimmed[:-1]
    return trimmed + ellipsis if trimmed else ellipsis


def draw_panel_text(frame, text, x, y, color=TEXT_COLOR, scale=0.62, thickness=1):
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        FONT,
        scale,
        SHADOW_COLOR,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_progress_bar(frame, x, y, width, height, progress, color):
    progress = max(0, min(1, progress))
    cv2.rectangle(frame, (x, y), (x + width, y + height), (70, 74, 82), -1)
    cv2.rectangle(
        frame,
        (x, y),
        (x + int(width * progress), y + height),
        color,
        -1,
    )
    cv2.rectangle(frame, (x, y), (x + width, y + height), TEXT_COLOR, 1)


def draw_metric(frame, label, value, x, y, width, value_color=TEXT_COLOR):
    draw_panel_text(frame, label.upper(), x, y, MUTED_TEXT, 0.44, 1)
    value = trim_text_to_width(str(value), width, 0.82, 2)
    draw_panel_text(frame, value, x, y + 32, value_color, 0.82, 2)


def draw_alert_banner(frame, text, color=ALERT_COLOR):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - 64), (width, height), color, -1)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)
    draw_panel_text(frame, text, 22, height - 22, (255, 255, 255), 0.86, 2)


def posture_tip(posture):
    tips = {
        "GOOD": "Nice. Keep your shoulders relaxed.",
        "LEAN LEFT": "Center your head and shoulders.",
        "LEAN RIGHT": "Center your head and shoulders.",
        "TOO CLOSE": "Move a little away from the screen.",
        "SIT UP": "Raise your head and sit taller.",
        "LOOK LEFT": "Face the screen directly.",
        "LOOK RIGHT": "Face the screen directly.",
    }
    return tips.get(posture, "Waiting for face posture.")


def posture_color(posture):
    if posture in POSTURE_ALERTS:
        return ALERT_COLOR
    if posture in FOCUS_ALERTS or posture is None:
        return WARNING_COLOR
    return GOOD_COLOR


def risk_color(score):
    if score >= 70:
        return ALERT_COLOR
    if score >= 40:
        return WARNING_COLOR
    return GOOD_COLOR


def draw_status_strip(frame, has_player, fps):
    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 58), (10, 12, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    status = "TRACKING" if has_player else "NO FACE"
    color = GOOD_COLOR if has_player else WARNING_COLOR
    draw_panel_text(frame, "Smart Gaming Prevention System", 18, 38, PANEL_ACCENT, 0.78, 2)
    draw_panel_text(frame, status, width - 260, 36, color, 0.7, 2)
    draw_panel_text(frame, f"{fps:.1f} FPS", width - 126, 36, MUTED_TEXT, 0.58, 1)


def draw_posture_panel(frame, posture):
    height, width = frame.shape[:2]
    panel_width = min(330, width - 40)
    panel_height = 164
    x = width - panel_width - 18 if width >= 850 else 18
    y = 76 if width >= 850 else 328
    status = posture if posture is not None else "NO FACE"
    status_color = posture_color(posture)

    if y + panel_height > height - 70:
        return

    draw_panel(frame, x, y, panel_width, panel_height, border_color=PANEL_BORDER)
    draw_panel_text(frame, "POSTURE", x + 18, y + 32, MUTED_TEXT, 0.5, 1)
    draw_panel_text(frame, status, x + 18, y + 82, status_color, 1.05, 3)
    tip = trim_text_to_width(posture_tip(posture), panel_width - 36, 0.55, 1)
    draw_panel_text(frame, tip, x + 18, y + 122, TEXT_COLOR, 0.55, 1)


def draw_risk_panel(frame, fatigue, rage, training_message):
    height, width = frame.shape[:2]
    panel_width = min(330, width - 40)
    panel_height = 270
    x = width - panel_width - 18 if width >= 850 else 18
    y = 258 if width >= 850 else 328

    if y + panel_height > height - 70:
        return

    fatigue_score = fatigue["score"]
    rage_score = rage["score"]
    draw_panel(frame, x, y, panel_width, panel_height, border_color=PANEL_BORDER)
    draw_panel_text(frame, "EYE FATIGUE", x + 18, y + 32, MUTED_TEXT, 0.5, 1)
    draw_panel_text(frame, f"{fatigue_score}%", x + 18, y + 78, risk_color(fatigue_score), 1.0, 3)
    draw_panel_text(frame, fatigue["status"], x + 126, y + 72, risk_color(fatigue_score), 0.72, 2)
    draw_progress_bar(
        frame,
        x + 18,
        y + 94,
        panel_width - 36,
        12,
        fatigue_score / 100,
        risk_color(fatigue_score),
    )
    draw_panel_text(
        frame,
        trim_text_to_width(fatigue["tip"], panel_width - 36, 0.5, 1),
        x + 18,
        y + 126,
        TEXT_COLOR,
        0.5,
        1,
    )

    draw_panel_text(frame, "RAGE SECTION", x + 18, y + 166, MUTED_TEXT, 0.5, 1)
    draw_panel_text(frame, f"{rage_score}%", x + 18, y + 216, risk_color(rage_score), 1.1, 3)
    draw_panel_text(
        frame,
        f"{rage['status']} ({rage['source']})",
        x + 126,
        y + 206,
        risk_color(rage_score),
        0.62,
        2,
    )
    draw_panel_text(
        frame,
        trim_text_to_width(rage["tip"], panel_width - 36, 0.48, 1),
        x + 18,
        y + 246,
        TEXT_COLOR,
        0.48,
        1,
    )

    if training_message:
        draw_panel_text(frame, training_message, x + 18, y + 264, WARNING_COLOR, 0.44, 1)


def draw_hud(frame, state, faces, eye_openness, posture, now, fps, fatigue, rage):
    has_player = len(faces) > 0
    play_time = state.display_time(now)
    time_since_blink = now - state.last_blink_time if has_player else 0
    break_progress = play_time / BREAK_REMINDER_SECONDS
    panel_width = min(440, frame.shape[1] - 40)
    panel_height = 238
    panel_x = 18
    panel_y = 76

    draw_status_strip(frame, has_player, fps)
    draw_panel(frame, panel_x, panel_y, panel_width, panel_height)
    draw_panel_text(frame, "SESSION", panel_x + 18, panel_y + 32, MUTED_TEXT, 0.5, 1)

    draw_metric(
        frame,
        "play time",
        format_duration(play_time),
        panel_x + 18,
        panel_y + 70,
        160,
        GOOD_COLOR if has_player else WARNING_COLOR,
    )
    draw_metric(frame, "blinks", state.blink_count, panel_x + 190, panel_y + 70, 95)
    draw_metric(
        frame,
        "eyes",
        f"{eye_openness:.2f}" if eye_openness is not None else "--",
        panel_x + 304,
        panel_y + 70,
        90,
    )

    draw_panel_text(frame, "Break progress", panel_x + 18, panel_y + 148, MUTED_TEXT, 0.5, 1)
    draw_progress_bar(
        frame,
        panel_x + 18,
        panel_y + 164,
        panel_width - 36,
        16,
        break_progress,
        GOOD_COLOR if break_progress < 0.8 else WARNING_COLOR,
    )

    eye_status = "Eyes closed" if eye_openness is not None and eye_openness < EYE_CLOSED_THRESHOLD else "Eyes open"
    if eye_openness is None:
        eye_status = "Waiting for face"
    draw_panel_text(
        frame,
        eye_status,
        panel_x + 18,
        panel_y + 210,
        ALERT_COLOR if eye_status == "Eyes closed" else MUTED_TEXT,
        0.58,
        1,
    )

    alert_text = None
    alert_color = ALERT_COLOR
    if has_player and time_since_blink > BLINK_ALERT_INTERVAL_SECONDS:
        alert_text = "Blink now - rest your eyes"
    if fatigue["score"] >= 70:
        alert_text = "Eye fatigue high - rest your eyes"
    if posture in FOCUS_ALERTS:
        alert_text = "Look back at the screen"
        alert_color = WARNING_COLOR
    if posture in POSTURE_ALERTS:
        alert_text = "Fix your posture"
    if rage["score"] >= 70:
        alert_text = "Rage risk high - pause and breathe"
    if has_player and play_time > BREAK_REMINDER_SECONDS:
        alert_text = "Take a short break"

    if alert_text is not None:
        draw_alert_banner(frame, alert_text, alert_color)


def draw_face_boxes(frame, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_BOX_COLOR, 2)
        draw_panel_text(frame, "FACE", x, max(24, y - 8), FACE_BOX_COLOR, 0.5, 1)


def draw_eye_points(frame, eye_points):
    for point in eye_points:
        cv2.circle(frame, point, 3, PANEL_ACCENT, -1, cv2.LINE_AA)


def face_center(faces):
    if not faces:
        return None

    x, y, w, h = faces[0]
    return x + w // 2, y + h // 2


def head_motion(previous_center, current_center):
    if previous_center is None or current_center is None:
        return 0

    dx = current_center[0] - previous_center[0]
    dy = current_center[1] - previous_center[1]
    return (dx * dx + dy * dy) ** 0.5


def build_rage_features(
    eye_openness,
    mouth_openness,
    motion,
    fatigue,
    posture,
    play_time,
):
    return {
        "eye_openness": eye_openness if eye_openness is not None else 0,
        "mouth_openness": mouth_openness if mouth_openness is not None else 0,
        "head_motion": motion,
        "blink_rate": fatigue["blink_rate"],
        "fatigue_score": fatigue["score"],
        "posture_bad": 1 if posture in POSTURE_ALERTS or posture in FOCUS_ALERTS else 0,
        "play_minutes": play_time / 60,
    }


def self_test():
    print(f"OpenCV version: {cv2.__version__}")

    tracker = create_mediapipe_tracker()
    if tracker is None:
        print("MediaPipe: not available, app will use OpenCV Haar fallback")
        print(f"MediaPipe reason: {get_mediapipe_error()}")
    else:
        print(f"MediaPipe: available, app will use {tracker.name}")
        tracker.close()

    rage_detector = RageDetector()
    print(f"Rage detector: {rage_detector.mode}")
    if rage_detector.model_error:
        print(f"Rage model reason: {rage_detector.model_error}")
    elif rage_detector.model is not None:
        print(f"Rage model path: {RAGE_MODEL_PATH}")

    print("Self-test completed")


def camera_backends():
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(("DirectShow", cv2.CAP_DSHOW))
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(("Media Foundation", cv2.CAP_MSMF))
    backends.append(("Default", cv2.CAP_ANY))
    return backends


def try_read_frame(cap):
    for _ in range(CAMERA_WARMUP_FRAMES):
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        time.sleep(0.05)

    return None


def open_camera(camera_index):
    errors = []
    sizes = [(CAMERA_WIDTH, CAMERA_HEIGHT), (640, 480), (0, 0)]

    for backend_name, backend in camera_backends():
        for width, height in sizes:
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                errors.append(f"{backend_name}: could not open")
                continue

            if width and height:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            first_frame = try_read_frame(cap)
            if first_frame is not None:
                return cap, first_frame, backend_name

            cap.release()
            size_name = f"{width}x{height}" if width and height else "default size"
            errors.append(f"{backend_name} {size_name}: opened but no frames")

    details = "; ".join(errors[-6:])
    raise RuntimeError(
        f"Could not read from webcam index {camera_index}. "
        "Close other apps using the camera or try --camera-index 1. "
        f"Attempts: {details}"
    )


def run(camera_index=CAMERA_INDEX):
    cap, pending_frame, _camera_backend = open_camera(camera_index)

    state = SessionState(last_blink_time=time.time())
    tracker = create_mediapipe_tracker()
    fatigue_monitor = EyeFatigueMonitor()
    rage_detector = RageDetector()
    previous_frame_time = time.time()
    training_message = ""
    training_message_until = 0

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Could not read a frame from the webcam.")
                    break

            if frame is None:
                print("Could not read a frame from the webcam.")
                break

            now = time.time()
            fps = 1 / max(0.001, now - previous_frame_time)
            previous_frame_time = now
            eye_openness = None
            mouth_openness = None
            blink_detected = False
            eye_points = []

            if tracker is not None:
                tracking = tracker.process(frame)
                faces = [tracking.face_box] if tracking.face_box is not None else []
                eye_openness = tracking.eye_openness
                mouth_openness = tracking.mouth_openness
                blink_detected = tracking.blink_detected
                posture = tracking.posture
                eye_points = tracking.eye_points
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detect_face(frame, gray)

                blink_result = detect_blink(frame, gray, faces)
                if blink_result is not None:
                    eye_openness, blink_detected = blink_result

                posture = detect_posture(frame, gray, faces)

            current_center = face_center(faces)
            motion = head_motion(state.previous_face_center, current_center)
            state.previous_face_center = current_center

            if len(faces) > 0:
                state.start_if_needed(now)
                if blink_detected:
                    state.blink_count += 1
                    state.last_blink_time = now
            else:
                state.pause_if_needed(now)

            play_time = state.display_time(now)
            fatigue = fatigue_monitor.update(
                now,
                len(faces) > 0,
                eye_openness,
                blink_detected,
                play_time,
            )
            rage_features = build_rage_features(
                eye_openness,
                mouth_openness,
                motion,
                fatigue,
                posture,
                play_time,
            )
            rage = rage_detector.predict(rage_features)
            if now > training_message_until:
                training_message = ""

            draw_face_boxes(frame, faces)
            draw_eye_points(frame, eye_points)
            draw_hud(frame, state, faces, eye_openness, posture, now, fps, fatigue, rage)
            draw_posture_panel(frame, posture)
            draw_risk_panel(frame, fatigue, rage, training_message)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), ord("n"), ord("r")):
                label = 1 if key == ord("r") else 0
                append_training_sample(DEFAULT_DATA_PATH, rage_features, label)
                training_message = f"Saved {'rage' if label else 'calm'} sample"
                training_message_until = now + 2
    finally:
        if tracker is not None:
            tracker.close()
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Gaming Addiction Prevention System"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="Webcam index to open. Try 1 if 0 does not work.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check OpenCV and MediaPipe without opening the webcam.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.camera_index)
