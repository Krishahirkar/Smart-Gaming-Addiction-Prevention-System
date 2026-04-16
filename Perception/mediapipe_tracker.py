from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    mp_tasks_python = None
    mp_vision = None


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"

CLOSED_EAR_THRESHOLD = 0.20
CLOSED_FRAME_THRESHOLD = 2
COOLDOWN_FRAMES = 5

LAST_MEDIAPIPE_ERROR = None


@dataclass
class TrackingResult:
    face_box: Optional[Tuple[int, int, int, int]]
    eye_openness: Optional[float]
    mouth_openness: Optional[float]
    blink_detected: bool
    posture: Optional[str]
    eye_points: List[Tuple[int, int]]


class MediaPipeTracker:
    def __init__(self):
        if mp is None:
            raise RuntimeError("MediaPipe is not installed.")

        self.backend = None
        self.face_mesh = None
        self.landmarker = None
        self.closed_frames = 0
        self.blink_cooldown = 0
        self.eye_was_closed = False

        if self._can_use_tasks_api():
            self._start_tasks_landmarker()
        elif self._can_use_solutions_api():
            self._start_solutions_face_mesh()
        else:
            raise RuntimeError(
                "MediaPipe is installed, but no supported face landmark API is available."
            )

    @property
    def name(self):
        if self.backend == "tasks":
            return "MediaPipe"
        if self.backend == "solutions":
            return "MediaPipe"
        return "MediaPipe"

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.landmarker is not None:
            self.landmarker.close()

    def process(self, frame):
        if self.backend == "tasks":
            return self._process_tasks(frame)

        return self._process_solutions(frame)

    def _can_use_tasks_api(self):
        return (
            mp_tasks_python is not None
            and mp_vision is not None
            and hasattr(mp_vision, "FaceLandmarker")
        )

    def _can_use_solutions_api(self):
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")

    def _start_tasks_landmarker(self):
        self._ensure_model()
        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.backend = "tasks"

    def _start_solutions_face_mesh(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.backend = "solutions"

    def _ensure_model(self):
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            return

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, MODEL_PATH)

    def _process_tasks(self, frame):
        frame_height, frame_width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.landmarker.detect(image)

        if not results.face_landmarks:
            return self._empty_result()

        landmarks = results.face_landmarks[0]
        return self._result_from_landmarks(landmarks, frame_width, frame_height)

    def _process_solutions(self, frame):
        frame_height, frame_width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return self._empty_result()

        landmarks = results.multi_face_landmarks[0].landmark
        return self._result_from_landmarks(landmarks, frame_width, frame_height)

    def _empty_result(self):
        self.closed_frames = 0
        self.eye_was_closed = False
        return TrackingResult(None, None, None, False, None, [])

    def _result_from_landmarks(self, landmarks, frame_width, frame_height):
        face_box = self._face_box(landmarks, frame_width, frame_height)
        eye_openness, eye_points = self._eye_openness(
            landmarks,
            frame_width,
            frame_height,
        )
        mouth_openness = self._mouth_openness(landmarks, frame_width, frame_height)
        blink_detected = self._update_blink_state(eye_openness)
        posture = self._posture(landmarks, frame_width, frame_height, face_box)

        return TrackingResult(
            face_box=face_box,
            eye_openness=eye_openness,
            mouth_openness=mouth_openness,
            blink_detected=blink_detected,
            posture=posture,
            eye_points=eye_points,
        )

    def _point(self, landmarks, index, width, height):
        landmark = landmarks[index]
        return int(landmark.x * width), int(landmark.y * height)

    def _face_box(self, landmarks, width, height):
        xs = [int(point.x * width) for point in landmarks]
        ys = [int(point.y * height) for point in landmarks]

        x_min = max(0, min(xs))
        y_min = max(0, min(ys))
        x_max = min(width - 1, max(xs))
        y_max = min(height - 1, max(ys))

        return x_min, y_min, x_max - x_min, y_max - y_min

    def _eye_openness(self, landmarks, width, height):
        left = [self._point(landmarks, index, width, height) for index in LEFT_EYE]
        right = [self._point(landmarks, index, width, height) for index in RIGHT_EYE]

        left_ear = self._eye_aspect_ratio(left)
        right_ear = self._eye_aspect_ratio(right)
        return (left_ear + right_ear) / 2, left + right

    def _mouth_openness(self, landmarks, width, height):
        top = self._point(landmarks, MOUTH_TOP, width, height)
        bottom = self._point(landmarks, MOUTH_BOTTOM, width, height)
        left = self._point(landmarks, MOUTH_LEFT, width, height)
        right = self._point(landmarks, MOUTH_RIGHT, width, height)

        vertical = hypot(top[0] - bottom[0], top[1] - bottom[1])
        horizontal = hypot(left[0] - right[0], left[1] - right[1])

        if horizontal == 0:
            return 0

        return vertical / horizontal

    def _eye_aspect_ratio(self, points):
        horizontal = hypot(
            points[0][0] - points[3][0],
            points[0][1] - points[3][1],
        )
        vertical_one = hypot(
            points[1][0] - points[5][0],
            points[1][1] - points[5][1],
        )
        vertical_two = hypot(
            points[2][0] - points[4][0],
            points[2][1] - points[4][1],
        )

        if horizontal == 0:
            return 0

        return (vertical_one + vertical_two) / (2 * horizontal)

    def _update_blink_state(self, eye_openness):
        blink_detected = False

        if eye_openness < CLOSED_EAR_THRESHOLD:
            self.closed_frames += 1
            if self.closed_frames >= CLOSED_FRAME_THRESHOLD:
                self.eye_was_closed = True
        else:
            if self.eye_was_closed and self.blink_cooldown == 0:
                blink_detected = True
                self.blink_cooldown = COOLDOWN_FRAMES
            self.closed_frames = 0
            self.eye_was_closed = False

        if self.blink_cooldown > 0:
            self.blink_cooldown -= 1

        return blink_detected

    def _posture(self, landmarks, frame_width, frame_height, face_box):
        left_cheek = landmarks[234]
        right_cheek = landmarks[454]
        nose = landmarks[1]
        face_center_x = face_box[0] + face_box[2] // 2
        face_center_y = face_box[1] + face_box[3] // 2
        frame_center_x = frame_width // 2
        horizontal_margin = max(50, frame_width // 10)
        close_face_width = int(frame_width * 0.36)

        if face_box[2] > close_face_width:
            return "TOO CLOSE"
        if face_center_y > frame_height * 0.62:
            return "SIT UP"
        if face_center_x < frame_center_x - horizontal_margin:
            return "LEAN LEFT"
        if face_center_x > frame_center_x + horizontal_margin:
            return "LEAN RIGHT"

        cheek_span = max(0.001, right_cheek.x - left_cheek.x)
        nose_offset = (nose.x - ((left_cheek.x + right_cheek.x) / 2)) / cheek_span
        if nose_offset < -0.12:
            return "LOOK RIGHT"
        if nose_offset > 0.12:
            return "LOOK LEFT"

        return "GOOD"


def get_mediapipe_error():
    return LAST_MEDIAPIPE_ERROR


def create_mediapipe_tracker():
    global LAST_MEDIAPIPE_ERROR

    if mp is None:
        LAST_MEDIAPIPE_ERROR = "MediaPipe is not installed."
        return None

    try:
        LAST_MEDIAPIPE_ERROR = None
        return MediaPipeTracker()
    except Exception as exc:
        LAST_MEDIAPIPE_ERROR = str(exc)
        return None
