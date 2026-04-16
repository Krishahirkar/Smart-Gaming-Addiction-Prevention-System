import cv2


eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_closed_frames = 0
blink_cooldown = 0
eye_was_closed = False

CLOSED_FRAME_THRESHOLD = 2
COOLDOWN_FRAMES = 5
MIN_EYE_SIZE = 20


def _largest_face(faces):
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def detect_blink(frame, gray=None, faces=None):
    global eye_closed_frames, blink_cooldown, eye_was_closed

    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if faces is None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face = _largest_face(faces)
    if face is None:
        eye_closed_frames = 0
        eye_was_closed = False
        return None

    (fx, fy, fw, fh) = face
    face_roi = gray[fy:fy + fh, fx:fx + fw]
    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

    filtered_eyes = [
        (ex, ey, ew, eh)
        for (ex, ey, ew, eh) in eyes
        if ew >= MIN_EYE_SIZE and eh >= MIN_EYE_SIZE
    ]
    eye_count = len(filtered_eyes)
    blink_detected = False

    if eye_count == 0:
        eye_closed_frames += 1
        if eye_closed_frames >= CLOSED_FRAME_THRESHOLD:
            eye_was_closed = True
    else:
        if eye_was_closed and blink_cooldown == 0:
            blink_detected = True
            blink_cooldown = COOLDOWN_FRAMES
        eye_closed_frames = 0
        eye_was_closed = False

    if blink_cooldown > 0:
        blink_cooldown -= 1

    if eye_count == 0:
        eye_openness = 0.1
    elif eye_count == 1:
        eye_openness = 0.22
    else:
        eye_openness = 0.32

    return eye_openness, blink_detected
