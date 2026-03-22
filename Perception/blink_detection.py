import cv2

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------- States --------
eye_closed_frames = 0
blink_cooldown = 0
eye_was_closed = False   # 🔥 IMPORTANT STATE

# -------- Parameters --------
CLOSED_FRAME_THRESHOLD = 2
COOLDOWN_FRAMES = 5

def detect_blink(frame):
    global eye_closed_frames, blink_cooldown, eye_was_closed

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (fx, fy, fw, fh) = faces[0]
    face_roi = gray[fy:fy+fh, fx:fx+fw]

    eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 5)

    # -------- Filter Noise --------
    filtered_eyes = []
    for (ex, ey, ew, eh) in eyes:
        if ew > 20 and eh > 20:
            filtered_eyes.append((ex, ey, ew, eh))

    eye_count = len(filtered_eyes)

    blink_detected = False

    # -------- CLOSED STATE --------
    if eye_count == 0:
        eye_closed_frames += 1
        if eye_closed_frames >= CLOSED_FRAME_THRESHOLD:
            eye_was_closed = True

    # -------- OPEN STATE --------
    else:
        if eye_was_closed and blink_cooldown == 0:
            blink_detected = True
            blink_cooldown = COOLDOWN_FRAMES

        eye_closed_frames = 0
        eye_was_closed = False

    # -------- COOLDOWN --------
    if blink_cooldown > 0:
        blink_cooldown -= 1

    # -------- Fake EAR --------
    if eye_count == 0:
        ear = 0.1
    elif eye_count == 1:
        ear = 0.22
    else:
        ear = 0.32

    return ear, blink_detected