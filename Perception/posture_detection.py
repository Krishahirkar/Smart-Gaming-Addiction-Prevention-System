import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_posture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]

    # Face center
    face_center_x = x + w // 2
    face_center_y = y + h // 2

    frame_center_x = frame.shape[1] // 2

    posture = "GOOD"

    # 🔹 Detect leaning left/right
    if face_center_x < frame_center_x - 50:
        posture = "LEAN LEFT"
    elif face_center_x > frame_center_x + 50:
        posture = "LEAN RIGHT"

    # 🔹 Detect too close (big face = close to screen)
    if w > 200:
        posture = "TOO CLOSE"

    return posture