import cv2


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _largest_face(faces):
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def detect_posture(frame, gray=None, faces=None):
    if gray is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if faces is None:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face = _largest_face(faces)
    if face is None:
        return None

    (x, y, w, h) = face
    face_center_x = x + w // 2
    face_center_y = y + h // 2
    frame_center_x = frame.shape[1] // 2
    frame_height = frame.shape[0]
    horizontal_margin = max(50, frame.shape[1] // 10)
    close_face_width = int(frame.shape[1] * 0.35)

    posture = "GOOD"

    if face_center_x < frame_center_x - horizontal_margin:
        posture = "LEAN LEFT"
    elif face_center_x > frame_center_x + horizontal_margin:
        posture = "LEAN RIGHT"

    if w > close_face_width:
        posture = "TOO CLOSE"
    elif face_center_y > frame_height * 0.62:
        posture = "SIT UP"

    return posture
