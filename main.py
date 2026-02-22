import cv2
import time
from perception import detect_face

cap = cv2.VideoCapture(0)

session_start = None
total_play_time = 0

while True:
    ret, frame = cap.read()
    faces = detect_face(frame)

    if len(faces) > 0:
        if session_start is None:
            session_start = time.time()

        current_session_time = time.time() - session_start
        display_time = total_play_time + current_session_time
        status = "Timer Running"

    else:
        if session_start is not None:
            total_play_time += time.time() - session_start
            session_start = None

        display_time = total_play_time
        status = "Timer Paused"

    minutes = int(display_time // 60)
    seconds = int(display_time % 60)

    cv2.putText(frame, f"Total Time: {minutes}:{seconds}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.putText(frame, status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,0,255), 2)

    # 🔵 ADD THIS PART
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Gaming Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()