import cv2
import time
import numpy as np
from perception import detect_face
from Perception.blink_detection import detect_blink
from Perception.posture_detection import detect_posture

cap = cv2.VideoCapture(0)

session_start = None
total_play_time = 0
blink_count = 0
last_blink_time = time.time()

EAR_THRESHOLD = 0.21
BLINK_ALERT_INTERVAL = 20
ear_below_threshold = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_face(frame)
    result = detect_blink(frame)
    ear = None
    blink_detected = False
    if result is not None:
        ear,blink_detected = result
    posture = detect_posture(frame)
    if posture:
        cv2.putText(frame, f"Posture: {posture}",
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    # -------- Timer Logic --------
    if len(faces) > 0:
        if session_start is None:
            session_start = time.time()
        current_session_time = time.time() - session_start
        display_time = total_play_time + current_session_time
        status = "Timer Running"

        # -------- Blink Count Logic --------
        
        if blink_detected:
            blink_count += 1
            last_blink_time = time.time()
            
    else:
        if session_start is not None:
            total_play_time += time.time() - session_start
            session_start = None

        display_time = total_play_time
        status = "Timer Paused"

    # -------- Display --------
    minutes = int(display_time // 60)
    seconds = int(display_time % 60)
    time_since_blink = time.time() - last_blink_time

    cv2.putText(frame, f"Time: {minutes}:{seconds:02d}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, status,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if ear is not None:
        cv2.putText(frame, f"EAR: {round(ear, 2)}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}",
                    (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if ear < EAR_THRESHOLD:
            cv2.putText(frame, "Eyes Closed!",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if time_since_blink > BLINK_ALERT_INTERVAL and len(faces) > 0:
        cv2.putText(frame, "BLINK! Eye strain warning",
                    (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Gaming Prevention System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()