class EyeFatigueMonitor:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds
        self.blink_times = []
        self.low_eye_started_at = None

    def update(self, now, has_player, eye_openness, blink_detected, play_time):
        if blink_detected:
            self.blink_times.append(now)

        cutoff = now - self.window_seconds
        self.blink_times = [timestamp for timestamp in self.blink_times if timestamp >= cutoff]

        if not has_player:
            self.low_eye_started_at = None
            return {
                "score": 0,
                "status": "NO FACE",
                "blink_rate": 0,
                "tip": "Face the camera to measure eye fatigue.",
            }

        blink_rate = len(self.blink_times)
        if eye_openness is not None and eye_openness < 0.22:
            if self.low_eye_started_at is None:
                self.low_eye_started_at = now
        else:
            self.low_eye_started_at = None

        low_eye_seconds = 0
        if self.low_eye_started_at is not None:
            low_eye_seconds = now - self.low_eye_started_at

        score = 0
        if blink_rate < 8:
            score += 30
        elif blink_rate < 12:
            score += 15

        if low_eye_seconds > 2:
            score += min(35, low_eye_seconds * 8)

        if play_time > 20 * 60:
            score += 20
        elif play_time > 10 * 60:
            score += 10

        score = int(max(0, min(100, score)))

        if score >= 70:
            status = "HIGH"
            tip = "Rest your eyes for 20 seconds."
        elif score >= 40:
            status = "MEDIUM"
            tip = "Blink slowly and look away briefly."
        else:
            status = "LOW"
            tip = "Eye strain looks low."

        return {
            "score": score,
            "status": status,
            "blink_rate": blink_rate,
            "tip": tip,
        }
