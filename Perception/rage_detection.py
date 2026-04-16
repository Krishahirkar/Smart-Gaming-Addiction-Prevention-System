import csv
from pathlib import Path


FEATURE_COLUMNS = [
    "eye_openness",
    "mouth_openness",
    "head_motion",
    "blink_rate",
    "fatigue_score",
    "posture_bad",
    "play_minutes",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rage_model.joblib"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "rage_training.csv"


class RageDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.model_error = None
        self._load_model()

    @property
    def mode(self):
        return "trained model" if self.model is not None else "rule based"

    def _load_model(self):
        if not self.model_path.exists():
            return

        try:
            import joblib

            self.model = joblib.load(self.model_path)
        except Exception as exc:
            self.model_error = str(exc)
            self.model = None

    def predict(self, features):
        values = [float(features.get(column, 0)) for column in FEATURE_COLUMNS]

        if self.model is not None:
            try:
                model_input = [values]
                try:
                    import pandas as pd

                    model_input = pd.DataFrame([values], columns=FEATURE_COLUMNS)
                except ImportError:
                    pass

                if hasattr(self.model, "predict_proba"):
                    probability = self.model.predict_proba(model_input)[0][1]
                else:
                    probability = float(self.model.predict(model_input)[0])
                score = int(max(0, min(100, probability * 100)))
                return self._result(score, "MODEL")
            except Exception as exc:
                self.model_error = str(exc)

        score = self._heuristic_score(features)
        return self._result(score, "RULE")

    def _heuristic_score(self, features):
        score = 0
        mouth = float(features.get("mouth_openness", 0))
        motion = float(features.get("head_motion", 0))
        fatigue = float(features.get("fatigue_score", 0))
        posture_bad = float(features.get("posture_bad", 0))

        if mouth > 0.42:
            score += 30
        elif mouth > 0.30:
            score += 15

        if motion > 55:
            score += 30
        elif motion > 28:
            score += 15

        if fatigue > 70:
            score += 20
        elif fatigue > 40:
            score += 10

        if posture_bad:
            score += 10

        return int(max(0, min(100, score)))

    def _result(self, score, source):
        if score >= 70:
            status = "HIGH"
            tip = "Pause, breathe, and take a short break."
        elif score >= 40:
            status = "MEDIUM"
            tip = "Slow down and relax your hands."
        else:
            status = "LOW"
            tip = "Rage signs look low."

        return {
            "score": score,
            "status": status,
            "source": source,
            "tip": tip,
        }


def append_training_sample(path, features, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_COLUMNS + ["label"])
        if not file_exists:
            writer.writeheader()

        row = {column: features.get(column, 0) for column in FEATURE_COLUMNS}
        row["label"] = label
        writer.writerow(row)
