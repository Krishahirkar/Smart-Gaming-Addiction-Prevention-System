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

MODEL_FEATURE_COLUMNS = [
    "eye_openness",
    "mouth_openness",
    "head_motion",
    "blink_rate",
    "fatigue_score",
    "posture_bad",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "rage_model.joblib"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "rage_training.csv"
HIGH_RAGE_THRESHOLD = 65
MEDIUM_RAGE_THRESHOLD = 40


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
        heuristic_score = self._heuristic_score(features)

        if self.model is not None:
            try:
                model_columns = self._model_columns()
                values = [float(features.get(column, 0)) for column in model_columns]
                model_input = [values]
                try:
                    import pandas as pd

                    model_input = pd.DataFrame([values], columns=model_columns)
                except ImportError:
                    pass

                if hasattr(self.model, "predict_proba"):
                    probability = self._positive_probability(model_input)
                else:
                    probability = float(self.model.predict(model_input)[0])
                model_score = int(max(0, min(100, probability * 100)))
                score = self._blend_scores(model_score, heuristic_score)
                return self._result(score, "MODEL")
            except Exception as exc:
                self.model_error = str(exc)

        return self._result(heuristic_score, "RULE")

    def _model_columns(self):
        feature_names = getattr(self.model, "feature_names_in_", None)
        if feature_names is not None:
            return [str(name) for name in feature_names]

        return MODEL_FEATURE_COLUMNS

    def _positive_probability(self, model_input):
        probabilities = self.model.predict_proba(model_input)[0]
        classes = list(getattr(self.model, "classes_", []))
        if 1 in classes:
            return probabilities[classes.index(1)]

        return probabilities[-1]

    def _blend_scores(self, model_score, heuristic_score):
        score = int(round((model_score * 0.65) + (heuristic_score * 0.35)))
        if heuristic_score >= HIGH_RAGE_THRESHOLD:
            score = max(score, heuristic_score)
        elif heuristic_score < 25 and model_score < HIGH_RAGE_THRESHOLD:
            score = min(score, 35)

        return int(max(0, min(100, score)))

    def _heuristic_score(self, features):
        score = 0
        eye = float(features.get("eye_openness", 0))
        mouth = float(features.get("mouth_openness", 0))
        motion = float(features.get("head_motion", 0))
        blink_rate = float(features.get("blink_rate", 0))
        fatigue = float(features.get("fatigue_score", 0))
        posture_bad = float(features.get("posture_bad", 0))

        if mouth > 0.24:
            score += 40
        elif mouth > 0.14:
            score += 30
        elif mouth > 0.08:
            score += 15

        if 0 < eye < 0.18:
            score += 20
        elif 0 < eye < 0.23:
            score += 10

        if motion > 45:
            score += 25
        elif motion > 18:
            score += 15

        if blink_rate < 4:
            score += 10
        elif blink_rate > 24:
            score += 10

        if mouth > 0.14 and blink_rate < 4:
            score += 10
        if mouth > 0.14 and 0 < eye < 0.20:
            score += 10

        if fatigue > 70:
            score += 20
        elif fatigue > 40:
            score += 10

        if posture_bad:
            score += 10

        return int(max(0, min(100, score)))

    def _result(self, score, source):
        if score >= HIGH_RAGE_THRESHOLD:
            status = "HIGH"
            tip = "Pause, breathe, and take a short break."
        elif score >= MEDIUM_RAGE_THRESHOLD:
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
