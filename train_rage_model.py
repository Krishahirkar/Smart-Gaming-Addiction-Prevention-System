import argparse
from pathlib import Path

from Perception.rage_detection import DEFAULT_DATA_PATH, FEATURE_COLUMNS, MODEL_PATH


def train(data_path, model_path):
    try:
        import pandas as pd
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise SystemExit(
            "Training needs pandas, scikit-learn, and joblib. "
            "Run: pip install -r requirements.txt"
        ) from exc

    data_path = Path(data_path)
    model_path = Path(model_path)

    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")

    data = pd.read_csv(data_path)
    missing_columns = [column for column in FEATURE_COLUMNS + ["label"] if column not in data]
    if missing_columns:
        raise SystemExit(f"Missing columns in training data: {missing_columns}")

    if data["label"].nunique() < 2:
        raise SystemExit("Training data needs at least two labels, for example calm and rage.")

    x = data[FEATURE_COLUMNS]
    y = data["label"].astype(int)

    if len(data) >= 10:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    else:
        x_train, x_test, y_train, y_test = x, x, y, y

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Saved rage model to: {model_path}")
    print(f"Training rows: {len(data)}")
    print(f"Validation accuracy: {accuracy:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the rage detection model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV training data path.")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Output model path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.model)
