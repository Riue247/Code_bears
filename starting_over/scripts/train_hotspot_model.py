import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.feature_extraction import DictVectorizer


REQUIRED_FEATURE_COLUMNS = [
    "location",
    "hour_of_day",
    "day_of_week",
    "crime_type",
    "incident_count_last_24h",
    "severity_score",
]
TARGET_COLUMN = "is_hotspot"


def validate_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_FEATURE_COLUMNS + [TARGET_COLUMN] if col not in frame.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Training CSV is empty; provide at least one sample")
    validate_columns(df)
    # Shuffle to avoid any ordering bias
    df = shuffle(df, random_state=42).reset_index(drop=True)
    return df


def vectorize_features(df: pd.DataFrame):
    dict_rows = df[REQUIRED_FEATURE_COLUMNS].to_dict(orient="records")
    vectorizer = DictVectorizer(sparse=False)
    features = vectorizer.fit_transform(dict_rows)
    return features.astype(np.float32), vectorizer


def train_model(features: np.ndarray, labels: np.ndarray, n_estimators: int, max_depth: int):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(features, labels)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train crime hotspot classifier and vectorizer.")
    parser.add_argument("--csv", required=True, help="Path to training CSV containing engineered features.")
    parser.add_argument("--model-out", default="crime_hotspot_model.pkl", help="Where to save the trained classifier.")
    parser.add_argument(
        "--vectorizer-out",
        default="crime_feature_vectorizer.pkl",
        help="Where to save the DictVectorizer fitted on training data.",
    )
    parser.add_argument(
        "--metrics-out",
        default="hotspot_training_metrics.json",
        help="Optional path to dump evaluation metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of samples reserved for validation.")
    parser.add_argument("--max-depth", type=int, default=12, help="Maximum depth for the RandomForest.")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of trees in the RandomForest.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    print(f"Loading dataset from {csv_path}")
    df = load_dataset(csv_path)
    labels = df[TARGET_COLUMN].astype(int).to_numpy()

    print("Vectorizing features …")
    features, vectorizer = vectorize_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    print("Training RandomForest classifier …")
    model = train_model(X_train, y_train, n_estimators=args.n_estimators, max_depth=args.max_depth)

    print("Evaluating on held-out split …")
    y_pred = model.predict(X_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)

    print(f"Saving model to {args.model_out}")
    joblib.dump(model, args.model_out)
    print(f"Saving vectorizer to {args.vectorizer_out}")
    joblib.dump(vectorizer, args.vectorizer_out)

    if args.metrics_out:
        metrics_path = Path(args.metrics_out)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Validation metrics written to {metrics_path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
