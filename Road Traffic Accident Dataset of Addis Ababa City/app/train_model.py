"""Train the crash severity model and record its held-out performance."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

APP_DIR = Path(__file__).resolve().parent
CATEGORIES = {
    "Light_conditions": "Lighting",
    "Weather_conditions": "Weather",
    "Road_surface_conditions": "Road surface",
    "Types_of_Junction": "Junction type",
}
FEATURES = [*CATEGORIES, "hour"]
TARGET = "is_high_severity"


def main():
    crashes = pd.read_csv(APP_DIR / "crashes_clean.csv")
    crashes = crashes.dropna(subset=[*FEATURES, TARGET])
    x_train, x_test, y_train, y_test = train_test_split(
        crashes[FEATURES], crashes[TARGET].astype(int),
        test_size=0.2, random_state=42, stratify=crashes[TARGET],
    )
    encoder = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), FEATURES),
    ])
    model = make_pipeline(encoder, LogisticRegression(max_iter=1000))
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "baseline": float((y_test == 0).mean()),
        "auroc": roc_auc_score(y_test, scores),
        "test_size": len(y_test),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    joblib.dump({"model": model, "metrics": metrics}, APP_DIR / "risk_model.joblib")
    print(metrics)


if __name__ == "__main__":
    main()
