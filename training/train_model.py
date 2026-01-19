import pandas as pd
from pathlib import Path
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

# Paths
PROCESSED_DATA_PATH = Path(
    "C:/Users/sajan/Documents/GitHub/Customer-Transaction-Risk-Pipeline/pipeline/data/processed/transaction_features_v1.csv"
)
MODEL_PATH = Path("models/fraud_model.pkl")


def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop(columns=["isFraud"])
    y = df["isFraud"]

    print("Class distribution:")
    print(y.value_counts(normalize=True), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # -----------------------------
    # Pipeline (scaling + model)
    # -----------------------------
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="saga",
            class_weight="balanced",
            max_iter=3000,
        ))
    ])

    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.6f}")

    # -----------------------------
    # Business-aware threshold
    # -----------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    idx = (precision >= 0.10).argmax()
    threshold = thresholds[idx]

    print(f"\nChosen threshold for precision≥10%: {threshold:.4f}")

    y_pred = (y_proba >= threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train_model()
