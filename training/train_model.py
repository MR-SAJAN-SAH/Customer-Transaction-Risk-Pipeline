import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

PROCESSED_DATA_PATH = Path("C:\\Users\\sajan\\Documents\\GitHub\\Customer-Transaction-Risk-Pipeline\\pipeline\\data\\processed\\customer_features_v1.csv")
MODEL_PATH = Path("models/risk_model.pkl")


def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Target variable: customer considered risky if any fraud occurred
    df["is_high_risk"] = (df["fraud_transaction_count"] > 0).astype(int)

    X = df.drop(
        columns=[
            "nameOrig",
            "is_high_risk"
        ]
    )
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
