import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/customer_features_v1.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data into customer-level and behavioral features WITHOUT label leakage.
    """

    df = df.copy()

    customer_features = df.groupby("nameOrig").agg(
        total_transactions=("amount", "count"),
        total_transaction_amount=("amount", "sum"),
        avg_transaction_amount=("amount", "mean"),
        std_transaction_amount=("amount", "std"),
        max_transaction_amount=("amount", "max"),
        min_transaction_amount=("amount", "min"),
        last_transaction_step=("step", "max"),

        # These are kept ONLY for label creation, not as predictors
        fraud_transaction_count=("isFraud", "sum"),
        flagged_transaction_count=("isFlaggedFraud", "sum")
    ).reset_index()

    # Replace NaN from std (customers with single transaction)
    customer_features = customer_features.fillna(0)

    return customer_features


def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
