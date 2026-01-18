import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path("data/processed/customer_features_v1.csv")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transaction-level data into customer-level features.
    """

    df = df.copy()

    # Create customer-level aggregation
    customer_features = df.groupby("nameOrig").agg(
        total_transactions=("amount", "count"),
        total_transaction_amount=("amount", "sum"),
        avg_transaction_amount=("amount", "mean"),
        std_transaction_amount=("amount", "std"),
        max_transaction_amount=("amount", "max"),
        fraud_transaction_count=("isFraud", "sum"),
        flagged_transaction_count=("isFlaggedFraud", "sum"),
        last_transaction_step=("step", "max")
    ).reset_index()

    # Derived risk indicators
    customer_features["fraud_ratio"] = (
        customer_features["fraud_transaction_count"]
        / customer_features["total_transactions"]
    )

    customer_features["flagged_ratio"] = (
        customer_features["flagged_transaction_count"]
        / customer_features["total_transactions"]
    )

    # Replace NaN values from std calculation (single transaction users)
    customer_features = customer_features.fillna(0)

    return customer_features


def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
