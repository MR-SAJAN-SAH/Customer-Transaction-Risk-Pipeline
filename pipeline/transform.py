import pandas as pd
from pathlib import Path

CLEAN_DATA_PATH = Path("data/clean/transactions_clean.csv")

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transaction data:
    - Type conversions
    - Remove invalid records
    - Deduplication
    """

    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Ensure numeric types
    numeric_cols = [
        "step", "amount",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid rows
    df = df[df["amount"] > 0]
    df = df.dropna(subset=["nameOrig", "nameDest", "type"])

    # Deduplicate
    df = df.drop_duplicates()

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
