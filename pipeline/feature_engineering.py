import pandas as pd
from pathlib import Path

PROCESSED_DATA_PATH = Path(
    "C:/Users/sajan/Documents/GitHub/Customer-Transaction-Risk-Pipeline/pipeline/data/processed/transaction_features_v1.csv"
)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction-level fraud features.
    Preserves PaySim fraud signal.
    """

    df = df.copy()

    # -----------------------
    # Balance dynamics
    # -----------------------
    df["orig_balance_delta"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["dest_balance_delta"] = df["newbalanceDest"] - df["oldbalanceDest"]

    # -----------------------
    # Transaction type encoding
    # -----------------------
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(int)
    df["is_cashout"] = (df["type"] == "CASH_OUT").astype(int)

    # -----------------------
    # Risk heuristics (strong signals in PaySim)
    # -----------------------
    df["amount_over_orig_balance"] = (
        df["amount"] > df["oldbalanceOrg"]
    ).astype(int)

    df["zero_dest_balance_before"] = (
        df["oldbalanceDest"] == 0
    ).astype(int)

    # -----------------------
    # Feature set
    # -----------------------
    features = [
        "step",
        "amount",
        "orig_balance_delta",
        "dest_balance_delta",
        "is_transfer",
        "is_cashout",
        "amount_over_orig_balance",
        "zero_dest_balance_before",
    ]

    final_df = df[features + ["isFraud"]].copy()

    return final_df


def save_features(df: pd.DataFrame) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
