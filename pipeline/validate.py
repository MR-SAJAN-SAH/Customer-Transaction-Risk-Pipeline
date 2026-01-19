import pandas as pd

class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_data(df: pd.DataFrame) -> None:
    """
    Validates cleaned transaction data.
    Pipeline must stop if validation fails.
    """

    # Required schema
    required_columns = {
        "step", "type", "amount",
        "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
        "isFraud", "isFlaggedFraud",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")

    # Logical checks
    if (df["amount"] <= 0).any():
        raise DataValidationError("Non-positive transaction amounts found")

    if df["step"].min() < 0 or df["step"].max() > 744:
        raise DataValidationError("Step out of expected range (0–744 hours)")

    if not set(df["isFraud"].unique()).issubset({0, 1}):
        raise DataValidationError("Invalid values in isFraud")

    if not set(df["isFlaggedFraud"].unique()).issubset({0, 1}):
        raise DataValidationError("Invalid values in isFlaggedFraud")

    # Balance consistency
    if (df["newbalanceOrig"] < 0).any() or (df["newbalanceDest"] < 0).any():
        raise DataValidationError("Negative balances detected")

    return
