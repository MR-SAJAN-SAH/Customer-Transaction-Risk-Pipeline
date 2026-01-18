import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("C:\\Users\\sajan\\Documents\\GitHub\\Customer-Transaction-Risk-Pipeline\\data\\raw\paysim_dataset.csv")

def extract_raw_data() -> pd.DataFrame:
    """
    Loads raw transaction data without modification and raw data must never be altered.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    return df
