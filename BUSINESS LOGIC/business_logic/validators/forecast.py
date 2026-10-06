import pandas as pd


def validate_history(df: pd.DataFrame) -> pd.DataFrame:
    """Validate historical data before forecasting.

    Requires a non-empty frame with 'date' and 'patients' columns,
    parseable dates, numeric non-negative patient values.
    """
    if df is None:
        raise ValueError("Historical data cannot be None.")

    if df.empty:
        raise ValueError("Historical data cannot be empty.")

    required = {"date", "patients"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["patients"] = pd.to_numeric(data["patients"], errors="coerce")

    if data["date"].isna().any():
        raise ValueError("Invalid date values found.")

    if data["patients"].isna().any():
        raise ValueError("Invalid patient values found.")

    if (data["patients"] < 0).any():
        raise ValueError("Patient values cannot be negative.")

    return data
