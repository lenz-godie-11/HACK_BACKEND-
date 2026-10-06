import pandas as pd


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw observations before any transformation.

    Requires a non-empty frame with 'date' and 'patients' columns,
    parseable dates, numeric non-negative patient values.
    """
    if df is None:
        raise ValueError("Input data cannot be None.")

    if df.empty:
        raise ValueError("Input data cannot be empty.")

    required_columns = {"date", "patients"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = df.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    if data["date"].isna().any():
        raise ValueError("Invalid date values found.")

    data["patients"] = pd.to_numeric(
        data["patients"],
        errors="coerce",
    )

    if data["patients"].isna().any():
        raise ValueError("Invalid patient values found.")

    if (data["patients"] < 0).any():
        raise ValueError("Patient values cannot be negative.")

    return data
