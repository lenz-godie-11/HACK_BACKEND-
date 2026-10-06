import numpy as np
import pandas as pd

from pipeline.config.settings import (
    INTERPOLATION_METHOD,
    PIPELINE_FREQUENCY,
    WINSOR_LOWER_IQR_MULTIPLIER,
    WINSOR_UPPER_IQR_MULTIPLIER,
)
from pipeline.validators.data import validate_data


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Sum same-day observations into one row per calendar date."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    return (
        data.groupby("date", as_index=False)["patients"]
        .sum()
        .sort_values("date")
    )


def reindex_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete daily calendar between the first and last date."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.set_index("date")

    full_index = pd.date_range(
        start=data.index.min(),
        end=data.index.max(),
        freq=PIPELINE_FREQUENCY,
    )

    data = data.reindex(full_index)
    data.index.name = "date"

    return data


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gaps left by reindexing with linear interpolation."""
    data = df.copy()

    data["patients"] = data["patients"].interpolate(
        method=INTERPOLATION_METHOD
    )

    return data


def winsorize(df: pd.DataFrame) -> pd.DataFrame:
    """Clip outliers using the validated IQR bounds.

    Lower bound: Q1 - 0.1 * IQR
    Upper bound: Q3 + 1.5 * IQR
    """
    data = df.copy()

    q1 = data["patients"].quantile(0.25)
    q3 = data["patients"].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (WINSOR_LOWER_IQR_MULTIPLIER * iqr)
    upper_bound = q3 + (WINSOR_UPPER_IQR_MULTIPLIER * iqr)

    data["patients"] = np.clip(
        data["patients"],
        lower_bound,
        upper_bound,
    )

    return data


def prepare_model_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing sequence and return model-ready data."""
    data = validate_data(df)
    data = aggregate_daily(data)
    data = reindex_calendar(data)
    data = interpolate_missing(data)
    data = winsorize(data)

    data["patients"] = data["patients"].astype(float)

    return data


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: raw observations to model-ready daily series."""
    return prepare_model_data(df)
