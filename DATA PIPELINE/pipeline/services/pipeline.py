import numpy as np
import pandas as pd

from pipeline.config.settings import (
    INTERPOLATION_METHOD,
    MAX_INTERPOLATION_GAP_DAYS,
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


def _detect_gaps(series: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Detect consecutive NaN gaps in a series.

    Returns list of (gap_start, gap_end, gap_length) for each gap.
    """
    isnan = series.isna()
    if not isnan.any():
        return []

    # Find consecutive NaN blocks
    gap_starts = isnan & ~isnan.shift(1, fill_value=False)
    gap_ends = isnan & ~isnan.shift(-1, fill_value=False)

    starts = series.index[gap_starts]
    ends = series.index[gap_ends]

    gaps = []
    for start, end in zip(starts, ends):
        gap_length = (end - start).days + 1
        gaps.append((start, end, gap_length))

    return gaps


def interpolate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill small gaps left by reindexing with linear interpolation.

    Gaps larger than MAX_INTERPOLATION_GAP_DAYS are not interpolated and
    cause a ValueError to be raised.
    """
    data = df.copy()
    patients = data["patients"]

    gaps = _detect_gaps(patients)

    for gap_start, gap_end, gap_length in gaps:
        if gap_length > MAX_INTERPOLATION_GAP_DAYS:
            raise ValueError(
                f"Gap of {gap_length} days from {gap_start.date()} to {gap_end.date()} "
                f"exceeds maximum interpolation threshold of {MAX_INTERPOLATION_GAP_DAYS} days. "
                f"History contains discontinuity too large for reliable forecasting."
            )

    # All gaps are within threshold, safe to interpolate
    data["patients"] = patients.interpolate(method=INTERPOLATION_METHOD)

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
