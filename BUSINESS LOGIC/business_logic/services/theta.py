import os
import pickle
from typing import Optional

import pandas as pd

from statsmodels.tsa.forecasting.theta import ThetaModel, ThetaModelResults

from business_logic.config.settings import FORECAST_HORIZON, MODEL_PERIOD


def _get_artifact_path() -> str:
    """Resolve the artifact path from environment or default."""
    return os.getenv(
        "MODEL_ARTIFACT_PATH",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "forecast-models", "theta_period7.pkl"
        ),
    )


class ArtifactLoadError(Exception):
    """Raised when the fitted Theta artifact cannot be loaded."""


_artifact_cache: Optional[ThetaModelResults] = None


def _load_artifact() -> ThetaModelResults:
    """Load the fitted ThetaModelResults artifact from disk.

    Uses a simple module-level cache to avoid repeated I/O.
    """
    global _artifact_cache
    if _artifact_cache is not None:
        return _artifact_cache

    path = _get_artifact_path()
    if not os.path.exists(path):
        raise ArtifactLoadError(f"Model artifact not found at {path}")

    try:
        with open(path, "rb") as f:
            _artifact_cache = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        raise ArtifactLoadError(f"Failed to load model artifact from {path}: {exc}") from exc

    if not isinstance(_artifact_cache, ThetaModelResults):
        raise ArtifactLoadError(
            f"Artifact at {path} is not a ThetaModelResults object "
            f"(got {type(_artifact_cache).__name__})"
        )

    return _artifact_cache


def _clear_artifact_cache() -> None:
    """Clear the artifact cache. Primarily for testing."""
    global _artifact_cache
    _artifact_cache = None


def forecast_theta_artifact() -> pd.DataFrame:
    """Generate a forecast using the pre-fitted Theta artifact.

    The artifact was fitted on 958 daily observations ending 2022-12-31.
    The forecast always starts from 2023-01-01 regardless of the current
    database history. The caller is responsible for validating that this
    static forecast is appropriate for the use case.

    Returns a DataFrame with 'date' and 'patients' columns.
    """
    artifact = _load_artifact()

    forecast = artifact.forecast(FORECAST_HORIZON)

    result = pd.DataFrame(
        {
            "date": forecast.index,
            "patients": forecast.values,
        }
    )

    result["patients"] = result["patients"].clip(lower=0)

    return result


def forecast_theta(history: pd.DataFrame) -> pd.DataFrame:
    """Fit ThetaModel(period=7) on the provided processed history and
    produce a ``FORECAST_HORIZON``-day forecast.

    DEPRECATED: This refits the model on every call. Kept for backward
    compatibility and testing. Use forecast_theta_artifact() for production.

    The forecast horizon starts immediately after the latest date in
    ``history``, so the result changes when the backend observation
    history changes.
    """
    series = history["patients"].copy().astype(float)
    index = pd.DatetimeIndex(history["date"])

    if index.freq is None and index.inferred_freq:
        index = pd.DatetimeIndex(index, freq=index.inferred_freq)

    series.index = index
    series = series.sort_index()

    model = ThetaModel(
        series,
        period=MODEL_PERIOD,
    )

    fitted_model = model.fit()

    forecast = fitted_model.forecast(FORECAST_HORIZON)

    result = pd.DataFrame(
        {
            "date": forecast.index,
            "patients": forecast.values,
        }
    )

    result["patients"] = result["patients"].clip(lower=0)

    return result
