import pandas as pd

from business_logic.config.settings import FORECAST_HORIZON
from business_logic.services.theta import ArtifactLoadError, forecast_theta_artifact
from business_logic.validators.forecast import validate_history


def generate_forecast(history: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate a forecast run: validate, preprocess, then fit and forecast."""
    data = validate_history(history)
    data = data.sort_values("date")

    return forecast_theta_artifact()


def forecast_to_records(forecast: pd.DataFrame) -> list[dict]:
    """Convert a forecast frame into JSON-ready records."""
    result = forecast.copy()

    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result["patients"] = result["patients"].round(2)

    return result.to_dict(orient="records")


def run_forecast(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Full flow: validate history -> artifact forecast.

    The history is validated for basic structure but the actual forecast
    is generated from the pre-fitted artifact, not from the input history.

    Returns exactly FORECAST_HORIZON daily points with 'date' and 'patients'.
    Raises ValueError when the input or the produced forecast is invalid.
    """
    validate_history(raw_history)

    if raw_history.empty or {"date", "patients"} - set(raw_history.columns):
        raise ValueError("History returned invalid data.")

    # Use artifact-based forecast (does not depend on raw_history content)
    try:
        forecast = forecast_theta_artifact()
    except ArtifactLoadError as exc:
        raise ValueError(f"Model artifact error: {exc}") from exc

    if len(forecast) != FORECAST_HORIZON:
        raise ValueError(f"Expected {FORECAST_HORIZON} forecast points.")

    # Strict date validation: must be actual datetime64, not integer positions
    if not pd.api.types.is_datetime64_any_dtype(forecast["date"]):
        raise ValueError(
            f"Forecast dates must be datetime64, got {forecast['date'].dtype}. "
            f"This indicates the model returned integer positions instead of timestamps."
        )

    dates = pd.to_datetime(forecast["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Forecast contains invalid dates.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be strictly ordered.")

    # Ensure consecutive daily frequency
    if dates.dt.normalize().diff().dropna().ne(pd.Timedelta(days=1)).any():
        raise ValueError("Forecast dates must be consecutive daily values.")

    if not pd.api.types.is_numeric_dtype(forecast["patients"]):
        raise ValueError("Predicted patients must be numeric.")

    if (forecast["patients"] < 0).any():
        raise ValueError("Predicted patients cannot be negative.")

    return forecast
