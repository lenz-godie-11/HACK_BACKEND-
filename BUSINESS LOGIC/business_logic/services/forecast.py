import pandas as pd

from pipeline.services.pipeline import run_pipeline

from business_logic.config.settings import FORECAST_HORIZON
from business_logic.services.theta import forecast_theta
from business_logic.validators.forecast import validate_history


def generate_forecast(history: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate a forecast run: validate, then fit and forecast."""
    data = validate_history(history)
    data = data.sort_values("date")

    return forecast_theta(data)


def forecast_to_records(forecast: pd.DataFrame) -> list[dict]:
    """Convert a forecast frame into JSON-ready records."""
    result = forecast.copy()

    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result["patients"] = result["patients"].round(2)

    return result.to_dict(orient="records")


def run_forecast(raw_history: pd.DataFrame) -> pd.DataFrame:
    """Full flow: raw observations -> DATA PIPELINE -> Theta forecast.

    Returns exactly FORECAST_HORIZON daily points with 'date' and 'patients'.
    Raises ValueError when the input or the produced forecast is invalid.
    """
    prepared = run_pipeline(raw_history).reset_index()

    if prepared.empty or {"date", "patients"} - set(prepared.columns):
        raise ValueError("Pipeline returned invalid model-ready data.")

    forecast = forecast_theta(prepared)

    if len(forecast) != FORECAST_HORIZON:
        raise ValueError(f"Expected {FORECAST_HORIZON} forecast points.")

    dates = pd.to_datetime(forecast["date"], errors="coerce")

    if dates.isna().any():
        raise ValueError("Forecast contains invalid dates.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Forecast dates must be strictly ordered.")

    if not pd.api.types.is_numeric_dtype(forecast["patients"]):
        raise ValueError("Predicted patients must be numeric.")

    if (forecast["patients"] < 0).any():
        raise ValueError("Predicted patients cannot be negative.")

    return forecast
