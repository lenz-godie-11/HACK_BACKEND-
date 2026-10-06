import pandas as pd

from statsmodels.tsa.forecasting.theta import ThetaModel

from business_logic.config.settings import FORECAST_HORIZON, MODEL_PERIOD


def fit_theta(history: pd.DataFrame):
    """Fit ThetaModel(period=7) on the daily patient series."""
    series = history["patients"].copy()
    index = pd.DatetimeIndex(history["date"])

    if index.freq is None and index.inferred_freq:
        index = pd.DatetimeIndex(index, freq=index.inferred_freq)

    series.index = index
    series = series.sort_index()

    model = ThetaModel(
        series,
        period=MODEL_PERIOD,
    )

    return model.fit()


def forecast_theta(history: pd.DataFrame) -> pd.DataFrame:
    """Produce a 90-day forecast as a DataFrame with 'date' and 'patients'."""
    fitted_model = fit_theta(history)

    forecast = fitted_model.forecast(FORECAST_HORIZON)

    result = pd.DataFrame(
        {
            "date": forecast.index,
            "patients": forecast.values,
        }
    )

    result["patients"] = result["patients"].clip(lower=0)

    return result
