import os

MODEL_NAME = "ThetaModel"
MODEL_PERIOD = 7
FORECAST_HORIZON = 90

MODEL_ARTIFACT_PATH = os.getenv(
    "MODEL_ARTIFACT_PATH",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "forecast-models", "theta_period7.pkl"
    ),
)
