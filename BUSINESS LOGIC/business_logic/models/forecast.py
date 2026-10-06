from dataclasses import dataclass


@dataclass
class ForecastResult:
    """A single forecast point: date plus predicted patient count."""

    date: str
    patients: float
