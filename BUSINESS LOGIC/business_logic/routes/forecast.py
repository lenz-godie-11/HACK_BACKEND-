from django.urls import path

from business_logic.controllers.forecast import ForecastController


urlpatterns = [
    path(
        "forecast/",
        ForecastController.as_view(),
        name="forecast",
    ),
    path(
        "forecast/<int:run_id>/",
        ForecastController.as_view(),
        name="forecast-detail",
    ),
]
