from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from src.auth.permissions import IsStaff
from src.services.forecast import ForecastError, get_forecast, trigger_forecast


class ForecastController(APIView):
    """Triggers and retrieves 90-day patient-demand forecasts.

    History is read from the database by the service layer;
    the caller does not send observations.

    Retrieval (GET) requires staff authentication (ADMIN or MANAGER).
    """

    authentication_classes = [JWTAuthentication]

    def get_permissions(self):
        if self.request.method == "GET":
            return [IsStaff()]

        return [AllowAny()]

    def post(self, request):
        try:
            result = trigger_forecast()
        except ForecastError as exc:
            return Response(
                {"success": False, "error": str(exc)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            return Response(
                {"success": False, "error": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(result, status=status.HTTP_200_OK)

    def get(self, request, run_id=None):
        try:
            result = get_forecast(run_id)
        except ForecastError as exc:
            return Response(
                {"success": False, "error": str(exc)},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(result, status=status.HTTP_200_OK)
