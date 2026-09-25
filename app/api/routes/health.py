from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.prediction import HealthResponse, ReadyResponse
from app.services.inference import inference_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    status = inference_service.status()
    return ReadyResponse(
        status="ready" if status["model_loaded"] else "not_ready",
        model_loaded=status["model_loaded"],
        device=status["device"],
        model_name=status["model_name"],
    )
