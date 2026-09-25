from fastapi import HTTPException, status

from app.services.inference import InferenceService, inference_service


def get_inference_service() -> InferenceService:
    if not inference_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet",
        )
    return inference_service
