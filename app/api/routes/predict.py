import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_inference_service
from app.core.config import get_settings
from app.schemas.prediction import EmbeddingResponse, GenderScores, PredictionResponse
from app.services.audio import AudioValidationError, load_wav_bytes
from app.services.inference import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict")


async def _read_upload(file: UploadFile) -> tuple[bytes, str | None]:
    settings = get_settings()
    content_type = (file.content_type or "").split(";")[0].strip().lower()

    if content_type and content_type not in settings.allowed_content_type_list:
        # Still allow if filename ends with .wav (browsers sometimes send octet-stream)
        name = (file.filename or "").lower()
        if not name.endswith(".wav"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported content type: {content_type}. Upload a WAV file.",
            )

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds max size of {settings.max_upload_bytes} bytes",
        )
    return data, file.filename


@router.post(
    "/age-gender",
    response_model=PredictionResponse,
    summary="Predict age and gender from a WAV upload",
)
async def predict_age_gender(
    file: UploadFile = File(..., description="WAV audio file"),
    service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    data, filename = await _read_upload(file)
    try:
        audio = load_wav_bytes(data)
        result = service.predict(audio, return_embeddings=False)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Inference failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        ) from None

    return PredictionResponse(
        age_years=round(result.age_years, 2),
        age_normalized=result.age_normalized,
        gender=GenderScores(**result.gender_probs),
        predicted_gender=result.predicted_gender,  # type: ignore[arg-type]
        duration_seconds=round(audio.duration_seconds, 3),
        sampling_rate=audio.sampling_rate,
        filename=filename,
    )


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Extract speaker embedding from a WAV upload",
)
async def predict_embeddings(
    file: UploadFile = File(..., description="WAV audio file"),
    service: InferenceService = Depends(get_inference_service),
) -> EmbeddingResponse:
    data, filename = await _read_upload(file)
    try:
        audio = load_wav_bytes(data)
        result = service.predict(audio, return_embeddings=True)
    except AudioValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception:
        logger.exception("Embedding extraction failed for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding extraction failed",
        ) from None

    assert result.embedding is not None
    emb = result.embedding.tolist()
    return EmbeddingResponse(
        embedding=emb,
        dim=len(emb),
        duration_seconds=round(audio.duration_seconds, 3),
        sampling_rate=audio.sampling_rate,
        filename=filename,
    )
