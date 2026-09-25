from typing import Literal

from pydantic import BaseModel, Field


class GenderScores(BaseModel):
    female: float = Field(..., ge=0.0, le=1.0)
    male: float = Field(..., ge=0.0, le=1.0)
    child: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    age_years: float = Field(..., description="Predicted age in years")
    age_normalized: float = Field(..., description="Raw model age output (age/100)")
    gender: GenderScores
    predicted_gender: Literal["female", "male", "child"]
    duration_seconds: float
    sampling_rate: int
    filename: str | None = None


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dim: int
    duration_seconds: float
    sampling_rate: int
    filename: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    environment: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    model_loaded: bool
    device: str
    model_name: str


class ErrorResponse(BaseModel):
    detail: str
