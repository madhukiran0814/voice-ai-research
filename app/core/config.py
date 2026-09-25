from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Age Gender Speech API"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Model
    model_name: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    model_device: Literal["cpu", "cuda", "mps"] = "cpu"
    model_local_files_only: bool = False
    hf_home: str | None = Field(default=None, description="Hugging Face cache root")
    target_sampling_rate: int = 16000

    # Inference limits
    max_audio_seconds: float = 60.0
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    allowed_content_types: str = "audio/wav,audio/x-wav,audio/wave,application/octet-stream"

    # API
    api_prefix: str = "/api/v1"
    cors_origins: str = "*"
    workers: int = 1

    # Age denormalization (audeering model outputs age / 100)
    age_scale: float = 100.0

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_content_type_list(self) -> list[str]:
        return [c.strip() for c in self.allowed_content_types.split(",") if c.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
