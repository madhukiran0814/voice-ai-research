"""Model lifecycle and inference service (singleton)."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import Wav2Vec2Processor

from app.core.config import Settings, get_settings
from app.ml.age_gender_model import AgeGenderModel
from app.services.audio import AudioSignal

logger = logging.getLogger(__name__)

GENDER_LABELS = ("female", "male", "child")


@dataclass(frozen=True)
class PredictionResult:
    age_normalized: float
    age_years: float
    gender_probs: dict[str, float]
    predicted_gender: str
    embedding: np.ndarray | None = None


class InferenceService:
    """Thread-safe wrapper around the audeering Wav2Vec2 age/gender model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: AgeGenderModel | None = None
        self._processor: Wav2Vec2Processor | None = None
        self._device: torch.device | None = None
        self._settings: Settings | None = None
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None and self._processor is not None

    @property
    def device_name(self) -> str:
        return str(self._device) if self._device is not None else "uninitialized"

    @property
    def model_name(self) -> str:
        if self._settings is None:
            return get_settings().model_name
        return self._settings.model_name

    def load(self, settings: Settings | None = None) -> None:
        with self._lock:
            if self._loaded:
                return

            settings = settings or get_settings()
            self._settings = settings

            if settings.hf_home:
                os.environ["HF_HOME"] = settings.hf_home

            device = torch.device(settings.model_device)
            if settings.model_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable; falling back to CPU")
                device = torch.device("cpu")

            logger.info(
                "Loading model %s on %s (local_files_only=%s)",
                settings.model_name,
                device,
                settings.model_local_files_only,
            )

            processor = Wav2Vec2Processor.from_pretrained(
                settings.model_name,
                local_files_only=settings.model_local_files_only,
            )
            model = AgeGenderModel.from_pretrained(
                settings.model_name,
                local_files_only=settings.model_local_files_only,
            )
            model.to(device)
            model.eval()

            self._processor = processor
            self._model = model
            self._device = device
            self._loaded = True
            logger.info("Model loaded successfully")

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._processor = None
            self._device = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict(
        self,
        audio: AudioSignal,
        *,
        return_embeddings: bool = False,
    ) -> PredictionResult:
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        assert self._processor is not None
        assert self._model is not None
        assert self._device is not None
        assert self._settings is not None

        # Processor normalizes; always returns a batch
        processed = self._processor(
            audio.samples,
            sampling_rate=audio.sampling_rate,
        )
        input_values = processed["input_values"][0]
        tensor = torch.from_numpy(np.asarray(input_values, dtype=np.float32)).reshape(1, -1)
        tensor = tensor.to(self._device)

        with torch.inference_mode():
            hidden, logits_age, logits_gender = self._model(tensor)

        age_normalized = float(logits_age.detach().cpu().numpy().reshape(-1)[0])
        gender_arr = logits_gender.detach().cpu().numpy().reshape(-1)
        gender_probs = {
            label: float(gender_arr[i]) for i, label in enumerate(GENDER_LABELS)
        }
        predicted = max(gender_probs, key=gender_probs.get)

        embedding = None
        if return_embeddings:
            embedding = hidden.detach().cpu().numpy().reshape(-1)

        return PredictionResult(
            age_normalized=age_normalized,
            age_years=age_normalized * self._settings.age_scale,
            gender_probs=gender_probs,
            predicted_gender=predicted,
            embedding=embedding,
        )

    def status(self) -> dict[str, Any]:
        return {
            "model_loaded": self.is_ready,
            "device": self.device_name,
            "model_name": self.model_name,
        }


# Process-wide singleton
inference_service = InferenceService()
