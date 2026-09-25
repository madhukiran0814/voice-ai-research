"""WAV loading and validation utilities."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np

from app.core.config import get_settings


class AudioValidationError(ValueError):
    """Raised when uploaded audio cannot be processed."""


@dataclass(frozen=True)
class AudioSignal:
    samples: np.ndarray
    sampling_rate: int
    duration_seconds: float


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        signal = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (signal - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise AudioValidationError(f"Unsupported sample width: {sample_width} bytes")


def _resample(signal: np.ndarray, sample_rate: int, target_sr: int) -> np.ndarray:
    if sample_rate == target_sr:
        return signal
    if sample_rate % target_sr == 0:
        return signal[:: sample_rate // target_sr]
    duration = len(signal) / sample_rate
    new_length = max(1, int(duration * target_sr))
    indices = np.linspace(0, len(signal) - 1, new_length)
    return np.interp(indices, np.arange(len(signal)), signal).astype(np.float32)


def load_wav_bytes(data: bytes, target_sr: int | None = None) -> AudioSignal:
    """Load mono float32 PCM from WAV bytes and resample to target_sr."""
    settings = get_settings()
    target_sr = target_sr or settings.target_sampling_rate

    if not data:
        raise AudioValidationError("Empty audio payload")
    if len(data) > settings.max_upload_bytes:
        raise AudioValidationError(
            f"File exceeds max size of {settings.max_upload_bytes} bytes"
        )

    try:
        with wave.open(io.BytesIO(data), "rb") as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            raw = wav_file.readframes(n_frames)
    except wave.Error as exc:
        raise AudioValidationError(f"Invalid WAV file: {exc}") from exc

    if n_frames <= 0:
        raise AudioValidationError("WAV file contains no frames")

    signal = _decode_pcm(raw, sample_width)
    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    signal = _resample(signal.astype(np.float32), sample_rate, target_sr)
    duration = float(len(signal) / target_sr)

    if duration > settings.max_audio_seconds:
        raise AudioValidationError(
            f"Audio duration {duration:.2f}s exceeds limit of {settings.max_audio_seconds}s"
        )
    if duration < 0.1:
        raise AudioValidationError("Audio too short (minimum 0.1s)")

    return AudioSignal(
        samples=signal.astype(np.float32),
        sampling_rate=target_sr,
        duration_seconds=duration,
    )
