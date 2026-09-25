from app.services.audio import AudioValidationError, load_wav_bytes
import pytest


def test_empty_audio_raises():
    with pytest.raises(AudioValidationError):
        load_wav_bytes(b"")


def test_invalid_wav_raises():
    with pytest.raises(AudioValidationError):
        load_wav_bytes(b"not-a-wav")
