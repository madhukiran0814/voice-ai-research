import argparse
import os
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

DEFAULT_AUDIO_PATH = os.environ.get(
    "TEST_AUDIO_PATH",
    r"C:\Users\madhu\Downloads\test\test-audio.wav"
    if os.name == "nt"
    else "./test-audio.wav",
)
TARGET_SAMPLING_RATE = 16000


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender



# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name)


def load_wav(path, target_sr: int = TARGET_SAMPLING_RATE):
    """Load a WAV file as mono float32 and resample to target_sr if needed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        raw = wav_file.readframes(n_frames)

    if sample_width == 1:
        signal = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        signal = (signal - 128.0) / 128.0
    elif sample_width == 2:
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        signal = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != target_sr:
        if sample_rate % target_sr == 0:
            step = sample_rate // target_sr
            signal = signal[::step]
        else:
            duration = len(signal) / sample_rate
            new_length = int(duration * target_sr)
            indices = np.linspace(0, len(signal) - 1, new_length)
            signal = np.interp(indices, np.arange(len(signal)), signal).astype(np.float32)

    return signal.astype(np.float32), target_sr


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run age/gender inference on a WAV file.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=DEFAULT_AUDIO_PATH,
        help=f"Path to input WAV file (default: {DEFAULT_AUDIO_PATH})",
    )
    args = parser.parse_args()

    signal, sampling_rate = load_wav(args.audio_path)
    duration_sec = len(signal) / sampling_rate
    print(f"Loaded: {args.audio_path}")
    print(f"Samples: {len(signal)}, rate: {sampling_rate} Hz, duration: {duration_sec:.2f}s")

    predictions = process_func(signal, sampling_rate)
    print("Age/gender predictions:")
    print("   Age        female     male       child")
    print(predictions)

    embeddings = process_func(signal, sampling_rate, embeddings=True)
    print("Embedding shape:", embeddings.shape)
    print("Embedding preview:", embeddings[0, :8], "...")
