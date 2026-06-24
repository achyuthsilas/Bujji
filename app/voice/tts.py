# Text-to-speech using Piper. Pipes text into the Piper binary, gets audio back,
# and plays it through the default speakers. Swappable: replace this file to use a
# different TTS engine without touching anything else.
#
# Two ways to use it:
#   - speak(text)            : synth the whole thing, then play (simple, blocking).
#   - synth_sentence(text)   : synth ONE sentence to raw PCM and return it, without
#                              playing. The streaming pipeline calls this per sentence
#                              so playback can start on the first sentence of a reply.

import json
import subprocess
import tempfile
import os
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

VOICE_DIR = Path(__file__).parent
PIPER_EXE = VOICE_DIR / "piper.exe"
MODELS_DIR = VOICE_DIR / "models"
VOICE_MODEL = MODELS_DIR / "en_US-lessac-medium.onnx"


def _read_sample_rate() -> int:
    """Piper writes raw audio with no header, so we need the model's sample rate.
    It lives in the voice model's .json sidecar (lessac-medium = 22050 Hz)."""
    cfg = VOICE_MODEL.with_suffix(VOICE_MODEL.suffix + ".json")
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            return int(json.load(f)["audio"]["sample_rate"])
    except Exception:
        return 22050


SAMPLE_RATE = _read_sample_rate()


def _check_setup():
    if not PIPER_EXE.exists():
        raise FileNotFoundError(f"piper.exe not found at {PIPER_EXE}")
    if not VOICE_MODEL.exists():
        raise FileNotFoundError(f"Voice model not found at {VOICE_MODEL}")


def synth_sentence(text: str) -> np.ndarray:
    """Synthesize ONE sentence and return it as float32 mono PCM (does NOT play it).
    Uses Piper's --output_raw so we get audio straight from stdout, no temp file."""
    _check_setup()
    text = text.strip()
    if not text:
        return np.zeros(0, dtype=np.float32)

    result = subprocess.run(
        [str(PIPER_EXE), "--model", str(VOICE_MODEL), "--output_raw"],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )
    # Raw int16 mono PCM at SAMPLE_RATE → float32 in [-1, 1].
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def speak(text: str) -> None:
    _check_setup()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run(
            [str(PIPER_EXE), "--model", str(VOICE_MODEL), "--output_file", tmp_path],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )

        with wave.open(tmp_path, "rb") as wav:
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            frames = wav.readframes(wav.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        print(f"[TTS] Speaking: {text!r}")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
