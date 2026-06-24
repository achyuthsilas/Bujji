# Speech-to-text using Groq Whisper Large v3 Turbo.
#
# Plain English: give this a chunk of recorded audio, get back the text.
# It does NOT record any more — recording + knowing when you stopped talking is now
# handled by the Silero VAD inside the pipeline (app/voice/vad.py). The pipeline
# buffers your speech and, the instant you go quiet, hands the whole buffer to
# transcribe() here. Groq Whisper is a single-shot file API (no streaming endpoint),
# so "fire the request the moment speech ends" is the fastest real option.
#
# Swappable: anything that turns audio into a string can replace transcribe().
# Requires GROQ_API_KEY in .env.

import io
import os
import wave

import numpy as np
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

SAMPLE_RATE = 16000       # Hz — what Whisper expects (and what the mic captures)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env — add it to use STT")
        _client = Groq(api_key=api_key)
    return _client


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Wrap float32 mono samples in a WAV container Groq can read."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    buf.seek(0)
    return buf.read()


def transcribe(audio: np.ndarray) -> str:
    """Send buffered audio to Groq Whisper and return the recognized text."""
    client = _get_client()
    wav_bytes = _audio_to_wav_bytes(audio)
    result = client.audio.transcriptions.create(
        file=("audio.wav", wav_bytes, "audio/wav"),
        model="whisper-large-v3-turbo",
        language="en",
        response_format="text",
    )
    text = result.strip() if isinstance(result, str) else result.text.strip()
    print(f"[STT] Transcript: {text!r}")
    return text
