# Speech-to-text using faster-whisper. Records from the mic until silence,
# then transcribes to text. Swappable: anything that returns a string works here.

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000       # Hz — what Whisper expects
CHUNK_SECS = 1            # record in 1-second chunks
SILENCE_RMS = 0.01        # volume threshold below which we call it silence
MIN_CHUNKS = 3            # always record at least 3 seconds before checking silence
MAX_SILENCE_CHUNKS = 3    # stop after 3 consecutive silent seconds
MAX_DURATION_SECS = 30    # hard cap — never record longer than this

_model = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        print("[STT] Loading faster-whisper base model (first time only)...")
        _model = WhisperModel("base", device="cpu", compute_type="int8")
        print("[STT] Model ready.")
    return _model


def _record_until_silence() -> np.ndarray:
    chunks = []
    silent_count = 0
    max_chunks = MAX_DURATION_SECS // CHUNK_SECS

    print("[STT] Listening... speak now. (silence stops recording)")

    for i in range(max_chunks):
        chunk = sd.rec(
            int(CHUNK_SECS * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        chunks.append(chunk)

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        print(f"[STT] chunk {i+1}: rms={rms:.4f}")

        if i >= MIN_CHUNKS:
            if rms < SILENCE_RMS:
                silent_count += 1
                if silent_count >= MAX_SILENCE_CHUNKS:
                    break
            else:
                silent_count = 0

    print("[STT] Recording done.")
    return np.concatenate(chunks, axis=0).flatten()


def transcribe(audio: np.ndarray) -> str:
    model = _get_model()
    segments, _ = model.transcribe(audio, language="en", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments).strip()


def listen_and_transcribe() -> str:
    audio = _record_until_silence()
    text = transcribe(audio)
    print(f"[STT] Transcript: {text!r}")
    return text
