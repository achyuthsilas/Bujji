# Wake-word detection — "Sunday" / "Hey Sunday".
#
# Plain English: this listens for the magic word so Sunday wakes up hands-free.
# We use livekit-wakeword (LiveKit's open wake-word library, built on openWakeWord but
# with a more accurate conv-attention model and a cleaner local training pipeline).
#
# How its model works (important): livekit-wakeword's predict() is STATELESS and wants a
# full ~2-second window of audio each call (it builds 16 embedding frames internally).
# So the engine keeps a rolling 2s buffer of mic audio and re-scores it every ~160 ms.
#
# IMPORTANT: there's no prebuilt "Sunday" model — you train one into models/sunday.onnx
# (see scripts/train_livekit_wakeword.md). Until then hands-free is off, but /listen works.
#
# Swappable on purpose: everything talks to the WakeWordEngine interface, so the engine
# can be replaced (openWakeWord, Porcupine, ...) without touching the pipeline. Each engine
# is fed raw mic frames via detect(frame) and buffers internally however it needs to.

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
SUNDAY_MODEL = MODELS_DIR / "sunday.onnx"
WAKE_THRESHOLD = float(os.getenv("WAKE_THRESHOLD", "0.5"))

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 2 * SAMPLE_RATE        # livekit-wakeword wants ~2s per prediction
PREDICT_EVERY_FRAMES = 5                # re-score every ~160ms (5 × 32ms frames)


class WakeWordEngine(ABC):
    """Anything that can answer 'did the user just say the wake word?'. Fed raw 16 kHz
    mic frames via detect(); the engine buffers internally as its model requires."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> bool:
        """Feed one audio frame (float32, 16 kHz). Return True on detection."""
        ...


class LiveKitWakeWordEngine(WakeWordEngine):
    def __init__(self, model_path: Path = SUNDAY_MODEL, threshold: float = WAKE_THRESHOLD):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Wake word model not found at {model_path}\n"
                "Train it first — see scripts/train_livekit_wakeword.md\n"
                "(Hands-free mode needs this; /listen single-turn works without it.)"
            )
        self.threshold = threshold
        self.key = model_path.stem        # predict() keys scores by the model filename stem

        from livekit.wakeword import WakeWordModel
        self._model = WakeWordModel(models=[str(model_path)])
        self._buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._count = 0
        print("[WAKE] livekit-wakeword ready.")

    def detect(self, frame: np.ndarray) -> bool:
        # Slide the rolling 2s window forward by one frame.
        n = len(frame)
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = frame

        # Only run the (heavier, stateless) classifier every few frames.
        self._count += 1
        if self._count % PREDICT_EVERY_FRAMES:
            return False

        scores = self._model.predict(self._buf)
        score = scores.get(self.key, max(scores.values()) if scores else 0.0)
        if score >= self.threshold:
            self._buf[:] = 0.0    # clear so we don't re-fire on the same utterance
            self._count = 0
            return True
        return False


class OpenWakeWordEngine(WakeWordEngine):
    """Fallback engine using openWakeWord directly (stateful, buffers internally)."""

    def __init__(self, model_path: Path = SUNDAY_MODEL, threshold: float = WAKE_THRESHOLD):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Wake word model not found at {model_path} (openWakeWord engine)."
            )
        self.threshold = threshold
        self.key = model_path.stem
        from openwakeword.model import Model
        from openwakeword.utils import download_models
        try:
            self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        except Exception:
            download_models()
            self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        print("[WAKE] openWakeWord ready.")

    def detect(self, frame: np.ndarray) -> bool:
        scores = self._model.predict((frame * 32767).astype(np.int16))
        score = scores.get(self.key, max(scores.values()) if scores else 0.0)
        if score >= self.threshold:
            self._model.reset()
            return True
        return False


_engine: WakeWordEngine | None = None


def get_engine() -> WakeWordEngine:
    """Return the configured wake-word engine (cached). Default: livekit-wakeword."""
    global _engine
    if _engine is None:
        which = os.getenv("WAKE_ENGINE", "livekit").lower()
        if which in ("livekit", "livekit-wakeword"):
            _engine = LiveKitWakeWordEngine()
        elif which == "openwakeword":
            _engine = OpenWakeWordEngine()
        else:
            raise ValueError(f"Unknown WAKE_ENGINE={which!r} (use 'livekit' or 'openwakeword')")
    return _engine
