# Voice Activity Detection using Silero VAD (the small ONNX model).
#
# Plain English: this tells us "is the person talking right now?" frame by frame.
# We use it to know exactly WHEN the user stops talking, instead of waiting a fixed
# number of seconds. The moment we see ~400ms of silence after speech, we declare the
# turn over — that's the signal that triggers transcription.
#
# We run the ONNX model directly through onnxruntime (no PyTorch), so it's light and
# CPU-friendly. The model is downloaded to models/silero_vad.onnx.
#
# Silero v5 contract (verified against the model file):
#   inputs : input [batch, 512 samples], state [2, batch, 128], sr (int64 scalar)
#   outputs: output [batch, 1] speech-probability, stateN (carried to next call)
# It must be fed EXACTLY 512 samples per call at 16 kHz (32 ms per frame).

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512                 # Silero v5 processes 512-sample chunks at 16 kHz
CONTEXT_SAMPLES = 64                # ...plus the previous chunk's last 64 samples,
                                    # prepended as context (the model REQUIRES this —
                                    # feeding bare 512 frames makes it blind to speech).
FRAME_MS = FRAME_SAMPLES / SAMPLE_RATE * 1000  # 32 ms

MODEL_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"


class SileroVAD:
    """Streaming speech endpointer. Feed it 512-sample frames; it tells you when the
    user has started talking and — the important bit — when they've stopped."""

    def __init__(
        self,
        threshold: float | None = None,   # prob >= this counts as speech (env VAD_THRESHOLD)
        silence_ms: int = 400,            # this much trailing silence ends the turn
        min_speech_ms: int = 200,         # ignore blips shorter than this
    ):
        if threshold is None:
            threshold = float(os.getenv("VAD_THRESHOLD", "0.5"))
        self.threshold = threshold
        self.neg_threshold = threshold - 0.15   # hysteresis: harder to "un-trigger"
        self.silence_frames_needed = int(silence_ms / FRAME_MS)
        self.min_speech_frames = int(min_speech_ms / FRAME_MS)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {MODEL_PATH}\n"
                "Download it once with:\n"
                "  curl -sL -o app/voice/models/silero_vad.onnx "
                "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear state between turns."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._triggered = False
        self._speech_frames = 0
        self._silence_frames = 0
        self.last_prob = 0.0   # most recent speech probability (for diagnostics)

    @property
    def triggered(self) -> bool:
        """True once real speech has started this turn."""
        return self._triggered

    def _prob(self, frame: np.ndarray) -> float:
        # Prepend the carried 64-sample context, then remember this frame's tail for next time.
        x = np.concatenate([self._context, frame]).reshape(1, -1).astype(np.float32)
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = frame[-CONTEXT_SAMPLES:]
        return float(out[0][0])

    def feed(self, frame: np.ndarray) -> str:
        """Process one 512-sample frame. Returns one of:
        - "speech"  : the user is talking (or just started)
        - "silence" : quiet, but the turn isn't over yet
        - "end"     : speech happened and now ~400ms of silence — turn is OVER
        """
        if frame.shape[0] != FRAME_SAMPLES:
            raise ValueError(f"VAD needs {FRAME_SAMPLES}-sample frames, got {frame.shape[0]}")

        prob = self._prob(frame)
        self.last_prob = prob

        if prob >= self.threshold:
            self._triggered = True
            self._speech_frames += 1
            self._silence_frames = 0
            return "speech"

        # prob below the (lower) negative threshold = confidently silent
        if self._triggered and prob < self.neg_threshold:
            self._silence_frames += 1
            # Only end if we heard enough real speech first (ignore short blips).
            if (
                self._silence_frames >= self.silence_frames_needed
                and self._speech_frames >= self.min_speech_frames
            ):
                self.reset()
                return "end"

        return "silence"
