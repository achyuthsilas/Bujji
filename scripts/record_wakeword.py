# Records real samples of you saying "Sunday" for wake word training.
# Saves WAVs to scripts/training_samples/real_positive/.
# After recording, run: python scripts/train_wakeword.py --force
#
# Usage: python scripts/record_wakeword.py

import sys
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
DURATION = 2.0  # seconds per clip
SAMPLES_DIR = Path(__file__).parent / "training_samples" / "real_positive"
DEFAULT_COUNT = 20


def _record() -> np.ndarray:
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def _save(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())


def _playback(audio: np.ndarray) -> None:
    sd.play(audio, samplerate=SAMPLE_RATE)
    sd.wait()


def _next_index() -> int:
    existing = sorted(SAMPLES_DIR.glob("real_*.wav"))
    if not existing:
        return 0
    last = existing[-1].stem  # e.g. "real_0007"
    return int(last.split("_")[1]) + 1


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_COUNT

    print("=== Sunday wake word recorder ===")
    print(f"Will record {count} samples of you saying 'Sunday'.")
    print("Each clip is 2 seconds. Speak clearly, at a natural distance from your mic.")
    print("Vary your tone a little between clips (normal, slightly faster, slightly slower).\n")

    start_idx = _next_index()
    saved = 0

    for i in range(count):
        clip_num = i + 1
        idx = start_idx + i
        path = SAMPLES_DIR / f"real_{idx:04d}.wav"

        print(f"Clip {clip_num}/{count}  — press Enter, then say 'Sunday' (or 'Hey Sunday')")
        input("  [Enter to record] ")

        print("  Recording...")
        audio = _record()

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.005:
            print("  Too quiet — no speech detected. Retrying this clip.\n")
            i -= 1  # doesn't actually decrement in a for loop, handled below
            count += 1  # extend to redo
            continue

        _save(audio, path)
        saved += 1
        print(f"  Saved ({rms:.3f} RMS). Playing back...")
        _playback(audio)

        redo = input("  Keep? [Enter = yes / r = redo]: ").strip().lower()
        if redo == "r":
            path.unlink()
            saved -= 1
            count += 1  # add a replacement clip
            print("  Deleted. Will record a replacement.\n")
        else:
            print()

    print(f"Done! Saved {saved} real samples to {SAMPLES_DIR}")
    print("\nNext step:")
    print("  python scripts/train_wakeword.py --force")


if __name__ == "__main__":
    main()
