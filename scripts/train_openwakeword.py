# Train a custom "Sunday" / "Hey Sunday" wake-word model for openWakeWord.
#
# Plain English: openWakeWord doesn't ship a "Sunday" model, so we make one.
# This script does the part that runs cleanly on your machine right now:
#   1. Generates synthetic "Sunday" / "Hey Sunday" voice clips with your local Piper.
#   2. Collects the real samples you already recorded under scripts/training_samples/.
# Then it hands off to openWakeWord's trainer to produce app/voice/models/sunday.onnx.
#
# Honest heads-up: the actual model-training step needs the heavy training stack
# (PyTorch + openWakeWord's training extras + a folder of background/negative audio).
# That stack is NOT required to RUN Sunday — only to build a new wake model. If it isn't
# installed, this script still generates the positive data and prints the exact next steps,
# rather than pretending it trained something. If training is too painful on Windows, the
# pipeline keeps a Porcupine drop-in seam (see app/voice/wake.py).
#
# Usage:
#   python scripts/train_openwakeword.py            # generate data (+ train if able)
#   python scripts/train_openwakeword.py --data-only

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_EXE = ROOT / "app" / "voice" / "piper.exe"
VOICE_MODEL = ROOT / "app" / "voice" / "models" / "en_US-lessac-medium.onnx"
OUT_MODEL = ROOT / "app" / "voice" / "models" / "sunday.onnx"

DATA_DIR = ROOT / "scripts" / "openww_data"
SYNTH_DIR = DATA_DIR / "positive_synthetic"
EXISTING_POSITIVES = [
    ROOT / "scripts" / "training_samples" / "positive",
    ROOT / "scripts" / "training_samples" / "real_positive",
]

# Phrasings to synthesize. Variety helps the model generalize to how you actually say it.
PHRASES = [
    "Sunday", "Sunday.", "Hey Sunday", "Hey, Sunday", "Hey Sunday.",
    "Okay Sunday", "Hi Sunday", "Hello Sunday", "Hey there Sunday",
]


def generate_positives() -> int:
    """Synthesize wake-word clips with the local Piper voice."""
    if not PIPER_EXE.exists() or not VOICE_MODEL.exists():
        print(f"!! Piper not found ({PIPER_EXE} / {VOICE_MODEL}); skipping synthesis.")
        return 0
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for i, phrase in enumerate(PHRASES):
        out = SYNTH_DIR / f"sunday_synth_{i:03d}.wav"
        subprocess.run(
            [str(PIPER_EXE), "--model", str(VOICE_MODEL), "--output_file", str(out)],
            input=phrase.encode("utf-8"), check=True, capture_output=True,
        )
        made += 1
    print(f"** Generated {made} synthetic positives -> {SYNTH_DIR}")
    return made


def collect_existing() -> int:
    total = sum(len(list(d.glob('*.wav'))) for d in EXISTING_POSITIVES if d.exists())
    print(f"** Found {total} existing real positives under scripts/training_samples/")
    return total


def print_training_handoff(stack_present: bool) -> None:
    """Print the authoritative next steps to turn the prepared data into sunday.onnx.

    We deliberately do NOT call an openWakeWord training function here: the supported
    custom-word flow is notebook-driven and needs a corpus of negative/background audio
    you supply, so a one-liner call would be guesswork. This keeps the script honest —
    it prepares data and points you at the real trainer."""
    print("\n" + "=" * 70)
    if stack_present:
        print("Training stack (torch + openwakeword.train) IS installed.")
    else:
        print("Training stack not installed (needs: pip install torch + openwakeword[train]).")
    print("Positive data is ready. To build sunday.onnx, do ONE of:")
    print("  A) Recommended — openWakeWord's official 'automatic_model_training' notebook:")
    print("       https://github.com/dscripka/openWakeWord  (notebooks/)")
    print("     Use target phrase 'Sunday' (also add 'Hey Sunday'). It synthesizes")
    print("     thousands of positives + pulls negative/background audio and trains.")
    print(f"     Download the resulting .onnx and save it as:\n       {OUT_MODEL}")
    print(f"     You can also feed it the clips this script just made in:\n       {SYNTH_DIR}")
    print("  B) If Windows training is painful, switch WAKE_ENGINE to Porcupine —")
    print("     a clean drop-in seam is already stubbed in app/voice/wake.py.")
    print("Until sunday.onnx exists, hands-free '!wake' is off, but '!listen' works fully.")
    print("=" * 70)


def try_train() -> bool:
    """Check whether the training stack is present and print the handoff. Returns True
    only if a finished sunday.onnx is actually present afterward."""
    try:
        import torch  # noqa: F401
        import openwakeword.train  # noqa: F401
        stack_present = True
    except Exception:
        stack_present = False

    print_training_handoff(stack_present)
    return OUT_MODEL.exists()


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the 'Sunday' wake-word model.")
    ap.add_argument("--data-only", action="store_true", help="generate data, skip training")
    args = ap.parse_args()

    print("=== Sunday wake-word data prep ===")
    generate_positives()
    collect_existing()

    if args.data_only:
        print("\n--data-only: stopping after data generation.")
        return 0

    return 0 if try_train() else 1


if __name__ == "__main__":
    sys.exit(main())
