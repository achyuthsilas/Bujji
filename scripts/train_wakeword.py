# Trains the "bujji" wake word model.
# Generates synthetic TTS samples, then mixes in any real recordings from
# scripts/training_samples/real_positive/ (recorded via record_wakeword.py).
#
# First run:  python scripts/train_wakeword.py
# After recording real samples: python scripts/train_wakeword.py --force
# --force deletes existing TTS samples and regenerates them (keeps real ones).
# Output: app/voice/models/bujji_wakeword.onnx

import argparse
import random
import subprocess
import wave
from pathlib import Path

import librosa
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

ROOT = Path(__file__).parent.parent
PIPER_EXE = ROOT / "app" / "voice" / "piper.exe"
VOICE_MODEL = ROOT / "app" / "voice" / "models" / "en_US-lessac-medium.onnx"
SAMPLES_DIR = ROOT / "scripts" / "training_samples"
REAL_POS_DIR = SAMPLES_DIR / "real_positive"   # your actual voice recordings
OUTPUT_MODEL = ROOT / "app" / "voice" / "models" / "sunday_wakeword.onnx"

SAMPLE_RATE = 16000
N_MFCC = 20

# Positive: things that sound like "bujji"
WAKE_PHRASES = ["sunday", "hey sunday", "sunday", "hey sunday", "sunday"]

# Negative: common words/phrases Bujji might mishear as the wake word
OTHER_PHRASES = [
    "hello", "okay", "add", "note", "remind", "today", "tomorrow",
    "please", "thank you", "yes", "no", "stop", "start", "open",
    "close", "set", "save", "create", "delete", "show", "list",
    "morning", "evening", "meeting", "boogie", "buddy", "body",
    "busy", "beauty", "buoy", "ready", "maybe", "baby",
]


def _generate_tts(text: str, path: Path, speed: float, noise: float) -> None:
    subprocess.run(
        [str(PIPER_EXE), "--model", str(VOICE_MODEL),
         "--output_file", str(path),
         "--length_scale", str(round(speed, 2)),
         "--noise_scale", str(round(noise, 2))],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )


def _extract_features(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, duration=3.0, mono=True)
    if len(y) < SAMPLE_RATE:
        y = np.pad(y, (0, SAMPLE_RATE - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    combined = np.concatenate([mfcc, delta], axis=0)
    return np.concatenate([combined.mean(axis=1), combined.std(axis=1)])  # 80 features


def _generate_samples(phrases: list, folder: Path, n: int, label: str, force: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    if force:
        for f in folder.glob("*.wav"):
            f.unlink()
        print(f"  {label}: cleared old TTS samples.")

    existing = len(list(folder.glob("*.wav")))
    if existing >= n:
        print(f"  {label}: {existing} samples already exist, skipping generation.")
        return

    print(f"  Generating {n} {label} samples...")
    for i in range(n):
        path = folder / f"{label}_{i:04d}.wav"
        if path.exists():
            continue
        phrase = phrases[i % len(phrases)]
        speed = random.uniform(0.75, 1.35)
        noise = random.uniform(0.3, 0.9)
        _generate_tts(phrase, path, speed, noise)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing TTS samples and regenerate them before training."
    )
    args = parser.parse_args()

    print("=== Sunday wake word trainer ===\n")

    pos_dir = SAMPLES_DIR / "positive"
    neg_dir = SAMPLES_DIR / "negative"

    print("Step 1: Generating TTS training samples...")
    _generate_samples(WAKE_PHRASES, pos_dir, n=200, label="wake", force=args.force)
    _generate_samples(OTHER_PHRASES, neg_dir, n=200, label="other", force=args.force)

    print("\nStep 2: Extracting MFCC features...")
    X, y = [], []

    # TTS-generated positive samples
    for f in sorted(pos_dir.glob("*.wav")):
        X.append(_extract_features(f))
        y.append(1)

    # Real voice samples (your recordings) — weighted 3x to help the model prioritise them
    real_files = sorted(REAL_POS_DIR.glob("*.wav")) if REAL_POS_DIR.exists() else []
    if real_files:
        print(f"  Found {len(real_files)} real voice samples — adding with 3x weight.")
        for f in real_files:
            feat = _extract_features(f)
            for _ in range(3):
                X.append(feat)
                y.append(1)
    else:
        print("  No real voice samples found. (Run record_wakeword.py to add them.)")

    # TTS-generated negative samples
    for f in sorted(neg_dir.glob("*.wav")):
        X.append(_extract_features(f))
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"  Dataset: {sum(y==1)} positive, {sum(y==0)} negative, {X.shape[1]} features each")

    print("\nStep 3: Training classifier...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )),
    ])
    clf.fit(X, y)

    train_acc = (clf.predict(X) == y).mean()
    print(f"  Training accuracy: {train_acc:.1%}")

    print("\nStep 4: Exporting to ONNX...")
    initial_type = [("mfcc_features", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=12)

    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MODEL, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\nDone! Model saved to: {OUTPUT_MODEL}")
    if real_files:
        print(f"Model trained with {len(real_files)} of your real voice samples.")
    print("Restart uvicorn — Sunday will now use the retrained wake word model.")


if __name__ == "__main__":
    main()
