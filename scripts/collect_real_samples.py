# Collect real human speech samples to retrain the wake word model.
# Records 2-second clips of you saying "Sunday" or other words.
# Saves them as positive (wake word) or negative (other words) samples.
#
# Run with: python scripts/collect_real_samples.py
#
# Usage:
#   1. Choose mode: collect positive (your "Sunday" samples) or negative (other words)
#   2. Hit Enter to start recording, say the word, hit Enter to stop
#   3. Repeat 15-20 times per category
#   4. After collecting, run: python scripts/train_wakeword.py

import sounddevice as sd
import numpy as np
from pathlib import Path
import sys

SAMPLE_RATE = 16000
SAMPLES_DIR = Path(__file__).parent / "training_samples"

def record_sample(duration=2.0):
    """Record and return 2 seconds of audio."""
    print("  Recording... (press Enter when done or speak for ~2 seconds)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()

def save_sample(audio, label, index):
    """Save audio as WAV."""
    import wave
    
    if label == "wake":
        folder = SAMPLES_DIR / "positive"
    else:
        folder = SAMPLES_DIR / "negative"
    
    folder.mkdir(parents=True, exist_ok=True)
    
    # Find next available filename
    existing = len(list(folder.glob(f"{label}_real_*.wav")))
    filename = folder / f"{label}_real_{existing:04d}.wav"
    
    with wave.open(str(filename), "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes((audio * 32767).astype(np.int16).tobytes())
    
    print(f"  ✓ Saved: {filename.name}")
    return filename

def main():
    print("=== Sunday Real Speech Sample Collector ===\n")
    
    print("Collect samples to train the wake word model with YOUR voice.\n")
    print("Options:")
    print("  1. Collect 'Sunday' wake word samples (say: 'Sunday' or 'Hey Sunday')")
    print("  2. Collect negative samples (say: 'Hello', 'Okay', 'Note', etc.)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "1":
        label = "wake"
        prompt = "Say 'Sunday' or 'Hey Sunday'"
    elif choice == "2":
        label = "other"
        prompt = "Say something else (e.g., 'Hello', 'Okay', 'Note')"
    else:
        print("Invalid choice.")
        sys.exit(1)
    
    num_samples = int(input(f"How many samples? (default: 10): ") or "10")
    
    print(f"\nWill collect {num_samples} samples.")
    print(f"For each: {prompt}\n")
    
    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}:")
        input("  Press Enter to record...")
        audio = record_sample()
        save_sample(audio, label, i)
        print()
    
    print(f"✓ Collected {num_samples} {label} samples!")
    print("\nNext: python scripts/train_wakeword.py")

if __name__ == "__main__":
    main()
