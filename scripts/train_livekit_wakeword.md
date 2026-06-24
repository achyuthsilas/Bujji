# Training the "Sunday" wake word with livekit-wakeword

This produces `app/voice/models/sunday.onnx`, which turns on hands-free mode
(say "Sunday"). The voice pipeline already knows how to USE the model — this is only
about building it.

> The runtime/inference deps are already installed. **Training** needs an extra, heavy
> set of deps (PyTorch etc.) that are NOT required to run Sunday. Install them only when
> you're ready to train.

## What you'll get into
- A one-time **download** of training assets (Piper voices, background noise, room impulse
  responses, ACAV100M speech features) — **several GB**. livekit-wakeword's `setup` handles
  this robustly (this is the step the openWakeWord Colab kept choking on).
- Training itself. On a **GPU** this is quick. On **CPU-only** (your machine) it's slow but
  works — the config in `configs/sunday_wakeword.yaml` is already trimmed (small model,
  5k samples, 20k steps) to keep it reasonable. Expect tens of minutes to a couple hours.
  If that's too slow, run the exact same steps on a free cloud GPU and copy the `.onnx` back.

## Windows gotchas (already solved once — do these before re-training)
1. **espeak-ng binary** — Piper sample-gen needs it on PATH (the `.dll` isn't enough):
   `winget install eSpeak-NG.eSpeak-NG` (installs to `C:\Program Files\eSpeak NG\`).
   If a new shell can't find it: `$env:PATH = "C:\Program Files\eSpeak NG;$env:PATH"`.
2. **onnxscript** — the `export` step needs it: `pip install onnxscript`.
3. **UTF-8 console** — `export` prints emoji and crashes the default cp1252 console.
   Set `$env:PYTHONUTF8 = "1"` before running `export`.

## Steps (run from the project root, with your venv active)

```bash
# 1. Install the training extra (heavy; one time) + export dep
pip install "livekit-wakeword[train]" onnxscript

# 2. Download training assets into ./data  (one time). NOTE: `setup` uses --config
#    (a flag), unlike the other commands which take a positional path.
#    This pulls Piper voices, backgrounds, RIRs, AND ACAV100M speech features (~16 GB).
python -m livekit.wakeword setup --config configs/sunday_wakeword.yaml
#    Low on disk/bandwidth? Add --skip-acav to skip the 16 GB ACAV100M download, BUT then
#    also set `ACAV100M_sample: 0` under batch_n_per_class in the config (untested; gives a
#    rougher model with more false wakes). Recommended: do the full download for quality.
#    python -m livekit.wakeword setup --config configs/sunday_wakeword.yaml --skip-acav

# 3. Generate synthetic "Sunday" clips + adversarial negatives
python -m livekit.wakeword generate configs/sunday_wakeword.yaml

# 4. Augment (noise/reverb) + extract features
python -m livekit.wakeword augment configs/sunday_wakeword.yaml

# 5. Train the classifier
python -m livekit.wakeword train configs/sunday_wakeword.yaml

# 6. Export to ONNX  → ./output/sunday.onnx
python -m livekit.wakeword export configs/sunday_wakeword.yaml

# 7. (optional) Check quality — prints AUT / false-positives-per-hour / recall
python -m livekit.wakeword eval configs/sunday_wakeword.yaml
```

## Install the model
Copy the exported model into place:

```bash
cp output/sunday.onnx app/voice/models/sunday.onnx
```

## Use it
- Restart the sidecar (`uvicorn app.main:app --reload`).
- In the test UI, click **Start** under "Hands-free", then say **"Sunday"**.
- The default `WAKE_ENGINE` is `livekit`, so it'll load `sunday.onnx` automatically.

## Tuning
- Too many false wakes → raise `WAKE_THRESHOLD` in `.env` (the model's optimal threshold is
  also printed by `eval`); add more `custom_negative_phrases` and retrain.
- Misses your voice → lower `WAKE_THRESHOLD`, or bump `n_samples`/`steps`/`model_size: medium`
  in `configs/sunday_wakeword.yaml` and retrain.

> Note: `scripts/train_openwakeword.py` (the earlier path) is superseded by this. You can
> ignore it.
