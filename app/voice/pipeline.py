# The concurrent voice pipeline — the heart of Sunday's "stop talking -> reply fast" loop.
#
# Plain English, the old way was a straight line: record everything, then transcribe,
# then think, then synthesize the whole reply, then play. Each step waited for the one
# before it to fully finish.
#
# This version overlaps the slow parts. As soon as you stop talking we transcribe; as the
# LLM streams its reply we cut it into sentences; each finished sentence is synthesized and
# starts PLAYING while the LLM is still writing the next sentence. Those three stages
# (LLM -> TTS -> playback) run as separate asyncio tasks joined by queues (a classic
# producer/consumer setup), so audio starts almost as soon as the model speaks its first
# sentence.
#
# The number we care about is "stop-talking -> first-audio". We anchor a stopwatch the
# instant the VAD says the turn ended (t_stop) and log every stage relative to it, tagged
# [TIMING]. first_audio is the headline.
#
# Capture is the one part that CAN'T overlap (you can't think before you've heard the
# words), so it runs in a worker thread; everything downstream is async.

import asyncio
import threading
import time

import numpy as np
import sounddevice as sd

from app.voice import stt
from app.voice import tts
from app.voice.vad import SileroVAD, FRAME_SAMPLES
from app.agent.stream_agent import stream_reply

MIC_RATE = stt.SAMPLE_RATE          # 16 kHz capture (Whisper + VAD + wake all want this)
PREROLL_FRAMES = 8                  # ~256 ms kept before speech triggers, so we don't clip
MAX_TURN_SECS = 15                  # hard cap on one utterance

# Cross-thread stop flag for the continuous loop / in-flight capture.
_stop_event = threading.Event()
_continuous_task: asyncio.Task | None = None
_last_metrics: dict = {}
_last_wake_ts: float | None = None   # perf_counter when the wake word last fired


# ── earcon ───────────────────────────────────────────────────────────────────
def _beep() -> None:
    t = np.linspace(0, 0.18, int(MIC_RATE * 0.18), dtype=np.float32)
    sd.play(np.sin(2 * np.pi * 880 * t) * 0.3, samplerate=MIC_RATE)
    sd.wait()


# ── capture (runs in a worker thread) ─────────────────────────────────────────
def _capture_blocking(do_wake: bool) -> tuple[np.ndarray | None, float, dict]:
    """Capture one utterance. In hands-free mode: wait for the wake word, beep, then record.

    IMPORTANT: the wake-listen and the command-recording use SEPARATE mic stream sessions,
    with the beep played in between while NO input stream is open. On Windows, playing the
    beep through sd.play() while an InputStream is open knocks that stream into delivering
    pure silence — which looked like a dead mic. Keeping them separate avoids that.
    """
    if do_wake:
        if not _wait_for_wake():            # opens + closes its own stream
            return None, time.perf_counter(), {"max_rms": 0.0, "max_prob": 0.0, "note": "stopped"}
        _beep()                             # no input stream open here — safe
    return _capture_speech()                # fresh stream for the actual command


def _wait_for_wake() -> bool:
    """Listen until the wake word fires (or we're stopped). Returns True on detection."""
    from app.voice.wake import get_engine   # lazy: only needed for hands-free mode
    engine = get_engine()
    print("[WAKE] Listening for 'Sunday'...")
    deadline = time.monotonic() + 3600
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES) as stream:
        while not _stop_event.is_set() and time.monotonic() < deadline:
            block, _ = stream.read(FRAME_SAMPLES)
            samples = block.reshape(-1).astype(np.float32)
            if engine.detect(samples):
                global _last_wake_ts
                _last_wake_ts = time.perf_counter()
                print("[TIMING] [WAKE] Wake word detected!")
                return True
    return False


def _capture_speech() -> tuple[np.ndarray | None, float, dict]:
    """Record one utterance using Silero VAD, ending on ~400ms of silence.
    Returns (audio, t_stop, diag); audio is None if nothing usable was heard."""
    vad = SileroVAD(silence_ms=400)
    captured: list[np.ndarray] = []
    started = False
    start_idx = 0
    leftover = np.zeros(0, dtype=np.float32)
    t_stop = 0.0
    speech_deadline = time.monotonic() + MAX_TURN_SECS

    # Diagnostics: a dead mic (low rms) vs a too-strict VAD (good rms, low prob).
    max_rms = 0.0
    max_prob = 0.0
    next_heartbeat = time.monotonic() + 1.0

    print(f"[MIC] capture open (device={sd.default.device}, rate={MIC_RATE})")
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES) as stream:
        while not _stop_event.is_set() and time.monotonic() < speech_deadline:
            block, _ = stream.read(FRAME_SAMPLES)
            samples = block.reshape(-1).astype(np.float32)

            leftover = np.concatenate([leftover, samples])
            while len(leftover) >= FRAME_SAMPLES:
                frame, leftover = leftover[:FRAME_SAMPLES], leftover[FRAME_SAMPLES:]
                captured.append(frame)
                status = vad.feed(frame)

                max_rms = max(max_rms, float(np.sqrt(np.mean(frame ** 2))))
                max_prob = max(max_prob, vad.last_prob)

                if not started and vad.triggered:
                    started = True
                    start_idx = max(0, len(captured) - 1 - PREROLL_FRAMES)
                    print("[VAD] speech start")

                if status == "end":
                    t_stop = time.perf_counter()
                    print(f"[TIMING] t_stop = 0 (end of speech)")
                    utterance = np.concatenate(captured[start_idx:])
                    return utterance, t_stop, {"max_rms": max_rms, "max_prob": max_prob, "note": ""}

            if not started and time.monotonic() >= next_heartbeat:
                print(f"[MIC] listening... max_rms={max_rms:.4f} max_speech_prob={max_prob:.2f}")
                next_heartbeat = time.monotonic() + 1.0

    note = _report_no_speech(started, max_rms, max_prob, vad.threshold)
    diag = {"max_rms": max_rms, "max_prob": max_prob, "note": note}
    audio = np.concatenate(captured[start_idx:]) if started else None
    return audio, (t_stop or time.perf_counter()), diag


def _report_no_speech(started: bool, max_rms: float, max_prob: float, threshold: float) -> str:
    """Explain a capture that ended without a clean utterance. Prints + returns the note."""
    if started:
        note = "Max turn length hit (had speech, used what we got)."
    elif max_rms < 0.005:
        note = (f"No audio signal (max_rms={max_rms:.4f}). Mic muted or wrong input device? "
                f"Pick the right mic in Windows sound settings.")
    else:
        note = (f"Heard audio (max_rms={max_rms:.4f}) but it never crossed the speech "
                f"threshold (max_prob={max_prob:.2f} < {threshold}). Speak sooner/louder, "
                f"or lower VAD_THRESHOLD in .env.")
    print(f"[VAD] {note}")
    return note


# ── turn handling: STT -> (LLM | TTS | playback overlapped) ───────────────────
# Each stage writes its elapsed-since-t_stop into the shared `timing` dict, which is
# published to _last_metrics so the UI can show the transcript, reply, and per-step times.
async def _llm_stage(transcript: str, tts_q: asyncio.Queue, t_stop: float,
                     reply_parts: list[str], timing: dict) -> None:
    first = True
    async for sentence in stream_reply(transcript):
        if first:
            timing["llm_ms"] = round((time.perf_counter() - t_stop) * 1000)
            print(f"[TIMING] llm_first_sentence = {timing['llm_ms']}ms")
            first = False
        reply_parts.append(sentence)
        await tts_q.put(sentence)
    await tts_q.put(None)   # sentinel: no more sentences


async def _tts_stage(tts_q: asyncio.Queue, play_q: asyncio.Queue, t_stop: float,
                     timing: dict) -> None:
    first = True
    while True:
        sentence = await tts_q.get()
        if sentence is None:
            await play_q.put(None)
            return
        pcm = await asyncio.to_thread(tts.synth_sentence, sentence)
        if first:
            timing["tts_ms"] = round((time.perf_counter() - t_stop) * 1000)
            print(f"[TIMING] tts_first_synth = {timing['tts_ms']}ms")
            first = False
        if pcm.size:
            await play_q.put(pcm)


async def _play_stage(play_q: asyncio.Queue, t_stop: float, timing: dict) -> None:
    first = True
    while True:
        pcm = await play_q.get()
        if pcm is None:
            return
        if first:
            timing["first_audio_ms"] = round((time.perf_counter() - t_stop) * 1000)
            print(f"[TIMING] *** first_audio = {timing['first_audio_ms']}ms "
                  f"(stop-talking -> first audio) ***")
            first = False
        # Sequential playback keeps sentences in order; we await each clip fully.
        await asyncio.to_thread(_play_blocking, pcm)


def _play_blocking(pcm: np.ndarray) -> None:
    sd.play(pcm, samplerate=tts.SAMPLE_RATE)
    sd.wait()


def _publish(metrics: dict) -> None:
    """Replace the published per-turn metrics the UI polls via /wake/status."""
    global _last_metrics
    metrics["turn"] = _last_metrics.get("turn", 0) + 1
    metrics["ts"] = time.time()
    _last_metrics = metrics


async def _handle_turn(utterance: np.ndarray, t_stop: float) -> dict:
    """Run one full turn from buffered audio to spoken reply. Returns transcript + reply
    and publishes per-step timings + transcript for the UI."""
    timing: dict = {}
    transcript = await asyncio.to_thread(stt.transcribe, utterance)
    timing["stt_ms"] = round((time.perf_counter() - t_stop) * 1000)
    print(f"[TIMING] stt_latency = {timing['stt_ms']}ms")

    if not transcript.strip():
        await asyncio.to_thread(_play_blocking, tts.synth_sentence("Sorry, I didn't catch that."))
        result = {"transcript": "", "reply": "Sorry, I didn't catch that.", **timing}
        _publish(result)
        return result

    tts_q: asyncio.Queue = asyncio.Queue()
    play_q: asyncio.Queue = asyncio.Queue()
    reply_parts: list[str] = []

    await asyncio.gather(
        _llm_stage(transcript, tts_q, t_stop, reply_parts, timing),
        _tts_stage(tts_q, play_q, t_stop, timing),
        _play_stage(play_q, t_stop, timing),
    )

    reply = " ".join(reply_parts).strip()
    timing["turn_total_ms"] = round((time.perf_counter() - t_stop) * 1000)
    print(f"[TIMING] turn_total = {timing['turn_total_ms']}ms")
    result = {"transcript": transcript, "reply": reply, **timing}
    _publish(result)
    return result


# ── public entry points ───────────────────────────────────────────────────────
async def run_single_turn() -> dict:
    """One turn, no wake word (used by /listen, the test harness, and the web UI)."""
    _stop_event.clear()
    utterance, t_stop, diag = await asyncio.to_thread(_capture_blocking, False)
    if utterance is None or utterance.size == 0:
        return {"transcript": "", "reply": "", "first_audio_ms": None,
                "note": diag.get("note", "No speech detected."), "diag": diag}
    result = await _handle_turn(utterance, t_stop)
    result["diag"] = diag
    return result


async def _continuous_loop() -> None:
    print("[PIPELINE] Continuous mode on. Say 'Sunday' to wake me.")
    while not _stop_event.is_set():
        utterance, t_stop, _diag = await asyncio.to_thread(_capture_blocking, True)
        if _stop_event.is_set():
            break
        if utterance is None or utterance.size == 0:
            continue
        try:
            result = await _handle_turn(utterance, t_stop)
            # Full hands-free latency: from the wake word firing to the answer's first audio.
            fa = result.get("first_audio_ms")
            if _last_wake_ts is not None and fa is not None:
                wake_to_answer = round((t_stop - _last_wake_ts) * 1000 + fa)
                _last_metrics["wake_to_answer_ms"] = wake_to_answer
                print(f"[TIMING] *** wake -> answer (first audio) = {wake_to_answer}ms "
                      f"(includes you speaking the command) ***")
        except Exception as e:
            print(f"[PIPELINE] turn error: {e}")
    print("[PIPELINE] Continuous mode stopped.")


def start_continuous() -> dict:
    global _continuous_task
    if _continuous_task and not _continuous_task.done():
        return {"status": "already running"}
    # Load the wake engine eagerly so we fail fast with a helpful message instead of
    # starting a background loop that dies on the first frame (e.g. missing sunday.onnx).
    try:
        from app.voice.wake import get_engine
        get_engine()
    except FileNotFoundError as e:
        return {"status": "error", "detail": str(e)}
    _stop_event.clear()
    _continuous_task = asyncio.create_task(_continuous_loop())
    return {"status": "started"}


def stop_continuous() -> dict:
    _stop_event.set()
    return {"status": "stopping"}


def is_running() -> bool:
    return _continuous_task is not None and not _continuous_task.done()


def last_metrics() -> dict:
    return dict(_last_metrics)


def mic_test(secs: float = 4.0) -> dict:
    """Diagnostic: record a fixed window, then report what the mic and VAD actually saw.
    Speak during the recording. Reveals clipping (peak≈1.0), DC offset, and whether Silero
    ever detects speech (prob timeline). Also writes debug_mic.wav for listening back."""
    import wave
    from pathlib import Path

    n = int(MIC_RATE * secs)
    print(f"[MICTEST] recording {secs}s — speak now...")
    buf = []
    with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="float32",
                        blocksize=FRAME_SAMPLES) as stream:
        got = 0
        while got < n:
            block, _ = stream.read(FRAME_SAMPLES)
            buf.append(block.reshape(-1))
            got += FRAME_SAMPLES
    audio = np.concatenate(buf)[:n].astype(np.float32)

    mean = float(audio.mean())
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))
    clipping = peak >= 0.98

    # Run Silero frame-by-frame; collect the probability timeline.
    vad = SileroVAD()
    probs = []
    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        vad.feed(audio[i:i + FRAME_SAMPLES])
        probs.append(round(vad.last_prob, 2))
    max_prob = max(probs) if probs else 0.0
    over = sum(1 for p in probs if p >= vad.threshold)

    # Save a wav so the audio can be inspected by ear.
    wav_path = Path(__file__).resolve().parent.parent.parent / "debug_mic.wav"
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(MIC_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    # Coarse timeline (≈20 buckets) so the UI can show where speech was detected.
    bucket = max(1, len(probs) // 20)
    timeline = [max(probs[i:i + bucket]) for i in range(0, len(probs), bucket)]

    verdict = (
        "CLIPPING — mic gain too high; speech is distorted so VAD can't see it. "
        "Lower the mic input level in Windows sound settings."
        if clipping else
        "VAD detected speech — pipeline should work."
        if over > 0 else
        "No speech detected by VAD despite audio. Check that you spoke during the window."
    )
    print(f"[MICTEST] peak={peak:.3f} rms={rms:.3f} dc={mean:+.3f} max_prob={max_prob:.2f} "
          f"frames_over_thresh={over} -> {verdict}")
    return {
        "secs": secs, "peak": round(peak, 3), "rms": round(rms, 4), "dc": round(mean, 4),
        "clipping": clipping, "max_prob": round(max_prob, 2),
        "frames_over_threshold": over, "threshold": vad.threshold,
        "timeline": [round(t, 2) for t in timeline],
        "wav": str(wav_path), "verdict": verdict,
    }
