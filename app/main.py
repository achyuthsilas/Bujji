# FastAPI sidecar entry point. Tauri (and the test harness) talk to this over HTTP.
#
# Endpoints:
#   POST /chat        → text in, Sunday's reply out (no audio)
#   POST /listen      → one voice turn (no wake word): records you, transcribes,
#                       streams the reply, and speaks it. Returns transcript + reply.
#   POST /wake/start  → hands-free mode: listen for "Sunday", then run a turn, repeat
#   POST /wake/stop   → stop hands-free mode
#   GET  /wake/status → is hands-free running + last stop-talking→first-audio time

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.agent.database import init_db
from app.agent.graph import run_agent
from app.voice import pipeline

app = FastAPI(title="Sunday", version="0.2.0")

UI_FILE = Path(__file__).parent / "static" / "index.html"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class ListenResponse(BaseModel):
    transcript: str
    reply: str
    first_audio_ms: int | None = None
    stt_ms: int | None = None
    llm_ms: int | None = None
    tts_ms: int | None = None
    turn_total_ms: int | None = None
    note: str = ""               # why nothing was heard, if applicable
    max_rms: float | None = None
    max_prob: float | None = None


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def index():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/mic-test")
async def mic_test():
    """Record a fixed 4s window and report mic level, clipping, and the VAD prob timeline."""
    import asyncio
    return await asyncio.to_thread(pipeline.mic_test, 4.0)


@app.get("/audio-devices")
def audio_devices():
    """List input devices so you can confirm the mic the server will record from."""
    import sounddevice as sd
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            devices.append({"index": i, "name": d["name"],
                            "channels": d["max_input_channels"]})
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    return {"default_input": default_in, "inputs": devices}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    reply = run_agent(req.message)
    return ChatResponse(reply=reply)


@app.post("/listen", response_model=ListenResponse)
async def listen():
    # Returns 200 even when nothing was heard, with a 'note' explaining why — so the
    # UI can show mic/VAD diagnostics instead of an opaque error.
    result = await pipeline.run_single_turn()
    diag = result.get("diag") or {}
    return ListenResponse(
        transcript=result.get("transcript", ""),
        reply=result.get("reply", ""),
        first_audio_ms=result.get("first_audio_ms"),
        stt_ms=result.get("stt_ms"),
        llm_ms=result.get("llm_ms"),
        tts_ms=result.get("tts_ms"),
        turn_total_ms=result.get("turn_total_ms"),
        note=result.get("note", ""),
        max_rms=diag.get("max_rms"),
        max_prob=diag.get("max_prob"),
    )


@app.post("/wake/start")
async def wake_start():
    # Must be async: start_continuous() calls asyncio.create_task(), which needs the
    # running event loop. A sync endpoint runs in a threadpool with no loop → crash.
    return pipeline.start_continuous()


@app.post("/wake/stop")
async def wake_stop():
    return pipeline.stop_continuous()


@app.get("/wake/status")
def wake_status():
    return {"running": pipeline.is_running(), **pipeline.last_metrics()}
