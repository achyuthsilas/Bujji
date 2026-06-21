# FastAPI sidecar entry point. Tauri talks to this over HTTP.
# POST /chat is the main endpoint — send a message, get Bujji's reply.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent.database import init_db
from app.agent.graph import run_agent
from app.voice.stt import listen_and_transcribe

app = FastAPI(title="Bujji", version="0.1.0")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


class ListenResponse(BaseModel):
    transcript: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    reply = run_agent(req.message)
    return ChatResponse(reply=reply)


@app.post("/listen", response_model=ListenResponse)
def listen():
    transcript = listen_and_transcribe()
    if not transcript:
        raise HTTPException(status_code=400, detail="No speech detected")
    reply = run_agent(transcript)
    return ListenResponse(transcript=transcript, reply=reply)
