# CLAUDE.md — Project Context for Bujji

> Claude Code reads this file automatically at the start of every session.
> It tells you what we're building, the rules, and how to work with me.

## What Bujji is
Bujji is a **local-first, voice-controlled personal AI assistant** for the desktop
(like a private Siri/Jarvis named "Bujji"). v1 scope is intentionally small and shippable.

## Who I am (the developer)
- New to Claude Code. Explain things in **plain English, short sentences, point form**.
- Define any jargon the first time you use it.
- I'm an AI Engineer, so I understand AI concepts — but I'm new to *this tool* and to Rust/Tauri.
- Always show me the plan BEFORE writing files. Let me approve.

## v1 Scope (LOCKED — do not add features beyond this)
A voice assistant that can:
1. Wake on "Hey Bujji"
2. Transcribe my speech
3. Do exactly 3 actions via a LangGraph agent: **add todo**, **add note**, **set reminder**
4. Remember preferences about me (memory)
5. Speak replies back
6. Show a glowing desktop tray icon that reacts while speaking

Everything else (news, stocks, sports, health, Apple Watch, feelings, custom voice)
is OUT OF SCOPE for v1. Note them as "v2" if relevant, but do not build them.

## Tech stack (LOCKED)
- **Wake word:** Picovoice Porcupine (on-device, custom "Bujji" keyword)
- **STT:** faster-whisper, `base` model, CPU
- **Agent:** LangGraph. Default LLM = Ollama `llama3.2:3b` (local, fast on CPU).
  Provide a single env-var switch `LLM_BACKEND=ollama|groq` so I can flip to Groq for speed.
- **TTS:** Piper (preset voice for v1; XTTS voice-clone is v2)
- **Storage:** SQLite for todos/notes/reminders. sqlite-vec (or Chroma) for preference memory.
- **Desktop UI:** Tauri (Rust backend + HTML/CSS/JS frontend)
- **Glue:** Python FastAPI sidecar. Tauri talks to it over HTTP.
- **OS:** Windows, AMD Radeon GPU. Assume **CPU-only** for all AI — do NOT rely on CUDA.

## Build order (one stage per session — do NOT skip ahead)
1. FastAPI + LangGraph agent + 3 tools (todo/note/reminder) + SQLite. **Text-only test harness.**
2. Add faster-whisper STT (mic -> text)
3. Add Piper TTS (text -> speech)
4. Add Porcupine wake word ("Hey Bujji")
5. Add memory (preference extraction + storage)
6. Wrap in Tauri: glowing tray icon + widget window
Each stage must run and be testable on its own before moving on.

## Working rules
- Use **Plan Mode** for anything touching multiple files. Show plan, wait for my OK.
- Keep modules **swappable**: STT, TTS, wake-word, LLM each behind a clean interface,
  so I can replace one without touching the others.
- After each working stage, remind me to `git commit`.
- Write a short comment at the top of each file explaining what it does, in plain English.
- Prefer small, readable functions over clever one-liners.
- If something needs an API key (Groq, Tavily later), put it in a `.env` file and
  tell me exactly what to paste where. Never hardcode keys.

## Commands (fill in as we build)
- Install deps: `pip install -r requirements.txt`
- Run sidecar: `uvicorn app.main:app --reload`
- Run tests: (TBD)
