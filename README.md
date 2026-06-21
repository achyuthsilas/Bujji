# Bujji 🎙️ — Local-First Voice AI Assistant

A private, voice-controlled desktop assistant. Say **"Hey Bujji"** and she listens,
understands, acts, and speaks back — all running locally on your machine.

> Personal project by Achyuth. Built to be genuinely useful *and* to demonstrate
> production-style agentic AI engineering (LangGraph orchestration, tool-calling,
> local STT/TTS, on-device wake word, persistent memory).

---

## ✨ v1 Features (what ships first)
- 🗣️ **Wake word** — responds to "Hey Bujji" (on-device, nothing sent to cloud)
- ✍️ **Voice todos / notes / reminders** — "Hey Bujji, add milk to my todo list"
- 🧠 **Memory** — learns your preferences over time
- 🔊 **Speaks back** — local text-to-speech
- 💡 **Glowing tray icon** — reacts while she's speaking

## 🔭 Roadmap (v2+, deliberately deferred)
News / stocks / sports / AI widgets · Apple Watch health tracking · custom cloned
voice · proactive interest updates · personality/emotional tone.

---

## 🏗️ Architecture
```
Tauri (tray icon + widget UI)
        │  HTTP
        ▼
FastAPI sidecar (Python)
   ├─ Porcupine        → wake-word detection (always listening, local)
   ├─ faster-whisper   → speech-to-text (local)
   ├─ LangGraph agent
   │     router → [add_todo | add_note | set_reminder | chitchat]
   │     ├─ memory read (your preferences)
   │     └─ memory write (learn about you)
   ├─ Piper            → text-to-speech (local)
   └─ SQLite + vector store
```
**Privacy:** your voice and personal data never leave the machine. (Only if you
later switch the LLM to Groq does text reasoning go to the cloud — your choice.)

---

## 🚀 Setup
1. Install [Claude Code](https://claude.com/download) and the prerequisites below.
2. `git init` in this folder (commit after every working stage).
3. Open Claude Code here: `claude`
4. Feed it the prompts from `PROMPTS.md`, one stage at a time.

### Prerequisites (Claude Code will help install these)
- Python 3.10+
- [Ollama](https://ollama.com) with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`)
- Rust + Tauri (only needed at Stage 6)
- A Picovoice access key (free) for the wake word (Stage 4)

---

## 🧩 Build order
| Stage | What | Test it by |
|------|------|-----------|
| 1 | FastAPI + LangGraph agent + 3 tools + SQLite | Typing commands in a terminal harness |
| 2 | faster-whisper STT | Speaking into your mic, see the transcript |
| 3 | Piper TTS | Hearing Bujji reply out loud |
| 4 | Porcupine wake word | Saying "Hey Bujji" triggers her |
| 5 | Memory | She recalls a preference you mentioned earlier |
| 6 | Tauri UI | Glowing tray icon + widget window |

---

## 🎯 Interview talking points (why this project sells you)
- **Agentic orchestration:** LangGraph router + tool nodes + parallel memory write —
  same pattern as production multi-agent systems.
- **Tool-augmented LLM:** the model doesn't "know" answers, it *calls tools* — the
  correct way to build reliable agents.
- **Local-first / privacy engineering:** on-device wake word + STT + TTS; explicit,
  minimal cloud surface.
- **Swappable module design:** STT/TTS/LLM behind clean interfaces — shows you think
  about maintainability, not just demos.
- **Latency awareness:** chose a 3B local model + measured trade-offs vs. cloud Groq.

---

## 📂 Project layout (target)
```
bujji/
├── CLAUDE.md          # context for Claude Code (already here)
├── README.md          # this file
├── PROMPTS.md         # copy-paste prompts, stage by stage
├── .env               # API keys (never commit this)
├── app/
│   ├── main.py        # FastAPI sidecar
│   ├── agent/         # LangGraph graph + tools
│   ├── voice/         # stt.py, tts.py, wake.py (swappable)
│   └── memory/        # preference store
└── ui/                # Tauri app (Stage 6)
```
