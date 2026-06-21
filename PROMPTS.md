# PROMPTS.md — Copy-paste prompts for Claude Code

How to use this file:
- Do ONE stage per session. Don't rush ahead.
- Paste the prompt, let Claude Code show its plan, read it, approve.
- After it works, `git add -A && git commit -m "stage N done"`, then `/clear` and move on.
- If you're confused at any point, just type: "Explain what you just did in plain English."

---

## 🧰 Stage 0 — First contact (do this once)
```
Read CLAUDE.md fully. Summarize back to me in plain English: what we're building,
the locked scope, and the build order. Then set up an empty Python project:
requirements.txt, a virtual environment, and the folder layout from the README.
Don't write any feature code yet — just scaffolding. Show me the plan first.
```

---

## 🧠 Stage 1 — The brain (text-only)
```
Use Plan Mode. Build Stage 1 only: a FastAPI sidecar with a LangGraph agent that has
exactly three tools — add_todo, add_note, set_reminder — backed by SQLite.

Requirements:
- A router node that classifies my message into one of: add_todo, add_note,
  set_reminder, or chitchat, and extracts the needed fields (e.g. reminder time).
- Each tool writes to SQLite and returns a short confirmation.
- Default LLM = Ollama llama3.2:3b. Add an env switch LLM_BACKEND=ollama|groq.
- A simple text test harness (a CLI loop) so I can type messages and see responses,
  with NO voice yet.
Explain the LangGraph graph to me in plain English when done. Show the plan first.
```
Test: run the harness, type *"add buy milk to my todo list"*, *"remind me about the
dentist tomorrow at 3pm"*, *"take a note: I like sci-fi movies"*. Check the SQLite rows.

---

## 🎤 Stage 2 — Ears (speech to text)
```
Use Plan Mode. Add Stage 2: faster-whisper speech-to-text, base model, CPU only.
Put it in app/voice/stt.py behind a clean interface so it's swappable.
Add a test script that records a few seconds from my mic and prints the transcript.
Wire it so the transcript can feed into the Stage 1 agent. Show the plan first.
```
Test: speak a command, confirm the transcript is right, confirm it triggers a tool.

---

## 🔊 Stage 3 — Voice (text to speech)
```
Use Plan Mode. Add Stage 3: Piper text-to-speech with a preset voice, CPU only.
Put it in app/voice/tts.py behind a clean interface. After the agent replies,
speak the reply out loud. Let me pick from a few Piper voices. Show the plan first.
```
Test: give a voice command, hear Bujji confirm it out loud.

---

## 👂 Stage 4 — Wake word ("Hey Bujji")
```
Use Plan Mode. Add Stage 4: Picovoice Porcupine wake-word detection in app/voice/wake.py.
It should listen continuously and trigger the listen->transcribe->agent->speak pipeline
when it hears the wake word. Tell me exactly how to create a custom "Bujji" keyword and
where to put my free Picovoice access key (.env). Show the plan first.
```
Test: say "Hey Bujji, add eggs to my todo list" with no key press.

---

## 🧠 Stage 5 — Memory (learns about me)
```
Use Plan Mode. Add Stage 5: a preference memory. On each turn, in parallel, extract any
durable facts about me ("Achyuth likes X", "Achyuth's standup is at 9am") and store them
with embeddings in sqlite-vec. Before the agent replies, retrieve relevant memories and
let it use them. Keep it in app/memory/. Show me the schema. Show the plan first.
```
Test: tell her a preference, start a fresh session, ask something that should use it.

---

## 🖥️ Stage 6 — The face (Tauri UI)
```
Use Plan Mode. Add Stage 6: a Tauri desktop app in ui/. A system tray icon that GLOWS /
animates while Bujji is speaking. Clicking it opens a small widget window; clicking again
enlarges it. The Tauri frontend talks to the FastAPI sidecar over HTTP. I'm new to Rust
and Tauri, so explain each step simply and tell me what to install. Show the plan first.
```
Test: launch the app, see the tray icon, watch it glow when she talks.

---

## 🆘 Handy prompts anytime
- "Explain what you just did in plain English, in bullet points."
- "I got this error: <paste>. What does it mean and how do we fix it?"
- "Before we continue, write/update the Commands section in CLAUDE.md."
- "Is this still inside our locked v1 scope? If not, note it as v2 and stop."
