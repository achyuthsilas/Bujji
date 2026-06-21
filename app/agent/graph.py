# Bujji's agent. Asks the LLM to classify intent as JSON, then calls the right tool.
# Avoids LangChain tool-binding (unreliable across model versions) in favour of
# a plain JSON parse → direct function call pattern.

import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from app.agent.database import insert_todo, insert_note, insert_reminder

load_dotenv()

# The LLM is only asked to classify and extract — no tool-binding involved.
CLASSIFY_PROMPT = """You are Bujji, a personal assistant. Classify the user's message into one of these intents and extract the relevant text.

Reply with ONLY valid JSON in this exact format — nothing else:
{"intent": "<intent>", "content": "<extracted text>", "remind_at": "<time or null>"}

Intents:
- "add_todo"    → user wants to add a task or to-do item
- "add_note"    → user wants to save a note or remember information
- "set_reminder"→ user mentions a time or says "remind me"
- "unknown"     → anything else

Examples:
User: "add milk to my todo list"
{"intent": "add_todo", "content": "milk", "remind_at": null}

User: "remind me to call mom at 5pm"
{"intent": "set_reminder", "content": "call mom", "remind_at": "5pm"}

User: "save a note that I prefer dark mode"
{"intent": "add_note", "content": "I prefer dark mode", "remind_at": null}

User: "what is the weather"
{"intent": "unknown", "content": "", "remind_at": null}"""


def _build_llm():
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    if backend == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("LLM_BACKEND=groq but GROQ_API_KEY is not set in .env")
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    return ChatOllama(model=model, base_url=base_url)


_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def _classify(message: str) -> dict:
    response = _get_llm().invoke([
        SystemMessage(content=CLASSIFY_PROMPT),
        HumanMessage(content=message),
    ])
    raw = response.content.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_agent(message: str) -> str:
    result = _classify(message)
    intent = result.get("intent", "unknown")
    content = result.get("content", "").strip()
    remind_at = result.get("remind_at") or None

    print(f"  [AGENT] intent={intent!r} content={content!r} remind_at={remind_at!r}")

    if intent == "add_todo":
        row_id = insert_todo(content)
        return f"Got it! I've added '{content}' to your todo list."

    if intent == "add_note":
        row_id = insert_note(content)
        return f"Noted! I've saved: '{content}'."

    if intent == "set_reminder":
        row_id = insert_reminder(content, remind_at)
        when = f" for {remind_at}" if remind_at else ""
        return f"Done! I've set a reminder{when}: '{content}'."

    return "Sorry, I can only add todos, notes, and reminders right now."
