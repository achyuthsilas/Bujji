# Bujji's agent. Asks the LLM to classify intent as JSON, then calls the right tool.
# Also extracts user preferences in the same pass and stores them in SQLite.
# Avoids LangChain tool-binding (unreliable across model versions) in favour of
# a plain JSON parse → direct function call pattern.

import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from app.agent.database import (
    insert_todo, insert_note, insert_reminder,
    upsert_preference, load_preferences,
)

load_dotenv()

# Base prompt — user preferences are prepended at call time via _build_system_prompt().
_CLASSIFY_BASE = """You are Sunday, a personal assistant. Do two things in one reply:
1. Classify the user's intent.
2. If the message reveals a personal fact or preference (name, location, job, habit, preference), capture it.

Reply with ONLY valid JSON — nothing else:
{"intent": "<intent>", "content": "<extracted text>", "remind_at": "<time or null>", "preference": {"key": "<short label>", "value": "<fact>"} or null}

Intents:
- "add_todo"     → user wants to add a task or to-do item
- "add_note"     → user wants to save a note or remember information
- "set_reminder" → user mentions a time or says "remind me"
- "unknown"      → anything else

Examples:
User: "add milk to my todo list"
{"intent": "add_todo", "content": "milk", "remind_at": null, "preference": null}

User: "remind me to call mom at 5pm"
{"intent": "set_reminder", "content": "call mom", "remind_at": "5pm", "preference": null}

User: "my name is Arjun, remind me to take meds at 8am"
{"intent": "set_reminder", "content": "take meds", "remind_at": "8am", "preference": {"key": "name", "value": "Arjun"}}

User: "I prefer dark mode"
{"intent": "unknown", "content": "", "remind_at": null, "preference": {"key": "ui preference", "value": "dark mode"}}

User: "what is the weather"
{"intent": "unknown", "content": "", "remind_at": null, "preference": null}"""


def _build_system_prompt() -> str:
    prefs = load_preferences()
    if not prefs:
        return _CLASSIFY_BASE
    facts = "\n".join(f"  - {p['key']}: {p['value']}" for p in prefs)
    return f"What you know about the user:\n{facts}\n\n{_CLASSIFY_BASE}"


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
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=message),
    ])
    raw = response.content.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _name_from_prefs() -> str | None:
    prefs = load_preferences()
    for p in prefs:
        if p["key"] in ("name", "user name", "my name"):
            return p["value"]
    return None


def run_agent(message: str) -> str:
    result = _classify(message)
    intent = result.get("intent", "unknown")
    content = result.get("content", "").strip()
    remind_at = result.get("remind_at") or None
    pref = result.get("preference")

    print(f"  [AGENT] intent={intent!r} content={content!r} remind_at={remind_at!r} pref={pref!r}")

    # Save any detected preference before replying
    if isinstance(pref, dict) and pref.get("key") and pref.get("value"):
        upsert_preference(pref["key"], pref["value"])
        print(f"  [MEMORY] saved: {pref['key']!r} = {pref['value']!r}")

    name = _name_from_prefs()
    greeting = f"{name}! " if name else ""

    if intent == "add_todo":
        insert_todo(content)
        return f"Got it, {greeting}I've added '{content}' to your todo list."

    if intent == "add_note":
        insert_note(content)
        return f"Noted, {greeting}I've saved: '{content}'."

    if intent == "set_reminder":
        insert_reminder(content, remind_at)
        when = f" for {remind_at}" if remind_at else ""
        return f"Done, {greeting}I've set a reminder{when}: '{content}'."

    if pref:
        return f"Got it, I'll remember that."

    return "Sorry, I can only add todos, notes, and reminders right now."
