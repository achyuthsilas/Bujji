# Streaming agent for the voice pipeline.
#
# Plain English: for each thing you say, this decides between two paths:
#   1. ACTION — you asked to add a todo / note / reminder (or shared a personal fact).
#      A fast model (llama-3.1-8b-instant) calls the matching tool; we run it and speak a
#      short confirmation.
#   2. QUESTION — anything else: "how are you", "what's the weather", "who won the match".
#      We route it to Groq's web-enabled "compound" model, which searches the web when
#      needed and answers. If it can't find an answer, it says so.
#
# We use the Groq SDK directly (not LangChain) for real token streaming + native
# function-calling, async so it fits the pipeline's producer/consumer tasks. Web search
# uses the SAME Groq key — no extra API key required.

import os
import re
import json
import time
from typing import AsyncIterator

from dotenv import load_dotenv
from groq import AsyncGroq

from app.agent.database import (
    insert_todo, insert_note, insert_reminder,
    upsert_preference, load_preferences,
)

load_dotenv()

# llama-3.1-8b-instant streams tool calls cleanly on Groq AND is faster. Note:
# llama-3.3-70b-versatile has a Groq bug where streaming + tool calls produces a
# malformed call ("add_todo{...}" merged name+args) — if you switch back to it, the
# non-streaming fallback below keeps it working.
MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")
MAX_TOKENS = 150          # cap output — replies are one or two short sentences

# Web-enabled model for general questions. Groq's "compound" systems do web search
# server-side using the SAME Groq key (no extra API key needed). compound-mini is fast.
COMPOUND_MODEL = os.getenv("GROQ_ANSWER_MODEL", "groq/compound-mini")

# Round-1 prompt: ONLY decide if this is one of the 3 actions. We route everything else
# (questions, chit-chat, current info) to the web-enabled model. Kept small.
_SYSTEM = (
    "You are Sunday's intent router. If the user wants to add a to-do, save a note, or set a "
    "reminder, call the matching tool. If they state a personal fact about themselves (name, "
    "job, location, a preference), also call remember_preference. For ANYTHING else — questions, "
    "chit-chat, weather, news, sports, prices, general knowledge — do NOT call any tool; just "
    "reply 'OK'."
)

# Round-2 prompt: actually answer a general question, using web search when needed.
_ANSWER_SYS = (
    "You are Sunday, a friendly personal voice assistant. Answer concisely — 1 to 2 short "
    "spoken sentences, suitable for text-to-speech (no markdown, no lists, no URLs). Use web "
    "search for anything current: weather, news, sports scores, prices, today's events. If "
    "after trying you still cannot find a reliable answer, reply exactly: 'Sorry, I don't have "
    "the answer to that.'"
)

# Tool schemas handed to Groq. These mirror app/agent/database.py inserts.
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a task to the user's todo list.",
            "parameters": {
                "type": "object",
                "properties": {"task": {"type": "string", "description": "the task text"}},
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a note for the user.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "the note text"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder, optionally for a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "what to be reminded of"},
                    "remind_at": {"type": "string", "description": "time/date, or empty"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": "Store a personal fact about the user (name, job, habit, preference).",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "short label, e.g. 'name'"},
                    "value": {"type": "string", "description": "the fact"},
                },
                "required": ["key", "value"],
            },
        },
    },
]

# Sentence boundary: end of string after . ! ? (optionally followed by quotes/space).
_SENTENCE_END = re.compile(r'([.!?]+["\')\]]*)(\s+|$)')

_client: AsyncGroq | None = None


def _get_client() -> AsyncGroq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env — needed for the streaming agent")
        _client = AsyncGroq(api_key=api_key)
    return _client


def _build_messages(transcript: str) -> list[dict]:
    """System prompt + a tiny 'things you already know' line so Sunday can be personal."""
    prefs = load_preferences()
    system = _SYSTEM
    if prefs:
        facts = "; ".join(f"{p['key']}={p['value']}" for p in prefs[:5])
        system = f"{_SYSTEM}\nYou already know: {facts}."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": transcript},
    ]


def _execute_tool(name: str, args: dict) -> str:
    """Run one tool against the database. Returns a short spoken confirmation."""
    if name == "add_todo":
        task = (args.get("task") or "").strip()
        insert_todo(task)
        return f"Added '{task}' to your todos."
    if name == "add_note":
        content = (args.get("content") or "").strip()
        insert_note(content)
        return f"Noted: '{content}'."
    if name == "set_reminder":
        content = (args.get("content") or "").strip()
        remind_at = (args.get("remind_at") or "").strip() or None
        insert_reminder(content, remind_at)
        when = f" for {remind_at}" if remind_at else ""
        return f"Reminder set{when}: '{content}'."
    if name == "remember_preference":
        key = (args.get("key") or "").strip()
        value = (args.get("value") or "").strip()
        if key and value:
            upsert_preference(key, value)
        return "Got it, I'll remember that."
    return ""


def _flush_sentences(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of the running buffer; keep the leftover tail."""
    sentences, last = [], 0
    for m in _SENTENCE_END.finditer(buffer):
        sentences.append(buffer[last:m.end()].strip())
        last = m.end()
    return [s for s in sentences if s], buffer[last:]


def _run_tool_calls(tool_calls: dict[int, dict]) -> list[str]:
    """Execute accumulated tool calls (DB writes) and return spoken confirmations."""
    confirmations: list[str] = []
    for slot in tool_calls.values():
        if not slot["name"]:
            continue
        try:
            args = json.loads(slot["args"]) if slot["args"] else {}
        except json.JSONDecodeError:
            args = {}
        confirmation = _execute_tool(slot["name"], args)
        print(f"  [AGENT] tool={slot['name']!r} args={args!r}")
        if confirmation:
            confirmations.append(confirmation)
    return confirmations


_NO_ACTION = "Sorry, I can only add todos, notes, and reminders right now."


async def _classify_action(client, transcript: str) -> dict[int, dict]:
    """Round 1 (fast, non-streaming): does the user want one of the 3 actions?
    Returns the tool calls to run (empty dict = it's a general question)."""
    try:
        resp = await client.chat.completions.create(
            model=MODEL, messages=_build_messages(transcript),
            tools=_TOOLS, tool_choice="auto", max_tokens=MAX_TOKENS, temperature=0.1,
        )
        msg = resp.choices[0].message
        return {i: {"name": tc.function.name, "args": tc.function.arguments or ""}
                for i, tc in enumerate(msg.tool_calls or [])}
    except Exception as e:
        print(f"[AGENT] action-classify failed: {str(e)[:90]}")
        return {}


async def _answer_question(client, transcript: str) -> AsyncIterator[str]:
    """Round 2: answer a general question with the web-enabled model, streamed by sentence.
    Falls back to 'I don't have the answer' if nothing comes back."""
    prefs = load_preferences()
    system = _ANSWER_SYS
    if prefs:
        facts = "; ".join(f"{p['key']}={p['value']}" for p in prefs[:5])
        system = f"{_ANSWER_SYS}\nAbout the user: {facts}."

    t0 = time.perf_counter()
    first = True
    buffer = ""
    spoke = False
    try:
        stream = await client.chat.completions.create(
            model=COMPOUND_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": transcript}],
            max_tokens=220, temperature=0.3, stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if first:
                    print(f"[TIMING] answer first_token={(time.perf_counter()-t0)*1000:.0f}ms")
                    first = False
                buffer += delta.content
                sentences, buffer = _flush_sentences(buffer)
                for s in sentences:
                    spoke = True
                    yield s
        tail = buffer.strip()
        if tail:
            spoke = True
            yield tail
        if not spoke:
            yield "Sorry, I don't have the answer to that."
    except Exception as e:
        msg = str(e)
        print(f"[AGENT] answer failed: {msg[:120]}")
        if "rate_limit" in msg or "429" in msg:
            yield "I'm getting a lot of requests right now — please ask again in a moment."
        else:
            yield "Sorry, I'm having trouble reaching the web right now."


async def stream_reply(transcript: str) -> AsyncIterator[str]:
    """Produce Sunday's spoken reply for one turn, yielded sentence by sentence.

    Two paths:
      • An action (add todo/note/reminder, remember a fact) -> do it, speak a confirmation.
      • Anything else (a question, chit-chat, weather, news, sports...) -> answer it using
        the web-enabled model, with web search, falling back to "I don't have the answer".
    """
    client = _get_client()
    t0 = time.perf_counter()

    tool_calls = await _classify_action(client, transcript)
    if tool_calls:
        print(f"[TIMING] action classified in {(time.perf_counter()-t0)*1000:.0f}ms")
        confirmations = _run_tool_calls(tool_calls)
        for c in (confirmations or [_NO_ACTION]):
            yield c
        return

    async for s in _answer_question(client, transcript):
        yield s
