# Streaming agent for the voice pipeline.
#
# Plain English: this talks to Groq's LLM and does two jobs in ONE pass:
#   1. Calls the right tool (add_todo / add_note / set_reminder / remember_preference)
#      so the action actually happens in the database.
#   2. Streams a short spoken reply back, sentence by sentence, so Piper can start
#      talking before the model has finished thinking.
#
# We use the Groq SDK directly (not LangChain) because we need real token streaming
# plus native function-calling, and we need it async so it fits the pipeline's
# producer/consumer tasks.
#
# Latency trick: it's a SINGLE LLM round. If the model speaks while calling the tool,
# we stream those words. If it returns only a tool call (no words), we instantly build
# a short confirmation from the tool's arguments — no second round-trip to Groq.

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

# System prompt — kept deliberately small (well under 200 tokens).
_SYSTEM = (
    "You are Sunday, a friendly local voice assistant. You can do exactly three things: "
    "add todos, add notes, and set reminders. When the user asks for one, call the matching "
    "tool. If the user reveals a personal fact (name, job, location, a preference), also call "
    "remember_preference. Always reply out loud in ONE short, warm sentence. If the request "
    "is not one of your three actions, say so briefly in one sentence."
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


async def _stream_attempt(client, messages, t0) -> AsyncIterator[str]:
    """Streaming path: yield sentences as tokens arrive; run tools at the end."""
    stream = await client.chat.completions.create(
        model=MODEL, messages=messages, tools=_TOOLS, tool_choice="auto",
        max_tokens=MAX_TOKENS, temperature=0.3, stream=True,
    )
    buffer = ""
    spoke = False
    first = False
    tool_calls: dict[int, dict] = {}
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if not first:
                print(f"[TIMING] llm: first_token={(time.perf_counter()-t0)*1000:.0f}ms")
                first = True
            buffer += delta.content
            sentences, buffer = _flush_sentences(buffer)
            for s in sentences:
                spoke = True
                yield s
        if delta.tool_calls:
            for tc in delta.tool_calls:
                slot = tool_calls.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

    tail = buffer.strip()
    if tail:
        spoke = True
        yield tail
    confirmations = _run_tool_calls(tool_calls)
    if not spoke:
        for c in (confirmations or [_NO_ACTION]):
            yield c


async def _nonstream_attempt(client, messages) -> AsyncIterator[str]:
    """Reliable fallback: one non-streaming call, run tools, yield the reply.
    Used if streaming errors (e.g. the llama-3.3 streaming+tools Groq bug)."""
    resp = await client.chat.completions.create(
        model=MODEL, messages=messages, tools=_TOOLS, tool_choice="auto",
        max_tokens=MAX_TOKENS, temperature=0.3,
    )
    msg = resp.choices[0].message
    tool_calls = {i: {"name": tc.function.name, "args": tc.function.arguments or ""}
                  for i, tc in enumerate(msg.tool_calls or [])}
    confirmations = _run_tool_calls(tool_calls)
    content = (msg.content or "").strip()
    if content:
        sentences, tail = _flush_sentences(content + " ")
        for s in sentences:
            yield s
        if tail.strip():
            yield tail.strip()
    else:
        for c in (confirmations or [_NO_ACTION]):
            yield c


async def stream_reply(transcript: str) -> AsyncIterator[str]:
    """Stream the spoken reply for one user turn, sentence by sentence.

    Yields each complete sentence as soon as it's ready (so TTS can start early) and
    executes tool calls against the database. If the streaming request fails before any
    audio is produced, it transparently falls back to a non-streaming call.
    """
    client = _get_client()
    t0 = time.perf_counter()
    messages = _build_messages(transcript)
    yielded = False
    try:
        async for s in _stream_attempt(client, messages, t0):
            yielded = True
            yield s
    except Exception as e:
        print(f"[AGENT] streaming failed ({str(e)[:90]}); retrying non-streaming")
        if yielded:
            return   # already spoke part of a reply; don't double up
        async for s in _nonstream_attempt(client, messages):
            yield s
