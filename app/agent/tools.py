# The three actions Sunday can take. Each is a LangChain tool the agent can call.
# add_todo, add_note, set_reminder — nothing else for v1.

from langchain_core.tools import tool
from app.agent.database import insert_todo, insert_note, insert_reminder


@tool
def add_todo(task: str) -> str:
    """Add a task to the user's todo list. Use this when the user wants to remember to do something."""
    row_id = insert_todo(task)
    return f"Added todo #{row_id}: '{task}'"


@tool
def add_note(content: str) -> str:
    """Save a note for the user. Use this when the user wants to write something down or remember information."""
    row_id = insert_note(content)
    return f"Saved note #{row_id}: '{content}'"


@tool
def set_reminder(content: str, remind_at: str | None = None) -> str:
    """Set a reminder for the user. Use this when the user mentions a time or says 'remind me'.
    remind_at should be a date/time string if the user specified one, otherwise leave it empty."""
    row_id = insert_reminder(content, remind_at)
    when = f" for {remind_at}" if remind_at else ""
    return f"Set reminder #{row_id}{when}: '{content}'"


TOOLS = [add_todo, add_note, set_reminder]
