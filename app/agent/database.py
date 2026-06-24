# Sets up the SQLite database and creates the four tables Sunday needs.
# Call init_db() once at startup to make sure the tables exist.

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "bujji.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS todos (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                task      TEXT    NOT NULL,
                done      INTEGER NOT NULL DEFAULT 0,
                created   TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS notes (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT    NOT NULL,
                created   TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                content   TEXT    NOT NULL,
                remind_at TEXT,
                created   TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS preferences (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
        """)


def insert_todo(task: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO todos (task) VALUES (?)", (task,))
        return cursor.lastrowid


def insert_note(content: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute("INSERT INTO notes (content) VALUES (?)", (content,))
        return cursor.lastrowid


def insert_reminder(content: str, remind_at: str | None = None) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reminders (content, remind_at) VALUES (?, ?)",
            (content, remind_at),
        )
        return cursor.lastrowid


def upsert_preference(key: str, value: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO preferences (key, value, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                              updated_at=excluded.updated_at""",
            (key.strip().lower(), value.strip()),
        )


def load_preferences() -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT key, value FROM preferences ORDER BY updated_at DESC LIMIT 30"
        ).fetchall()
    return [{"key": row["key"], "value": row["value"]} for row in rows]
