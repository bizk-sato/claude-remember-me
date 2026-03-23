"""SQLite + sqlite-vec database management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec
from sqlite_vec import serialize_float32

from claude_remember_me.models import Memory

DEFAULT_DB_PATH = Path.home() / ".claude" / "claude-remember-me" / "data" / "memory.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    project_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, chunk_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    user_message, assistant_message,
    content=memories, content_rowid=id,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, user_message, assistant_message)
    VALUES (new.id, new.user_message, new.assistant_message);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, user_message, assistant_message)
    VALUES ('delete', old.id, old.user_message, old.assistant_message);
END;

CREATE TABLE IF NOT EXISTS ingest_state (
    session_id TEXT PRIMARY KEY,
    last_chunk_index INTEGER NOT NULL
);
"""

VEC_TABLE_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768] distance_metric=cosine
);
"""


def _row_to_memory(row: tuple) -> Memory:
    """Convert a DB row (id, session_id, chunk_index, user_message, assistant_message, project_path, created_at) to a Memory object."""
    return Memory(
        id=row[0],
        session_id=row[1],
        chunk_index=row[2],
        user_message=row[3],
        assistant_message=row[4],
        project_path=row[5],
        created_at=row[6],
    )


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.execute(VEC_TABLE_SQL)
    conn.commit()


def insert_memory(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    chunk_index: int,
    user_message: str,
    assistant_message: str,
    project_path: str | None,
    embedding: list[float],
) -> None:
    cursor = conn.execute(
        """INSERT OR IGNORE INTO memories
           (session_id, chunk_index, user_message, assistant_message, project_path)
           VALUES (?, ?, ?, ?, ?)""",
        (session_id, chunk_index, user_message, assistant_message, project_path),
    )
    if cursor.rowcount > 0:
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO memories_vec(id, embedding) VALUES (?, ?)",
            (row_id, serialize_float32(embedding)),
        )


def get_last_chunk_index(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute(
        "SELECT last_chunk_index FROM ingest_state WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    return row[0] if row else -1


def update_ingest_state(conn: sqlite3.Connection, session_id: str, last_chunk_index: int) -> None:
    conn.execute(
        """INSERT INTO ingest_state (session_id, last_chunk_index)
           VALUES (?, ?)
           ON CONFLICT(session_id) DO UPDATE SET last_chunk_index = excluded.last_chunk_index""",
        (session_id, last_chunk_index),
    )
