"""Lightweight SQLite conversation store.

Stores ``ConversationRun`` objects keyed by ``conversation_id``.
Rows older than 24 hours are treated as expired (TTL).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

from src.models.schemas import ConversationRun

_DB_PATH = Path("data/conversations.db")
_TTL_HOURS = 24


def _ensure_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_runs (
            conversation_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    _ensure_db(conn)
    return conn


def guardar(run: ConversationRun) -> None:
    """Upsert a conversation run."""
    conn = _conn()
    conn.execute(
        "INSERT OR REPLACE INTO conversation_runs (conversation_id, data, created_at) "
        "VALUES (?, ?, ?)",
        (str(run.conversation_id), run.model_dump_json(), run.created_at.isoformat()),
    )
    conn.commit()
    conn.close()


def cargar(conversation_id: UUID) -> ConversationRun | None:
    """Load a conversation run by ID. Returns ``None`` if not found or expired."""
    conn = _conn()
    row = conn.execute(
        "SELECT data, created_at FROM conversation_runs WHERE conversation_id = ?",
        (str(conversation_id),),
    ).fetchone()
    conn.close()

    if not row:
        return None

    created_at = datetime.fromisoformat(row[1])
    if datetime.utcnow() - created_at > timedelta(hours=_TTL_HOURS):
        eliminar(conversation_id)
        return None

    return ConversationRun.model_validate_json(row[0])


def eliminar(conversation_id: UUID) -> None:
    conn = _conn()
    conn.execute(
        "DELETE FROM conversation_runs WHERE conversation_id = ?", (str(conversation_id),)
    )
    conn.commit()
    conn.close()


def limpiar_expiradas() -> int:
    """Delete rows older than TTL. Returns count deleted."""
    conn = _conn()
    cutoff = (datetime.utcnow() - timedelta(hours=_TTL_HOURS)).isoformat()
    cursor = conn.execute(
        "DELETE FROM conversation_runs WHERE created_at < ?", (cutoff,)
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
