"""Transactional conversation and message persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..database import Database


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    sequence: int
    created_at: float


class ConversationRepository:
    def __init__(self, database: Database):
        self._db = database

    def get_or_create(self, conversation_id: str, *, now: float | None = None) -> str:
        now = time.time() if now is None else now
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations(id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
        return conversation_id

    def history(self, conversation_id: str) -> list[StoredMessage]:
        with self._db.read() as conn:
            rows = conn.execute(
                "SELECT role, content, sequence, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY sequence",
                (conversation_id,),
            ).fetchall()
        return [StoredMessage(row["role"], row["content"], row["sequence"], row["created_at"]) for row in rows]

    def append_turn(self, conversation_id: str, user: str, assistant: str) -> None:
        now = time.time()
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations(id, created_at, updated_at) VALUES (?, ?, ?)",
                (conversation_id, now, now),
            )
            next_sequence = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()[0]
            )
            conn.executemany(
                "INSERT INTO messages(conversation_id, sequence, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    (conversation_id, next_sequence, "user", user, now),
                    (conversation_id, next_sequence + 1, "assistant", assistant, now),
                ),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

    def trim_to_turns(self, conversation_id: str, max_turns: int) -> None:
        keep_messages = max(0, max_turns) * 2
        with self._db.transaction() as conn:
            if not keep_messages:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                return
            conn.execute(
                "DELETE FROM messages WHERE id IN ("
                "SELECT id FROM messages WHERE conversation_id = ? ORDER BY sequence DESC LIMIT -1 OFFSET ?"
                ")",
                (conversation_id, keep_messages),
            )

    def clear(self, conversation_id: str) -> None:
        with self._db.transaction() as conn:
            conn.execute("DELETE FROM summaries WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conversation_id))

    def ids(self) -> list[str]:
        with self._db.read() as conn:
            rows = conn.execute("SELECT id FROM conversations ORDER BY updated_at").fetchall()
        return [str(row[0]) for row in rows]

    def upsert_summary(self, conversation_id: str, content: str, through_sequence: int) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO summaries(conversation_id, content, through_sequence, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(conversation_id, through_sequence) DO UPDATE SET content = excluded.content, created_at = excluded.created_at",
                (conversation_id, content, through_sequence, time.time()),
            )

    def latest_summary(self, conversation_id: str) -> str | None:
        with self._db.read() as conn:
            row = conn.execute(
                "SELECT content FROM summaries WHERE conversation_id = ? "
                "ORDER BY through_sequence DESC LIMIT 1",
                (conversation_id,),
            ).fetchone()
        return str(row[0]) if row else None

    def import_legacy_json(self, path: Path) -> None:
        """Import the old sessions file once, without overwriting live data."""
        if not path.is_file() or self._already_imported(path):
            return
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        with self._db.transaction() as conn:
            for conversation_id, blob in payload.items():
                if not isinstance(conversation_id, str) or not isinstance(blob, dict):
                    continue
                created = float(blob.get("created", time.time()))
                updated = float(blob.get("updated", created))
                conn.execute(
                    "INSERT OR IGNORE INTO conversations(id, created_at, updated_at) VALUES (?, ?, ?)",
                    (conversation_id, created, updated),
                )
                exists = conn.execute("SELECT 1 FROM messages WHERE conversation_id = ? LIMIT 1", (conversation_id,)).fetchone()
                if exists:
                    continue
                rows = []
                for sequence, message in enumerate(blob.get("messages", []), start=1):
                    if not isinstance(message, dict):
                        continue
                    role, content = message.get("role"), message.get("content")
                    if role in {"system", "user", "assistant", "tool"} and isinstance(content, str):
                        rows.append((conversation_id, sequence, role, content, updated))
                conn.executemany(
                    "INSERT INTO messages(conversation_id, sequence, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
            self._mark_imported(conn, path)

    def _already_imported(self, path: Path) -> bool:
        with self._db.read() as conn:
            return bool(conn.execute("SELECT 1 FROM legacy_imports WHERE source_path = ?", (str(path.resolve()),)).fetchone())

    @staticmethod
    def _mark_imported(conn, path: Path) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO legacy_imports(source_path, imported_at) VALUES (?, ?)",
            (str(path.resolve()), time.time()),
        )
