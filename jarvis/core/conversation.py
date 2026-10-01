"""Conversation state shared by every entry point.

This is what makes an opencode `jarvis_ask` and a spoken "Hey Jarvis" feel like
the same assistant: both read and write the same session, and both draw from
the same provider budget. Sessions live in the data directory so they survive
a daemon restart.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ..storage import Database
from ..storage.repositories import ConversationRepository
from .providers.base import Message

DEFAULT_SYSTEM = (
    "You are JARVIS, a concise voice assistant. You are spoken to aloud, so keep "
    "replies short and natural: no markdown, no bullet lists, no code blocks "
    "unless explicitly asked. When a tool would help, call it rather than "
    "guessing. If you are unsure, say so plainly."
)


class SessionStore:
    """Backwards-compatible facade over transactional SQLite conversations.

    ``path`` may still be the historic ``sessions.json`` location. It is used
    once as an import source; all subsequent reads and writes use the sibling
    ``jarvis.db`` database.
    """

    def __init__(self, path: Path, max_turns: int = 40):
        self._path = path
        self._max_turns = max_turns
        self._repository = ConversationRepository(Database(path))
        if path.suffix.lower() == ".json":
            self._repository.import_legacy_json(path)

    def get_or_create(self, session_id: str | None = None) -> str:
        sid = session_id or f"s_{uuid.uuid4().hex[:12]}"
        return self._repository.get_or_create(sid)

    def history(self, session_id: str, *, system: str | None = DEFAULT_SYSTEM) -> list[Message]:
        msgs = [Message(message.role, message.content) for message in self._repository.history(session_id)]
        if system:
            return [Message("system", system), *msgs]
        return msgs

    def append(self, session_id: str, user: str, assistant: str) -> None:
        self._repository.append_turn(session_id, user, assistant)
        # Trim from the front: the oldest turns stop being useful long before
        # the newest ones do. Both inserts and trimming are individually
        # transactional, so another process cannot corrupt the conversation.
        self._repository.trim_to_turns(session_id, self._max_turns)

    def clear(self, session_id: str) -> None:
        self._repository.clear(session_id)

    def ids(self) -> list[str]:
        return self._repository.ids()
