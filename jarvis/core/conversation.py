"""Conversation state shared by every entry point.

This is what makes an opencode `jarvis_ask` and a spoken "Hey Jarvis" feel like
the same assistant: both read and write the same session, and both draw from
the same provider budget. Sessions live in the data directory so they survive
a daemon restart.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from .providers.base import Message

DEFAULT_SYSTEM = (
    "You are JARVIS, a concise voice assistant. You are spoken to aloud, so keep "
    "replies short and natural: no markdown, no bullet lists, no code blocks "
    "unless explicitly asked. When a tool would help, call it rather than "
    "guessing. If you are unsure, say so plainly."
)


class SessionStore:
    def __init__(self, path: Path, max_turns: int = 40):
        self._path = path
        self._max_turns = max_turns
        self._lock = threading.RLock()
        self._sessions: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._sessions = json.loads(self._path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            self._sessions = {}

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._sessions, indent=1), "utf-8")
        except OSError:
            pass

    def get_or_create(self, session_id: str | None = None) -> str:
        with self._lock:
            sid = session_id or f"s_{uuid.uuid4().hex[:12]}"
            if sid not in self._sessions:
                self._sessions[sid] = {
                    "id": sid,
                    "created": time.time(),
                    "updated": time.time(),
                    "messages": [],
                }
                self._persist()
            return sid

    def history(self, session_id: str, *, system: str | None = DEFAULT_SYSTEM) -> list[Message]:
        with self._lock:
            blob = self._sessions.get(session_id) or {}
            msgs = [Message(**m) for m in blob.get("messages", [])]
        if system:
            return [Message("system", system), *msgs]
        return msgs

    def append(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            blob = self._sessions.setdefault(
                session_id,
                {"id": session_id, "created": time.time(), "updated": time.time(), "messages": []},
            )
            blob["messages"].append({"role": "user", "content": user})
            blob["messages"].append({"role": "assistant", "content": assistant})
            # Trim from the front: the oldest turns stop being useful long
            # before the newest ones do.
            if len(blob["messages"]) > self._max_turns * 2:
                blob["messages"] = blob["messages"][-self._max_turns * 2:]
            blob["updated"] = time.time()
            self._persist()

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["messages"] = []
                self._persist()

    def ids(self) -> list[str]:
        with self._lock:
            return sorted(self._sessions, key=lambda s: self._sessions[s].get("updated", 0))
