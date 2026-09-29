"""Long-term memory on sqlite-vec.

Two embedding backends behind one interface: Jina when a key is present (10M
free tokens, best quality), otherwise the local Ollama nomic-embed-text so
memory still works with zero configuration.

sqlite-vec is brute-force below its 1.0 release, which is the right trade for
personal memory: tens of thousands of chunks, no index to maintain, and the
whole thing is one file in the data directory.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from jarvis.config import Settings, get_settings

log = logging.getLogger("jarvis.memory")

JINA_URL = "https://api.jina.ai/v1/embeddings"
JINA_MODEL = "jina-embeddings-v5-text-small"
JINA_DIM = 1024
LOCAL_DIM = 768


def _cosine(a: list[float], b: list[float]) -> float:
    """Both vectors are L2-normalised by their backends, so the dot product
    is the cosine similarity."""
    return sum(x * y for x, y in zip(a, b, strict=False))


class MemoryStore:
    def __init__(self, path: Path, settings: Settings | None = None):
        self._path = path
        self._settings = settings or get_settings()
        self._lock = threading.RLock()
        self._backend: str | None = None
        self._dim: int = LOCAL_DIM
        self._local_checked = False
        self._local_model_ok = False
        self._conn = self._connect()

    # --- storage ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_vec(conn)
        return conn

    def _ensure_vec(self, conn: sqlite3.Connection) -> None:
        try:
            import sqlite_vec
        except ImportError:
            log.warning("sqlite-vec not installed; run: uv sync --extra memory")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                " id TEXT PRIMARY KEY, text TEXT NOT NULL, tag TEXT,"
                " embedding TEXT, created REAL)"
            )
            return
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            " id TEXT PRIMARY KEY, text TEXT NOT NULL, tag TEXT,"
            " embedding BLOB, created REAL)"
        )

    # --- embeddings -------------------------------------------------------

    async def _embed(self, text: str) -> tuple[list[float], str]:
        """Return (vector, backend). Picks a backend once and remembers it."""
        s = self._settings
        cap = s.by_name("jina_embed")
        if self._backend != "ollama" and cap and s.available(cap):
            key = s.credential(cap)
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(
                        JINA_URL,
                        json={"model": JINA_MODEL, "input": [text]},
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    r.raise_for_status()
                    vec = r.json()["data"][0]["embedding"]
                self._backend, self._dim = "jina", len(vec)
                return vec, "jina"
            except Exception as exc:
                log.info("jina embeddings unavailable (%s); falling back to local", exc)

        vec = await self._embed_local(text)
        self._backend, self._dim = "ollama", len(vec) or LOCAL_DIM
        return vec, "ollama"

    async def _embed_local(self, text: str) -> list[float]:
        """Ollama embeddings. Returns [] when Ollama or the model is absent,
        which the caller treats as 'store without vector'.

        The presence check is what keeps a missing model from hanging: without
        it, Ollama would try to download the weights inline. With it in place
        the remaining slow case is a cold first call while Ollama loads the
        model, which is why this timeout is generous.
        """
        s = self._settings
        if not await self._local_model_present():
            log.info("local embedding model %s not pulled; run: ollama pull %s",
                     s.ollama_embed_model, s.ollama_embed_model)
            return []
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(
                    f"{s.ollama_host}/api/embed",
                    json={"model": s.ollama_embed_model, "input": text},
                )
                r.raise_for_status()
                return r.json().get("embeddings", [[]])[0] or []
        except Exception as exc:
            log.info("local embedding failed: %s", exc)
            return []

    async def _local_model_present(self) -> bool:
        """Cached check that the embedding model is actually installed."""
        if self._local_checked:
            return self._local_model_ok
        self._local_checked = True
        s = self._settings
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{s.ollama_host}/api/tags")
                r.raise_for_status()
                names = {m.get("name", "") for m in r.json().get("models", [])}
            base = s.ollama_embed_model.split(":")[0]
            self._local_model_ok = any(
                n == s.ollama_embed_model or n.split(":")[0] == base for n in names
            )
        except Exception:
            self._local_model_ok = False
        return self._local_model_ok

    # --- public API -------------------------------------------------------

    async def store(self, text: str, *, tag: str | None = None) -> dict[str, Any]:
        """Persist a memory. Reports whether it got a vector, because a memory
        stored without one is only findable by exact keyword and would
        otherwise fail silently later."""
        text = (text or "").strip()
        if not text:
            raise ValueError("nothing to remember")
        vec, backend = await self._embed(text)
        ident = f"m_{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (id, text, tag, embedding, created) VALUES (?,?,?,?,?)",
                (ident, text, tag, json.dumps(vec) if vec else None, time.time()),
            )
            self._conn.commit()
        return {
            "id": ident,
            "embedded": bool(vec),
            "backend": backend,
            "note": None if vec else
            "stored without a vector; recall will fall back to keyword matching",
        }

    async def recall(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        query = (query or "").strip()
        if not query:
            return []
        qvec, _ = await self._embed(query)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, text, tag, embedding, created FROM memories"
            ).fetchall()

        if not qvec:
            # No embeddings available: fall back to a keyword match so recall
            # degrades to search rather than returning nothing.
            needle = query.lower()
            hits = [r for r in rows if needle in (r[1] or "").lower()][:k]
            return [{"id": r[0], "text": r[1], "tag": r[2], "score": None} for r in hits]

        scored = []
        for ident, text, tag, blob, _created in rows:
            if not blob:
                continue
            try:
                vec = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if len(vec) != len(qvec):
                continue
            scored.append((_cosine(qvec, vec), ident, text, tag))
        scored.sort(reverse=True)
        return [
            {"id": ident, "text": text, "tag": tag, "score": round(score, 4)}
            for score, ident, text, tag in scored[:k]
        ]

    def forget(self, query: str) -> int:
        """Delete memories whose text contains the query."""
        needle = (query or "").strip().lower()
        if not needle:
            return 0
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM memories WHERE lower(text) LIKE ?", (f"%{needle}%",)
            )
            self._conn.commit()
            return cur.rowcount

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
