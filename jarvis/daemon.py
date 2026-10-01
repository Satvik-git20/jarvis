"""The HTTP daemon that opencode's plugin talks to.

Security posture: loopback bind only, and a required token header. The second
one is not ceremony -- any web page your browser loads can issue requests to
127.0.0.1, so an unauthenticated port here is reachable by anything you visit.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .core.budget import BudgetLedger
from .core.conversation import SessionStore
from .core.providers.base import Message
from .core.router import AllProvidersExhausted, Router
from .core.tools.image_gen import generate_image
from .core.tools.memory import MemoryStore
from .core.tools.web_search import search_web

log = logging.getLogger("jarvis.daemon")
STARTED = time.time()


class State:
    """Process-wide singletons, built once at startup."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.ensure_data_dir()
        data = self.settings.data_dir
        self.ledger = BudgetLedger(data / "budget.json")
        self.router = Router(self.ledger, self.settings)
        self.sessions = SessionStore(data / "sessions.json")
        self.memory = MemoryStore(data / "memory.db", self.settings)
        self._ollama_models: tuple[float, set[str]] = (0.0, set())

    def ollama_models(self, max_age: float = 60.0) -> set[str]:
        """Installed Ollama models, cached briefly.

        Used to name a missing local model in /status. Without this, "Ollama is
        ready" and "Ollama is ready but has no chat model pulled" look
        identical, and the only symptom is silent failover to a cloud provider
        that may have a 50-requests-a-day ceiling.
        """
        import httpx as _httpx

        now = time.time()
        stamp, cached = self._ollama_models
        if now - stamp < max_age:
            return cached
        names: set[str] = set()
        try:
            r = _httpx.get(f"{self.settings.ollama_host}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
        except Exception:
            names = set()
        self._ollama_models = (now, names)
        return names

    def setup_hints(self) -> list[str]:
        """Actionable setup steps. Empty means nothing is obviously missing."""
        hints: list[str] = []
        want = self.settings.ollama_chat_model
        installed = self.ollama_models()
        if not installed:
            hints.append(
                f"Ollama has no models. Run: ollama pull {want}  "
                "(the local brain is the only unlimited provider)"
            )
        elif not any(
            n == want or n.split(":")[0] == want.split(":")[0] for n in installed
        ):
            hints.append(
                f"Chat model {want} is not pulled. Run: ollama pull {want}  "
                "(until then the router falls over to a rate-limited cloud provider)"
            )
        embed = self.settings.ollama_embed_model
        if installed and not any(
            n == embed or n.split(":")[0] == embed.split(":")[0] for n in installed
        ):
            hints.append(
                f"Embedding model {embed} is not pulled. Run: ollama pull {embed}  "
                "(memory will fall back to keyword matching without it)"
            )
        if not self.settings.daemon_token:
            hints.append("JARVIS_DAEMON_TOKEN is empty; the daemon will refuse to start")
        if not self.settings.secret("JARVIS_CONTACT"):
            hints.append(
                "JARVIS_CONTACT is empty; Wikipedia returns 403 to a User-Agent "
                "with no contact address"
            )
        return hints

    def status(self) -> dict[str, Any]:
        return {
            "version": __version__,
            "uptime_seconds": round(time.time() - STARTED, 1),
            "ready_providers": self.router.ready(),
            "blocked_providers": self.router.blocked(),
            "last_provider_used": self.router.last_used,
            "setup_hints": self.setup_hints(),
            "budget": self.ledger.snapshot(),
            "sessions": len(self.sessions.ids()),
        }


STATE: State | None = None


def get_state() -> State:
    global STATE
    if STATE is None:
        STATE = State()
    return STATE


def require_token(x_jarvis_token: str | None = Header(default=None)) -> None:
    """Reject callers without the shared token.

    Compared with hmac.compare_digest so a wrong token cannot be discovered by
    timing. If no token is configured the daemon refuses to start rather than
    serving unauthenticated -- see `serve()`.
    """
    expected = get_state().settings.daemon_token
    if not expected:
        raise HTTPException(500, "JARVIS_DAEMON_TOKEN is not set; refusing to serve")
    if not x_jarvis_token or not hmac.compare_digest(x_jarvis_token, expected):
        raise HTTPException(401, "missing or invalid X-Jarvis-Token")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_state()
    log.info("jarvis daemon ready")
    yield


app = FastAPI(title="JARVIS", version=__version__, lifespan=lifespan)
auth = [Depends(require_token)]

# Only the Chrome side-panel extension may read responses cross-origin.
# Ordinary web pages can already *reach* this loopback port -- that was always
# true -- but without matching CORS headers they cannot read a byte back, and
# every state-changing route still demands X-Jarvis-Token either way.
# Chrome extension ids are 32 characters from the range a-p.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST"],
    allow_headers=["X-Jarvis-Token", "Content-Type"],
    max_age=86_400,
)


# --- request models ---------------------------------------------------------


class AskRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    system: str | None = None
    prefer: str | None = Field(default=None, description="Force a specific provider")
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    remember: bool = Field(default=True, description="Append the turn to session history")
    timeout: float = Field(default=60.0, ge=1.0, le=600.0,
                           description="Per-provider completion timeout in seconds")


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5
    fetch_content: bool = False


class RememberRequest(BaseModel):
    op: str = Field(pattern="^(store|recall|forget)$")
    text: str | None = None
    query: str | None = None
    tag: str | None = None
    k: int = 5


class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024


# --- routes -----------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, Any]:
    """Unauthenticated so the plugin can distinguish "down" from "bad token"."""
    return {"ok": True, "version": __version__, "uptime_seconds": round(time.time() - STARTED, 1)}


@app.get("/status", dependencies=auth)
async def status() -> dict[str, Any]:
    return get_state().status()


@app.post("/ask", dependencies=auth)
async def ask(req: AskRequest) -> dict[str, Any]:
    st = get_state()
    # Generate the id but do NOT insert a row yet: an unanswered or
    # remember=false request must not leave an empty conversation behind.
    # append() below creates the row as part of the turn transaction.
    sid = req.session_id or f"s_{uuid.uuid4().hex[:12]}"
    msgs = st.sessions.history(sid, system=req.system)
    msgs.append(Message("user", req.prompt))
    try:
        result = await st.router.complete(
            msgs, prefer=req.prefer, model=req.model,
            temperature=req.temperature, max_tokens=req.max_tokens,
            timeout=req.timeout,
        )
    except AllProvidersExhausted as exc:
        raise HTTPException(503, {
            "error": "all providers exhausted",
            "attempts": exc.attempts,
            "hint": "run `uv run python scripts/doctor.py` to see which keys are missing",
        }) from exc

    if req.remember:
        st.sessions.append(sid, req.prompt, result.text)
    return {
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "tokens": result.total_tokens,
        "latency_ms": result.latency_ms,
        "session_id": sid,
    }


@app.post("/search", dependencies=auth)
async def do_search(req: SearchRequest) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    results = await search_web(req.query, max_results=req.max_results,
                               fetch_content=req.fetch_content)
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/remember", dependencies=auth)
async def remember(req: RememberRequest) -> dict[str, Any]:
    st = get_state()
    if req.op == "store":
        if not req.text:
            raise HTTPException(400, "op=store requires `text`")
        return {"op": "store", **await st.memory.store(req.text, tag=req.tag)}
    if req.op == "recall":
        if not req.query:
            raise HTTPException(400, "op=recall requires `query`")
        hits = await st.memory.recall(req.query, k=req.k)
        return {"op": "recall", "count": len(hits), "memories": hits}
    if not req.query:
        raise HTTPException(400, "op=forget requires `query`")
    return {"op": "forget", "removed": st.memory.forget(req.query)}


@app.post("/image", dependencies=auth)
async def image(req: ImageRequest) -> dict[str, Any]:
    try:
        return await generate_image(req.prompt, width=req.width, height=req.height)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/session/{session_id}", dependencies=auth)
async def session(session_id: str) -> dict[str, Any]:
    st = get_state()
    if session_id not in st.sessions.ids():
        raise HTTPException(404, "no such session")
    return {
        "id": session_id,
        "messages": [{"role": m.role, "content": m.content}
                     for m in st.sessions.history(session_id, system=None)],
    }


@app.post("/session/{session_id}/clear", dependencies=auth)
async def clear_session(session_id: str) -> dict[str, str]:
    get_state().sessions.clear(session_id)
    return {"cleared": session_id}


def _selector_loop() -> asyncio.AbstractEventLoop:
    """A selector event loop, the one that survives failed accepts on Windows."""
    return asyncio.SelectorEventLoop()


def uvicorn_loop_setting() -> str:
    """Loop setting passed to uvicorn.run().

    uvicorn hard-codes ProactorEventLoop on Windows (uvicorn/loops/asyncio.py)
    and an asyncio policy set beforehand is ignored, so the selector loop has
    to be injected as a custom loop factory. The proactor accept path issues
    one overlapped AcceptEx at a time and, when that accept fails (a client
    reset in the accept queue -- WinError 64/10054), it stops re-arming: the
    daemon then sits idle in GetQueuedCompletionStatus forever while new
    connections pile up in the backlog, with nothing in the logs. The selector
    loop drains the backlog in a loop and keeps its read handler armed across
    a bad accept, so it heals. The daemon needs no proactor-only features (it
    spawns no asyncio subprocesses).
    """
    if sys.platform == "win32":
        return "jarvis.daemon:_selector_loop"
    return "asyncio"


def serve() -> None:
    import uvicorn

    s = get_settings()
    if not s.daemon_token:
        raise SystemExit(
            "JARVIS_DAEMON_TOKEN is empty. Generate one and put it in .env:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
            "The daemon will not start unauthenticated."
        )
    print(f"JARVIS daemon on http://{s.daemon_host}:{s.daemon_port}")
    uvicorn.run(
        "jarvis.daemon:app",
        host=s.daemon_host,
        port=s.daemon_port,
        loop=uvicorn_loop_setting(),
        log_level=os.environ.get("JARVIS_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    serve()
