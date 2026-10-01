"""DaemonClient contract tests.

This client is how the voice loop, `python -m jarvis talk`, and anything else
outside the daemon talks to conversational state. Three failure modes must be
distinguishable at a terminal: daemon down, bad token, providers exhausted --
each gets its own exception and its own actionable message. No real sockets:
httpx.MockTransport stands in for the daemon.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from jarvis.client import (
    Answer,
    DaemonClient,
    DaemonDown,
    DaemonError,
    DaemonRejected,
    ProvidersExhausted,
)
from jarvis.config import Settings

TOKEN = "client-token-xyz"


def make_client(handler, tmp_path) -> DaemonClient:
    settings = Settings(
        data_dir=tmp_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3:4b",
        ollama_embed_model="nomic-embed-text",
        daemon_host="127.0.0.1",
        daemon_port=8765,
        daemon_token=TOKEN,
        exclude_privacy_unsafe=False,
    )
    return DaemonClient(settings, transport=httpx.MockTransport(handler))


def test_ask_parses_the_response_and_sends_the_token(tmp_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["token"] = request.headers.get("x-jarvis-token")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "text": "hi there", "provider": "ollama", "model": "qwen3:4b",
            "tokens": 42, "latency_ms": 5, "session_id": "s_abc",
        })

    client = make_client(handler, tmp_path)

    async def run() -> Answer:
        return await client.ask("hello", session_id="s_abc", system="sys",
                                prefer="ollama", remember=True)

    answer = asyncio.run(run())
    assert answer == Answer(text="hi there", provider="ollama", model="qwen3:4b",
                            tokens=42, latency_ms=5, session_id="s_abc")
    assert seen["path"] == "/ask"
    assert seen["token"] == TOKEN
    assert seen["body"]["remember"] is True
    assert seen["body"]["system"] == "sys"
    assert seen["body"]["prefer"] == "ollama"


def test_one_shot_ask_omits_optional_fields_and_disables_remember(tmp_path):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "text": "4", "provider": "ollama", "model": "m",
            "tokens": 1, "latency_ms": 2, "session_id": "s_x",
        })

    client = make_client(handler, tmp_path)
    asyncio.run(client.ask("2+2", remember=False))
    assert seen["body"]["remember"] is False
    assert "session_id" not in seen["body"]
    assert "system" not in seen["body"]
    assert "prefer" not in seen["body"]


def test_connection_refused_reports_how_to_start_the_daemon(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(handler, tmp_path)
    with pytest.raises(DaemonDown) as exc:
        asyncio.run(client.ask("hi"))
    assert "uv run python -m jarvis daemon" in str(exc.value)


def test_bad_token_gets_its_own_error(tmp_path):
    client = make_client(
        lambda request: httpx.Response(401, json={"detail": "invalid token"}),
        tmp_path,
    )
    with pytest.raises(DaemonRejected) as exc:
        asyncio.run(client.status())
    assert "JARVIS_DAEMON_TOKEN" in str(exc.value)


def test_exhausted_providers_are_reraised_with_attempts(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={
            "error": "all providers exhausted",
            "attempts": [["ollama", "429 cooldown"], ["openrouter", "not configured"]],
            "hint": "run doctor",
        })

    client = make_client(handler, tmp_path)
    with pytest.raises(ProvidersExhausted) as exc:
        asyncio.run(client.ask("hi"))
    assert exc.value.attempts == [
        ("ollama", "429 cooldown"),
        ("openrouter", "not configured"),
    ]
    assert "ollama: 429 cooldown" in str(exc.value)


def test_health_and_status_hit_the_right_paths(tmp_path):
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"ok": True, "ready_providers": ["ollama"]})

    client = make_client(handler, tmp_path)

    async def run():
        assert (await client.health())["ok"] is True
        assert (await client.status())["ready_providers"] == ["ollama"]

    asyncio.run(run())
    assert paths == ["/health", "/status"]


def test_bind_any_daemon_host_is_dialled_as_loopback(tmp_path):
    """daemon_host may be 0.0.0.0 (bind-all); clients must dial 127.0.0.1."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.headers["host"]
        return httpx.Response(200, json={"ok": True})

    settings = Settings(
        data_dir=tmp_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3:4b",
        ollama_embed_model="nomic-embed-text",
        daemon_host="0.0.0.0",
        daemon_port=8765,
        daemon_token=TOKEN,
        exclude_privacy_unsafe=False,
    )
    client = DaemonClient(settings, transport=httpx.MockTransport(handler))
    asyncio.run(client.health())
    assert seen["host"].startswith("127.0.0.1:")


def test_non_dict_json_response_raises_instead_of_crashing(tmp_path):
    client = make_client(
        lambda request: httpx.Response(200, json=["not", "a", "dict"]),
        tmp_path,
    )
    with pytest.raises(DaemonError):
        asyncio.run(client.status())
