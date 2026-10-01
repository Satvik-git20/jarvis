"""Daemon integration tests.

The auth tests are the important ones. This daemon listens on loopback, which
any web page the browser loads can reach, so "requires a token" has to be
verified rather than assumed.

Provider and tool calls are stubbed so the suite never spends a token or
depends on a quota being available.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import jarvis.daemon as daemon_module
from jarvis.config import Settings
from jarvis.core.budget import BudgetLedger, Limits
from jarvis.core.providers.base import Completion, Message, ProviderError

TOKEN = "test-token-abc123"


class FakeProvider:
    """Stands in for a real brain so no tokens are spent."""

    def __init__(self, name: str, *, fail: bool = False, empty: bool = False):
        self.name = name
        self.fail = fail
        self.empty = empty
        self.calls: list[list[Message]] = []

    def healthy(self) -> bool:
        return True

    async def complete(self, messages, **kw) -> Completion:
        self.calls.append(messages)
        if self.fail:
            raise ProviderError(f"{self.name}: simulated 429", status=429)
        return Completion(
            text="" if self.empty else f"answered by {self.name}",
            model="fake-model",
            provider=self.name,
            tokens_in=10,
            tokens_out=5,
            latency_ms=1,
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    s = Settings(
        data_dir=tmp_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3:4b",
        ollama_embed_model="nomic-embed-text",
        daemon_host="127.0.0.1",
        daemon_port=8765,
        daemon_token=TOKEN,
        exclude_privacy_unsafe=False,
    )

    from jarvis.core.conversation import SessionStore

    class StubState:
        def __init__(self):
            self.settings = s
            self.ledger = BudgetLedger(tmp_path / "budget.json", limits={})
            self.router = None
            self.sessions = SessionStore(tmp_path / "sessions.json")
            self.memory = None

    stub = StubState()
    monkeypatch.setattr(daemon_module, "get_state", lambda: stub)
    monkeypatch.setattr(daemon_module, "STATE", stub)
    with TestClient(daemon_module.app) as c:
        c.stub_state = stub
        yield c


# --- event loop (Windows accept-wedge regression) ---------------------------


def test_daemon_opts_out_of_the_proactor_accept_loop():
    """Windows' default proactor loop stops accepting forever after one
    failed AcceptEx (a client reset in the accept queue): the loop idles in
    GetQueuedCompletionStatus while new connections pile up in the backlog.
    uvicorn hard-codes the proactor on Windows, so the daemon must inject the
    self-healing selector loop as a custom loop factory."""
    import sys

    from jarvis.daemon import _selector_loop, uvicorn_loop_setting

    if sys.platform == "win32":
        assert uvicorn_loop_setting() == "jarvis.daemon:_selector_loop"
        loop = _selector_loop()
        try:
            assert isinstance(loop, asyncio.SelectorEventLoop)
        finally:
            loop.close()
    else:
        assert uvicorn_loop_setting() == "asyncio"


# --- auth -------------------------------------------------------------------


def test_health_needs_no_token(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_status_requires_token(client):
    assert client.get("/status").status_code == 401


def test_status_rejects_wrong_token(client):
    assert client.get("/status", headers={"X-Jarvis-Token": "nope"}).status_code == 401


def test_status_accepts_correct_token(client):
    """A valid token gets past auth. 404 (no such session) proves it reached
    the handler, where a 401 would mean auth rejected it."""
    r = client.get("/session/does-not-exist", headers={"X-Jarvis-Token": TOKEN})
    assert r.status_code == 404


def test_ask_requires_token(client):
    r = client.post("/ask", json={"prompt": "hello"})
    assert r.status_code == 401


def test_ask_without_remember_leaves_no_session_row(client):
    """remember=false must not create an empty conversation.

    The handler used to call get_or_create before the model answered, so
    every one-shot question left a dead zero-message session behind."""
    client.stub_state.router = FakeProvider("ollama")
    auth = {"X-Jarvis-Token": TOKEN}
    r = client.post("/ask", json={"prompt": "one-shot", "remember": False},
                    headers=auth)
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]
    assert sid.startswith("s_")
    assert sid not in client.stub_state.sessions.ids()
    assert client.get(f"/session/{sid}", headers=auth).status_code == 404


def test_ask_remembers_the_turn_and_history_contains_both_sides(client):
    client.stub_state.router = FakeProvider("ollama")
    auth = {"X-Jarvis-Token": TOKEN}
    r = client.post("/ask", json={"prompt": "hi", "session_id": "s_t",
                                  "remember": True}, headers=auth)
    assert r.status_code == 200, r.text
    history = client.get("/session/s_t", headers=auth).json()
    contents = [m["content"] for m in history["messages"]]
    assert "hi" in contents
    assert "answered by ollama" in contents


def test_ask_passes_the_client_timeout_through_to_the_router(client):
    """The CLI needs a 300s budget for cold local models; the router default
    is 60s, so /ask has to forward the caller's value."""
    fake = FakeProvider("ollama")
    client.stub_state.router = fake
    seen: dict = {}
    original = fake.complete

    async def spy(messages, **kw):
        seen.update(kw)
        return await original(messages, **kw)

    fake.complete = spy
    r = client.post("/ask", json={"prompt": "x", "timeout": 123},
                    headers={"X-Jarvis-Token": TOKEN})
    assert r.status_code == 200, r.text
    assert seen["timeout"] == 123


def test_daemon_refuses_to_serve_without_a_configured_token(tmp_path, monkeypatch):
    from jarvis.config import get_settings

    monkeypatch.setattr(
        get_settings, "cache_clear", lambda: None, raising=False
    )
    s = Settings(
        data_dir=tmp_path, ollama_host="", ollama_chat_model="", ollama_embed_model="",
        daemon_host="127.0.0.1", daemon_port=8765, daemon_token="",
        exclude_privacy_unsafe=False,
    )
    stub = type("S", (), {"settings": s})()
    monkeypatch.setattr(daemon_module, "get_state", lambda: stub)
    with TestClient(daemon_module.app) as c:
        # 500 not 200: loud failure beats silently serving an open port.
        assert c.get("/status", headers={"X-Jarvis-Token": "anything"}).status_code == 500


# --- memory endpoints -------------------------------------------------------


def test_remember_store_and_recall_round_trip(client, monkeypatch):
    """Regression: the daemon forgot to await the coroutine, so a coroutine
    object was splatted into a dict and every store returned 500."""
    from jarvis.core.tools.memory import MemoryStore

    store = client.stub_state.memory = MemoryStore(
        client.stub_state.settings.data_dir / "memory.db", client.stub_state.settings
    )

    async def fake_embed(text: str):
        # Deterministic bag-of-characters vector. Deliberately not hash():
        # Python randomises string hashing per process, so the same word would
        # land in a different bucket on the next run and recall would flake.
        vec = [0.0] * 8
        for ch in text.lower():
            vec[ord(ch) % 8] += 1.0
        return vec, "stub"

    monkeypatch.setattr(store, "_embed", fake_embed)
    auth = {"X-Jarvis-Token": TOKEN}

    r = client.post("/remember", json={"op": "store", "text": "the sky is blue",
                                       "tag": "colour"}, headers=auth)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embedded"] is True
    assert body["backend"] == "stub"
    assert body["id"].startswith("m_")

    r = client.post("/remember", json={"op": "recall", "query": "sky"}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["count"] >= 1
    assert "sky" in r.json()["memories"][0]["text"]

    r = client.post("/remember", json={"op": "forget", "query": "sky"}, headers=auth)
    assert r.status_code == 200
    assert r.json()["removed"] == 1


def test_remember_reports_when_no_vector_was_stored(client, monkeypatch):
    """A memory stored without a vector is only findable by keyword, so the
    degradation must be reported rather than silent."""
    from jarvis.core.tools.memory import MemoryStore

    store = client.stub_state.memory = MemoryStore(
        client.stub_state.settings.data_dir / "memory.db", client.stub_state.settings
    )

    async def no_embed(text: str):
        return [], "ollama"

    monkeypatch.setattr(store, "_embed", no_embed)
    r = client.post("/remember", json={"op": "store", "text": "no vectors here"},
                    headers={"X-Jarvis-Token": TOKEN})
    assert r.status_code == 200
    body = r.json()
    assert body["embedded"] is False
    assert body["note"]


def test_remember_recall_falls_back_to_keyword_without_embeddings(client, monkeypatch):
    from jarvis.core.tools.memory import MemoryStore

    store = client.stub_state.memory = MemoryStore(
        client.stub_state.settings.data_dir / "memory.db", client.stub_state.settings
    )

    async def no_embed(text: str):
        return [], "ollama"

    monkeypatch.setattr(store, "_embed", no_embed)
    auth = {"X-Jarvis-Token": TOKEN}
    client.post("/remember", json={"op": "store", "text": "pineapple"}, headers=auth)
    r = client.post("/remember", json={"op": "recall", "query": "pineapple"}, headers=auth)
    assert r.json()["count"] == 1


# --- router failover --------------------------------------------------------


def test_router_fails_over_past_a_quota_failure(tmp_path):
    from jarvis.core.router import Router

    good, bad = FakeProvider("good"), FakeProvider("bad", fail=True)
    router = Router(BudgetLedger(tmp_path / "b.json", limits={}), None)
    router._providers = {"ollama": bad, "openrouter": good}

    result = asyncio.run(router.complete([Message("user", "hi")]))
    assert result.provider == "good"
    # The failure should have parked the bad provider, not just been logged.
    assert router.ledger.cooldown_for("ollama") > 0


def test_router_raises_only_when_all_providers_fail(tmp_path):
    from jarvis.core.router import AllProvidersExhausted, Router

    router = Router(BudgetLedger(tmp_path / "b.json", limits={}), None)
    router._providers = {"ollama": FakeProvider("a", fail=True), "openrouter": FakeProvider("b", fail=True)}
    with pytest.raises(AllProvidersExhausted) as exc:
        asyncio.run(router.complete([Message("user", "hi")]))
    assert {name for name, _ in exc.value.attempts} == {"ollama", "openrouter"}


def test_router_prefers_named_provider_but_still_fails_over(tmp_path):
    from jarvis.core.router import Router

    failing, ok = FakeProvider("gemini", fail=True), FakeProvider("ollama")
    router = Router(BudgetLedger(tmp_path / "b.json", limits={}), None)
    router._providers = {"ollama": ok, "gemini": failing}
    result = asyncio.run(router.complete([Message("user", "hi")], prefer="gemini"))
    assert result.provider == "ollama"  # honoured the preference, then recovered


def test_empty_completion_is_treated_as_failure(tmp_path):
    """Regression: a free-roster classifier returns HTTP 200 with empty text.
    Accepting that ended the request in silence instead of failing over."""
    from jarvis.core.router import Router

    silent, talker = FakeProvider("openrouter", empty=True), FakeProvider("ollama")
    router = Router(BudgetLedger(tmp_path / "b.json", limits={}), None)
    router._providers = {"ollama": talker, "openrouter": silent}
    result = asyncio.run(router.complete([Message("user", "hi")]))
    assert result.provider == "ollama"
    assert result.text.strip()


def test_empty_completion_does_not_penalise_the_provider(tmp_path):
    """The provider is healthy, just not a chat model. Cooling it down would
    wrongly disable a working endpoint."""
    from jarvis.core.router import AllProvidersExhausted, Router

    router = Router(BudgetLedger(tmp_path / "b.json", limits={}), None)
    router._providers = {"ollama": FakeProvider("ollama", empty=True)}
    with pytest.raises(AllProvidersExhausted):
        asyncio.run(router.complete([Message("user", "hi")]))
    assert router.ledger.cooldown_for("ollama") == 0


# --- openrouter model selection --------------------------------------------


def test_openrouter_skips_non_chat_free_models():
    from jarvis.core.providers.openai_compat import OpenRouter

    roster = [
        "nvidia/nemotron-3.5-content-safety:free",
        "liquid/lfm-2.5-2.6b:free",
        "some/thing-embedding:free",
        "cohere/rerank-v3:free",
        "qwen/qwen3.8-27b:free",
    ]
    picked = OpenRouter()._pick_default(roster)
    assert picked == "qwen/qwen3.8-27b:free"
    assert "content-safety" not in picked


def test_openrouter_prefers_a_known_good_model():
    from jarvis.core.providers.openai_compat import OpenRouter

    roster = [
        "nvidia/nemotron-3.5-content-safety:free",
        "openai/gpt-oss-120b:free",
        "qwen/qwen3.8-27b:free",
    ]
    assert OpenRouter()._pick_default(roster) == "qwen/qwen3.8-27b:free"


def test_openrouter_returns_none_when_only_non_chat_models_are_free():
    from jarvis.core.providers.openai_compat import OpenRouter

    assert OpenRouter()._pick_default(["nvidia/nemotron-3.5-content-safety:free"]) is None


def test_openrouter_rotation_advances_past_a_rate_limited_model():
    from jarvis.core.providers.openai_compat import OpenRouter

    o = OpenRouter()
    o._candidates = ["a:free", "b:free", "c:free"]
    o._default = "a:free"
    assert o._advance() == "b:free"
    assert o._advance() == "c:free"
    assert o._advance() == "a:free"  # wraps


def test_openrouter_rotation_ignores_empty_pool():
    from jarvis.core.providers.openai_compat import OpenRouter

    o = OpenRouter()
    assert o._advance() is None  # nothing to rotate to; stays None


def test_openrouter_candidate_order_puts_preferred_first():
    from jarvis.core.providers.openai_compat import OpenRouter

    o = OpenRouter()
    got = o._ordered_candidates([
        "some/unknown:free",
        "liquid/lfm-2.5-2.6b:free",
        "nvidia/nemotron-3.5-content-safety:free",
        "qwen/qwen3.8-27b:free",
    ])
    assert "nvidia/nemotron-3.5-content-safety:free" not in got
    assert got[0] == "qwen/qwen3.8-27b:free"
    assert got[-1] == "some/unknown:free"


# --- budget ledger ----------------------------------------------------------


def test_ledger_enforces_minute_limit(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limits={"t": Limits(rpm=2, rpd=100)})
    ledger.record("t")
    ledger.record("t")
    assert ledger.available("t") is False
    assert ledger.exhausted("t") == "rpm"


def test_ledger_enforces_daily_limit(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limits={"t": Limits(rpm=100, rpd=1)})
    ledger.record("t")
    assert ledger.available("t") is False
    assert ledger.exhausted("t") == "rpd"


def test_ledger_token_cap_binds_before_request_cap(tmp_path):
    """Groq allows 1000 rpd but only 200k tpd, so tokens are the real limit."""
    ledger = BudgetLedger(tmp_path / "b.json", limits={"groq": Limits(rpm=1000, rpd=1000, tpd=200_000)})
    ledger.record("groq", tokens=200_000)
    assert ledger.available("groq") is False
    assert ledger.exhausted("groq") == "tpd"


def test_ledger_ignores_old_entries(tmp_path):
    import time

    ledger = BudgetLedger(tmp_path / "b.json", limits={"t": Limits(rpm=5, rpd=5)})
    st = ledger.state("t")
    stale = time.time() - 3600
    st.requests_min.append(stale)
    st.requests_day.append(stale)
    assert ledger.available("t") is True


def test_ledger_survives_restart(tmp_path):
    path = tmp_path / "b.json"
    first = BudgetLedger(path, limits={"t": Limits(rpm=100, rpd=1)})
    first.record("t")
    second = BudgetLedger(path, limits={"t": Limits(rpm=100, rpd=1)})
    assert second.available("t") is False  # daily cap not reset by a restart


def test_ledger_penalty_clears_on_success(tmp_path):
    ledger = BudgetLedger(tmp_path / "b.json", limits={})
    ledger.penalize("t", "429", 60)
    assert ledger.available("t") is False
    ledger.record("t")
    assert ledger.available("t") is True


# --- CORS: the Chrome side panel is the only cross-origin reader ------------

EXTENSION_ORIGIN = f"chrome-extension://{('a' * 32)}"


def test_extension_preflight_is_accepted(client):
    r = client.options("/ask", headers={
        "Origin": EXTENSION_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-jarvis-token,content-type",
    })
    assert r.status_code in (200, 204)
    assert r.headers["access-control-allow-origin"] == EXTENSION_ORIGIN
    assert "x-jarvis-token" in r.headers["access-control-allow-headers"].lower()
    assert "POST" in r.headers["access-control-allow-methods"]


def test_extension_reads_responses_cross_origin(client):
    r = client.get("/health", headers={"Origin": EXTENSION_ORIGIN})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == EXTENSION_ORIGIN


def test_ordinary_web_pages_get_no_cors_headers(client):
    """A web page may reach the port, but it must not be able to read replies."""
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_extension_still_needs_the_token(client):
    """CORS is a readability grant, not an auth bypass."""
    assert client.post("/ask", json={"prompt": "hi"},
                       headers={"Origin": EXTENSION_ORIGIN}).status_code == 401
    assert client.get("/session/x", headers={"Origin": EXTENSION_ORIGIN}).status_code == 401
