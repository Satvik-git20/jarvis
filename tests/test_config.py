"""Key resolution is the highest-risk logic in config.py: it decides whether a
provider is used at all, and it reads a second file that holds real secrets.
These tests pin the precedence and guarantee no secret escapes in the process.
"""

from __future__ import annotations

import json

from jarvis.config import Capability, Settings

OPENROUTER = Capability(
    "openrouter", "llm", "OpenRouter", requires_key=True,
    secret_env=("OPENROUTER_API_KEY",), opencode_auth_id="openrouter",
)


def make_settings(tmp_path, monkeypatch, auth: dict | None = None, **overrides) -> Settings:
    """Build a Settings without touching the module-level lru_cache or .env."""
    if auth is not None:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps(auth), "utf-8")
        monkeypatch.setattr("jarvis.config.OPENCODE_AUTH_CANDIDATES", (path,))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    defaults = dict(
        data_dir=tmp_path,
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3:4b",
        ollama_embed_model="nomic-embed-text",
        daemon_host="127.0.0.1",
        daemon_port=8765,
        daemon_token="",
        exclude_privacy_unsafe=False,
    )
    return Settings(**{**defaults, **overrides})


def test_env_wins_over_opencode_auth(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch,
                      auth={"openrouter": {"type": "api", "key": "from-opencode"}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert s.credential(OPENROUTER) == "from-env"


def test_falls_back_to_opencode_auth(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch,
                      auth={"openrouter": {"type": "api", "key": "from-opencode"}})
    assert s.credential(OPENROUTER) == "from-opencode"


def test_blank_env_value_is_not_a_credential(tmp_path, monkeypatch):
    """Whitespace must fall through to the fallback, not count as present."""
    s = make_settings(tmp_path, monkeypatch,
                      auth={"openrouter": {"type": "api", "key": "from-opencode"}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    assert s.credential(OPENROUTER) == "from-opencode"


def test_missing_everywhere_is_none(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch, auth={})
    assert s.credential(OPENROUTER) is None
    assert s.available(OPENROUTER) is False


def test_malformed_auth_file_is_survivable(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    path.write_text("{ not json", "utf-8")
    monkeypatch.setattr("jarvis.config.OPENCODE_AUTH_CANDIDATES", (path,))
    s = make_settings(tmp_path, monkeypatch)
    assert s.credential(OPENROUTER) is None


def test_keyless_capability_needs_no_credential(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch)
    cap = Capability("ddgs", "search", "DuckDuckGo", requires_key=False)
    assert s.available(cap) is True
    assert s.credential(cap) is None


def test_privacy_unsafe_can_be_excluded(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch, exclude_privacy_unsafe=True)
    cap = Capability("gemini", "llm", "Gemini", requires_key=True, privacy_unsafe=True,
                     secret_env=("GEMINI_API_KEY",))
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    assert s.credential(cap) == "secret"   # still resolvable
    assert s.available(cap) is False        # but never selected for use


def test_secret_never_appears_in_repr(tmp_path, monkeypatch):
    s = make_settings(tmp_path, monkeypatch,
                      auth={"openrouter": {"type": "api", "key": "super-secret"}})
    s.opencode_key("openrouter")
    assert "super-secret" not in repr(s)
