"""Settings, key resolution, and the free-AI capability registry.

Secrets never leave this module unmasked. Callers ask for a capability name and
get back either a usable credential or None -- never a partially-populated
provider that will fail at request time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where opencode stores provider credentials on this machine. We read it only
# as a fallback so you do not have to paste the same key into two places.
OPENCODE_AUTH_CANDIDATES = (
    Path.home() / ".local" / "share" / "opencode" / "auth.json",
    Path.home() / ".config" / "opencode" / "auth.json",
    Path(os.environ.get("APPDATA", "")) / "opencode" / "auth.json" if os.environ.get("APPDATA") else None,
)


@dataclass(frozen=True)
class Capability:
    """One integration point and how to reach it.

    kind groups services that serve the same job so the router can try them in
    cost order. `requires_key` False means it works with no signup at all.
    `privacy_unsafe` marks free tiers whose vendor states prompt data is used
    for product improvement.
    """

    name: str
    kind: str
    label: str
    requires_key: bool
    privacy_unsafe: bool = False
    free_limit: str = ""
    # Environment variables that hold an actual secret. Checked in order.
    secret_env: tuple[str, ...] = ()
    # Key to look up in opencode's auth.json when every secret_env is unset.
    opencode_auth_id: str | None = None


REGISTRY: tuple[Capability, ...] = (
    # --- brains (LLM) -----------------------------------------------------
    Capability("ollama", "llm", "Ollama (local)", requires_key=False,
               free_limit="unlimited, offline", secret_env=()),
    Capability("openrouter", "llm", "OpenRouter :free", requires_key=True,
               free_limit="20 rpm / 50 rpd (1000 rpd after $10 lifetime)",
               secret_env=("OPENROUTER_API_KEY",), opencode_auth_id="openrouter"),
    Capability("groq", "llm", "Groq", requires_key=True,
               free_limit="30 rpm / 1000 rpd / 200k tpd",
               secret_env=("GROQ_API_KEY",)),
    Capability("cerebras", "llm", "Cerebras", requires_key=True,
               free_limit="30 rpm / 14.4k rpd / 1M tpd",
               secret_env=("CEREBRAS_API_KEY",)),
    Capability("gemini", "llm", "Google Gemini (free tier)", requires_key=True,
               privacy_unsafe=True,
               free_limit="free on Flash; limits per project",
               secret_env=("GEMINI_API_KEY",)),
    # --- voice in ---------------------------------------------------------
    Capability("whisper_local", "stt", "faster-whisper (local)", requires_key=False,
               free_limit="unlimited, offline", secret_env=()),
    Capability("groq_stt", "stt", "Groq whisper-large-v3-turbo", requires_key=True,
               free_limit="~8 audio-hours/day", secret_env=("GROQ_API_KEY",)),
    # --- voice out --------------------------------------------------------
    Capability("kokoro", "tts", "Kokoro-82M (local)", requires_key=False,
               free_limit="unlimited, offline", secret_env=()),
    Capability("edge_tts", "tts", "Microsoft Edge TTS", requires_key=False,
               free_limit="unlimited, no key", secret_env=()),
    # --- search -----------------------------------------------------------
    Capability("tavily", "search", "Tavily", requires_key=True,
               free_limit="1000 credits/mo", secret_env=("TAVILY_API_KEY",)),
    Capability("jina_reader", "search", "Jina Reader", requires_key=False,
               free_limit="20 rpm keyless / 500 rpm with key", secret_env=("JINA_API_KEY",)),
    Capability("ddgs", "search", "DuckDuckGo (ddgs)", requires_key=False,
               free_limit="keyless, rate-limited", secret_env=()),
    # --- embeddings / memory ---------------------------------------------
    Capability("jina_embed", "embed", "Jina embeddings", requires_key=True,
               free_limit="10M tokens on signup", secret_env=("JINA_API_KEY",)),
    Capability("ollama_embed", "embed", "Ollama nomic-embed-text", requires_key=False,
               free_limit="unlimited, offline", secret_env=()),
    # --- image ------------------------------------------------------------
    Capability("pollinations", "image", "Pollinations flux", requires_key=True,
               free_limit="flux model is free", secret_env=("POLLINATIONS_KEY",)),
    # --- vision -----------------------------------------------------------
    Capability("rapidocr", "ocr", "RapidOCR (local)", requires_key=False,
               free_limit="unlimited, offline", secret_env=()),
    # --- keyless utilities ------------------------------------------------
    Capability("open_meteo", "weather", "Open-Meteo", requires_key=False,
               free_limit="unlimited, no key", secret_env=()),
    Capability("wikipedia", "knowledge", "Wikipedia API", requires_key=False,
               free_limit="unlimited, no key", secret_env=()),
    Capability("hn_algolia", "knowledge", "Hacker News (Algolia)", requires_key=False,
               free_limit="no key", secret_env=()),
)


@dataclass
class Settings:
    data_dir: Path
    ollama_host: str
    ollama_chat_model: str
    ollama_embed_model: str
    daemon_host: str
    daemon_port: int
    daemon_token: str
    exclude_privacy_unsafe: bool
    # Backstop only. The Recorder tracks the real noise level at runtime and
    # gates on noise_multiplier x that, so these are not a calibration.
    rms_floor: float = 0.004
    vad_threshold: float = 0.5
    whisper_model: str = "small"
    _opencode_auth: dict[str, Any] | None = field(default=None, repr=False)

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    def secret(self, env_var: str) -> str | None:
        """Read one secret from the environment, trimmed, None if blank."""
        val = os.environ.get(env_var, "").strip()
        return val or None

    def opencode_key(self, auth_id: str) -> str | None:
        """Fallback credential lookup against opencode's auth.json.

        Read lazily and cached for the process. Never logged, never returned
        anywhere except a provider's Authorization header.
        """
        if self._opencode_auth is None:
            for path in OPENCODE_AUTH_CANDIDATES:
                if path and path.is_file():
                    try:
                        self._opencode_auth = json.loads(path.read_text("utf-8"))
                        break
                    except (OSError, json.JSONDecodeError):
                        continue
            else:
                self._opencode_auth = {}
        entry = self._opencode_auth.get(auth_id) or {}
        key = entry.get("key")
        return key.strip() if isinstance(key, str) and key.strip() else None

    def credential(self, cap: Capability) -> str | None:
        """Resolve the usable credential for a capability, or None.

        Order: our own .env first, then opencode's auth.json. That way adding a
        key to .env always wins, and a key opencode already has is picked up
        automatically without you pasting it twice.
        """
        for var in cap.secret_env:
            if val := self.secret(var):
                return val
        if cap.opencode_auth_id:
            return self.opencode_key(cap.opencode_auth_id)
        return None

    def available(self, cap: Capability) -> bool:
        """True when this capability can actually be used right now."""
        if cap.privacy_unsafe and self.exclude_privacy_unsafe:
            return False
        if not cap.requires_key:
            return True
        return self.credential(cap) is not None

    def by_kind(self, kind: str) -> list[Capability]:
        return [c for c in REGISTRY if c.kind == kind]

    def by_name(self, name: str) -> Capability | None:
        return next((c for c in REGISTRY if c.name == name), None)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env", override=False)
    data_dir = Path(_env("JARVIS_DATA_DIR", str(REPO_ROOT / ".jarvis"))).expanduser()
    return Settings(
        data_dir=data_dir,
        ollama_host=_env("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
        ollama_chat_model=_env("OLLAMA_CHAT_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
        ollama_embed_model=_env("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        daemon_host=_env("JARVIS_DAEMON_HOST", "127.0.0.1"),
        daemon_port=int(_env("JARVIS_DAEMON_PORT", "8765")),
        daemon_token=_env("JARVIS_DAEMON_TOKEN", ""),
        exclude_privacy_unsafe=_env("JARVIS_EXCLUDE_PRIVACY_UNSAFE", "0") in ("1", "true", "yes"),
        # Run `jarvis test-voice --calibrate` in a quiet room to get a floor
        # that suits your environment; the defaults suit a fairly quiet one.
        rms_floor=float(_env("JARVIS_RMS_FLOOR", "0.015")),
        vad_threshold=float(_env("JARVIS_VAD_THRESHOLD", "0.5")),
        whisper_model=_env("JARVIS_WHISPER_MODEL", "small"),
    )
