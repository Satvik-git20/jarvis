"""Provider construction, in failover order.

Order is the whole design. Local first because it has no quota and no data
egress; then the fastest free tiers; then the widest-context ones. Anything with
no credential is skipped silently, so a half-configured .env still works.
"""

from __future__ import annotations

from jarvis.config import Settings, get_settings

from .base import Provider
from .gemini import Gemini
from .ollama import Ollama
from .openai_compat import Cerebras, Groq, OpenRouter

# Failover order. Ollama leads because it is the only unbounded option.
FAILOVER_ORDER = ("ollama", "openrouter", "groq", "cerebras", "gemini")


def build_providers(settings: Settings | None = None) -> dict[str, Provider]:
    s = settings or get_settings()
    providers: dict[str, Provider] = {
        "ollama": Ollama(s),
        "openrouter": OpenRouter(s),
        "groq": Groq(s),
        "cerebras": Cerebras(s),
        "gemini": Gemini(s),
    }
    return {name: p for name, p in providers.items() if name in FAILOVER_ORDER}
