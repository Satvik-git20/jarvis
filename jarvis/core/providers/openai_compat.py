"""OpenAI-shaped providers: OpenRouter, Groq, Cerebras.

Same wire format, different house. Only the base URL, the auth header, and
which model ids are actually free differ, so they share one implementation.
"""

from __future__ import annotations

import httpx

from jarvis.config import Settings, get_settings

from .base import (
    Completion,
    Message,
    ProviderError,
    ToolSpec,
    openai_payload,
    parse_openai_response,
    post_json,
)


class OpenAICompatible:
    """A provider speaking the OpenAI chat-completions dialect."""

    def __init__(
        self,
        name: str,
        base_url: str,
        capability: str,
        default_model: str,
        extra_headers: dict[str, str] | None = None,
        settings: Settings | None = None,
    ):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._capability = capability
        self._extra_headers = extra_headers or {}
        self._settings = settings or get_settings()

    def _key(self) -> str | None:
        cap = self._settings.by_name(self._capability)
        return self._settings.credential(cap) if cap else None

    def healthy(self) -> bool:
        cap = self._settings.by_name(self._capability)
        return bool(cap and self._settings.available(cap))

    def _headers(self) -> dict[str, str]:
        key = self._key()
        if not key:
            raise ProviderError(f"{self.name}: no API key", retryable=False)
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> Completion:
        import time

        started = time.perf_counter()
        chosen = model or self.default_model
        payload = openai_payload(chosen, messages, temperature=temperature,
                                 max_tokens=max_tokens, tools=tools)
        async with httpx.AsyncClient(timeout=timeout) as client:
            data = await post_json(
                client, self.name, f"{self.base_url}/chat/completions",
                payload, self._headers(), started,
            )
        return parse_openai_response(self.name, chosen, data, started)

    async def list_models(self, timeout: float = 10.0) -> list[str]:
        """Model ids this key can reach. Used to discover the free roster."""
        cap = self._settings.by_name(self._capability)
        key = self._settings.credential(cap) if cap else None
        if not key:
            return []
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self.base_url}/models",
                                     headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                return [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []


class OpenRouter(OpenAICompatible):
    """16 `:free` models behind one key.

    The free roster rotates constantly, so the default is discovered at runtime
    rather than hardcoded -- and deliberately NOT `openrouter/free`. That router
    picks any free model at random, including non-chat models: on this machine
    it handed back `nvidia/nemotron-3.5-content-safety:free`, a classifier
    that returns an empty completion.
    """

    # Free models that are not chat models. Any id matching these is skipped
    # even if it is listed as free.
    NON_CHAT = ("content-safety", "embedding", "rerank", "moderation",
                "guard", "classif", "safety")

    # Preferred, in descending order of usefulness for an assistant turn.
    PREFERRED = (
        "qwen/qwen3.8-27b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-31b-it:free",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b:free",
        "liquid/lfm-2.5-2.6b:free",
    )

    def __init__(self, settings: Settings | None = None):
        super().__init__(
            "openrouter",
            "https://openrouter.ai/api/v1",
            "openrouter",
            "",  # resolved lazily
            extra_headers={
                "HTTP-Referer": "http://localhost:8765",
                "X-Title": "JARVIS",
            },
            settings=settings,
        )
        self._default: str | None = None
        self._candidates: list[str] = []

    @classmethod
    def _is_chat_model(cls, model_id: str) -> bool:
        low = model_id.lower()
        return not any(bad in low for bad in cls.NON_CHAT)

    def _pick_default(self, free: list[str]) -> str | None:
        chatty = [m for m in free if self._is_chat_model(m)]
        if not chatty:
            return None
        for want in self.PREFERRED:
            for m in chatty:
                if m == want:
                    return m
        # Fall back to the largest-context chat model we can see.
        return sorted(chatty, key=lambda m: ("free" not in m, len(m)))[0]

    async def resolve_model(self, timeout: float = 10.0) -> str | None:
        """Discover a usable free chat model, caching the choice for the process.

        Refreshes periodically because the free roster churns; a model that
        disappears should not disable the provider for the whole run.
        """
        if self._default and not await self._default_still_free():
            self._default = None
        if self._default:
            return self._default
        free = await self.free_models(timeout)
        self._candidates = self._ordered_candidates(free)
        self._default = self._candidates[0] if self._candidates else None
        return self._default

    def _ordered_candidates(self, free: list[str]) -> list[str]:
        """Free chat models, best-first. Used as a rotation pool."""
        chatty = [m for m in free if self._is_chat_model(m)]
        rank = {m: i for i, m in enumerate(self.PREFERRED)}
        # Unlisted models rank after every known-good one. Ties keep the
        # catalog's own order, so rotation stays stable run to run.
        return sorted(chatty, key=lambda m: rank.get(m, len(self.PREFERRED)))

    def _advance(self) -> str | None:
        """Move to the next candidate after a rate-limited model."""
        if not self._candidates:
            return self._default
        try:
            idx = self._candidates.index(self._default)
        except ValueError:
            return self._default
        nxt = self._candidates[(idx + 1) % len(self._candidates)]
        self._default = nxt
        return nxt

    async def _default_still_free(self) -> bool:
        try:
            return self._default in await self.free_models()
        except Exception:
            return True  # network trouble is not evidence the model went away

    async def complete(self, messages, *, model: str | None = None, max_attempts: int = 3, **kw):
        """Try each candidate free model until one answers.

        Free models are individually rate-limited upstream and fail
        independently, so a single fixed choice gets stuck: `qwen3.8-27b:free`
        returned 429 "temporarily rate-limited" while the rest of the roster
        was idle. Rotating turns a dead provider into a slower one.
        """
        if model:
            return await super().complete(messages, model=model, **kw)

        last: ProviderError | None = None
        for _ in range(max(1, max_attempts)):
            chosen = await self.resolve_model() or self.PREFERRED[0]
            try:
                return await super().complete(messages, model=chosen, **kw)
            except ProviderError as exc:
                last = exc
                # Only rate limits and upstream failures justify trying a
                # different model. A 401 means the key is wrong and every
                # model will fail the same way.
                if exc.status not in (429, 502, 503, 504) or len(self._candidates) <= 1:
                    raise
                self._advance()
        assert last is not None
        raise last

    async def free_models(self, timeout: float = 10.0) -> list[str]:
        return sorted(m for m in await self.list_models(timeout) if m.endswith(":free"))


class Groq(OpenAICompatible):
    def __init__(self, settings: Settings | None = None):
        super().__init__("groq", "https://api.groq.com/openai/v1", "groq",
                         "llama-3.3-70b-versatile", settings=settings)


class Cerebras(OpenAICompatible):
    def __init__(self, settings: Settings | None = None):
        super().__init__("cerebras", "https://api.cerebras.ai/v1", "cerebras",
                         "llama-3.3-70b", settings=settings)
