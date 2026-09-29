"""Google Gemini free tier.

Not OpenAI-shaped. The endpoint differs, the auth is a query parameter or
`x-goog-api-key` header, and the schema nests parts differently. Marked
`privacy_unsafe` in the registry because Google's pricing page states free-tier
content is used to improve their products.
"""

from __future__ import annotations

import time

import httpx

from jarvis.config import Settings, get_settings

from .base import Completion, Message, ProviderError, ToolSpec

BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"


class Gemini:
    name = "gemini"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    def _key(self) -> str | None:
        cap = self._settings.by_name("gemini")
        return self._settings.credential(cap) if cap else None

    def healthy(self) -> bool:
        cap = self._settings.by_name("gemini")
        return bool(cap and self._settings.available(cap))

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[dict]]:
        """Gemini takes system text as a separate field, not a message role."""
        system: str | None = None
        rest: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = (system + "\n\n" + m.content) if system else m.content
            else:
                # "assistant" is Gemini's "model"; "user" is as-is.
                role = "model" if m.role == "assistant" else "user"
                rest.append({"role": role, "parts": [{"text": m.content}]})
        return system, rest

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
        started = time.perf_counter()
        chosen = model or DEFAULT_MODEL
        key = self._key()
        if not key:
            raise ProviderError("gemini: no API key", retryable=False)

        system, contents = self._split_system(messages)
        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{
                "functionDeclarations": [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ]
            }]

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{BASE}/{chosen}:generateContent",
                    json=payload,
                    headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                )
        except httpx.TimeoutException as exc:
            raise ProviderError("gemini: timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"gemini: {type(exc).__name__}", retryable=True) from exc

        if r.status_code >= 400:
            raise ProviderError(f"gemini: HTTP {r.status_code} {r.text[:160]}",
                                status=r.status_code,
                                retryable=r.status_code in (408, 429) or r.status_code >= 500)
        try:
            data = r.json()
        except ValueError as exc:
            raise ProviderError("gemini: response was not JSON", retryable=False) from exc

        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"gemini: unexpected shape {str(data)[:120]}",
                                retryable=False) from exc

        usage = data.get("usageMetadata") or {}
        return Completion(
            text=text,
            model=chosen,
            provider=self.name,
            tokens_in=int(usage.get("promptTokenCount", 0)),
            tokens_out=int(usage.get("candidatesTokenCount", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )
