"""Ollama - the local brain.

Kept first in the failover order because it is the only provider with no quota
and no data leaving the machine. It also speaks its own dialect, so it does not
share the OpenAI-shaped path.
"""

from __future__ import annotations

import time

import httpx

from jarvis.config import Settings, get_settings

from .base import Completion, Message, ProviderError, ToolSpec, parse_openai_response


class Ollama:
    name = "ollama"

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    def healthy(self) -> bool:
        # Local and free; only unusable if the daemon is down, which the router
        # discovers on first call and treats as a skip.
        return True

    @property
    def model(self) -> str:
        return self._settings.ollama_chat_model

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 300.0,
    ) -> Completion:
        # Local generation of a 4B model on a 1650 is slow but not failed;
        # the ceiling is generous on purpose.
        started = time.perf_counter()
        chosen = model or self.model
        payload: dict = {
            "model": chosen,
            "messages": [m.to_openai() for m in messages],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [
                {"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters}}
                for t in tools
            ]
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{self._settings.ollama_host}/api/chat", json=payload
                )
        except httpx.ConnectError as exc:
            raise ProviderError("ollama: not running on "
                                f"{self._settings.ollama_host}", retryable=False) from exc
        except httpx.TimeoutException as exc:
            raise ProviderError("ollama: timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"ollama: {type(exc).__name__}", retryable=True) from exc

        if r.status_code >= 400:
            raise ProviderError(f"ollama: HTTP {r.status_code} {r.text[:160]}",
                                status=r.status_code,
                                retryable=r.status_code in (408, 429) or r.status_code >= 500)
        try:
            data = r.json()
        except ValueError as exc:
            raise ProviderError("ollama: response was not JSON", retryable=False) from exc

        # Normalise onto the OpenAI shape so one parser handles every provider.
        message = data.get("message") or {}
        normalised = {
            "model": chosen,
            "choices": [{"message": {"content": message.get("content", "")}}],
            "usage": {
                "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                "completion_tokens": int(data.get("eval_count", 0)),
            },
        }
        return parse_openai_response(self.name, chosen, normalised, started)

    async def list_models(self, timeout: float = 5.0) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self._settings.ollama_host}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, KeyError, ValueError):
            return []

    async def has_model(self, name: str | None = None) -> bool:
        want = name or self.model
        return want in await self.list_models()
