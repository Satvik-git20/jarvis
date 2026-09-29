"""The one interface every free-AI backend implements.

Three of the five brains speak the OpenAI chat-completions shape, so they share
a single implementation and differ only by base URL, auth header, and model
list. Ollama speaks a different dialect and Gemini a third; both get their own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx


class ProviderError(RuntimeError):
    """A provider failed in a way the router should route around.

    `retryable` separates "this provider is out of quota" from "you sent
    something malformed", because only the first is worth cooling down.
    """

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable

    @property
    def cooldown(self) -> float:
        """How long to skip this provider.

        Quota errors park it for a long window; everything else is brief.
        """
        if self.status in (402, 429):
            return 900.0
        if self.status and 500 <= self.status < 600:
            return 30.0
        return 15.0


@dataclass
class Message:
    role: str
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out


@runtime_checkable
class Provider(Protocol):
    """What the router needs from any backend."""

    name: str

    def healthy(self) -> bool:
        """Whether this provider is configured and worth trying."""
        ...

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
        """Run one chat completion, or raise ProviderError."""
        ...


# --- shared plumbing --------------------------------------------------------


def parse_openai_response(name: str, model: str, data: dict[str, Any], started: float) -> Completion:
    """Turn an OpenAI-shaped chat completion into a Completion."""
    try:
        text = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"{name}: unexpected response shape: {str(data)[:120]}",
                            retryable=False) from exc
    usage = data.get("usage") or {}
    return Completion(
        text=text,
        model=data.get("model", model),
        provider=name,
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
        latency_ms=int((time.perf_counter() - started) * 1000),
        raw=data,
    )


def openai_payload(
    model: str,
    messages: list[Message],
    *,
    temperature: float,
    max_tokens: int,
    tools: list[ToolSpec] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [m.to_openai() for m in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.parameters}}
            for t in tools
        ]
        payload["tool_choice"] = "auto"
    return payload


async def post_json(
    client: httpx.AsyncClient,
    provider: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    started: float,
) -> dict[str, Any]:
    """POST and translate transport/HTTP failures into ProviderError."""
    try:
        resp = await client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise ProviderError(f"{provider}: timed out", retryable=True) from exc
    except httpx.HTTPError as exc:
        raise ProviderError(f"{provider}: {type(exc).__name__}", retryable=True) from exc
    if resp.status_code >= 400:
        # 4xx other than 401/403/408/429 means our request was wrong, so do
        # not park the provider.
        retryable = resp.status_code in (408, 429) or resp.status_code >= 500 or resp.status_code in (401, 403)
        raise ProviderError(
            f"{provider}: HTTP {resp.status_code} {resp.text[:160]}",
            status=resp.status_code,
            retryable=retryable,
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(f"{provider}: response was not JSON", retryable=False) from exc
