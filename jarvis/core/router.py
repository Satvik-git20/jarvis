"""The router: pick a brain, fail over when it cannot answer.

The rule that matters is that exhausting a free tier must not end the request.
Every provider here is bounded, so a 429 from one is normal operation, not an
error worth surfacing. The router records it, cools that provider down, and
moves to the next one -- only reporting failure when all of them are spent.
"""

from __future__ import annotations

import logging

from .budget import BudgetLedger
from .providers.base import Completion, Message, Provider, ProviderError, ToolSpec
from .providers.registry import FAILOVER_ORDER, build_providers

log = logging.getLogger("jarvis.router")


class AllProvidersExhausted(RuntimeError):
    """Every configured brain refused. Carries why, for /status."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts
        detail = "; ".join(f"{n}: {why}" for n, why in attempts)
        super().__init__(f"no provider could answer ({detail})")


class Router:
    def __init__(self, ledger: BudgetLedger, settings=None):
        self.ledger = ledger
        self._settings = settings
        self._providers: dict[str, Provider] = build_providers(settings)
        self.last_used: str | None = None

    @property
    def providers(self) -> dict[str, Provider]:
        return self._providers

    def ready(self) -> list[str]:
        """Providers that are configured and not currently blocked."""
        return [n for n in FAILOVER_ORDER
                if n in self._providers
                and self._providers[n].healthy()
                and self.ledger.available(n)]

    def blocked(self) -> dict[str, str]:
        """Why each unavailable provider is unavailable, for /status."""
        out: dict[str, str] = {}
        for name in FAILOVER_ORDER:
            p = self._providers.get(name)
            if p is None:
                continue
            if not p.healthy():
                out[name] = "no key configured"
            elif reason := self.ledger.exhausted(name):
                out[name] = f"{reason} exhausted"
        return out

    async def complete(
        self,
        messages: list[Message],
        *,
        prefer: str | None = None,
        model: str | None = None,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> Completion:
        order = list(FAILOVER_ORDER)
        if prefer and prefer in order:
            # An explicit request goes first but is not exclusive: if the
            # caller names a provider, honour it, then still fail over.
            order.remove(prefer)
            order.insert(0, prefer)

        attempts: list[tuple[str, str]] = []
        for name in order:
            provider = self._providers.get(name)
            if provider is None:
                continue
            if not provider.healthy():
                attempts.append((name, "not configured"))
                continue
            reason = self.ledger.exhausted(name)
            if reason:
                attempts.append((name, f"{reason} exhausted"))
                continue

            try:
                result = await provider.complete(
                    messages, model=model, tools=tools, temperature=temperature,
                    max_tokens=max_tokens, timeout=timeout,
                )
            except ProviderError as exc:
                attempts.append((name, str(exc)[:80]))
                if exc.retryable:
                    self.ledger.penalize(name, str(exc), exc.cooldown)
                log.warning("provider %s failed: %s", name, exc)
                continue

            # An empty completion is a failure, not an answer. Non-chat models
            # on a free roster (classifiers, embedding endpoints) return 200
            # with no text, and accepting that would end the request with
            # silence instead of trying the next provider.
            if not result.text.strip():
                attempts.append((name, "empty completion"))
                log.warning("provider %s returned an empty completion; skipping", name)
                # Do not penalise: the provider is healthy, it just is not a
                # chat model. A cooldown would wrongly disable it.
                continue

            self.ledger.record(name, result.total_tokens)
            self.last_used = name
            return result

        raise AllProvidersExhausted(attempts)
