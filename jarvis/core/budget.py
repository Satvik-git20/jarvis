"""Per-provider rate-limit accounting.

Every free tier is bounded, and the bounds differ in kind: OpenRouter counts
requests per minute and per day, Groq also caps tokens per day, Cerebras caps
tokens per minute. A router that ignores this will happily burn its whole daily
budget on one provider and then fail over with nothing left.

The ledger is optimistic: it records what we send and trusts the provider's
429 to correct us. That is the right bias, because a counter that guesses low
would throttle us before the provider actually would.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..storage import Database
from ..storage.repositories import ProviderLogRepository, ProviderUsage


@dataclass
class Limits:
    """Published free-tier ceilings. None means unbounded for that axis."""

    rpm: int | None = None
    rpd: int | None = None
    tpm: int | None = None
    tpd: int | None = None


# Free-tier ceilings as published by each provider. Kept in one place so
# /status can explain *why* a provider was skipped.
KNOWN_LIMITS: dict[str, Limits] = {
    # 20 rpm; 50/day until $10 of lifetime credits, then 1000/day.
    "openrouter": Limits(rpm=20, rpd=50),
    # 30 rpm, 1000/day, 8000 tpm, 200k tpd. The token cap binds before rpd
    # for anything but very short prompts.
    "groq": Limits(rpm=30, rpd=1000, tpm=8_000, tpd=200_000),
    # 30 rpm, 14400/day, ~60k tpm, 1M tpd.
    "cerebras": Limits(rpm=30, rpd=14_400, tpm=60_000, tpd=1_000_000),
    # Per-project and not published; we self-throttle to stay polite.
    "gemini": Limits(rpm=10, rpd=1_000),
    "ollama": Limits(),  # local, no quota
}


@dataclass
class ProviderState:
    """Rolling counters plus cooldown state for one provider."""

    requests_min: deque[float] = field(default_factory=deque)
    requests_day: deque[float] = field(default_factory=deque)
    tokens_min: deque[tuple[float, int]] = field(default_factory=deque)
    tokens_day: deque[tuple[float, int]] = field(default_factory=deque)
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    errors: int = 0
    last_error: str = ""
    last_used: float = 0.0

    def prune(self, now: float) -> None:
        while self.requests_min and now - self.requests_min[0] > 60:
            self.requests_min.popleft()
        while self.tokens_min and now - self.tokens_min[0][0] > 60:
            self.tokens_min.popleft()
        cutoff = now - 86_400
        while self.requests_day and self.requests_day[0] < cutoff:
            self.requests_day.popleft()
        while self.tokens_day and self.tokens_day[0][0] < cutoff:
            self.tokens_day.popleft()

    def token_totals(self) -> tuple[int, int]:
        return sum(t for _, t in self.tokens_min), sum(t for _, t in self.tokens_day)


class BudgetLedger:
    """Quota accounting backed by transactional provider event logs.

    The constructor accepts the historic ``budget.json`` path for backwards
    compatibility. The file is imported once and the sibling ``jarvis.db`` is
    authoritative from then on.
    """

    def __init__(self, path: Path | None = None, limits: dict[str, Limits] | None = None):
        self._lock = threading.RLock()
        self._state: dict[str, ProviderState] = {}
        self._limits = limits if limits is not None else KNOWN_LIMITS
        self._path = path
        self._repository = ProviderLogRepository(Database(path)) if path else None
        if path and path.suffix.lower() == ".json":
            self._repository.import_legacy_json(path)

    # --- introspection ----------------------------------------------------

    def limits(self, provider: str) -> Limits:
        return self._limits.get(provider, Limits())

    def state(self, provider: str) -> ProviderState:
        with self._lock:
            return self._state.setdefault(provider, ProviderState())

    def remaining(self, provider: str) -> dict[str, int | None]:
        """How much of each ceiling is left. None means unbounded."""
        now = time.time()
        st, lim = self.state(provider), self.limits(provider)
        st.prune(now)
        usage = self._usage(provider, now)
        manual_tpm, manual_tpd = st.token_totals()
        return {
            "rpm": None if lim.rpm is None else max(0, lim.rpm - usage.requests_minute - len(st.requests_min)),
            "rpd": None if lim.rpd is None else max(0, lim.rpd - usage.requests_day - len(st.requests_day)),
            "tpm": None if lim.tpm is None else max(0, lim.tpm - usage.tokens_minute - manual_tpm),
            "tpd": None if lim.tpd is None else max(0, lim.tpd - usage.tokens_day - manual_tpd),
        }

    def available(self, provider: str) -> bool:
        """Whether this provider can be tried right now."""
        if self.cooldown_for(provider) > 0:
            return False
        rem = self.remaining(provider)
        return all(v is None or v > 0 for v in rem.values())

    def cooldown_for(self, provider: str) -> float:
        manual = self.state(provider).cooldown_until
        return max(0.0, max(manual, self._usage(provider, time.time()).cooldown_until) - time.time())

    def exhausted(self, provider: str) -> str | None:
        """Which axis is spent, or None if the provider is usable."""
        now = time.time()
        st = self.state(provider)
        if max(st.cooldown_until, self._usage(provider, now).cooldown_until) > now:
            return "cooldown"
        rem = self.remaining(provider)
        for axis, val in rem.items():
            if val == 0:
                return axis
        return None

    # --- mutation ---------------------------------------------------------

    def record(self, provider: str, tokens: int = 0) -> None:
        if self._repository:
            self._repository.record_completion(provider, tokens)
            return
        self._record_memory(provider, tokens)

    def penalize(self, provider: str, error: str, seconds: float = 60.0) -> None:
        """Put a provider in cooldown after a failure.

        429 and 402 mean the quota is gone, so back off far longer than a
        transient 500.
        """
        if self._repository:
            self._repository.record_error(provider, error, time.time() + seconds)
            return
        st = self.state(provider)
        st.errors += 1
        st.last_error = error[:200]
        st.cooldown_until = max(st.cooldown_until, time.time() + seconds)

    def reset(self, provider: str | None = None) -> None:
        with self._lock:
            if provider:
                self._state.pop(provider, None)
            else:
                self._state.clear()
        if self._repository:
            self._repository.clear(provider)

    # --- persistence ------------------------------------------------------

    def snapshot(self) -> dict:
        names = set(self._state)
        if self._repository:
            names.update(self._repository.providers())
        return {
            provider: self._snapshot_provider(provider)
            for provider in names
        }

    def _usage(self, provider: str, now: float):
        if self._repository:
            return self._repository.usage(provider, now=now)
        # No path: pure in-memory mode. Rolling counters live in self._state
        # and are added by the callers, so the durable slice is all zeros.
        return ProviderUsage(0, 0, 0, 0, 0, 0, 0, "", 0.0)

    def _snapshot_provider(self, provider: str) -> dict:
        now = time.time()
        usage, manual = self._usage(provider, now), self.state(provider)
        return {
            "total_requests": usage.total_requests + manual.total_requests,
            "total_tokens": usage.total_tokens + manual.total_tokens,
            "errors": usage.errors + manual.errors,
            "last_error": usage.last_error or manual.last_error,
            "cooldown_seconds": round(self.cooldown_for(provider), 1),
            "remaining": self.remaining(provider),
        }

    def _record_memory(self, provider: str, tokens: int) -> None:
        now, st = time.time(), self.state(provider)
        st.requests_min.append(now)
        st.requests_day.append(now)
        if tokens:
            st.tokens_min.append((now, tokens))
            st.tokens_day.append((now, tokens))
            st.total_tokens += tokens
        st.total_requests += 1
        st.last_used = now
        st.cooldown_until = 0.0
