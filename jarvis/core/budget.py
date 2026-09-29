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

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path


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
    """Thread-safe quota accounting across the daemon's request handlers."""

    def __init__(self, path: Path | None = None, limits: dict[str, Limits] | None = None):
        self._lock = threading.RLock()
        self._state: dict[str, ProviderState] = {}
        self._limits = limits if limits is not None else KNOWN_LIMITS
        self._path = path
        if path:
            self._load(path)

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
        tpm_used, tpd_used = st.token_totals()
        return {
            "rpm": None if lim.rpm is None else max(0, lim.rpm - len(st.requests_min)),
            "rpd": None if lim.rpd is None else max(0, lim.rpd - len(st.requests_day)),
            "tpm": None if lim.tpm is None else max(0, lim.tpm - tpm_used),
            "tpd": None if lim.tpd is None else max(0, lim.tpd - tpd_used),
        }

    def available(self, provider: str) -> bool:
        """Whether this provider can be tried right now."""
        now = time.time()
        st = self.state(provider)
        with self._lock:
            if st.cooldown_until > now:
                return False
            rem = self.remaining(provider)
        return all(v is None or v > 0 for v in rem.values())

    def cooldown_for(self, provider: str) -> float:
        return max(0.0, self.state(provider).cooldown_until - time.time())

    def exhausted(self, provider: str) -> str | None:
        """Which axis is spent, or None if the provider is usable."""
        now = time.time()
        st = self.state(provider)
        if st.cooldown_until > now:
            return "cooldown"
        rem = self.remaining(provider)
        for axis, val in rem.items():
            if val == 0:
                return axis
        return None

    # --- mutation ---------------------------------------------------------

    def record(self, provider: str, tokens: int = 0) -> None:
        now = time.time()
        with self._lock:
            st = self.state(provider)
            st.requests_min.append(now)
            st.requests_day.append(now)
            if tokens:
                st.tokens_min.append((now, tokens))
                st.tokens_day.append((now, tokens))
                st.total_tokens += tokens
            st.total_requests += 1
            st.last_used = now
            st.cooldown_until = 0.0
        self._persist()

    def penalize(self, provider: str, error: str, seconds: float = 60.0) -> None:
        """Put a provider in cooldown after a failure.

        429 and 402 mean the quota is gone, so back off far longer than a
        transient 500.
        """
        now = time.time()
        with self._lock:
            st = self.state(provider)
            st.errors += 1
            st.last_error = error[:200]
            st.cooldown_until = max(st.cooldown_until, now + seconds)
        self._persist()

    def reset(self, provider: str | None = None) -> None:
        with self._lock:
            if provider:
                self._state.pop(provider, None)
            else:
                self._state.clear()
        self._persist()

    # --- persistence ------------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            return {
                p: {
                    "total_requests": st.total_requests,
                    "total_tokens": st.total_tokens,
                    "errors": st.errors,
                    "last_error": st.last_error,
                    "cooldown_seconds": round(self.cooldown_for(p), 1),
                    "remaining": self.remaining(p),
                }
                for p, st in self._state.items()
            }

    def _persist(self) -> None:
        """Counters and failure state survive restarts.

        Persisting last_error matters: when the router quietly falls over to a
        different provider, the only way to find out why the preferred one was
        skipped is its recorded error, and that is gone if the daemon restarts.
        """
        if not self._path:
            return
        with self._lock:
            payload = {"saved_at": time.time(), "providers": {}}
            for p, st in self._state.items():
                cutoff = time.time() - 86_400
                payload["providers"][p] = {
                    "requests_day": [t for t in st.requests_day if t >= cutoff],
                    "tokens_day": [[t, n] for t, n in st.tokens_day if t >= cutoff],
                    "total_requests": st.total_requests,
                    "total_tokens": st.total_tokens,
                    "errors": st.errors,
                    "last_error": st.last_error,
                    "cooldown_until": st.cooldown_until,
                    "last_used": st.last_used,
                }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload), "utf-8")
        except OSError:
            pass  # accounting is best-effort; never fail a request over it

    def _load(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        cutoff = time.time() - 86_400
        for name, blob in (payload.get("providers") or {}).items():
            st = ProviderState()
            st.requests_day = deque(t for t in blob.get("requests_day", []) if t >= cutoff)
            st.tokens_day = deque(
                (float(t), int(n)) for t, n in blob.get("tokens_day", []) if t >= cutoff
            )
            st.total_requests = int(blob.get("total_requests", 0))
            st.total_tokens = int(blob.get("total_tokens", 0))
            st.errors = int(blob.get("errors", 0))
            st.last_error = str(blob.get("last_error", ""))
            st.last_used = float(blob.get("last_used", 0.0))
            # A cooldown that outlived the restart would silently keep a
            # recovered provider out of rotation, so drop expired ones and
            # honour the rest.
            st.cooldown_until = float(blob.get("cooldown_until", 0.0))
            self._state[name] = st
