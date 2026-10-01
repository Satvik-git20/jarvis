"""HTTP client for the JARVIS daemon.

The daemon is the single owner of conversational state: it holds the router,
the provider budget, and the session store, and it is the only process that
writes ``jarvis.db``. Everything else -- the voice loop, one-shot commands,
the Chrome panel, opencode -- is a client of this API. If the daemon is down,
the honest answer is "the daemon is down", not a silent write to a second
copy of the state.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .config import Settings, get_settings

DEFAULT_ASK_TIMEOUT = 300.0  # cold local model load plus a long generation


class DaemonError(RuntimeError):
    """Base class; the message is written for a human at a terminal."""


class DaemonDown(DaemonError):
    """The daemon is not reachable at all."""

    def __init__(self, detail: str = ""):
        message = "daemon unreachable -- start it with: uv run python -m jarvis daemon"
        super().__init__(f"{message} ({detail})" if detail else message)


class DaemonRejected(DaemonError):
    """Reachable, but our token was refused."""

    def __init__(self):
        super().__init__(
            "daemon rejected the token -- check JARVIS_DAEMON_TOKEN in .env "
            "and restart the daemon"
        )


class ProvidersExhausted(DaemonError):
    """The daemon answered 503: every provider is spent."""

    def __init__(self, attempts: list[tuple[str, str]], hint: str = ""):
        self.attempts = attempts
        self.hint = hint
        detail = "; ".join(f"{name}: {why}" for name, why in attempts)
        super().__init__(f"no provider could answer ({detail})" if detail else "no provider could answer")


@dataclass(frozen=True)
class Answer:
    """One completed turn, as returned by ``POST /ask``."""

    text: str
    provider: str
    model: str
    tokens: int
    latency_ms: int
    session_id: str


class DaemonClient:
    """Async client for one daemon; safe to share within a single event loop."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._settings = settings or get_settings()
        host = self._settings.daemon_host
        if host in ("0.0.0.0", "::"):  # bind-any addresses are not dial-able
            host = "127.0.0.1"
        self._client = httpx.AsyncClient(
            base_url=f"http://{host}:{self._settings.daemon_port}",
            headers={"X-Jarvis-Token": self._settings.daemon_token},
            transport=transport,
        )

    # --- read paths --------------------------------------------------------

    async def health(self) -> dict:
        return await self._request("GET", "/health")

    async def status(self) -> dict:
        return await self._request("GET", "/status")

    # --- the one call the CLI and one-shots need ---------------------------

    async def ask(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        system: str | None = None,
        prefer: str | None = None,
        remember: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float = DEFAULT_ASK_TIMEOUT,
    ) -> Answer:
        payload: dict = {"prompt": prompt, "remember": remember,
                         "temperature": temperature, "max_tokens": max_tokens,
                         "timeout": timeout}
        if session_id is not None:
            payload["session_id"] = session_id
        if system is not None:
            payload["system"] = system
        if prefer is not None:
            payload["prefer"] = prefer
        data = await self._request("POST", "/ask", json=payload, read_timeout=timeout)
        return Answer(
            text=str(data["text"]),
            provider=str(data["provider"]),
            model=str(data["model"]),
            tokens=int(data["tokens"]),
            latency_ms=int(data["latency_ms"]),
            session_id=str(data.get("session_id") or ""),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # --- plumbing -------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        read_timeout: float | None = None,
    ) -> dict:
        timeout = httpx.Timeout(10.0, read=read_timeout) if read_timeout else httpx.Timeout(10.0)
        try:
            res = await self._client.request(method, path, json=json, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise DaemonError(f"daemon did not answer within {read_timeout or 10:.0f}s") from exc
        except httpx.TransportError as exc:
            raise DaemonDown(str(exc)) from exc

        if res.status_code == 401:
            raise DaemonRejected()
        if res.status_code == 503:
            body = _json(res)
            attempts = [
                (str(pair[0]), str(pair[1])) if isinstance(pair, (list, tuple)) and len(pair) >= 2
                else (str(pair), "")
                for pair in body.get("attempts", [])
            ]
            raise ProvidersExhausted(attempts, hint=str(body.get("hint", "")))
        if res.status_code >= 400:
            body = _json(res)
            detail = body.get("detail", res.text)
            raise DaemonError(f"daemon returned HTTP {res.status_code}: {detail}")
        try:
            data = res.json()
        except ValueError as exc:
            raise DaemonError(f"daemon returned a non-JSON body: {res.text[:120]}") from exc
        if not isinstance(data, dict):
            raise DaemonError(f"daemon returned a malformed response: {res.text[:120]}")
        return data


def _json(res: httpx.Response) -> dict:
    try:
        data = res.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}
