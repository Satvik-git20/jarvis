"""Durable provider usage and failure-event persistence."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..database import Database


@dataclass(frozen=True)
class ProviderUsage:
    requests_minute: int
    requests_day: int
    tokens_minute: int
    tokens_day: int
    total_requests: int
    total_tokens: int
    errors: int
    last_error: str
    cooldown_until: float


class ProviderLogRepository:
    def __init__(self, database: Database):
        self._db = database

    def record_completion(self, provider: str, tokens: int = 0) -> None:
        self._append(provider, "completion", request_count=1, token_count=max(0, tokens))

    def record_error(self, provider: str, error: str, cooldown_until: float) -> None:
        self._append(provider, "error", error_count=1, error_text=error[:200], cooldown_until=cooldown_until)

    def usage(self, provider: str, *, now: float | None = None) -> ProviderUsage:
        now = time.time() if now is None else now
        minute, day = now - 60, now - 86_400
        with self._db.read() as conn:
            rolling = conn.execute(
                "SELECT "
                "COALESCE(SUM(CASE WHEN event_type = 'completion' AND occurred_at >= ? THEN request_count ELSE 0 END), 0), "
                "COALESCE(SUM(CASE WHEN event_type = 'completion' AND occurred_at >= ? THEN request_count ELSE 0 END), 0), "
                "COALESCE(SUM(CASE WHEN event_type = 'completion' AND occurred_at >= ? THEN token_count ELSE 0 END), 0), "
                "COALESCE(SUM(CASE WHEN event_type = 'completion' AND occurred_at >= ? THEN token_count ELSE 0 END), 0), "
                "COALESCE(SUM(request_count), 0), COALESCE(SUM(token_count), 0), COALESCE(SUM(error_count), 0) "
                "FROM provider_logs WHERE provider = ?",
                (minute, day, minute, day, provider),
            ).fetchone()
            last_error = conn.execute(
                "SELECT error_text FROM provider_logs WHERE provider = ? AND error_text != '' ORDER BY id DESC LIMIT 1",
                (provider,),
            ).fetchone()
            last_state = conn.execute(
                "SELECT cooldown_until FROM provider_logs WHERE provider = ? ORDER BY id DESC LIMIT 1",
                (provider,),
            ).fetchone()
        return ProviderUsage(
            *(int(value) for value in rolling[:7]),
            last_error=str(last_error[0]) if last_error else "",
            cooldown_until=float(last_state[0]) if last_state else 0.0,
        )

    def providers(self) -> set[str]:
        with self._db.read() as conn:
            rows = conn.execute("SELECT DISTINCT provider FROM provider_logs").fetchall()
        return {str(row[0]) for row in rows}

    def clear(self, provider: str | None = None) -> None:
        with self._db.transaction() as conn:
            if provider:
                conn.execute("DELETE FROM provider_logs WHERE provider = ?", (provider,))
            else:
                conn.execute("DELETE FROM provider_logs")

    def import_legacy_json(self, path: Path) -> None:
        if not path.is_file() or self._already_imported(path):
            return
        try:
            payload = json.loads(path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        providers = payload.get("providers", {}) if isinstance(payload, dict) else {}
        if not isinstance(providers, dict):
            return
        with self._db.transaction() as conn:
            for provider, state in providers.items():
                if not isinstance(provider, str) or not isinstance(state, dict):
                    continue
                requests = [float(value) for value in state.get("requests_day", [])]
                token_by_time = {float(ts): int(tokens) for ts, tokens in state.get("tokens_day", [])}
                for occurred_at in requests:
                    conn.execute(
                        "INSERT INTO provider_logs(provider, event_type, occurred_at, request_count, token_count) VALUES (?, 'completion', ?, 1, ?)",
                        (provider, occurred_at, token_by_time.pop(occurred_at, 0)),
                    )
                for occurred_at, tokens in token_by_time.items():
                    conn.execute(
                        "INSERT INTO provider_logs(provider, event_type, occurred_at, token_count) VALUES (?, 'completion', ?, ?)",
                        (provider, occurred_at, tokens),
                    )
                total_requests = max(0, int(state.get("total_requests", 0)) - len(requests))
                total_tokens = max(0, int(state.get("total_tokens", 0)) - sum(int(x) for x in state.get("tokens_day", []) for x in [x[1]]))
                total_errors = max(0, int(state.get("errors", 0)))
                if total_requests or total_tokens or total_errors:
                    conn.execute(
                        "INSERT INTO provider_logs(provider, event_type, occurred_at, request_count, token_count, error_count) "
                        "VALUES (?, 'legacy_summary', ?, ?, ?, ?)",
                        (provider, float(state.get("last_used", time.time())), total_requests, total_tokens, total_errors),
                    )
                last_error = str(state.get("last_error", ""))[:200]
                cooldown = float(state.get("cooldown_until", 0.0))
                conn.execute(
                    "INSERT INTO provider_logs(provider, event_type, occurred_at, error_text, cooldown_until) VALUES (?, 'legacy_state', ?, ?, ?)",
                    (provider, time.time(), last_error, cooldown),
                )
            self._mark_imported(conn, path)

    def _append(
        self,
        provider: str,
        event_type: str,
        *,
        request_count: int = 0,
        token_count: int = 0,
        error_count: int = 0,
        error_text: str = "",
        cooldown_until: float = 0.0,
    ) -> None:
        with self._db.transaction() as conn:
            conn.execute(
                "INSERT INTO provider_logs(provider, event_type, occurred_at, request_count, token_count, error_count, error_text, cooldown_until) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (provider, event_type, time.time(), request_count, token_count, error_count, error_text, cooldown_until),
            )

    def _already_imported(self, path: Path) -> bool:
        with self._db.read() as conn:
            return bool(conn.execute("SELECT 1 FROM legacy_imports WHERE source_path = ?", (str(path.resolve()),)).fetchone())

    @staticmethod
    def _mark_imported(conn, path: Path) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO legacy_imports(source_path, imported_at) VALUES (?, ?)",
            (str(path.resolve()), time.time()),
        )
