"""Web search as a cascade: Tavily, then Jina, then DuckDuckGo.

Ordered by result quality per unit of quota, so the free tiers that cost you
nothing are spent last. Tavily returns pre-extracted clean content and is worth
its 1,000 monthly credits; Jina Reader is free but rate-limited per minute;
ddgs is keyless and will occasionally throttle, which is why it is the floor
rather than the entry point.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from jarvis.config import Settings, get_settings

log = logging.getLogger("jarvis.search")

UA = {"User-Agent": "JARVIS/0.1 (personal assistant)"}


def _contact_ua(settings: Settings) -> str:
    contact = settings.secret("JARVIS_CONTACT") or "set JARVIS_CONTACT in .env"
    return f"JARVIS/0.1 (personal assistant; contact: {contact})"


def _clean(text: str, limit: int) -> str:
    """Collapse whitespace so a result snippet is one readable line."""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


async def _tavily(client: httpx.AsyncClient, s: Settings, query: str, n: int) -> list[dict]:
    cap = s.by_name("tavily")
    key = s.credential(cap) if cap else None
    if not key:
        return []
    r = await client.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": n, "search_depth": "basic"},
        headers={"Authorization": f"Bearer {key}"},
    )
    r.raise_for_status()
    out = []
    for item in r.json().get("results", []):
        out.append({
            "title": _clean(item.get("title", ""), 160),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("content", ""), 600),
            "source": "tavily",
        })
    return out


async def _jina(client: httpx.AsyncClient, s: Settings, query: str, n: int) -> list[dict]:
    """Jina's search endpoint returns top results *with* full content."""
    cap = s.by_name("jina_reader")
    key = s.credential(cap) if cap else None
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    r = await client.get("https://s.jina.ai/", params={"q": query},
                         headers=headers, timeout=20.0)
    r.raise_for_status()
    data = r.json()
    results = data.get("data", data) if isinstance(data, dict) else data
    if isinstance(results, dict):
        results = results.get("results", [])
    out = []
    for item in (results or [])[:n]:
        if not isinstance(item, dict):
            continue
        out.append({
            "title": _clean(item.get("title", ""), 160),
            "url": item.get("url", ""),
            "snippet": _clean(item.get("description") or item.get("content", ""), 600),
            "source": "jina",
        })
    return out


async def _ddgs(query: str, n: int) -> list[dict]:
    """Keyless fallback. Runs the sync client in a thread so it cannot stall
    the event loop."""
    def _run() -> list[dict]:
        from ddgs import DDGS

        with DDGS() as ddg:
            rows = list(ddg.text(query, max_results=n))
        return [
            {
                "title": _clean(r.get("title", ""), 160),
                "url": r.get("href", ""),
                "snippet": _clean(r.get("body", ""), 600),
                "source": "ddgs",
            }
            for r in rows
        ]

    return await asyncio.to_thread(_run)


async def search_web(
    query: str,
    *,
    max_results: int = 5,
    fetch_content: bool = False,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Return search results, trying each backend until one yields anything."""
    s = settings or get_settings()
    n = max(1, min(max_results, 10))
    headers = {**UA, "User-Agent": _contact_ua(s)}

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        for name, fn in (("tavily", _tavily), ("jina", _jina)):
            try:
                got = await fn(client, s, query, n)
            except Exception as exc:
                log.info("search backend %s failed: %s", name, exc)
                continue
            if got:
                if fetch_content:
                    await _enrich(got)
                return got[:n]

    try:
        got = await _ddgs(query, n)
    except Exception as exc:
        log.info("search backend ddgs failed: %s", exc)
        got = []
    if got and fetch_content:
        await _enrich(got)
    return got[:n]


async def _enrich(results: list[dict[str, Any]]) -> None:
    """Replace short snippets with readable page text via Jina Reader.

    Best effort: a page that will not convert is left with its snippet rather
    than failing the whole search.
    """
    s = get_settings()
    cap = s.by_name("jina_reader")
    key = s.credential(cap) if cap else None
    headers = {"User-Agent": _contact_ua(s)}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
        for item in results:
            url = item.get("url", "")
            if not url:
                continue
            try:
                r = await client.get(f"https://r.jina.ai/{url}", timeout=25.0)
                if r.status_code == 200:
                    item["content"] = r.text[:4000]
            except Exception:
                continue
