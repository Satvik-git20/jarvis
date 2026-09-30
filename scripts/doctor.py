"""Health-check every JARVIS integration.

Run this before trusting anything. It answers three questions per service:
do we have a credential, does the endpoint answer, and is it actually usable.

Usage:
    uv run python scripts/doctor.py
    uv run python scripts/doctor.py --key-only     # no network calls
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from jarvis.config import REGISTRY, get_settings

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

WIKIPEDIA = (
    "https://en.wikipedia.org/w/api.php?action=query&list=search"
    "&srsearch=JARVIS&format=json&srlimit=1"
)
HN = "https://hn.algolia.com/api/v1/search?query=test&tags=story&hitsPerPage=1"
OPEN_METEO_GEO = "https://geocoding-api.open-meteo.com/v1/search?name=Berlin&count=1"
DDG_HTML = "https://html.duckduckgo.com/html/?q=test"
JINA_READER = "https://r.jina.ai/https://example.com"
POLLINATIONS = "https://gen.pollinations.ai/image/prompt/ping?model=flux&width=64&height=64"
OLLAMA_TAGS = "/api/tags"

# Wikipedia returns 403 unless the User-Agent carries a way to contact the
# client, per their UA policy. Their edge rejects a descriptive-but-anonymous
# UA and a browser UA alike; only one with contact info is accepted. Set
# JARVIS_CONTACT in .env to your email or a URL you actually answer to.
CONTACT = os.environ.get("JARVIS_CONTACT", "").strip() or "set JARVIS_CONTACT in .env"
UA = {"User-Agent": f"JARVIS/0.1 (personal assistant; contact: {CONTACT})"}


def client_factory(**kw) -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, headers=UA, **kw)


async def probe(
    client: httpx.AsyncClient, url: str, *, timeout: float = 8.0, expect_json: bool = False
) -> tuple[bool, str]:
    """GET with one retry.

    Wikipedia in particular returns 403 when a client bursts several requests
    from one IP, which reads as a hard block but is not. One short retry
    separates "throttled" from "actually broken".
    """
    start = time.perf_counter()
    last = ""
    for attempt in range(2):
        if attempt:
            await asyncio.sleep(1.5)
        try:
            r = await client.get(url, timeout=timeout)
            ms = int((time.perf_counter() - start) * 1000)
            if r.status_code >= 400:
                last = f"HTTP {r.status_code} ({ms}ms)"
                if r.status_code in (403, 429, 503):
                    continue
                return False, last
            if expect_json:
                r.json()
            return True, f"ok ({ms}ms)"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:60]}"
            if attempt:
                return False, last
    return False, f"{last} (retried)"


async def check(kind: str, cap_name: str, url_fn, key_only: bool, label: str) -> tuple[str, str, str]:
    """Return (status_char, name, detail) for one capability."""
    s = get_settings()
    cap = s.by_name(cap_name)
    if cap is None:
        return "?", cap_name, "not in registry"

    if cap.privacy_unsafe and s.exclude_privacy_unsafe:
        return "-", cap.label, "excluded by JARVIS_EXCLUDE_PRIVACY_UNSAFE"

    has_cred = s.available(cap)
    if cap.requires_key and not has_cred:
        return "x", cap.label, "no key set"

    if key_only:
        return ("+" if has_cred else "?"), cap.label, ("credential present" if cap.requires_key else "no key needed")

    async with client_factory() as client:
        ok, detail = await url_fn(client, s)

    if not ok:
        return ("!" if has_cred else "x"), cap.label, detail
    return "+", cap.label, f"{cap.free_limit} - {detail}" if cap.free_limit else detail


def _llm_probe(cap_name: str):
    def url_fn(client: httpx.AsyncClient, s):
        if cap_name == "ollama":
            return probe(client, s.ollama_host + OLLAMA_TAGS, expect_json=True)
        base = {
            "openrouter": "https://openrouter.ai/api/v1/models",
            "groq": "https://api.groq.com/openai/v1/models",
            "cerebras": "https://api.cerebras.ai/v1/models",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
        }[cap_name]
        key = s.credential(s.by_name(cap_name))
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return probe(client, base, expect_json=True, timeout=10.0) if not headers else _authed(
            client, base, headers
        )

    return url_fn


async def _authed(client: httpx.AsyncClient, url: str, headers: dict) -> tuple[bool, str]:
    start = time.perf_counter()
    try:
        r = await client.get(url, headers=headers, timeout=10.0)
        ms = int((time.perf_counter() - start) * 1000)
        if r.status_code in (401, 403):
            return False, f"HTTP {r.status_code} - key rejected"
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}"
        return True, f"auth ok ({ms}ms)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


PROBES = {
    "ollama": lambda c, s: probe(c, s.ollama_host + OLLAMA_TAGS, expect_json=True),
    "openrouter": lambda c, s: _authed(c, "https://openrouter.ai/api/v1/models",
                                       {"Authorization": f"Bearer {s.credential(s.by_name('openrouter'))}"}),
    "groq": lambda c, s: _authed(c, "https://api.groq.com/openai/v1/models",
                                 {"Authorization": f"Bearer {s.credential(s.by_name('groq'))}"}),
    "cerebras": lambda c, s: _authed(c, "https://api.cerebras.ai/v1/models",
                                      {"Authorization": f"Bearer {s.credential(s.by_name('cerebras'))}"}),
    "gemini": lambda c, s: _authed(c, "https://generativelanguage.googleapis.com/v1beta/models",
                                    {"x-goog-api-key": s.credential(s.by_name("gemini")) or ""}),
    "jina_reader": lambda c, s: probe(c, JINA_READER, timeout=15.0),
    "ddgs": lambda c, s: probe(c, DDG_HTML, timeout=10.0),
    "tavily": lambda c, s: _authed_post(c, "https://api.tavily.com/search", s, "tavily"),
    "pollinations": lambda c, s: probe(c, POLLINATIONS, timeout=25.0),
    "open_meteo": lambda c, s: probe(c, OPEN_METEO_GEO, expect_json=True),
    "wikipedia": lambda c, s: probe(c, WIKIPEDIA, expect_json=True),
    "hn_algolia": lambda c, s: probe(c, HN, expect_json=True),
}


async def _authed_post(client: httpx.AsyncClient, url: str, s, cap_name: str):
    key = s.credential(s.by_name(cap_name))
    start = time.perf_counter()
    try:
        r = await client.post(url, json={"query": "test"}, headers={"Authorization": f"Bearer {key}"}, timeout=10.0)
        ms = int((time.perf_counter() - start) * 1000)
        if r.status_code in (401, 403):
            return False, f"HTTP {r.status_code} - key rejected"
        if r.status_code >= 400:
            return False, f"HTTP {r.status_code}"
        return True, f"auth ok ({ms}ms)"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:60]}"


ICON = {"+": f"{GREEN}OK  {RESET}", "!": f"{YELLOW}WARN{RESET}", "x": f"{RED}FAIL{RESET}",
        "-": f"{DIM}SKIP{RESET}", "?": f"{DIM}INFO{RESET}"}

# Where to get each missing key. All of these free tiers need no credit card.
SIGNUP = {
    "groq":        "https://console.groq.com/keys",
    "cerebras":    "https://cloud.cerebras.ai",
    "gemini":      "https://aistudio.google.com/apikey",
    "tavily":      "https://app.tavily.com",
    "pollinations": "https://enter.pollinations.ai",
    "jina_embed":  "https://jina.ai/embeddings",
    "jina_reader": "https://jina.ai/reader (optional, raises 20->500 rpm)",
    "openrouter":  "https://openrouter.ai/keys (optional: the opencode fallback already supplies this)",
}

ENV_NAME = {
    "groq": "GROQ_API_KEY", "cerebras": "CEREBRAS_API_KEY", "gemini": "GEMINI_API_KEY",
    "tavily": "TAVILY_API_KEY", "pollinations": "POLLINATIONS_KEY", "jina_embed": "JINA_API_KEY",
    "jina_reader": "JINA_API_KEY", "openrouter": "OPENROUTER_API_KEY",
    # Shares the Groq key, so it dedupes against the `groq` row above.
    "groq_stt": "GROQ_API_KEY",
}

KIND_LABEL = {
    "llm": "Brains (LLM)", "stt": "Voice in", "tts": "Voice out",
    "search": "Search", "embed": "Embeddings", "image": "Image",
    "ocr": "OCR", "weather": "Utilities", "knowledge": "Knowledge",
}


async def main(argv: list[str] | None = None) -> int:
    """Accept argv so `python -m jarvis doctor` can pass an explicit empty list.

    Re-parsing sys.argv would see the word `doctor` as an unknown argument.
    """
    ap = argparse.ArgumentParser(prog="jarvis doctor")
    ap.add_argument("--key-only", action="store_true", help="skip network probes")
    args = ap.parse_args(argv)

    s = get_settings()
    s.ensure_data_dir()
    print(f"\nJARVIS doctor  {DIM}(data: {s.data_dir}){RESET}\n")

    results: list[tuple[str, str, str, str]] = []
    for cap in REGISTRY:
        probe_fn = PROBES.get(cap.name)
        if cap.name in ("whisper_local", "kokoro", "ollama_embed", "rapidocr", "edge_tts", "groq_stt"):
            status, label, detail = _local_status(cap.name, args.key_only)
        elif probe_fn is None:
            status, label, detail = ("+" if s.available(cap) else "?"), cap.label, "credential present" if s.available(cap) else "no key"
        else:
            status, label, detail = await check(cap.kind, cap.name, probe_fn, args.key_only, cap.label)
        results.append((status, cap.kind, label, detail))

    last_kind = None
    for status, kind, label, detail in results:
        if kind != last_kind:
            print(f"{KIND_LABEL.get(kind, kind)}")
            last_kind = kind
        print(f"  {ICON.get(status, '?')}  {label:<32} {DIM}{detail}{RESET}")

    # Dedupe by env var: several capabilities share one key (groq_stt reuses
    # GROQ_API_KEY), and listing the same signup twice is just noise.
    seen_env: set[str] = set()
    missing = []
    for cap in REGISTRY:
        if not cap.requires_key or s.available(cap):
            continue
        marker = ENV_NAME.get(cap.name) or cap.name
        if marker in seen_env:
            continue
        seen_env.add(marker)
        missing.append(cap)

    ok = sum(1 for r in results if r[0] == "+")
    fail = sum(1 for r in results if r[0] == "x")
    warn = sum(1 for r in results if r[0] == "!")

    if missing:
        print(f"\n{BOLD}To switch these on{RESET}  {DIM}(all free, no card){RESET}")
        for cap in missing:
            env = ENV_NAME.get(cap.name, "")
            url = SIGNUP.get(cap.name, "")
            print(f"  {YELLOW}{cap.name:<14}{RESET} {DIM}{env:<18}{RESET} {url}")
        print(f"\n  {DIM}add the value to .env, then re-run the doctor. Nothing here is")
        print(f"  required: keyless services already work.{RESET}")

    print(f"\n  {GREEN}{ok} ok{RESET}  {YELLOW}{warn} warn{RESET}  {RED}{fail} unavailable\n")
    return 0


def _local_status(name: str, key_only: bool):
    """Local extras are not installed yet in early phases."""
    labels = {
        "whisper_local": "faster-whisper (local)", "kokoro": "Kokoro-82M (local)",
        "ollama_embed": "Ollama nomic-embed-text", "rapidocr": "RapidOCR (local)",
        "edge_tts": "Microsoft Edge TTS", "groq_stt": "Groq whisper-large-v3-turbo",
    }
    s = get_settings()
    if key_only:
        return "?", labels[name], "local, installed in a later phase"
    if name == "ollama_embed":
        return "?", labels[name], f"model {s.ollama_embed_model} (pull it with: ollama pull {s.ollama_embed_model})"
    if name == "groq_stt":
        return ("+" if s.available(s.by_name("groq")) else "x"), labels[name], "same key as Groq"
    return "?", labels[name], "local, installed in a later phase"


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
