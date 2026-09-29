"""Real-socket smoke test for the daemon.

TestClient bypasses uvicorn, the loopback bind, and header parsing. This boots
the actual server and drives it over TCP, because the whole security model
depends on the socket really being loopback-only and really requiring a token.

Self-contained: starts, probes, and stops in one process. Usage:

    uv run python scripts/smoke_daemon.py
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
HOST, PORT = "127.0.0.1", 8765
BASE = f"http://{HOST}:{PORT}"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"  {GREEN + 'PASS' + RESET if ok else RED + 'FAIL' + RESET}  {name}"
          f"  {DIM}{detail}{RESET}")


def attempt(name: str, fn) -> None:
    """Run one probe without letting a timeout or transport error abort the
    whole run -- a single unreachable endpoint should not hide every result
    after it."""
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {str(exc)[:70]}"
    check(name, ok, detail)


def status_of(fn) -> tuple[bool, str]:
    code = fn()
    return code == 200, f"HTTP {code}"


def exact(expected: int):
    def probe(fn):
        def run():
            code = fn()
            return code == expected, f"HTTP {code} (want {expected})"
        return run
    return probe


def token() -> str:
    for line in (REPO / ".env").read_text("utf-8").splitlines():
        if line.startswith("JARVIS_DAEMON_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("JARVIS_DAEMON_TOKEN not set in .env")


def wait_for_server(proc: subprocess.Popen, tries: int = 40) -> bool:
    for _ in range(tries):
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            time.sleep(0.5)
    return False


def main() -> int:
    tok = token()
    python = REPO / ".venv" / "Scripts" / "python.exe"

    print(f"\nBooting daemon on {BASE}")
    proc = subprocess.Popen(
        [str(python), "-m", "jarvis.daemon"],
        cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_for_server(proc):
            out = proc.stdout.read() if proc.stdout else ""
            print(f"{RED}daemon did not start{RESET}\n{out}")
            return 1

        auth = {"X-Jarvis-Token": tok}
        print("\nAuth boundary")
        attempt("/health needs no token",
                exact(200)(lambda: httpx.get(f"{BASE}/health", timeout=5).status_code))
        attempt("/status without token is 401",
                exact(401)(lambda: httpx.get(f"{BASE}/status", timeout=5).status_code))
        attempt("/status with wrong token is 401",
                exact(401)(lambda: httpx.get(f"{BASE}/status", headers={"X-Jarvis-Token": "x"},
                                             timeout=5).status_code))
        attempt("/ask without token is 401",
                exact(401)(lambda: httpx.post(f"{BASE}/ask", json={"prompt": "hi"},
                                              timeout=5).status_code))

        print("\nReal status")
        attempt("GET /status with token is 200",
                exact(200)(lambda: httpx.get(f"{BASE}/status", headers=auth, timeout=10).status_code))
        try:
            body = httpx.get(f"{BASE}/status", headers=auth, timeout=10).json()
            print(f"       {DIM}ready:   {body['ready_providers']}{RESET}")
            print(f"       {DIM}blocked: {body['blocked_providers']}{RESET}")
        except Exception as exc:
            print(f"       {DIM}status unreadable: {exc}{RESET}")

        print("\nValidation")
        attempt("/search rejects an empty query",
                exact(400)(lambda: httpx.post(f"{BASE}/search", json={"query": "  "},
                                              headers=auth, timeout=5).status_code))
        attempt("/remember store requires text",
                exact(400)(lambda: httpx.post(f"{BASE}/remember", json={"op": "store"},
                                              headers=auth, timeout=5).status_code))
        attempt("/remember rejects a bad op",
                exact(422)(lambda: httpx.post(f"{BASE}/remember", json={"op": "drop table"},
                                              headers=auth, timeout=5).status_code))

        print("\nLive tool paths (keyless only)")

        def do_search():
            r = httpx.post(f"{BASE}/search", json={"query": "python programming",
                                                    "max_results": 3}, headers=auth, timeout=60)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            got = r.json()
            src = got["results"][0]["source"] if got["count"] else "-"
            return got["count"] > 0, f"{got['count']} results via {src}"

        attempt("/search returns results", do_search)

        def do_store():
            r = httpx.post(f"{BASE}/remember", json={"op": "store", "text": "smoke-test memory"},
                           headers=auth, timeout=120)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code} {r.text[:80]}"
            body = r.json()
            return True, f"id={body['id']} embedded={body['embedded']} ({body['backend']})"

        attempt("/remember store (vector embedded)", do_store)

        def do_recall():
            r = httpx.post(f"{BASE}/remember", json={"op": "recall", "query": "smoke-test",
                                                     "k": 3}, headers=auth, timeout=120)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            body = r.json()
            score = body["memories"][0]["score"] if body["count"] else None
            return body["count"] > 0, f"{body['count']} hits, top score={score}"

        attempt("/remember recall finds it", do_recall)

        def do_forget():
            r = httpx.post(f"{BASE}/remember", json={"op": "forget", "query": "smoke-test"},
                           headers=auth, timeout=30)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            return True, f"removed {r.json()['removed']}"

        attempt("/remember forget", do_forget)

        print("\nBind scope")
        s = socket.socket()
        s.settimeout(2)
        try:
            hostip = socket.gethostbyname(socket.gethostname())
            reachable = s.connect_ex((hostip, PORT)) == 0
        except OSError:
            reachable = False
        finally:
            s.close()
        check("not listening on the LAN address", not reachable,
              "loopback only" if not reachable else "REACHABLE OFF-HOST")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    passed = sum(1 for ok, _, _ in results if ok)
    failed = len(results) - passed
    colour = GREEN if not failed else RED
    print(f"\n  {colour}{passed} passed{RESET}  {RED if failed else DIM}{failed} failed{RESET}\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
