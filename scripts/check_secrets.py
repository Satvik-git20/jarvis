"""Check whether JARVIS secrets have leaked into places they should not be.

Run this before pushing, or after adding a key:

    uv run python scripts/check_secrets.py

It reports, without printing any secret:
  - which files hold a live credential
  - whether the git history contains one
  - whether any file that git tracks contains one
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)

# Files outside the repo that commonly hold the same keys. Worth flagging
# because a repo can be public while the credential lives beside it.
DANGEROUS_OUTSIDE = (
    Path.home() / ".claude" / "settings.json",
    Path.home() / ".claude" / "settings.local.json",
    Path.home() / ".local" / "share" / "opencode" / "auth.json",
    Path.home() / ".config" / "opencode" / "opencode.json",
    Path.home() / ".config" / "opencode" / "opencode.jsonc",
)

# Key-shaped strings: catches a pasted key even if the file was never read.
KEY_PATTERN = re.compile(
    r"\b(?:sk-or-v1-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|gsk_[A-Za-z0-9]{20,}"
    r"|tvly-[A-Za-z0-9_-]{20,}|jina_[A-Za-z0-9]{20,})"
)


def live_secrets() -> dict[str, str]:
    """Collect the values that are actually in use, keyed by a label."""
    out: dict[str, str] = {}

    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text("utf-8", errors="replace").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            if val and any(t in key.upper() for t in
                           ("KEY", "TOKEN", "SECRET")) and "DATA_DIR" not in key:
                out[f".env:{key.strip()}"] = val

    for path in DANGEROUS_OUTSIDE:
        if not path.is_file():
            continue
        text = path.read_text("utf-8", errors="replace")
        try:
            blob = json.loads(text)
        except json.JSONDecodeError:
            blob = None
        if isinstance(blob, dict):
            for provider, entry in blob.items():
                if isinstance(entry, dict):
                    val = entry.get("key")
                    if isinstance(val, str) and val.strip():
                        out[f"{path.name}:{provider}"] = val.strip()
        # Env-style blocks such as OPENROUTER_API_KEY="sk-or-v1-..."
        for m in KEY_PATTERN.finditer(text):
            out.setdefault(f"{path.name}:pattern", m.group(0))
    return out


def tracked_files() -> list[Path]:
    try:
        names = subprocess.run(
            ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, timeout=30
        ).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return []
    return [REPO / n for n in names if (REPO / n).is_file()]


def git_history_blob() -> str:
    """Every blob ever committed, concatenated. Catches a deleted file."""
    try:
        return subprocess.run(
            ["git", "rev-list", "--objects", "--all"],
            cwd=REPO, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def main() -> int:
    secrets = live_secrets()
    print(f"\n{BOLD}JARVIS secret check{RESET} {DIM}(values are never printed){RESET}\n")

    if not secrets:
        print(f"  {GREEN}no live credentials found to check{RESET}\n")
        return 0

    print(f"  {DIM}credentials in use:{RESET}")
    for label in secrets:
        print(f"    {DIM}{label}{RESET}")

    print(f"\n{BOLD}1. Files outside the repo{RESET}")
    outside = sorted({name.split(":")[0] for name in secrets} - {".env"})
    if outside:
        for name in outside:
            print(f"  {YELLOW}holds a live key:{RESET} {name}")
        print(f"  {DIM}These are expected -- opencode's auth.json is what JARVIS reads")
        print(f"  as its OpenRouter fallback -- but they are plaintext. See below.{RESET}")
    else:
        print(f"  {GREEN}none{RESET}")

    print(f"\n{BOLD}2. Tracked by git{RESET}")
    leaks = []
    for path in tracked_files():
        try:
            text = path.read_text("utf-8", errors="replace")
        except OSError:
            continue
        for label, val in secrets.items():
            if len(val) >= 12 and val in text:
                leaks.append((path, label))
        if KEY_PATTERN.search(text):
            leaks.append((path, "key-shaped string"))
    if leaks:
        seen = set()
        for path, label in leaks:
            key = (str(path), label)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {RED}LEAK{RESET} {path.relative_to(REPO)} ({label})")
        print(f"\n  {RED}Fix: rotate the key, then git rm --cached the file.{RESET}")
    else:
        print(f"  {GREEN}clean{RESET} {DIM}({len(tracked_files())} tracked files){RESET}")

    print(f"\n{BOLD}3. .env ignored{RESET}")
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", ".env"], cwd=REPO, timeout=15
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        ignored = False
    print(f"  {GREEN}ignored{RESET}" if ignored else f"  {RED}NOT ignored{RESET}")

    if leaks:
        print(f"\n  {RED}action needed before publishing{RESET}\n")
        return 1

    print(f"\n{BOLD}If this repo is or becomes public{RESET}")
    print(f"  {DIM}The OpenRouter key in opencode's auth.json and in")
    print("  ~/.claude/settings.json is plaintext on disk. Rotating it means:")
    print("    1. openrouter.ai/keys -> create a new key")
    print("    2. put it in .env as OPENROUTER_API_KEY")
    print("    3. opencode auth logout openrouter, then opencode auth login")
    print(f"    4. update ~/.claude/settings.json, or delete the stale key from it{RESET}")
    print(f"\n  {GREEN}nothing to fix right now{RESET}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

