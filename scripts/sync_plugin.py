"""Install the opencode plugin from this repo into opencode's config dir.

The plugin lives at opencode/jarvis.ts in the repo, but opencode only auto-loads
plugins from ~/.config/opencode/plugins/. This copies one to the other so the
repo stays the single source of truth.

    uv run python scripts/sync_plugin.py           # install
    uv run python scripts/sync_plugin.py --check   # report drift, change nothing
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "opencode" / "jarvis.ts"
PLUGIN_DIR = Path.home() / ".config" / "opencode" / "plugins"
TARGET = PLUGIN_DIR / "jarvis.ts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report drift only")
    args = ap.parse_args()

    if not SOURCE.is_file():
        print(f"missing source plugin: {SOURCE}")
        return 1

    if not TARGET.exists():
        action = "missing"
    elif filecmp.cmp(SOURCE, TARGET, shallow=False):
        print(f"plugin is in sync: {TARGET}")
        return 0
    else:
        action = "differs"

    if args.check:
        print(f"plugin {action}: {SOURCE} vs {TARGET}")
        print("run: uv run python scripts/sync_plugin.py")
        return 1

    PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, TARGET)
    print(f"installed {TARGET}")
    print("opencode picks up plugin changes on next session start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
