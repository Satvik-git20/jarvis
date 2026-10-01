"""Static checks that the Chrome extension stays loadable and safe.

Chrome MV3 refuses inline scripts, a missing ``side_panel`` path silently
disables the feature, and model output must never be parsed as HTML. These
tests guard all three without needing a browser in the loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXT = Path(__file__).resolve().parents[1] / "chrome-extension"


def _manifest() -> dict:
    return json.loads((EXT / "manifest.json").read_text("utf-8"))


def test_manifest_declares_a_loadable_mv3_side_panel():
    m = _manifest()
    assert m["manifest_version"] == 3
    assert (EXT / m["side_panel"]["default_path"]).is_file()
    assert (EXT / m["background"]["service_worker"]).is_file()
    assert (EXT / m["options_page"]).is_file()
    assert {"sidePanel", "storage"} <= set(m["permissions"])
    assert "http://127.0.0.1:8765/*" in m["host_permissions"]


def test_every_referenced_asset_exists():
    for html in EXT.glob("*.html"):
        text = html.read_text("utf-8")
        for ref in re.findall(r"(?:src|href)=\"([^\"]+)\"", text):
            if ref.startswith(("http://", "https://", "#", "data:")):
                continue
            assert (EXT / ref).is_file(), f"{html.name} references missing {ref}"


def test_no_inline_scripts_or_event_handlers():
    for html in EXT.glob("*.html"):
        text = html.read_text("utf-8")
        for tag in re.findall(r"<script\b[^>]*>", text):
            assert "src=" in tag, f"{html.name} contains an inline script (blocked by MV3 CSP)"
        assert not re.search(r"\son[a-z]+\s*=", text), f"{html.name} uses an inline event handler"


def test_sidepanel_never_parses_model_output_as_html():
    js = (EXT / "sidepanel.js").read_text("utf-8")
    assert "innerHTML" not in js
    assert "insertAdjacentHTML" not in js
    assert "eval(" not in js


def test_token_is_stored_locally_and_never_synced():
    for name in ("sidepanel.js", "options.js"):
        js = (EXT / name).read_text("utf-8")
        assert "chrome.storage.local" in js, f"{name} should use chrome.storage.local"
        assert "chrome.storage.sync" not in js, f"{name} must not sync the daemon token"
