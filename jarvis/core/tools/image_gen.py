"""Image generation via Pollinations.

The `flux` model is listed as always-free, but Pollinations now requires a key
for every generation -- their auth doc and their older FAQ disagree, and the
auth doc is the current one. So this requires POLLINATIONS_KEY.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import httpx

from jarvis.config import Settings, get_settings

log = logging.getLogger("jarvis.image")

GEN_URL = "https://gen.pollinations.ai/image/{prompt}"


async def generate_image(
    prompt: str,
    *,
    width: int = 1024,
    height: int = 1024,
    model: str = "flux",
    settings: Settings | None = None,
) -> dict[str, Any]:
    s = settings or get_settings()
    cap = s.by_name("pollinations")
    key = s.credential(cap) if cap else None
    if not key:
        raise RuntimeError(
            "Pollinations needs a free key. Get one at enter.pollinations.ai and put it "
            "in .env as POLLINATIONS_KEY."
        )

    out_dir = s.ensure_data_dir() / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"img_{int(time.time())}.jpg"
    dest = out_dir / name

    url = GEN_URL.format(prompt=prompt.strip().replace(" ", "%20"))
    params = {
        "model": model,
        "width": max(64, min(width, 2048)),
        "height": max(64, min(height, 2048)),
        "nologo": "true",
        "seed": int(time.time()) % 1_000_000,
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
        r = await client.get(url, params=params,
                             headers={"Authorization": f"Bearer {key}"})
        if r.status_code >= 400:
            raise RuntimeError(f"Pollinations HTTP {r.status_code}: {r.text[:200]}")
        content_type = r.headers.get("content-type", "")
        if "image" not in content_type:
            raise RuntimeError(f"expected an image, got {content_type}: {r.text[:160]}")

    dest.write_bytes(r.content)
    return {
        "path": str(dest),
        "url": str(r.url),
        "model": model,
        "bytes": len(r.content),
        "width": params["width"],
        "height": params["height"],
    }


def list_images(limit: int = 20, settings: Settings | None = None) -> list[Path]:
    s = settings or get_settings()
    out_dir = s.data_dir / "images"
    if not out_dir.is_dir():
        return []
    files = sorted(out_dir.glob("img_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]
