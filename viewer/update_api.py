"""Is there a newer release? Asked of GitHub at most once a day, only while the viewer runs.

Nothing is installed and nothing runs in the background. When the page asks, this looks
at a small cache file in `output/`; if the answer is under a day old it is reused, else
one request goes to GitHub's public releases API. That request sends nothing about the
user. Offline, the last answer stands and nothing breaks.

Off switches: `I3D_NO_UPDATE_CHECK=1` in the environment (for agents and servers), or the
checkbox on the About page, in which case the page never asks.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import image_to_3dlab
from image_to_3dlab.host import os_family

LATEST_URL = "https://api.github.com/repos/Bingeljell/image-to-3dlab/releases/latest"
RAW = "https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main"
CACHE = REPO / "output" / ".update-check.json"
MAX_AGE = 24 * 3600


def fetch_latest() -> dict[str, Any]:
    request = urllib.request.Request(LATEST_URL, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "image-to-3dlab-update-check",
    })
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _version(tag: str | None) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag or "")
    return tuple(int(part) for part in match.groups()) if match else None


def update_command(repo: Path, family: str) -> str:
    """The installer line that updates *this* folder, not a fresh copy in the default."""
    if family == "windows":
        return f'$env:I3D_DIR = "{repo}"; irm {RAW}/install.ps1 | iex'
    return f"curl -fsSL {RAW}/install.sh | bash -s -- --dir {repo}"


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check(current: str = image_to_3dlab.__version__, now: float | None = None,
          cache: Path = CACHE, fetch: Callable[[], dict[str, Any]] = fetch_latest,
          env: dict[str, str] | None = None, repo: Path = REPO,
          family: str | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    now = time.time() if now is None else now
    result: dict[str, Any] = {
        "schema_version": 1, "enabled": env.get("I3D_NO_UPDATE_CHECK") != "1",
        "current": current, "latest": None, "url": None, "newer": False,
        "command": update_command(repo, family or os_family()),
    }
    if not result["enabled"]:
        return result

    cached = _read_cache(cache)
    checked_at = cached.get("checked_at")
    if not isinstance(checked_at, (int, float)) or now - checked_at >= MAX_AGE:
        try:
            latest = fetch()
            cached = {"checked_at": now, "tag": latest.get("tag_name"),
                      "url": latest.get("html_url")}
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            # Offline or rate-limited: keep the last answer, and do not ask again for a
            # day, so a flaky network does not mean a request on every page load.
            cached = {**cached, "checked_at": now}
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cached), encoding="utf-8")
        except OSError:
            pass

    latest_version, current_version = _version(cached.get("tag")), _version(current)
    if latest_version:
        result["latest"] = ".".join(map(str, latest_version))
        result["url"] = cached.get("url")
        result["newer"] = bool(current_version and latest_version > current_version)
    return result
