"""Notices from us: a new release, AssetFurnace Cloud, a known problem and its fix.

The studio shows them behind a bell in its header and never pops them over someone's work; each
one stays until it is read. Agents see the unread ones in the MCP's `status`.

Where they come from: a small file we edit on our site (NOTICES_URL), asked for at most once a
day, only while the studio runs, sending nothing about the user (the same rules as update_api).
Offline, the last answer stands. A newer release becomes a notice too, from update_api's answer.

Off switch: `I3D_NO_NOTICES=1` (agents, servers). What was read is kept in output/, so it
survives restarts and updates.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
NOTICES_URL = os.environ.get("I3D_NOTICES_URL") or "https://assetfurnace.com/notices.json"   # override: testing a draft feed
CACHE = REPO / "output" / ".notices-cache.json"
READ = REPO / "output" / ".notices-read.json"
MAX_AGE = 24 * 3600
ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")   # used with fullmatch
# where a notice may link: our site, our GitHub, our Discord invite. Nothing else.
LINK = re.compile(r"https://(assetfurnace\.com/[^\s]*|github\.com/Bingeljell/[^\s]*|discord\.gg/EuY9cQTRUt)")
LIMITS = {"title": 120, "text": 600, "url": 300, "link_text": 60}
MAX_NOTICES = 20


def fetch_notices() -> dict[str, Any]:
    request = urllib.request.Request(NOTICES_URL, headers={"User-Agent": "assetfurnace-notices"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read(64 * 1024))


def _version(text: str | None) -> tuple[int, ...] | None:
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", str(text or ""))
    return tuple(int(part) for part in match.groups()) if match else None


def clean(feed: Any, current: str, today: str) -> list[dict[str, str]]:
    """The notices from the feed that apply to this copy today, checked strictly: plain text, our
    links only, ids we can remember. Anything malformed is dropped, never shown half-right."""
    notices = feed.get("notices") if isinstance(feed, dict) else None
    if not isinstance(notices, list):
        return []
    mine = _version(current)
    out = []
    for item in notices[:MAX_NOTICES]:
        if not isinstance(item, dict) or not ID.fullmatch(str(item.get("id", ""))):
            continue
        fields = {k: item.get(k) for k in LIMITS}
        if not isinstance(fields["title"], str) or not fields["title"].strip():
            continue
        if any(v is not None and (not isinstance(v, str) or len(v) > LIMITS[k]) for k, v in fields.items()):
            continue
        url = fields["url"] or ""
        if url and not safe_link(url):
            continue
        if (start := item.get("from")) and str(start) > today:
            continue
        if (end := item.get("until")) and str(end) < today:
            continue
        low, high = _version(item.get("min_version")), _version(item.get("max_version"))
        if mine and ((low and mine < low) or (high and mine > high)):
            continue
        out.append({"id": item["id"], "title": fields["title"].strip(), "text": (fields["text"] or "").strip(),
                    "url": url, "link_text": (fields["link_text"] or "").strip() or ("Read more" if url else "")})
    return out


def safe_link(url: str) -> bool:
    return bool(LINK.fullmatch(url)) and ".." not in url and "\\" not in url


def release_notice(update: dict[str, Any]) -> dict[str, str] | None:
    """A newer release, from update_api.check's answer."""
    if not update.get("newer") or not update.get("latest"):
        return None
    return {"id": f"release-{update['latest']}", "title": f"Version {update['latest']} is out",
            "text": f"Update with: {update.get('command') or './lab update'}",
            "url": url if safe_link(url := str(update.get("url") or "")) else "", "link_text": "What's new"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path: Path, value) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    except OSError:
        pass


def notices(current: str, update: dict[str, Any] | None = None, now: float | None = None,
            cache: Path | None = None, read_path: Path | None = None,
            fetch: Callable[[], dict[str, Any]] | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    """What the News button shows: {enabled, notices: [... with read: bool], unread}."""
    env = os.environ if env is None else env
    cache, read_path, fetch = cache or CACHE, read_path or READ, fetch or fetch_notices
    if env.get("I3D_NO_NOTICES") == "1":
        return {"enabled": False, "notices": [], "unread": 0}
    now = time.time() if now is None else now
    cached = _read_json(cache, {})
    if not isinstance(cached, dict):
        cached = {}
    checked_at = cached.get("checked_at")
    if not isinstance(checked_at, (int, float)) or now - checked_at >= MAX_AGE:
        try:
            cached = {"checked_at": now, "feed": fetch()}
        except (OSError, ValueError, urllib.error.URLError, TimeoutError):
            cached = {**cached, "checked_at": now}   # offline: keep the last feed, ask again tomorrow
        _write_json(cache, cached)
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    shown = clean(cached.get("feed"), current, today)
    if update and (release := release_notice(update)):
        shown.insert(0, release)
    stored = _read_json(read_path, [])
    read = {i for i in stored if isinstance(i, str)} if isinstance(stored, list) else set()
    for notice in shown:
        notice["read"] = notice["id"] in read
    shown.sort(key=lambda n: n["read"])          # unread first, in feed order
    return {"enabled": True, "notices": shown, "unread": sum(not n["read"] for n in shown)}


def mark_read(ids: list[str], read_path: Path | None = None) -> None:
    read_path = read_path or READ
    stored = _read_json(read_path, [])
    read = [i for i in stored if isinstance(i, str)] if isinstance(stored, list) else []
    for notice_id in ids:
        if isinstance(notice_id, str) and ID.fullmatch(notice_id) and notice_id not in read:
            read.append(notice_id)
    _write_json(read_path, read[-500:])
