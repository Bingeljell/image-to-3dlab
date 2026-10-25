"""The welcome card: who we are, what this machine can run, and what changed.

It doubles as the announcements channel. The page remembers the last version it showed
and asks for news since then, so an update is announced once, from the `CHANGELOG.md` that
shipped with the code on disk.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import backend_catalog

import image_to_3dlab

BRAND_FILE = Path(__file__).resolve().parent / "brand.json"
CHANGELOG = REPO / "CHANGELOG.md"

_RELEASE = re.compile(r"^## \[(\d+\.\d+\.\d+)\](?:\s*-\s*(\S+))?", re.MULTILINE)
_SECTION = re.compile(r"^### (\w+)", re.MULTILINE)


def brand() -> dict[str, str]:
    """The name shown to people. One file, so a rename is a one-line change."""
    return json.loads(BRAND_FILE.read_text(encoding="utf-8"))


def read_changelog() -> str:
    return CHANGELOG.read_text(encoding="utf-8")


HEADLINE_MAX = 140


def headline(item: str) -> str:
    """A changelog bullet cut to one line.

    Its bold lead when that is a sentence; when the lead is only a name (a script, a
    flag), the lead plus the rest of its first sentence; else the first sentence.
    """
    flat = " ".join(item.split())
    bold = re.match(r"\*\*(.+?)\*\*", flat)
    if bold and re.search(r"[.!?]$", bold[1].strip()):
        text = bold[1].strip()
    else:
        if bold:
            flat = bold[1].strip() + flat[bold.end():]
        sentence = re.match(r"(.+?[.!?])(\s|$)", flat)
        text = sentence[1] if sentence else flat
    if len(text) > HEADLINE_MAX:
        text = text[:HEADLINE_MAX - 1].rsplit(" ", 1)[0] + "…"
    return text


def _items(body: str) -> list[str]:
    """Top-level bullets, each with its wrapped continuation lines joined back on."""
    items: list[str] = []
    for line in body.splitlines():
        if line.startswith("- "):
            items.append(line[2:])
        elif items and line.startswith("  "):
            items[-1] += " " + line.strip()
    return items


def parse_changelog(text: str) -> list[dict[str, Any]]:
    """Released versions, newest first. `[Unreleased]` is never announced."""
    marks = list(_RELEASE.finditer(text))
    releases = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        # Stop at any later "## " heading (e.g. link references), not only releases.
        body = text[mark.end():end]
        body = body.split("\n## ", 1)[0]
        heads = list(_SECTION.finditer(body))
        sections = {}
        for j, head in enumerate(heads):
            chunk_end = heads[j + 1].start() if j + 1 < len(heads) else len(body)
            sections.setdefault(head[1], []).extend(
                headline(x) for x in _items(body[head.end():chunk_end]))
        releases.append({"version": mark[1], "date": mark[2], "sections": sections})
    return releases


def _version_key(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.split("."))
    except (AttributeError, ValueError):
        return None


def news_since(releases: list[dict[str, Any]], since: str | None) -> list[dict[str, Any]]:
    """Releases newer than `since`; on a first visit, just the newest one."""
    seen = _version_key(since) if since else None
    if seen is None:
        return releases[:1]
    return [r for r in releases if (_version_key(r["version"]) or ()) > seen]


def payload(since: str | None = None) -> dict[str, Any]:
    status = backend_catalog.catalog_status()
    routes = [
        {"id": b["id"], "label": b["label"], "kind": b["kind"], "state": b["state"]}
        for b in status["backends"] if b["supported_here"]
    ]
    return {
        "schema_version": 1,
        "brand": brand(),
        "version": image_to_3dlab.__version__,
        "host": {"id": status["host"]["id"], "label": status["host"]["label"]},
        "routes": routes,
        "news": news_since(parse_changelog(read_changelog()), since),
    }
