"""Post a GitHub release to Discord as a tidy card instead of a bare link.

Discord's own `/github` webhook address turns a release into a one-line link. This
script builds a proper embed from the release notes instead: the title linked to the
release, the brand colour, the notes with their headings and links kept readable, and
the social preview image. The release workflow runs it; `--dry-run` prints the card
without sending it, so the formatting can be checked locally.

Usage:
    gh api repos/OWNER/REPO/releases/tags/v0.4.3 > release.json
    python scripts/discord_release_card.py release.json --dry-run
    DISCORD_RELEASE_WEBHOOK=https://discord.com/api/webhooks/... \\
        python scripts/discord_release_card.py release.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

#: AssetFurnace ember orange, as on the landing site.
EMBER = 0xFD6D14
#: Discord rejects an embed description longer than 4096 characters.
DESCRIPTION_LIMIT = 4096
PREVIEW_IMAGE = "docs/images/social-preview.jpg"


def repo_from_release_url(url: str) -> str:
    """`owner/repo` from a release's html_url."""
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/releases/", url)
    if not match:
        raise ValueError(f"not a GitHub release URL: {url}")
    return match.group(1)


def _linkify(line: str, repo: str) -> str:
    """Turn bare `#123` and `@user` into links, leaving existing links and code alone."""
    parts = re.split(r"(`[^`]*`|\[[^\]]*\]\([^)]*\))", line)
    for i, part in enumerate(parts):
        if i % 2:
            continue
        part = re.sub(
            r"(?<![\w/&])#(\d+)\b",
            lambda m: f"[#{m.group(1)}](https://github.com/{repo}/issues/{m.group(1)})",
            part,
        )
        part = re.sub(
            r"(?<![\w/])@([A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))\b",
            lambda m: f"[@{m.group(1)}](https://github.com/{m.group(1)})",
            part,
        )
        parts[i] = part
    return "".join(parts)


def format_notes(body: str, repo: str) -> str:
    """Release-note Markdown rewritten for a Discord embed.

    The top `# Title` line goes (the embed has its own title), `##` headings become
    bold lines, and bare issue numbers and usernames become links. Code blocks pass
    through untouched.
    """
    out: list[str] = []
    in_code = False
    for line in body.replace("\r\n", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if re.match(r"#\s", line) and not any(o.strip() for o in out):
            continue
        heading = re.match(r"#{2,6}\s+(.*)", line)
        if heading:
            out.append(f"**{heading.group(1).strip()}**")
            continue
        out.append(_linkify(line, repo))
    text = "\n".join(out).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def fit(text: str, url: str, limit: int = DESCRIPTION_LIMIT) -> str:
    """Cut `text` to fit Discord's limit, at a paragraph break, with a link to the rest."""
    if len(text) <= limit:
        return text
    tail = f"\n\n… [Read the full notes]({url})"
    room = limit - len(tail)
    cut = text.rfind("\n\n", 0, room)
    if cut <= 0:
        cut = room
    head = text[:cut].rstrip()
    if head.count("```") % 2:
        head = head[: head.rfind("```")].rstrip()
    return head + tail


def build_payload(release: dict) -> dict:
    """The Discord webhook JSON for one GitHub release object."""
    url = release["html_url"]
    repo = repo_from_release_url(url)
    tag = release["tag_name"]
    title = release.get("name") or tag
    if release.get("prerelease"):
        title += " (pre-release)"
    notes = format_notes(release.get("body") or "", repo)
    embed = {
        "title": title[:256],
        "url": url,
        "color": EMBER,
        "description": fit(notes or f"[Release notes]({url})", url),
        "image": {"url": f"https://raw.githubusercontent.com/{repo}/{tag}/{PREVIEW_IMAGE}"},
        "footer": {"text": f"{repo} · {tag}"},
    }
    if release.get("published_at"):
        embed["timestamp"] = release["published_at"]
    return {"username": "AssetFurnace", "embeds": [embed], "allowed_mentions": {"parse": []}}


def send(payload: dict, webhook: str) -> None:
    request = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "assetfurnace-release-card"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status >= 300:
            raise SystemExit(f"Discord answered {response.status}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("release_json", type=Path, help="a GitHub release object, as JSON")
    parser.add_argument("--dry-run", action="store_true", help="print the card, send nothing")
    args = parser.parse_args(argv)

    payload = build_payload(json.loads(args.release_json.read_text()))
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    webhook = os.environ.get("DISCORD_RELEASE_WEBHOOK", "").strip()
    if not webhook:
        print("DISCORD_RELEASE_WEBHOOK is not set", file=sys.stderr)
        return 1
    send(payload, webhook.removesuffix("/github"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
