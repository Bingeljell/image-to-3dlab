"""Post a GitHub pull request to Discord as a tidy card, like the release cards.

Discord's own `/github` webhook address turns a PR into a one-line link. This builds an
embed instead: "#114 Title" linked to the PR, who made it, and the start of the description
with its headings and links kept readable. Only a merged PR gets a card: opening, reopening,
marking ready and closing without merging all stay quiet. The PR workflow runs it on the
event GitHub hands it; `--dry-run` prints the card without sending it.

Usage:
    gh api repos/OWNER/REPO/pulls/114 > pr.json
    python scripts/discord_pr_card.py pr.json --dry-run
    DISCORD_PR_WEBHOOK=https://discord.com/api/webhooks/... \\
        python scripts/discord_pr_card.py "$GITHUB_EVENT_PATH"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from discord_release_card import fit, format_notes, send

#: A PR card says what it is about; the full description is one click away.
SUMMARY_LIMIT = 1200
#: GitHub's own colour for a merged PR.
MERGED = 0x8957E5


def repo_from_pr_url(url: str) -> str:
    """`owner/repo` from a pull request's html_url."""
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/pull/\d+", url)
    if not match:
        raise ValueError(f"not a GitHub pull request URL: {url}")
    return match.group(1)


def what_happened(pr: dict, action: str) -> str | None:
    """"merged" for a pull request that has just been merged; None for everything else (opened,
    reopened, ready for review, closed without merging, edits, label changes)."""
    return "merged" if action == "closed" and pr.get("merged") else None


def build_payload(pr: dict, action: str = "opened") -> dict | None:
    """The Discord webhook JSON for one pull request event, or None to stay quiet."""
    event = what_happened(pr, action)
    if event is None:
        return None
    url = pr["html_url"]
    repo = repo_from_pr_url(url)
    user = pr.get("user") or {}
    notes = format_notes(pr.get("body") or "", repo)
    embed = {
        "title": f"#{pr['number']} {pr['title']}"[:256],
        "url": url,
        "color": MERGED,
        "description": fit(notes or f"[Open the pull request]({url})", url, SUMMARY_LIMIT),
        "footer": {"text": f"{repo} · {event}"},
    }
    if user.get("login"):
        embed["author"] = {"name": user["login"], "url": user.get("html_url") or f"https://github.com/{user['login']}"}
        if user.get("avatar_url"):
            embed["author"]["icon_url"] = user["avatar_url"]
    stamp = pr.get("merged_at")
    if stamp:
        embed["timestamp"] = stamp
    return {"username": "AssetFurnace", "embeds": [embed], "allowed_mentions": {"parse": []}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pr_json", type=Path,
                        help="a pull_request webhook event, or a bare pull request object")
    parser.add_argument("--action", help="the event's action, when the file is a bare PR (default: closed)")
    parser.add_argument("--dry-run", action="store_true", help="print the card, send nothing")
    args = parser.parse_args(argv)

    data = json.loads(args.pr_json.read_text())
    pr, action = (data["pull_request"], data.get("action", "closed")) if "pull_request" in data else (data, "closed")
    payload = build_payload(pr, args.action or action)
    if payload is None:
        print(f"nothing to post for '{args.action or action}'")
        return 0
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    webhook = os.environ.get("DISCORD_PR_WEBHOOK", "").strip()
    if not webhook:
        print("DISCORD_PR_WEBHOOK is not set", file=sys.stderr)
        return 1
    send(payload, webhook.removesuffix("/github"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
