"""Report a bug: the report the studio fills in, and the three ways to send it.

Nothing here sends anything. It builds the text and two links; the person reads the report,
edits it, and sends it themselves through assetfurnace.com, GitHub, or a paste into Discord.

The home folder is replaced with ~ everywhere, so a report never carries the user's name
in a path, and the prompt stays out unless they tick the box.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from activity_api import LABELS, read_history

import image_to_3dlab
from image_to_3dlab.host import os_family

# I3D_REPORT_URL points the studio at a local copy of the site while testing it.
SITE_REPORT_URL = os.environ.get("I3D_REPORT_URL", "https://assetfurnace.com/report")
GITHUB_NEW_ISSUE = "https://github.com/Bingeljell/image-to-3dlab/issues/new"
DISCORD_INVITE = "https://discord.gg/3D4bcEhGx"
GITHUB_URL_MAX = 7500     # GitHub refuses much longer links; the log tail is what gets cut
LOG_LINES = 40


def scrub(text: str, home: str) -> str:
    """`text` with the home folder shown as ~, so no path carries the user's name."""
    return text.replace(home, "~") if home else text


def _chip(system: str) -> str:
    """The chip or graphics card, best effort: an empty string when it can't be told."""
    commands = {"macos": ["sysctl", "-n", "machdep.cpu.brand_string"],
                "linux": ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]}
    command = commands.get(system)
    if not command:
        return ""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.strip().splitlines()[0] if out.strip() else ""


def machine_summary(sys_platform: str | None = None, chip: Callable[[], str] | None = None) -> str:
    """'macOS 26.0 · Apple M5', 'Linux · NVIDIA GeForce RTX 4090', and so on."""
    family = os_family(sys_platform)
    name = {"macos": f"macOS {platform.mac_ver()[0]}".strip(), "linux": "Linux", "windows": "Windows"}.get(family, family)
    if sys_platform not in (None, sys.platform):
        name = name.split(" ")[0]  # describing another system: this machine's version number means nothing
    card = (chip or (lambda: _chip(family)))()
    return " · ".join(part for part in (name, card, platform.machine() if sys_platform is None else "") if part)


def find_failure(output: Path, chain: str | None = None, step: str | None = None) -> dict[str, Any] | None:
    """The failure to report: the one named, or else the newest."""
    for record in read_history(output):
        if record.get("result") != "error":
            continue
        if chain and (record.get("chain") != chain or (step and record.get("step") != step)):
            continue
        return record
    return None


def details(output: Path, *, home: str | None = None, machine: str | None = None, version: str | None = None,
            include_prompt: bool = False, chain: str | None = None, step: str | None = None) -> str:
    """Everything the studio adds to a report, as plain text the person can read and edit."""
    home = str(Path.home()) if home is None else home
    lines = [f"AssetFurnace {version or image_to_3dlab.__version__} · {machine or machine_summary()}"]
    failure = find_failure(output, chain, step)
    if failure:
        lines.append(f"Step: {LABELS.get(failure.get('step'), failure.get('step'))} · {failure.get('time', '')}")
        if failure.get("note"):
            lines.append(f"What it said: {failure['note']}")
        if include_prompt and failure.get("prompt"):
            lines.append(f"Prompt: {failure['prompt']}")
        tail = "\n".join(str(failure.get("log") or "").splitlines()[-LOG_LINES:])
        if tail:
            lines += ["", "Last lines of the log:", tail]
    return scrub("\n".join(lines), home)


def _site_link(message: str, details_text: str, version: str) -> str:
    draft = json.dumps({"message": message, "details": details_text, "version": version}, ensure_ascii=False)
    encoded = base64.urlsafe_b64encode(draft.encode()).decode().rstrip("=")
    return f"{SITE_REPORT_URL}#report={encoded}"


def _github_link(message: str, details_text: str) -> str:
    title = (message.strip().splitlines() or ["Bug report"])[0][:100]
    body = f"{message.strip()}\n\n```\n{details_text}\n```"
    url = f"{GITHUB_NEW_ISSUE}?{urlencode({'title': title, 'body': body})}"
    while len(url) > GITHUB_URL_MAX and details_text:
        details_text = details_text[len(details_text) // 4:]       # drop the oldest log lines first
        body = f"{message.strip()}\n\n```\n…{details_text}\n```"
        url = f"{GITHUB_NEW_ISSUE}?{urlencode({'title': title, 'body': body})}"
    return url


def links(message: str, details_text: str, version: str | None = None) -> dict[str, str]:
    """The three ways to send: assetfurnace.com, a new GitHub issue, and text to paste in Discord."""
    version = version or image_to_3dlab.__version__
    return {
        "site": _site_link(message, details_text, version),
        "github": _github_link(message, details_text),
        "text": f"{message.strip()}\n\n{details_text}",
        "discord": DISCORD_INVITE,
    }
