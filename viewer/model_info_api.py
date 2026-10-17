"""A model's stats and its place on disk: the studio's stats panel, Show in folder, and the
MCP tools that mirror them.

Paths come from the page as `output/`-relative strings, so every one is resolved here and
refused if it leaves `output/`. Opening a folder happens on the machine running the server,
so it is only allowed when the request comes from that machine too.
"""

from __future__ import annotations

import ipaddress
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab.glb_stats import file_stats
from image_to_3dlab.host import os_family
from image_to_3dlab.processes import group_popen_kwargs


def resolve(output: Path, relative: str) -> Path:
    """The file `relative` names inside `output`, or ValueError."""
    root = output.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("that file is outside the output folder")
    if not path.is_file():
        raise ValueError(f"no such file: {relative}")
    return path


def model_info(output: Path, relative: str) -> dict[str, Any]:
    path = resolve(output, relative)
    return {
        "file": path.relative_to(output.resolve()).as_posix(),
        "path": str(path),
        "folder": str(path.parent),
        "stats": file_stats(path) if path.suffix.lower() == ".glb" else None,
    }


def reveal_command(path: Path, family: str) -> list[str]:
    """The command that shows `path` in this system's file manager."""
    if family == "macos":
        return ["open", "-R", str(path)]
    if family == "windows":
        return ["explorer", f"/select,{path}"]
    return ["xdg-open", str(path.parent)]


def _launch(command: list[str]) -> None:
    subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **group_popen_kwargs())


def reveal(output: Path, relative: str, *, family: str | None = None,
           launch: Callable[[list[str]], None] = _launch) -> Path:
    path = resolve(output, relative)
    launch(reveal_command(path, family or os_family()))
    return path


def is_local(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback
