"""A model's stats and its place on disk: the studio's stats panel, Show in folder, and the
MCP tools that mirror them.

Paths come from the page as `output/`-relative strings, so every one is resolved here and
refused if it leaves `output/`. Opening a folder happens on the machine running the server,
so it is only allowed when the request comes from that machine too.
"""

from __future__ import annotations

import ipaddress
import shutil
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from image_to_3dlab import roblox_export as rx
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


# A prop with no real-world size gets its longest side this long in studs unless the person picks
# a height: about a crate or a chest beside a 5-6 stud character. Longest side, not height, so a rug
# or a sword lying flat doesn't come out hundreds of studs wide. Rigged models are person-sized.
PROP_LONGEST_STUDS = 4.0


def roblox_export(output: Path, relative: str, *, kind: str | None = None,
                  height_studs: float | None = None) -> dict[str, Any]:
    """Write `<model>_roblox/` beside the model and say how it did against Roblox's limits."""
    path = resolve(output, relative)
    if path.suffix.lower() != ".glb":
        raise ValueError("only GLB files can be exported for Roblox")
    data = path.read_bytes()
    try:
        longest = PROP_LONGEST_STUDS if height_studs is None and not file_stats(path)["bones"] else None
        result = rx.export(data, kind=kind, height_studs=height_studs, longest_studs=longest)
        folder = rx.write_folder(result, path.with_name(f"{path.stem}_roblox"))
    except rx.RobloxExportError as exc:
        raise ValueError(str(exc)) from exc
    except (KeyError, IndexError, struct.error, OSError) as exc:
        # A damaged or unusual file, or a full disk: say so, not "is the studio running?".
        raise ValueError(f"{path.name} couldn't be exported: it may be damaged ({exc.__class__.__name__})") from exc
    # The licence record travels with the file (repo rule), so the Roblox copy carries it too.
    record = path.with_suffix(".provenance.json")
    if record.is_file():
        shutil.copyfile(record, folder / "model.provenance.json")
    root = output.resolve()
    return {
        "folder": folder.relative_to(root).as_posix(),
        "file": (folder / "model.glb").relative_to(root).as_posix(),
        "path": str(folder),
        "passed": result.passed,
        "checks": result.checks,
        "notes": result.notes,
        "size_studs": result.size_studs,
        "triangles": result.triangles,
        "maps": sorted(result.maps),
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
