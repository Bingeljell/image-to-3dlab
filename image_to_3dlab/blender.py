"""Find Blender on any machine this repo supports, and say how to get it when it is missing.

Finish (retopology and the detail bake) and the rig tools run Blender headless. Blender used
to be looked up at the macOS app path only, so on Linux and Windows every Finish run failed
before it started. Nothing here installs Blender: it is a ~400 MB application the user
chooses, like any other.

Search order: `I2L_BLENDER` (an explicit override always wins, and a wrong one is reported
rather than silently skipped), `blender` on the PATH, then each platform's usual places.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from collections.abc import Callable

from image_to_3dlab.host import os_family

ENV_VAR = "I2L_BLENDER"
MAC_APP = Path("/Applications/Blender.app/Contents/MacOS/Blender")
DOWNLOAD_URL = "https://www.blender.org/download/"
# Tested on 5.2; the scripts use nothing newer than 4.2 LTS, so older than that is flagged.
MIN_VERSION = (4, 2)


def candidates(family: str, home: Path | None = None) -> list[Path]:
    """The usual install locations for one OS family, most likely first."""
    home = home or Path.home()
    if family == "macos":
        return [MAC_APP, home / "Applications" / "Blender.app" / "Contents" / "MacOS" / "Blender"]
    if family == "windows":
        roots = [Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                 Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))]
        found = []
        for root in roots:
            # One folder per version ("Blender 4.2", "Blender 5.2"); newest name first.
            found += sorted((root / "Blender Foundation").glob("Blender*/blender.exe"), reverse=True)
        return found
    # Linux: distro package, snap, flatpak export, and the blender.org tarball unpacked in
    # /opt or the home folder.
    unpacked = sorted(Path("/opt").glob("blender*/blender"), reverse=True)
    unpacked += sorted(home.glob("blender*/blender"), reverse=True)
    return [Path("/usr/bin/blender"), Path("/usr/local/bin/blender"), Path("/snap/bin/blender"),
            Path("/var/lib/flatpak/exports/bin/org.blender.Blender"), *unpacked]


def find_blender(env: dict[str, str] | None = None, which: Callable = shutil.which,
                 family: str | None = None, home: Path | None = None, system_ok: bool = True) -> Path | None:
    """The Blender executable to use, or None when there is none."""
    env = os.environ if env is None else env
    configured = env.get(ENV_VAR)
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    on_path = which("blender")
    if on_path and system_ok:
        return Path(on_path)
    # When system_ok is False, skip PATH and also skip standard system locations
    candidates_list = candidates(family or os_family(), home)
    if not system_ok:
        candidates_list = [p for p in candidates_list if not p.as_posix().startswith(('/usr', '/snap', '/var/lib/flatpak'))]
    return next((p for p in candidates_list if p.is_file()), None)


def parse_version(text: str) -> tuple[int, int] | None:
    """`Blender 5.2.0 LTS` -> (5, 2)."""
    match = re.search(r"Blender\s+(\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2])) if match else None


def blender_version(path: Path, run: Callable = subprocess.run) -> tuple[int, int] | None:
    try:
        result = run([str(path), "--version"], capture_output=True, text=True, timeout=60,
                     check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return parse_version(result.stdout or "")


def missing_help(family: str | None = None) -> str:
    """What to tell someone who has no Blender, in their OS's terms."""
    family = family or os_family()
    how = {
        "macos": "Install Blender from blender.org (drag it into Applications).",
        "windows": "Install Blender from blender.org with the standard installer.",
    }.get(family, "Install Blender from blender.org (the tarball, unpacked in /opt or your "
                  "home folder) or with `sudo snap install blender --classic`. Distro packages "
                  "are often too old.")
    return (f"Finish needs Blender {MIN_VERSION[0]}.{MIN_VERSION[1]} or newer, and none was "
            f"found. {how} {DOWNLOAD_URL} If it is installed somewhere unusual, set "
            f"{ENV_VAR} to its path.")


def version_problem(version: tuple[int, int] | None) -> str | None:
    """Why this Blender may not work, or None when it is new enough (or unreadable)."""
    if version is not None and version < MIN_VERSION:
        return (f"Blender {version[0]}.{version[1]} is older than {MIN_VERSION[0]}."
                f"{MIN_VERSION[1]}; Finish may fail. Get a newer one from {DOWNLOAD_URL}")
    return None


def can_install(system: str | None = None, machine: str | None = None) -> bool:
    """Whether scripts/bootstrap_blender.py can install Blender here: blender.org's Linux
    x86_64 tarball, unpacked in the home folder. Elsewhere Blender is an app install."""
    import platform

    system = system or platform.system()
    machine = machine or platform.machine()
    return system == "Linux" and machine in {"x86_64", "AMD64"}
