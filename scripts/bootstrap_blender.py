#!/usr/bin/env python3
"""Install Blender 4.2 LTS for Finish on Linux, from blender.org.

    python scripts/bootstrap_blender.py          # says what it will fetch, then asks
    python scripts/bootstrap_blender.py --yes    # non-interactive (the viewer's button)

Finish (retopology, bake, Pixel Match) runs Blender in the background. On a Mac Blender is
an ordinary app install; on a Linux server it meant leaving the viewer to download and
unpack blender.org's tarball by hand. This does exactly that: the newest 4.2 LTS build,
unpacked into `~/blender-lts/`, where the lab already looks (image_to_3dlab/blender.py).
It needs no root and no extra libraries to run headless (checked on an NVIDIA pod).

Nothing is fetched without an explicit yes.
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import __version__
from image_to_3dlab.blender import can_install

RELEASES = "https://download.blender.org/release/Blender4.2/"
TARBALL = re.compile(r"blender-4\.2\.(\d+)-linux-x64\.tar\.xz(?![.\w])")


def supported(system: str, machine: str) -> bool:
    return can_install(system, machine)


# download.blender.org sits behind Cloudflare, which refuses Python-urllib's anonymous
# default User-Agent (403, error 1010). We say who we are instead, honestly; that passes.
# No browser disguise: a site that turns this away has decided, and we respect it.
USER_AGENT = f"image-to-3dlab/{__version__} (+https://github.com/Bingeljell/image-to-3dlab)"


def open_url(url: str, method: str = "GET"):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def target(home: Path | None = None) -> Path:
    return (home or Path.home()) / "blender-lts"


def newest_tarball(listing: str) -> str:
    """The newest 4.2 LTS Linux build named in blender.org's release listing."""
    versions = {int(m.group(1)) for m in TARBALL.finditer(listing)}
    if not versions:
        raise SystemExit(f"no Linux build of Blender 4.2 listed at {RELEASES}")
    return f"blender-4.2.{max(versions)}-linux-x64.tar.xz"


def announcement(name: str, size: int, home: Path) -> str:
    return (f"Blender {name.split('-')[1]} LTS for Finish (GPL, from the Blender Foundation)\n"
            f"  from   {RELEASES}{name}\n"
            f"  size   {size / 1024 ** 2:.0f} MB download, about 1 GB unpacked\n"
            f"  to     {target(home)}")


def unpack(archive: Path, home: Path) -> Path:
    """Extract into the home folder and rename the versioned folder to blender-lts/."""
    dest = target(home)
    with tempfile.TemporaryDirectory(dir=home) as scratch:
        with tarfile.open(archive) as tar:
            tar.extractall(scratch, filter="data")
        (top,) = [p for p in Path(scratch).iterdir() if p.is_dir()]
        if dest.exists():
            shutil.rmtree(dest)
        top.rename(dest)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    if not supported(platform.system(), platform.machine()):
        raise SystemExit("This installs the Linux x86_64 build. On a Mac or Windows, install "
                         "Blender from https://www.blender.org/download/")
    home = Path.home()
    with open_url(RELEASES) as response:
        name = newest_tarball(response.read().decode("utf-8", "replace"))
    with open_url(RELEASES + name, method="HEAD") as response:
        size = int(response.headers.get("Content-Length", 0))
    print(announcement(name, size, home), flush=True)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    archive = home / f".{name}.partial"
    print(f"Downloading {name}...", flush=True)
    with open_url(RELEASES + name) as response, archive.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    print("Unpacking...", flush=True)
    dest = unpack(archive, home)
    archive.unlink(missing_ok=True)
    check = subprocess.run([str(dest / "blender"), "--background", "--version"],
                           capture_output=True, text=True, check=False)
    first = (check.stdout or check.stderr).splitlines()[:1]
    if check.returncode != 0:
        print(f"Blender is unpacked in {dest} but did not start: {first}", flush=True)
        return 1
    print(f"Done: {first[0] if first else 'Blender'} in {dest}. Finish will find it.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
