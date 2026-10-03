#!/usr/bin/env python3
"""Install gltfpack for the Props tab, from meshoptimizer's GitHub release.

    python scripts/bootstrap_gltfpack.py          # says what it will fetch, then asks
    python scripts/bootstrap_gltfpack.py --yes    # non-interactive (the viewer's button)

The Props tab compresses each prop's LODs with gltfpack: mesh reordered for the GPU,
textures re-encoded as WebP, which is where the weight is. Without it the LODs stay
uncompressed and there are no `.web.glb` files. This fetches the native build for this
machine (the npm build cannot write WebP), checks it against a pinned checksum, and puts
it in `vendor/gltfpack/`, where `scripts/finish_props.py` already looks.

Nothing is fetched without an explicit yes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import platform
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import __version__  # noqa: E402

VERSION = "v1.3"
RELEASES = f"https://github.com/zeux/meshoptimizer/releases/download/{VERSION}/"
USER_AGENT = f"image-to-3dlab/{__version__} (+https://github.com/Bingeljell/image-to-3dlab)"


@dataclass(frozen=True)
class Asset:
    zip: str
    sha256: str
    size: int
    binary: str


# Pinned so a new upstream release, or a swapped file, cannot change what gets installed.
# The macOS build is Apple Silicon only and the Linux one x86_64 only.
ASSETS = {
    ("darwin", "arm64"): Asset(
        "gltfpack-macos.zip",
        "1002fe6a437aa005eca541502c27b8d2b184f91f4ad0c6d282e3614a1e35e75a", 1663661, "gltfpack"),
    ("linux", "x86_64"): Asset(
        "gltfpack-ubuntu.zip",
        "0666d9dc40d60fe5b9a45f3fc24f8e6ca87112974bd5de9ca57152aa13d06017", 1977681, "gltfpack"),
    ("windows", "amd64"): Asset(
        "gltfpack-windows.zip",
        "f6e9c09d66af23da3da71b86f0652d2413e81c734e0aecd1cd3d0e6e6e8645c0", 1482895,
        "gltfpack.exe"),
}


def asset_for(system: str, machine: str) -> Asset | None:
    return ASSETS.get((system.lower(), machine.lower()))


def can_install() -> bool:
    return asset_for(platform.system(), platform.machine()) is not None


def target_dir(repo: Path = REPO) -> Path:
    return repo / "vendor" / "gltfpack"


def announcement(asset: Asset) -> str:
    return (f"gltfpack {VERSION} for the Props tab (MIT, from meshoptimizer by Arseny Kapoulkine)\n"
            f"  from   {RELEASES}{asset.zip}\n"
            f"  size   {asset.size / 1024 ** 2:.1f} MB download\n"
            f"  to     vendor/gltfpack/{asset.binary}")


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def install(archive: bytes, asset: Asset, dest: Path) -> Path:
    """Check the zip against its pinned checksum, then write the binary into `dest`."""
    if hashlib.sha256(archive).hexdigest() != asset.sha256:
        raise SystemExit(f"{asset.zip} does not match its pinned checksum; nothing installed")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if asset.binary not in bundle.namelist():
            raise SystemExit(f"{asset.zip} has no {asset.binary} in it; nothing installed")
        payload = bundle.read(asset.binary)
    dest.mkdir(parents=True, exist_ok=True)
    binary = dest / asset.binary
    binary.write_bytes(payload)
    binary.chmod(0o755)
    return binary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    asset = asset_for(platform.system(), platform.machine())
    if asset is None:
        raise SystemExit(f"No gltfpack build for {platform.system()} {platform.machine()}. "
                         f"See https://github.com/zeux/meshoptimizer/releases")
    print(announcement(asset), flush=True)
    if not args.yes:
        if not sys.stdin.isatty():
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1
    print(f"Downloading {asset.zip}...", flush=True)
    binary = install(download(RELEASES + asset.zip), asset, target_dir())
    check = subprocess.run([str(binary), "-h"], capture_output=True, text=True, check=False)
    first = (check.stdout or check.stderr).splitlines()[:1]
    if not first or "gltfpack" not in first[0]:
        print(f"gltfpack is in {binary} but did not start: {first}", flush=True)
        return 1
    print(f"Done: {first[0]} in {binary.parent}. The Props tab will compress LODs now.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
