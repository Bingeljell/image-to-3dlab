#!/usr/bin/env python3
"""Install BiRefNet-lite, the background remover every backend uses once it is present.

    python scripts/bootstrap_matte.py          # says what it will fetch, then asks
    python scripts/bootstrap_matte.py --yes    # non-interactive
    python scripts/bootstrap_matte.py --check  # exit 0 if installed, 1 if not

Without it, runs fall back to u2net, which ate a white robot's arms on a light backdrop, an
axe handle and a sword + shield in a side-by-side test (2026-09-28); BiRefNet-lite kept
all of them and matched full BiRefNet. One file, 224 MB, MIT licensed, from rembg's own
release, saved where rembg looks (`~/.u2net`, or `$U2NET_HOME`) and checked against
rembg's published MD5 before it is kept.

Nothing is fetched without an explicit yes: AGENTS.md forbids weight downloads the user
has not chosen.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import matte


def announcement() -> str:
    return (f"Background remover: BiRefNet-lite ({matte.LITE_BYTES / 1e6:.0f} MB, MIT)\n"
            f"  from   {matte.LITE_URL}\n"
            f"  to     {matte.model_file(matte.LITE_MODEL)}")


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_if_missing(target: Path | None = None, download=None) -> None:
    """BiRefNet-lite, unless it is already there. Setups that cut out pictures call this
    so a first generation never fetches a remover nobody agreed to."""
    target = target or matte.model_file(matte.LITE_MODEL)
    if target.is_file():
        return
    download = download or globals()["download"]
    print(f"\nFetching BiRefNet-lite ({matte.LITE_BYTES / 1e6:.0f} MB)...", flush=True)
    download(target)
    print(f"  background remover in {target}")


def download(target: Path, opener=urllib.request.urlopen) -> Path:
    """Fetch to a temporary name, verify, then move into place. A bad file is never kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    with opener(matte.LITE_URL) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    actual = md5_of(partial)
    if actual != matte.LITE_MD5:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch ({actual}); nothing was installed")
    partial.replace(target)
    return target


def main(argv: list[str] | None = None, ask=input) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="download without asking")
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args(argv)

    target = matte.model_file(matte.LITE_MODEL)
    if target.is_file():
        print(f"BiRefNet-lite is installed: {target}")
        return 0
    if args.check:
        print("BiRefNet-lite is not installed; runs use u2net")
        return 1

    print(announcement(), flush=True)
    if not args.yes:
        try:
            answer = ask("Download it? [y/N] ")
        except EOFError:  # no terminal: nobody to ask, so the answer is no
            print("Refusing to download without --yes when there is nobody to ask.")
            return 1
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing downloaded.")
            return 1
    download(target)
    print(f"Installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
