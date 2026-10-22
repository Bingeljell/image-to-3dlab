#!/usr/bin/env python3
"""Install Real-ESRGAN x4plus, the 4x upscaler Hero face uses on the head crop.

    python scripts/bootstrap_upscaler.py          # says what it will fetch, then asks
    python scripts/bootstrap_upscaler.py --yes    # non-interactive
    python scripts/bootstrap_upscaler.py --check  # exit 0 if installed, 1 if not

One file, 67 MB, BSD-3-Clause, from the Real-ESRGAN project's own release, saved under
vendor/realesrgan/. It runs in the project's own environment (image_to_3dlab/upscale.py),
so Hero no longer needs the multi-GB Hunyuan3D-MLX paint install for it. The file is kept
only if its size matches the release and PyTorch reads the network's weights from it.

Nothing is fetched without an explicit yes: AGENTS.md forbids weight downloads the user
has not chosen.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import upscale


def announcement() -> str:
    return (f"Hero face upscaler: Real-ESRGAN x4plus ({upscale.WEIGHTS_BYTES / 1e6:.0f} MB, BSD-3-Clause)\n"
            f"  from   {upscale.WEIGHTS_URL}\n"
            f"  to     {upscale.WEIGHTS_FILE}")


def verify(path: Path) -> str | None:
    """Why PATH is not the upscaler's weights, or None when it is."""
    if path.stat().st_size != upscale.WEIGHTS_BYTES:
        return f"size {path.stat().st_size} is not the release's {upscale.WEIGHTS_BYTES}"
    try:
        upscale.network().load_state_dict(upscale.state_dict(path))
    except Exception as exc:   # a truncated or foreign file: say so, keep nothing
        return f"PyTorch could not read it ({exc})"
    return None


def download(target: Path, opener=urllib.request.urlopen, check=verify) -> Path:
    """Fetch to a temporary name, verify, then move into place. A bad file is never kept."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with opener(upscale.WEIGHTS_URL) as response, partial.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)
    problem = check(partial)
    if problem:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"download rejected: {problem}; nothing was installed")
    partial.replace(target)
    return target


def main(argv: list[str] | None = None, ask=input, fetch=download) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="download without asking")
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args(argv)

    target = upscale.WEIGHTS_FILE
    if upscale.installed(target):
        print(f"Real-ESRGAN is installed: {target}")
        return 0
    if args.check:
        print("Real-ESRGAN is not installed; Hero face needs it")
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
    fetch(target)
    print(f"Installed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
