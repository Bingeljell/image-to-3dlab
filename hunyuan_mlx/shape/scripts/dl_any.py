"""Generic resumable ModelScope downloader. Reads the exact size from the repo API.

Usage: python scripts/dl_any.py <repo> <filepath> <outpath> [--yes]

Asks before it fetches. Pass --yes only when there is nobody to ask.
Importing this module does not download anything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

TOLERANCE = 4096


def complete(size: int, expected: int) -> bool:
    """True only when the file is within a short tail of the expected length."""
    return size >= expected - TOLERANCE


def write_body(response, path: str, have: int) -> int:
    """Append a 206 resume. Any other status replaces the file from the start."""
    status = getattr(response, "status", None)
    if status is None:
        status = getattr(response, "code", None)
    mode = "ab" if have and status == 206 else "wb"
    base = have if mode == "ab" else 0
    got = 0
    started = time.time()
    with open(path, mode) as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            got += len(chunk)
            if got % (64 << 20) < (1 << 20):
                rate = got / 1e6 / max(1e-3, time.time() - started)
                print(f"  {path}: {(base + got) / 1e6:.0f} MB  ({rate:.1f} MB/s)", flush=True)
    return got


def confirm(yes: bool, *, route: str, size: str) -> bool:
    """True on --yes or an interactive y/yes. EOF is a no."""
    if yes:
        return True
    print(
        f"About to download Hunyuan3D weights ({route}, {size}).\n"
        "These are Tencent Hunyuan Community License weights, not licensed for use in "
        "the EU, the UK or South Korea.\n"
        "Proceed? [y/N]",
        flush=True,
    )
    try:
        answer = input().strip().lower()
    except EOFError:
        print("Refusing to download without --yes when there is nobody to ask.", flush=True)
        return False
    return answer in ("y", "yes")


def _opener():
    return urllib.request.build_opener(
        urllib.request.ProxyHandler(urllib.request.getproxies())
    )


def file_size(repo: str, filepath: str) -> int | None:
    api = f"https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true"
    with _opener().open(api, timeout=60) as response:
        for entry in json.load(response)["Data"]["Files"]:
            if entry.get("Path") == filepath:
                return int(entry["Size"])
    return None


def download(repo: str, filepath: str, out: str, expected: int) -> bool:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    base = (
        f"https://modelscope.cn/api/v1/models/{repo}/repo?Revision=master&FilePath={filepath}"
    )
    handle = _opener()
    for attempt in range(1, 81):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if complete(have, expected):
            print(f"complete ({have} bytes)", flush=True)
            return True
        request = urllib.request.Request(base)
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            with handle.open(request, timeout=120) as response:
                write_body(response, out, have)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                have = os.path.getsize(out) if os.path.exists(out) else 0
                if complete(have, expected):
                    print("complete (416)", flush=True)
                    return True
            print(f"  attempt {attempt}: HTTP {exc.code}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
        except Exception as exc:  # noqa: BLE001
            print(
                f"  attempt {attempt}: {type(exc).__name__} {str(exc)[:80]}",
                flush=True,
            )
            time.sleep(min(20, 2 ** (attempt % 6)))
    have = os.path.getsize(out) if os.path.exists(out) else 0
    print("DONE" if complete(have, expected) else "INCOMPLETE", flush=True)
    return complete(have, expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo")
    parser.add_argument("filepath")
    parser.add_argument("out")
    parser.add_argument(
        "--yes", action="store_true",
        help="download without prompting. Defaulting to yes is not allowed.",
    )
    args = parser.parse_args(argv)
    size = file_size(args.repo, args.filepath)
    if size is None:
        print(f"could not find {args.filepath} in {args.repo}", flush=True)
        return 1
    if not confirm(
        args.yes,
        route=f"ModelScope {args.repo}",
        size=f"about {size / 1e9:.2f} GB",
    ):
        return 1
    return 0 if download(args.repo, args.filepath, args.out, size) else 1


if __name__ == "__main__":
    raise SystemExit(main())
