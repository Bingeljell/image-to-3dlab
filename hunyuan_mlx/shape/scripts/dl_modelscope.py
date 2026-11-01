"""Resumable ModelScope download of Hunyuan3D-2mini safetensors.

ModelScope serves the LFS blobs (hf-mirror 308-redirects them to blocked
huggingface.co). Honors HTTPS_PROXY from env. Pure stdlib.

Asks before it fetches. Pass --yes only when there is nobody to ask.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

TOLERANCE = 4096
ROUTE = "ModelScope Hunyuan3D-2mini"
SIZE = "about 4.2 GB"

BASE = ("https://modelscope.cn/api/v1/models/Tencent-Hunyuan/"
        "Hunyuan3D-2mini/repo?Revision=master&FilePath=")
ROOT = os.environ.get("HY3D_WEIGHTS_ROOT", "weights/Hunyuan3D-2mini")

FILES = [
    ("hunyuan3d-vae-v2-mini/model.fp16.safetensors", 428_455_666),
    ("hunyuan3d-dit-v2-mini/model.fp16.safetensors", 3_819_958_234),
]

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(urllib.request.getproxies())
)


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


def download(path, approx) -> bool:
    out = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for attempt in range(1, 61):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if complete(have, approx):
            print(f"  complete {path} ({have} bytes)", flush=True)
            return True
        req = urllib.request.Request(BASE + path)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with opener.open(req, timeout=120) as response:
                got = write_body(response, out, have)
            print(f"  segment done {path}: +{got / 1e6:.0f} MB", flush=True)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                have = os.path.getsize(out) if os.path.exists(out) else 0
                if complete(have, approx):
                    print(f"  complete {path} (416, {have} bytes)", flush=True)
                    return True
            print(f"  attempt {attempt} {path}: HTTP {exc.code}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
        except Exception as exc:  # noqa: BLE001
            print(f"  attempt {attempt} {path}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            time.sleep(min(20, 2 ** (attempt % 6)))
    have = os.path.getsize(out) if os.path.exists(out) else 0
    return complete(have, approx)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--yes", action="store_true",
        help="download without prompting. Defaulting to yes is not allowed.",
    )
    args = parser.parse_args(argv)
    if not confirm(args.yes, route=ROUTE, size=SIZE):
        return 1
    ok = True
    for path, approx in FILES:
        print(f">>> {path}", flush=True)
        if not download(path, approx):
            ok = False
    print("DOWNLOADS_DONE" if ok else "DOWNLOADS_INCOMPLETE", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
