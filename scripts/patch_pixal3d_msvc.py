#!/usr/bin/env python3
"""Let Pixal3D's uv_bake.cpp build with MSVC (Windows): make one lambda-read constant static.

    python scripts/patch_pixal3d_msvc.py [pixal3d/src/uv_bake.cpp]     # then build

Upstream's weld reads `BIAS`, a local `constexpr`, from inside a capture-less lambda. Clang and
GCC accept that; MSVC stops with C3493 ("cannot be implicitly captured"), which broke the
Windows NVIDIA build at commit c89126d. A `static constexpr` local needs no capture, so every
compiler accepts it and the maths is unchanged. Safe to run repeatedly; refuses if upstream
moved the line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "vendor" / "pixal3d-cpp" / "src" / "uv_bake.cpp"
ANCHOR = "    constexpr int64_t BIAS = (int64_t)1 << 20;\n"
PATCHED = "    static constexpr int64_t BIAS = (int64_t)1 << 20;\n"


def is_patched(source: str) -> bool:
    return PATCHED in source


def apply(source: str) -> str:
    if is_patched(source):
        return source
    if source.count(ANCHOR) != 1:
        raise SystemExit("anchor not found once in uv_bake.cpp: upstream moved the BIAS line, check by hand")
    return source.replace(ANCHOR, PATCHED)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text(encoding="utf-8")
    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
