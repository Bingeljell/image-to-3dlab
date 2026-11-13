#!/usr/bin/env python3
"""Fix `post-replay` on Windows: read its dump header in one `fread`, and say what fell short.

    python scripts/patch_pixal3d_header_read.py     # then rebuild post-replay
    python scripts/patch_pixal3d_header_read.py --check

**Why.** `post_replay.cpp` read the 16-byte dump header as

    if (fread(&V,4,1,f)+fread(&F,4,1,f)+fread(&Mv,4,1,f)+fread(&res,4,1,f) != 4) return 1;

The order in which the operands of `+` run is up to the compiler. MSVC runs them right to
left, so on Windows `V` got `res`, `F` got `Mv` and so on; the body reads then came up
short and the tool exited 1 without printing anything at all. GCC and Clang go left to
right, which is why the Mac and Linux builds never showed it. Reported by Ahnyth in
issue #192, with the proof (a dump with its header written backwards loaded fine).

One `fread` for the whole header removes the question. Every body read now says which dump
ended early and what its header claimed, so the next failure of this kind is not silent.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone, then
`cmake --build vendor/pixal3d-cpp/build --target post-replay`. Safe to run repeatedly.
Once upstream carries the fix the anchor is gone and this refuses rather than guesses;
drop it from `.github/workflows/pixal3d-nvidia.yml` at that point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "i2l_header_read"

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "vendor" / "pixal3d-cpp" / "src" / "post_replay.cpp"

HEADER_ANCHOR = (
    "    if (fread(&V,4,1,f)+fread(&F,4,1,f)+fread(&Mv,4,1,f)+fread(&res,4,1,f) != 4) "
    "return 1;\n"
)
HEADER_REPLACEMENT = (
    f"    // {MARKER}: one read for the whole header. The order in which the operands of\n"
    "    // '+' run is unspecified: MSVC runs those four freads right to left, so on Windows\n"
    "    // V got res, F got Mv and the body reads below came up short (issue #192).\n"
    "    int32_t hdr[4];\n"
    '    if (fread(hdr,4,4,f) != 4) { fprintf(stderr, "%s: header is short\\n", dump); '
    "return 1; }\n"
    "    V = hdr[0]; F = hdr[1]; Mv = hdr[2]; res = hdr[3];\n"
)

BODY_ANCHOR = (
    "    if (fread(verts.data(),4,verts.size(),f) != verts.size()) return 1;\n"
    "    if (fread(faces.data(),4,faces.size(),f) != faces.size()) return 1;\n"
    "    for (auto& c : coords) if (fread(c.data(),4,3,f) != 3) return 1;\n"
    "    if (fread(pbr6.data(),4,pbr6.size(),f) != pbr6.size()) return 1;\n"
)
BODY_REPLACEMENT = (
    f"    // {MARKER}: name the dump and its header, so a short read is not a bare exit code.\n"
    "    bool ok = fread(verts.data(),4,verts.size(),f) == verts.size()\n"
    "           && fread(faces.data(),4,faces.size(),f) == faces.size();\n"
    "    for (auto& c : coords) ok = ok && fread(c.data(),4,3,f) == 3;\n"
    "    ok = ok && fread(pbr6.data(),4,pbr6.size(),f) == pbr6.size();\n"
    '    if (!ok) { fprintf(stderr, "%s ends early: its header says V=%d F=%d Mv=%d\\n", '
    "dump, V, F, Mv); return 1; }\n"
)

# What a built binary still contains, for the workflow to grep for. A comment is stripped;
# a format string is not.
BINARY_MARKER = "ends early: its header says"


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent.

    Both anchors must appear exactly once. `post_replay.cpp` has one dump reader, so a
    second copy means upstream grew another path and this must be looked at again.
    """
    if is_patched(source):
        return source
    for name, anchor in (("header", HEADER_ANCHOR), ("body", BODY_ANCHOR)):
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                f"{name} anchor found {count} times in post_replay.cpp (expected 1): the "
                "dump reader changed upstream. Check whether upstream's own fix now reads "
                "the header in one call before patching, and if it does, drop this script "
                "from .github/workflows/pixal3d-nvidia.yml."
            )
    source = source.replace(HEADER_ANCHOR, HEADER_REPLACEMENT, 1)
    return source.replace(BODY_ANCHOR, BODY_REPLACEMENT, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", nargs="?", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if not args.target.is_file():
        raise SystemExit(f"not found: {args.target}")
    source = args.target.read_text(encoding="utf-8")

    if args.check:
        print(("patched: " if is_patched(source) else "NOT PATCHED: ") + str(args.target))
        return 0 if is_patched(source) else 1

    patched = apply(source)
    if patched == source:
        print(f"already patched: {args.target}")
        return 0
    args.target.write_text(patched, encoding="utf-8")
    print(f"patched: {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
