#!/usr/bin/env python3
"""Let Pixal3D save its clean-up input and still write the model, via `PIXAL3D_DUMP_CONTINUE=1`.

    python scripts/patch_pixal3d_dump_continue.py     # then rebuild trellis-cli
    PIXAL3D_DUMP_CONTINUE=1 trellis-cli ... --dump-post run.post.bin run.glb

**Why.** Pixal3D ends every run with a clean-up (weld, remesh, simplify to 1,000,000
triangles, unwrap, bake the paint from its own colour data). Asked for 20,000 triangles
instead, that same clean-up makes the cleanest game model we have, and its `post-replay`
tool reruns it from a saved input in about 13 seconds. `--dump-post` saves that input,
but upstream then stops: no model is written. This keeps upstream's behaviour by default
and, with the environment variable set, carries on to write the normal model as well, so
one generation gives both the usual 1M model and the input for any later face count.

**The weld.** Before writing the dump, upstream welds the mesh in place without its
per-vertex colours. Carrying on from there would leave the colour list longer than the
welded vertex list. So the dump's weld now takes the colours too, exactly as the normal
path's weld does; the normal path's own weld then finds nothing left to merge.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone, then
`cmake --build vendor/pixal3d-cpp/build --target trellis-cli`. Safe to run repeatedly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "i2l_dump_continue"
ENV_VAR = "PIXAL3D_DUMP_CONTINUE"

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO / "vendor" / "pixal3d-cpp" / "src" / "trellis_cli.cpp"

INCLUDE_ANCHOR = "#include <filesystem>\n"
# What the injected code calls. Never assume the host file's includes.
INCLUDE_ADDITION = f"#include <cstdlib>   // {MARKER}: std::getenv\n"

WELD_ANCHOR = (
    "        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr, "
    "1.0f / ((float)so.res * 8.0f));\n"
)
WELD_REPLACEMENT = (
    f"        // {MARKER}: weld the colours too, so the run can carry on after the dump.\n"
    "        trellis::weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors, "
    "1.0f / ((float)so.res * 8.0f));\n"
)

# Starts at the "[6/6]" line: the "done in ... return 0;" tail alone also ends the
# single-image TRELLIS route's dump ("[7/7]"), which this must not touch.
DUMP_LINE = (
    '        printf("[6/6] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\\n", '
    "cfg.dump_post.c_str(), dV, dFc,\n"
    '               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);\n'
)
EXIT_ANCHOR = (
    DUMP_LINE
    + '        printf("done in %.1fs -> %s\\n", now() - t0, cfg.dump_post.c_str());\n'
    "        return 0;\n"
    "    }\n"
)
EXIT_REPLACEMENT = (
    DUMP_LINE
    + '        printf("done in %.1fs -> %s\\n", now() - t0, cfg.dump_post.c_str());\n'
    f"        // {MARKER}: {ENV_VAR}=1 carries on to write the normal model as well.\n"
    f'        const char* {MARKER}_env = std::getenv("{ENV_VAR}");\n'
    f"        if (!{MARKER}_env || {MARKER}_env[0] != '1') return 0;\n"
    f'        printf("      [dump] {ENV_VAR}=1: carrying on to write the model\\n");\n'
    "    }\n"
)


def continues(raw: str | None) -> bool:
    """The same decision the injected C++ makes: only a value starting with "1" carries on."""
    return bool(raw) and raw[0] == "1"


def is_patched(source: str) -> bool:
    return MARKER in source


def apply(source: str) -> str:
    """Return the patched source. Idempotent.

    Every anchor must appear exactly once: the exit and the weld each have a twin in the
    single-image TRELLIS path, and patching the wrong one would change a route we do not use.
    """
    if is_patched(source):
        return source
    for name, anchor in (("include", INCLUDE_ANCHOR), ("weld", WELD_ANCHOR),
                         ("exit", EXIT_ANCHOR)):
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                f"{name} anchor found {count} times in trellis_cli.cpp (expected 1): the "
                "multi-view dump branch changed upstream. Check whether --dump-post grew its "
                "own way to keep going before patching."
            )
    source = source.replace(INCLUDE_ANCHOR, INCLUDE_ANCHOR + INCLUDE_ADDITION, 1)
    source = source.replace(WELD_ANCHOR, WELD_REPLACEMENT, 1)
    return source.replace(EXIT_ANCHOR, EXIT_REPLACEMENT, 1)


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
