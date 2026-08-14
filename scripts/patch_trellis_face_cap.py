#!/usr/bin/env python3
"""Lift the hard 200,000-face cap in the Mac port's `generate.py`.

**The defect.** Before o_voxel's postprocess runs at all, `generate.py` pre-simplifies the
decoded mesh with `fast_simplification`:

```python
# Pre-simplify mesh to avoid mtlbvh crash on large meshes.
# Target ~200K faces — keeps detail, avoids Metal BVH issues.
target_faces = min(args.bake_target_faces, 200000, len(faces_np))
```

From a real Flicker run:

```
Mesh: 1,601,340 vertices, 3,207,582 triangles
  Simplifying mesh: 3,207,582 -> ~200,000 faces
```

**94% of the decode is destroyed by a cruder decimator than the one o_voxel would use**,
and every subsequent step -- hole filling, non-manifold repair, the weld patch, QEM
simplification, UV unwrapping and the texture bake -- then runs on the wreckage. The
result is a surface crazed with cracks across the whole body.

**How we found it.** The user ran the same artwork through the official demo at
`huggingface.co/spaces/microsoft/TRELLIS.2`. It came back at **281,889 faces** -- above our
cap -- smooth, crisply marked and essentially watertight. Every comparison we had run until
then was our output against our other output, so a defect present in all of them looked
like a limitation of TRELLIS. It was ours.

**The second consequence.** `bake_target_faces` was *inert above 200,000*. Requesting
300,000 and 3,000,000 both produced ~197k faces. Every sweep of that parameter in this
repo's history, and the conclusion "do not raise bake_target_faces", measured a clamped
value.

**About the crash this guarded against.** The comment blames an `mtlbvh` crash on large
meshes, and that risk is real but unquantified -- nobody measured the quality it cost.
Verified working at 300,000 (290,662 faces out, no crash). The true Metal limit is not
known, so this patch takes a ceiling rather than removing the clamp entirely: raise it
deliberately, and if a run crashes, lower it.

Safe to run repeatedly; it detects its own marker and does nothing on a second run.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = "i2l_face_cap"

DEFAULT_TARGET = Path("vendor/trellis-mac/generate.py")
# The official demo simplifies the decoded mesh to 2**24 faces before to_glb. This remains
# a safety ceiling while allowing the demo UI's multi-million-face export targets through.
DEFAULT_CEILING = 16_777_216

ORIGINAL = "target_faces = min(args.bake_target_faces, 200000, len(faces_np))"


def patched_line(ceiling: int) -> str:
    """The replacement, carrying its own explanation and the marker."""
    return (
        f"target_faces = min(args.bake_target_faces, {ceiling}, len(faces_np))  "
        f"# {MARKER}: was 200000, which crushed a 3.2M-tri decode by 94% "
        f"before o_voxel ran"
    )


def is_patched(source: str) -> bool:
    return MARKER in source


def find_cap(source: str) -> int | None:
    """The current ceiling, or None when the anchor is missing.

    Matches the numeric literal rather than the whole line so the patch still finds an
    already-patched file (to report or re-target it) and so upstream whitespace changes
    do not silently turn this into a no-op.
    """
    match = re.search(
        r"target_faces\s*=\s*min\(\s*args\.bake_target_faces\s*,\s*(\d+)\s*,\s*len\(faces_np\)\s*\)",
        source,
    )
    return int(match.group(1)) if match else None


def apply(source: str, ceiling: int) -> str:
    """Return the patched source. Idempotent and able to retarget an older patch."""
    current = find_cap(source)
    if current is None:
        raise SystemExit(
            "anchor not found: generate.py no longer contains the "
            "`min(args.bake_target_faces, <n>, len(faces_np))` pre-simplification. "
            "Check whether the Mac port changed it before assuming this patch is needed."
        )
    if is_patched(source):
        if current == ceiling:
            return source
        return re.sub(
            r"(target_faces\s*=\s*min\(\s*args\.bake_target_faces\s*,\s*)\d+(\s*,\s*len\(faces_np\)\s*\))",
            rf"\g<1>{ceiling}\g<2>",
            source,
            count=1,
        )
    return re.sub(
        r"target_faces\s*=\s*min\(\s*args\.bake_target_faces\s*,\s*\d+\s*,\s*len\(faces_np\)\s*\)",
        patched_line(ceiling).replace("\\", "\\\\"),
        source,
        count=1,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--ceiling",
        type=int,
        default=DEFAULT_CEILING,
        help=f"new face ceiling (default {DEFAULT_CEILING:,}). Verified at 300,000; the "
        f"real Metal BVH limit is unknown, so lower this if a run crashes",
    )
    parser.add_argument("--check", action="store_true", help="report status, change nothing")
    args = parser.parse_args()

    path = args.target.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{path} does not exist -- bootstrap the vendored port first")
    source = path.read_text()

    if args.check:
        if is_patched(source):
            print(f"PATCHED (cap now {find_cap(source)})")
        else:
            print(f"NOT PATCHED (cap is {find_cap(source)})")
        return 0

    updated = apply(source, args.ceiling)
    if updated == source:
        print(f"already patched, cap is {find_cap(source)} -- nothing to do")
        return 0
    path.write_text(updated)
    print(f"lifted the face cap: {find_cap(source):,} -> {args.ceiling:,} in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
