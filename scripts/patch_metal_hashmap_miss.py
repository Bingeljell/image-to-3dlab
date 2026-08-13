#!/usr/bin/env python3
"""Fix the unchecked hashmap miss in the Metal dual-contouring kernel.

    python scripts/patch_metal_hashmap_miss.py            # apply
    python scripts/patch_metal_hashmap_miss.py --check
    python scripts/patch_metal_hashmap_miss.py --revert

Then rebuild: `scripts/rebuild_metallib.sh`

**The bug.** `src/metal/remesh.metal`:

```c
inline float get_vertex_val_u32(...) {
    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);
    return udf[idx];                    // no check that the lookup hit
}
```

`linear_probing_lookup_u32` returns `0xFFFFFFFF` on a miss (line 36). So a missed lookup
indexes `udf[0xFFFFFFFF]` — four billion elements past the end of the buffer.

**Why that makes a lattice.** Metal defines out-of-bounds buffer reads as returning **zero**.
The crossing test is `(val1 < 0 && val2 >= 0) || (val1 >= 0 && val2 < 0)`, and `0.0` is not
`< 0`, so a missed lookup reads as *outside the surface* — deterministically, every time. Two
misses on an edge give the same sign, no crossing is detected, and the voxel falls back to
its grid centre. Quads then connect grid centres, which is exactly the wireframe cage this
port produces, with 22-29% of vertices stranded.

Zero is the worst possible substitute: it means *exactly on the surface*. A vertex outside
the narrow band is far outside it, so the honest answer is a large positive distance.

**Why CUDA does not show it.** Same source, different out-of-bounds semantics. CUDA leaves
OOB reads undefined; in practice they land in adjacent allocated memory and return values
with varied signs, which produce crossings often enough that the defect stays hidden. Metal's
defined zero makes it systematic.

This is consistent with everything the parameter sweeps could not explain: the result is
deterministic to 0.3% across runs (zeros, not garbage), MtlBVH's distances are precise to
2.7% of eps (the value is never read), and band, subdivision threshold and project_back all
failed because none of them changes miss behaviour.

`vendor/` is git-ignored, so re-apply after every bootstrap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "vendor/trellis-mac/deps/mtlmesh/src/metal/remesh.metal"

MARKER = "// i2l: guard the hashmap miss"

# (function signature line, the body line to guard, the sentinel, the u32/u64 label)
PATCHES = (
    (
        "inline float get_vertex_val_u32(",
        (
            "    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);\n"
            "    return udf[idx];\n"
        ),
        (
            "    uint idx = linear_probing_lookup_u32(hashmap_keys, hashmap_vals, flat_idx, M);\n"
        f"    {MARKER}: a miss returns 0xFFFFFFFF, and udf[0xFFFFFFFF] is an out-of-bounds\n"
        "    // read that Metal defines as 0.0 - which the crossing test reads as \"exactly on\n"
        "    // the surface\", the most wrong possible answer. A vertex outside the narrow band\n"
        "    // is far outside it, so report a large positive distance.\n"
            "    if (idx == 0xFFFFFFFFu) return 1.0e30f;\n"
            "    return udf[idx];\n"
        ),
    ),
    (
        "inline float get_vertex_val_u64(",
        (
            "    uint idx = linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, M);\n"
            "    return udf[idx];\n"
        ),
        (
            "    uint idx = linear_probing_lookup_u64(hashmap_keys, hashmap_vals, flat_idx, M);\n"
            f"    {MARKER} (u64 variant, same defect)\n"
            "    if (idx == 0xFFFFFFFFu) return 1.0e30f;\n"
            "    return udf[idx];\n"
        ),
    ),
)


def state(source: str) -> str:
    """'applied', 'absent', or raise if the anchors have moved."""
    for signature, original, _patched in PATCHES:
        if signature not in source:
            raise RuntimeError(
                f"anchor missing from remesh.metal: {signature!r}. "
                "The shader has changed shape; inspect it before patching."
            )
        if original not in source and MARKER not in source:
            raise RuntimeError(
                f"the body under {signature!r} does not match what this patch expects"
            )
    applied = source.count(MARKER)
    if applied == len(PATCHES):
        return "applied"
    if applied == 0:
        return "absent"
    raise RuntimeError("remesh.metal is half-patched; revert and re-apply")


def apply(source: str) -> str:
    for _signature, original, patched in PATCHES:
        source = source.replace(original, patched, 1)
    return source


def remove(source: str) -> str:
    for _signature, original, patched in PATCHES:
        source = source.replace(patched, original, 1)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--path", type=Path, default=TARGET)
    args = parser.parse_args()

    if not args.path.is_file():
        print(f"not found: {args.path}", file=sys.stderr)
        return 2

    source = args.path.read_text()
    current = state(source)

    if args.check:
        print("APPLIED" if current == "applied" else "ABSENT")
        return 0

    want = "absent" if args.revert else "applied"
    if current == want:
        print(f"already {want} — no change")
        return 0
    args.path.write_text((remove if args.revert else apply)(source))
    print("reverted" if args.revert else "applied: hashmap miss guarded in both variants")
    print("now rebuild: scripts/rebuild_metallib.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
