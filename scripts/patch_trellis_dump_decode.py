#!/usr/bin/env python3
"""Teach `generate.py` to cache the decoded mesh, so baking can be re-run without sampling.

    python scripts/patch_trellis_dump_decode.py           # apply
    python scripts/patch_trellis_dump_decode.py --check
    python scripts/patch_trellis_dump_decode.py --revert

**The problem this solves.** A full run is ~20 minutes: ~90s loading a 4B model, ~15 min
sampling, ~4 min decode and bake. Everything we currently want to test — `remesh` on or off,
`decimation_target` in the right unit, `texture_size`, material mode, the Branch 1 vs
Branch 2 split in `o_voxel.postprocess.to_glb` — lives in that last stage. So every
one-line question costs 20 minutes, and 16 of those re-derive a sampling result that never
changes.

The official TRELLIS.2 demo already works this way: `app.py` splits `image_to_3d` (sample,
cache latents in `state`) from `extract_glb` (decode and bake from that state), which is why
its sliders re-extract instantly. This patch gives our CLI the same seam, cached one stage
later — at the decoded mesh rather than the latents, because `to_glb` is the only thing we
need to vary and caching post-decode also skips the model load entirely.

Pairs with `scripts/trellis_rebake.py`, which consumes the dump and never imports the model.

`vendor/` is git-ignored, so re-apply after every bootstrap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "vendor/trellis-mac/generate.py"

ANCHOR_ARG = '    parser.add_argument("--remesh-band", type=float, default=1.0)\n'
ADDED_ARG = (
    '    parser.add_argument(\n'
    '        "--dump-decode", default=None,\n'
    '        help="i2l: write the decoded mesh here (torch .pt) so scripts/trellis_rebake.py "\n'
    '             "can re-bake with different to_glb settings without re-sampling",\n'
    '    )\n'
)

ANCHOR_MESH = "    mesh_out = outputs[0] if isinstance(outputs, list) else outputs\n"
ADDED_DUMP = (
    "    # i2l: cache the decode so baking can be re-run without the 16-minute sample.\n"
    "    if getattr(args, 'dump_decode', None):\n"
    "        import torch as _torch\n"
    "        _payload = {\n"
    "            'vertices': mesh_out.vertices.cpu(),\n"
    "            'faces': mesh_out.faces.cpu(),\n"
    "            'attrs': mesh_out.attrs.cpu(),\n"
    "            'coords': mesh_out.coords.cpu(),\n"
    "            'layout': mesh_out.layout,\n"
    "            'voxel_size': mesh_out.voxel_size,\n"
    "        }\n"
    "        _torch.save(_payload, args.dump_decode)\n"
    "        print(f'Decode cached: {args.dump_decode} '\n"
    "              f'({mesh_out.vertices.shape[0]} vertices, {mesh_out.faces.shape[0]} faces)')\n"
)


def state(source: str) -> str:
    """'applied', 'absent', or raise if the anchors have moved."""
    for anchor, what in ((ANCHOR_ARG, "--remesh-band argument"), (ANCHOR_MESH, "mesh_out assignment")):
        if anchor not in source:
            raise RuntimeError(
                f"anchor missing from generate.py: {what}. The port has changed shape; "
                "inspect it before patching."
            )
    has_arg = ADDED_ARG in source
    has_dump = ADDED_DUMP in source
    if has_arg and has_dump:
        return "applied"
    if not has_arg and not has_dump:
        return "absent"
    raise RuntimeError("generate.py is half-patched; revert and re-apply")


def apply(source: str) -> str:
    """Insert both halves. The argument must import nothing the host file lacks."""
    source = source.replace(ANCHOR_ARG, ANCHOR_ARG + ADDED_ARG, 1)
    return source.replace(ANCHOR_MESH, ANCHOR_MESH + ADDED_DUMP, 1)


def remove(source: str) -> str:
    return source.replace(ADDED_ARG, "", 1).replace(ADDED_DUMP, "", 1)


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
    print("reverted" if args.revert else "applied: --dump-decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
