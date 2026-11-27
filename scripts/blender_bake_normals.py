#!/usr/bin/env python3
"""Bake TRELLIS' discarded high-poly detail into a normal map for the low-poly mesh.

Run (live): python scripts/blender_bake_normals.py low.glb high.ply normal_out.png

TRELLIS generates ~6.2 million triangles and the Mac port decimates to at most 200,000
before texture baking. That discarded 98% is real surface detail we already paid for.
A **normal map** captures it: the lighting engine then renders bumps that do not exist in
the geometry, so a 101k mesh lights like a multi-million-triangle one.

Unusually, this needs no sculpting step — the high-poly source already exists. Dump it
with `scripts/patch_trellis_highpoly.py` (`--dump-highpoly`), then bake with this.

Two details that decide whether the bake is usable:

* **Alignment.** Compare the two *inside Blender*, not in file space. The glTF importer
  already applies Y-up to Z-up, so by the time both are loaded they are in the same
  orientation and no permutation is needed. Reasoning from raw file-space sizes suggests
  a Y/Z swap that does not exist, and "correcting" it misaligns them — caught here by the
  per-axis ratios, which read 1.42 spread when rotated and 1.04 when left alone.
* **Ray distance.** Blender casts rays outward from the low-poly surface to find the
  high-poly. Too short and detail is missed; too long and rays hit the wrong surface
  across a gap — very visible on thin geometry like ears and leaf tips. Scaled from the
  model's size rather than hardcoded.
* **Sign.** The decode is non-manifold — the Snag arrives with 237k non-manifold edges —
  so winding cannot propagate across it and per-component repair does not converge. Rays
  then hit the right surface at the right distance and come back with the normal
  *reversed*: measured on the Snag, 48.9% of hits sat more than 90 degrees from the
  low-poly normal, with the upper quartile at 164 degrees, near-exactly opposite. Baked,
  that is rainbow confetti over the relief rather than relief. A tangent-space normal map
  cannot have a negative Z, so the ambiguity is resolvable after the fact: negate any
  texel that points into the surface. That single step took mean blue from 132 to 221 and
  negative-Z texels from 48.1% to 0.1%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blender_bake_normals_worker as worker
from blender_joint_markers import send
from blender_worker_source import build_worker_code

AXIS_PERMUTATIONS = worker.AXIS_PERMUTATIONS
axis_size_ratios = worker.axis_size_ratios
best_permutation = worker.best_permutation


def resolve_tangent_sign(pixels):
    """Negate texels whose normal points into the surface, and say how many.

    Returns (pixels, flipped_fraction). A tangent-space normal has Z > 0 by construction:
    it is a deviation *from* the surface, not through it. So a texel with Z < 0 can only
    have come from a source face whose winding was reversed, and the true normal is its
    negation — the sign is recoverable without touching the mesh, which matters because
    the mesh is non-manifold and cannot be repaired by orientation propagation.

    The flipped fraction is worth reading: near zero means the source was consistently
    wound, and near half means it was not, which is a property of the source rather than
    of this bake.
    """
    import numpy as np

    normals = np.asarray(pixels, dtype=np.float32) / 255.0 * 2.0 - 1.0
    flipped = normals[..., 2] < 0.0
    normals[flipped] = -normals[flipped]
    # Rounded, not truncated: every value from 1 to 63 decodes and re-encodes a hair
    # under itself (1 comes back as 0.99999994), so truncating moved it down a step
    # even in texels that were never flipped.
    encoded = np.clip(np.rint((normals + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
    return encoded, float(flipped.mean())


def blender_code(
    low: Path,
    high: Path,
    out: Path,
    size: int,
    ray: float,
    smooth_source: bool,
    device: str,
    max_ray: float,
    decimate_to: int,
    samples: int,
    rot_x: bool = False,
) -> str:
    return build_worker_code(
        Path(__file__).with_name("blender_bake_normals_worker.py"),
        {
            "low_path": str(low), "high_path": str(high), "out_path": str(out),
            "size": size, "ray": ray, "smooth_source": smooth_source, "device": device,
            "max_ray": max_ray, "decimate_to": decimate_to, "samples": samples, "rot_x": rot_x,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("low", type=Path, help="low-poly GLB (the bake target)")
    parser.add_argument("high", type=Path, help="high-poly PLY (the bake source)")
    parser.add_argument("out", type=Path, help="normal map PNG to write")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--ray", type=float, default=0.02,
        help="How far rays search for the high-poly surface, in world units. Too short "
             "misses detail; too long jumps a gap and bakes the wrong surface, which is "
             "very visible on thin geometry like ears and leaf tips",
    )
    parser.add_argument(
        "--decimate-to", type=int, default=1_500_000,
        help="Decimate the bake source to this many triangles first. 6M is slow to "
             "ray-cast against and a 2048 map cannot represent that much detail anyway. "
             "0 disables",
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    parser.add_argument(
        "--keep-sign", action="store_true",
        help="Leave texels that point into the surface alone. Only for diagnosis: the "
             "fraction negated is a direct read on how inconsistently the source is wound",
    )
    parser.add_argument(
        "--max-ray", type=float, default=0.0,
        help="Hard limit on how far a ray may travel, in world units. 0 is no limit and "
             "is the right default: the cage already decides where rays start, and "
             "limiting them to the extrusion distance stops them at the surface itself",
    )
    parser.add_argument(
        "--flat-source", action="store_true",
        help="Leave the high-poly flat-shaded. Only useful for diagnosis: a decode PLY "
             "has no vertex normals, and flat shading bakes each facet as one constant "
             "normal, which reads as rainbow confetti rather than detail",
    )
    parser.add_argument(
        "--permutation", choices=sorted(AXIS_PERMUTATIONS), default="none",
        help="Axis mapping from high-poly space to the low-poly's, compared *inside "
             "Blender*. The glTF importer already applies the Y-up to Z-up conversion, "
             "so the two arrive aligned and the default is correctly 'none'. Comparing "
             "raw file-space sizes instead suggests a swap that does not exist",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    perm_name = args.permutation
    code = blender_code(
        args.low.expanduser().resolve(),
        args.high.expanduser().resolve(),
        args.out.expanduser().resolve(),
        args.size, args.ray, not args.flat_source, args.device, args.max_ray,
        args.decimate_to, args.samples, rot_x=(perm_name == "gltf_to_blender"),
    )
    print(send(code, args.host, args.port))

    if args.keep_sign or not args.out.exists():
        return 0

    import numpy as np
    from PIL import Image

    with Image.open(args.out) as image:
        pixels = np.array(image.convert("RGB"))
    resolved, flipped = resolve_tangent_sign(pixels)
    Image.fromarray(resolved).save(args.out)
    print(f"sign resolved: negated {flipped * 100:.1f}% of texels (source winding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
