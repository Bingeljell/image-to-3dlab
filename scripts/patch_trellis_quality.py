#!/usr/bin/env python3
"""Expose the quality controls TRELLIS already has and the Mac port never passes.

`o_voxel.postprocess.to_glb` accepts roughly ten parameters governing remeshing, UV
clustering and bake resolution. The port passes three. Everything this repo has fought
downstream — 11,340 UV islands, 53% atlas coverage, 16,467 boundary edges — has a knob
here that was never turned. See `docs/trellis-prescribed-flow.md`.

This patch adds:

* `--remesh` (plus band and projection) — narrow-band DC remeshing, which snaps vertices
  back to the original surface. Targets boundary edges at source.
* `--uv-refine-iterations`, `--uv-global-iterations`, `--uv-cone-degrees`,
  `--uv-smooth-strength` — the UV clustering controls. Refinement ships **off**
  (`refine_iterations=0`) and the cone threshold sits at a permissive 90 degrees, which
  is what produces the island count.
* Removal of the `--texture-size` choice restriction. There is no cap inside `to_glb` —
  no assert, clamp, min or max — so the 2048 ceiling is purely this argparse list.

`decimation_target` is deliberately left alone. Its 200,000 guard carries the comment
"avoid mtlbvh crash on large meshes", and raising it would also *lower* texels per
triangle, which is the opposite of the goal.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "vendor" / "trellis-mac" / "generate.py"

# The port restricts texture size by argparse alone; to_glb has no such limit.
SIZE_NEEDLE = '        "--texture-size", type=int, default=1024,\n        choices=[512, 1024, 2048],\n'
SIZE_REPLACEMENT = (
    '        "--texture-size", type=int, default=1024,\n'
)

OPTION_ANCHOR = '    parser.add_argument(\n        "--no-texture", action="store_true",\n'
OPTIONS = '''    parser.add_argument(
        "--remesh", action="store_true",
        help="Narrow-band DC remeshing before UV unwrap; snaps vertices back to the "
             "original surface. Targets boundary edges at source",
    )
    parser.add_argument("--remesh-band", type=float, default=1.0)
    parser.add_argument("--remesh-project", type=float, default=0.9)
    parser.add_argument(
        "--uv-refine-iterations", type=int, default=0,
        help="Cluster refinement during UV unwrapping. Ships at 0, which is a large "
             "part of why the atlas fragments",
    )
    parser.add_argument("--uv-global-iterations", type=int, default=1)
    parser.add_argument(
        "--uv-cone-degrees", type=float, default=90.0,
        help="Cone half-angle threshold for UV clustering. 90 is permissive; lower "
             "values should yield fewer, cleaner charts",
    )
    parser.add_argument("--uv-smooth-strength", type=float, default=1.0)
'''

CALL_NEEDLE = """                    decimation_target=target_faces,
                    texture_size=tex_size,
                    verbose=True,
                )"""

CALL_REPLACEMENT = """                    decimation_target=target_faces,
                    texture_size=tex_size,
                    # image-to-3dlab: the quality controls the port never passed.
                    remesh=args.remesh,
                    remesh_band=args.remesh_band,
                    remesh_project=args.remesh_project,
                    mesh_cluster_threshold_cone_half_angle_rad=__import__("math").radians(
                        args.uv_cone_degrees
                    ),
                    mesh_cluster_refine_iterations=args.uv_refine_iterations,
                    mesh_cluster_global_iterations=args.uv_global_iterations,
                    mesh_cluster_smooth_strength=args.uv_smooth_strength,
                    verbose=True,
                )"""


def patch(path: Path) -> None:
    source = path.read_text()

    if "choices=[512, 1024, 2048]" in source:
        if SIZE_NEEDLE not in source:
            raise RuntimeError(f"texture-size option not in the expected form in {path}")
        source = source.replace(SIZE_NEEDLE, SIZE_REPLACEMENT)

    if '"--uv-refine-iterations"' not in source:
        if OPTION_ANCHOR not in source:
            raise RuntimeError(f"expected CLI option anchor not found in {path}")
        source = source.replace(OPTION_ANCHOR, OPTIONS + OPTION_ANCHOR)

    if "mesh_cluster_refine_iterations=args.uv_refine_iterations" not in source:
        if CALL_NEEDLE not in source:
            raise RuntimeError(f"expected to_glb call not found in {path}")
        source = source.replace(CALL_NEEDLE, CALL_REPLACEMENT)

    path.write_text(source)


patch(GENERATOR)
print("Patched TRELLIS to expose remesh, UV clustering, and unrestricted texture size.")
