"""End-to-end Hunyuan3D-MLX generation: image -> textured GLB.

Chains dgrauet's shape stage (MLX, octree-resolution marching-cubes decode), a
fast_simplification remesh, and ZimengXiong's multi-view PBR paint stage (a *separate* venv,
subprocess-invoked) into one process. Mirrors scripts/trellis_space_generate.py's shape: one
CLI, per-stage timed prints, a ``<out>.json`` manifest.

Usage:
    vendor/hunyuan-mlx/.venv/bin/python scripts/hunyuan_mlx_generate.py \
        input.png output.glb [--octree-resolution 512] [--seed 42] \
        [--decimation-target 300000] [--paint-seed 0] [--paint-res 512] \
        [--paint-steps 15] [--paint-tex 4096]

The octree=512 / decimation-target~300-500k recipe was validated end-to-end on 2026-08-18
(Controller, Fox, Snag, Flicker) — see docs/STATE-OF-REPO-2026-08-17.md. The paint stage's
xatlas UV-unwrap step has a hard wall between 500k-700k faces (confirmed same day); do not
raise --decimation-target past ~500k.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHAPE_ROOT = REPO / "vendor" / "hunyuan-mlx"
PAINT_ROOT = REPO / "vendor" / "hunyuan-mlx-paint" / "python" / "paint"
PAINT_PYTHON = PAINT_ROOT / ".venv" / "bin" / "python"
PAINT_SCRIPT = PAINT_ROOT / "scripts" / "run_paint_pbr.py"

sys.path.insert(0, str(SHAPE_ROOT / "hy3dshape"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--octree-resolution", type=int, default=512,
                         choices=[256, 384, 512, 1024])
    parser.add_argument("--seed", type=int, default=42, help="shape-stage seed")
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    return parser.parse_args()


def run_shape(image: Path, octree_resolution: int, seed: int, t0: float):
    from hy3dshape.pipeline_mlx import ShapePipeline

    print(f"loading shape pipeline... ({time.time() - t0:.0f}s)", flush=True)
    shape_pipe = ShapePipeline.from_pretrained("dgrauet/hunyuan3d-2.1-mlx")
    print(f"shape pipeline ready ({time.time() - t0:.0f}s)", flush=True)
    mesh = shape_pipe(
        str(image), num_inference_steps=50, guidance_scale=7.5,
        octree_resolution=octree_resolution, seed=seed,
    )
    print(f"shape generated ({time.time() - t0:.0f}s): "
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)
    return mesh


def run_remesh(mesh, decimation_target: int, t0: float) -> tuple[object, bool]:
    if len(mesh.faces) <= decimation_target:
        print(f"mesh at/under decimation target ({len(mesh.faces):,} <= "
              f"{decimation_target:,}); no remesh needed ({time.time() - t0:.0f}s)", flush=True)
        return mesh, False
    import fast_simplification
    import trimesh

    v_out, f_out = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_count=decimation_target
    )
    print(f"simplified to {len(f_out):,} faces ({time.time() - t0:.0f}s)", flush=True)
    return trimesh.Trimesh(vertices=v_out, faces=f_out, process=False), True


def run_paint(mesh_path: Path, image: Path, output: Path, paint_seed: int, paint_res: int,
              paint_steps: int, paint_tex: int, t0: float) -> None:
    env = {
        **os.environ, "PYTHONUNBUFFERED": "1",
        "PAINT_MESH": str(mesh_path.resolve()), "PAINT_IMG": str(image.resolve()),
        "PAINT_SEED": str(paint_seed), "PAINT_RES": str(paint_res),
        "PAINT_STEPS": str(paint_steps), "PAINT_TEX": str(paint_tex),
    }
    print(f"starting paint stage ({time.time() - t0:.0f}s)", flush=True)
    proc = subprocess.run(
        [str(PAINT_PYTHON), "-u", str(PAINT_SCRIPT)], cwd=str(PAINT_ROOT), env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"paint stage exited with code {proc.returncode}")

    result = PAINT_ROOT / "outputs" / "textured_mesh_pbr.glb"
    if not result.is_file():
        raise RuntimeError(f"paint stage did not produce {result}")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.replace(output)
    for name in ("pbr_albedo_texture.png", "pbr_albedo_views.png",
                 "pbr_mr_texture.png", "pbr_mr_views.png"):
        src = PAINT_ROOT / "outputs" / name
        if src.is_file():
            src.replace(output.with_name(f"{output.stem}_{name}"))
    print(f"paint stage done ({time.time() - t0:.0f}s) -> {output}", flush=True)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(f"[hunyuan-mlx] image={args.image} octree={args.octree_resolution}", flush=True)

    mesh = run_shape(args.image, args.octree_resolution, args.seed, t0)
    shape_seconds = time.time() - t0

    mesh, remeshed = run_remesh(mesh, args.decimation_target, t0)
    remesh_seconds = time.time() - t0 - shape_seconds

    tmp_mesh = args.output.parent / f"{args.output.stem}_shape.obj"
    tmp_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(tmp_mesh))

    paint_t0 = time.time()
    run_paint(tmp_mesh, args.image, args.output, args.paint_seed, args.paint_res,
              args.paint_steps, args.paint_tex, t0)
    paint_seconds = time.time() - paint_t0

    total = time.time() - t0
    manifest = {
        "schema_version": 1,
        "backend": "hunyuan-mlx",
        "input": str(args.image),
        "output": str(args.output),
        "parameters": {
            "octree_resolution": args.octree_resolution,
            "seed": args.seed,
            "decimation_target": args.decimation_target,
            "remeshed": remeshed,
            "paint_seed": args.paint_seed,
            "paint_res": args.paint_res,
            "paint_steps": args.paint_steps,
            "paint_tex": args.paint_tex,
        },
        "faces": len(mesh.faces),
        "timings_seconds": {
            "shape": round(shape_seconds, 1),
            "remesh": round(remesh_seconds, 1),
            "paint": round(paint_seconds, 1),
            "total": round(total, 1),
        },
    }
    manifest_path = args.output.with_name(f"{args.output.stem}.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Total: {total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
