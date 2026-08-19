"""End-to-end, single-repo Hunyuan3D-MLX generation: image -> textured GLB.

Unlike scripts/hunyuan_mlx_generate.py (dgrauet shape + ZimengXiong paint, two repos), this
uses ZimengXiong's *own* shape stage (vendor/hunyuan-mlx-paint/python/shape/hy3dmlx) chained
into their own paint stage (vendor/hunyuan-mlx-paint/python/paint) -- one repo, one author,
end to end. Otherwise mirrors hunyuan_mlx_generate.py's shape: one CLI, per-stage timed
prints, a ``<out>.json`` manifest.

Usage:
    vendor/hunyuan-mlx-paint/python/shape/.venv/bin/python \
        scripts/hunyuan_mlx_xiong_generate.py input.png output.glb \
        [--octree-resolution 512] [--seed 42] [--quantize 8] [--compile-dit] \
        [--octree-decode] [--decimation-target 300000] [--paint-seed 0] \
        [--paint-res 512] [--paint-steps 15] [--paint-tex 4096]

Caveat, unvalidated at speed: a real run of this shape stage (2.1, --quantize 0, no
--compile-dit, no --octree-decode, octree=512) took ~48 minutes just for the shape stage on
this machine -- dgrauet's equivalent shape stage takes ~5 minutes. --quantize 8
--compile-dit --octree-decode are default-on here specifically to claw that back, but no
timed run at those settings exists yet; treat the first real run as the actual benchmark.
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
SHAPE_ROOT = REPO / "vendor" / "hunyuan-mlx-paint" / "python" / "shape"
SHAPE_WEIGHTS = SHAPE_ROOT / "weights" / "Hunyuan3D-2.1" / "hunyuan3d-dit-v2-1"
PAINT_ROOT = REPO / "vendor" / "hunyuan-mlx-paint" / "python" / "paint"
PAINT_PYTHON = PAINT_ROOT / ".venv" / "bin" / "python"
PAINT_SCRIPT = PAINT_ROOT / "scripts" / "run_paint_pbr.py"

sys.path.insert(0, str(SHAPE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--octree-resolution", type=int, default=512,
                         choices=[256, 384, 512, 1024])
    parser.add_argument("--seed", type=int, default=42, help="shape-stage seed")
    parser.add_argument("--quantize", type=int, default=8, choices=[0, 4, 8],
                         help="4/8-bit DiT+DINO; 0 = full precision (much slower)")
    parser.add_argument("--compile-dit", dest="compile_dit", action="store_true", default=True)
    parser.add_argument("--no-compile-dit", dest="compile_dit", action="store_false")
    parser.add_argument("--octree-decode", dest="octree_decode", action="store_true",
                         default=True, help="FlashVDM-style near-surface-only decode (faster)")
    parser.add_argument("--no-octree-decode", dest="octree_decode", action="store_false")
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--paint-seed", type=int, default=0)
    parser.add_argument("--paint-res", type=int, default=512)
    parser.add_argument("--paint-steps", type=int, default=15)
    parser.add_argument("--paint-tex", type=int, default=4096)
    return parser.parse_args()


def run_shape(image: Path, octree_resolution: int, seed: int, quantize: int,
              compile_dit: bool, octree_decode: bool, t0: float):
    import mlx.core as mx
    from hy3dmlx.pipeline import Hunyuan3DShapePipeline

    print(f"loading shape pipeline (quantize={quantize or 'off'})... "
          f"({time.time() - t0:.0f}s)", flush=True)
    pipe = Hunyuan3DShapePipeline.from_pretrained(
        str(SHAPE_WEIGHTS), dtype=mx.float16, quantize=quantize or None)
    print(f"shape pipeline ready ({time.time() - t0:.0f}s)", flush=True)
    mesh = pipe.generate(
        str(image), num_inference_steps=30, guidance_scale=5.0,
        octree_resolution=octree_resolution, seed=seed,
        octree_decode=octree_decode, compile_dit=compile_dit,
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
    # Same self-sufficiency fix as hunyuan_mlx_generate.py: run_paint_pbr.py's bare
    # Image.save("outputs/...") calls crash post-diffusion on a fresh outputs/ dir.
    (PAINT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
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
    print(f"[hunyuan-mlx-xiong] image={args.image} octree={args.octree_resolution} "
          f"quantize={args.quantize}", flush=True)

    mesh = run_shape(args.image, args.octree_resolution, args.seed, args.quantize,
                      args.compile_dit, args.octree_decode, t0)
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
        "backend": "hunyuan-mlx-xiong",
        "input": str(args.image),
        "output": str(args.output),
        "parameters": {
            "octree_resolution": args.octree_resolution,
            "seed": args.seed,
            "quantize": args.quantize,
            "compile_dit": args.compile_dit,
            "octree_decode": args.octree_decode,
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
