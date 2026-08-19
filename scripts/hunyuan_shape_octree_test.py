"""Shape-only generation at a given octree_resolution, with visible progress.

Usage:
    hunyuan_mlx/paint/.venv/bin/python \
        scripts/hunyuan_shape_octree_test.py <octree_resolution> <image_path> <out_dir>

Mirrors the Stage-1 call in vendor/hunyuan-mlx/tests/test_stage1_to_stage2.py
(same model, guidance_scale, seed) so results are comparable across resolutions.
Writes <out_dir>/shape.glb.
"""

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUNYUAN_MLX = os.path.join(REPO_ROOT, "vendor", "hunyuan-mlx")
sys.path.insert(0, os.path.join(HUNYUAN_MLX, "hy3dshape"))


def main() -> None:
    octree_resolution = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    image_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        REPO_ROOT, "assets_to_test", "game-controller-test.png")
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
        REPO_ROOT, "output", "hunyuan_mlx_zimeng_test", f"controller_octree{sys.argv[1] if len(sys.argv) > 1 else 512}")
    os.makedirs(out_dir, exist_ok=True)
    shape_glb = os.path.join(out_dir, "shape.glb")

    t0 = time.time()
    print(f"[shape octree={octree_resolution}] image={image_path}", flush=True)

    from hy3dshape.pipeline_mlx import ShapePipeline

    print(f"loading pipeline... ({time.time() - t0:.0f}s)", flush=True)
    shape_pipe = ShapePipeline.from_pretrained("dgrauet/hunyuan3d-2.1-mlx")
    print(f"pipeline ready ({time.time() - t0:.0f}s)", flush=True)

    print(f"running denoise + marching cubes at octree={octree_resolution}... ({time.time() - t0:.0f}s)", flush=True)
    mesh = shape_pipe(
        image_path,
        num_inference_steps=50,
        guidance_scale=7.5,
        octree_resolution=octree_resolution,
        seed=42,
    )
    print(f"shape generated ({time.time() - t0:.0f}s): "
          f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces", flush=True)

    mesh.export(shape_glb)
    print(f"DONE ({time.time() - t0:.0f}s) -> {shape_glb}", flush=True)


if __name__ == "__main__":
    main()
