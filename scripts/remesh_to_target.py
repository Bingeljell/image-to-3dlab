"""Decimate a mesh to a target face count via fast_simplification, with timing.

Usage:
    vendor/hunyuan-mlx/.venv/bin/python scripts/remesh_to_target.py <in.glb> <target_faces> <out.obj>
"""

import sys
import time

import trimesh
import fast_simplification


def main() -> None:
    in_path, target_faces, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    t0 = time.time()
    mesh = trimesh.load(in_path, force="mesh")
    print(f"loaded {in_path}: {len(mesh.faces)} faces ({time.time() - t0:.1f}s)", flush=True)

    v_out, f_out = fast_simplification.simplify(
        mesh.vertices, mesh.faces, target_count=target_faces
    )
    print(f"simplified to {len(f_out)} faces ({time.time() - t0:.1f}s)", flush=True)

    out = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    out.export(out_path)
    print(f"DONE ({time.time() - t0:.1f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
