"""Time xatlas.parametrize in isolation, to characterize its face-count scaling.

Usage:
    hunyuan_mlx/paint/.venv/bin/python scripts/xatlas_timing_probe.py <mesh_path>
"""

import sys
import time

import trimesh
import xatlas


def main() -> None:
    mesh_path = sys.argv[1]

    mesh = trimesh.load(mesh_path, force="mesh")
    n_faces = len(mesh.faces)
    print(f"[{mesh_path}] {n_faces} faces — starting xatlas.parametrize...", flush=True)

    t0 = time.time()
    xatlas.parametrize(mesh.vertices, mesh.faces)
    elapsed = time.time() - t0

    print(f"RESULT faces={n_faces} xatlas_seconds={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
