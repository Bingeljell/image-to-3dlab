"""macOS replacement for the Linux/Windows-only binvox binary.

RigNet's `binvox` executable ships as a Linux x86-64 ELF (plus a `binvox.exe` for
Windows), with no macOS build, so `quick_start.py` cannot voxelize on darwin. This
rebuilds the same solid occupancy grid `binvox_rw.read_as_3d_array` would hand back --
only `.data` / `.dims` / `.translate` / `.scale` are ever read downstream (see
`utils/mst_utils.py:inside_check` and `utils/vis_utils.py`).

**Do not query `trimesh.Trimesh.contains()` directly on a large point batch** -- it
was tried first here and used ~9GB RSS for just 20,000 of the 681,472 grid points
this needs (extrapolates to killing the process well before the full grid, which is
exactly what happened: two runs died with SIGKILL). `Trimesh.voxelized(pitch).fill()`
takes <1s and ~600MB for the same mesh: it builds a solid grid once via surface
rasterization + flood-fill, then `VoxelGrid.is_filled(points)` is a cheap index
lookup into that already-computed grid, not a fresh raycast per point.

Trimesh's own `voxelized()` sizes its grid to best-fit the bounding box per axis, so
its shape usually isn't a cube (e.g. (55, 89, 87)) -- `mst_utils.py`'s coordinate
formula assumes a single uniform scale across all three axes (it indexes `dims[0]`
for every axis), matching binvox's own convention. So this resamples that fast grid
onto an explicit `resolution`-cubed grid built from our own translate/scale, via
`is_filled`, rather than trusting trimesh's native shape.

Installed by scripts/patch_rignet_macos_compat.py; re-run that script after any fresh
`git clone` of vendor/rignet since vendor/ is git-ignored.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:
    from utils.binvox_rw import Voxels


def voxelize_normalized_mesh(obj_path: str, resolution: int = 88) -> Voxels:
    """Solid occupancy grid for a mesh already normalized by quick_start.normalize_obj.

    Cell (i, j, k) is True iff its center -- (i+.5)/resolution etc., scaled and
    translated by the mesh's own bounding box -- lies inside the mesh surface.
    """
    from utils.binvox_rw import Voxels

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    bmin, bmax = mesh.bounds
    scale = float((bmax - bmin).max())
    translate = bmin.astype(float)
    pitch = scale / resolution

    filled = mesh.voxelized(pitch=pitch).fill()

    centers = (np.arange(resolution) + 0.5) / resolution
    xs = scale * centers + translate[0]
    ys = scale * centers + translate[1]
    zs = scale * centers + translate[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)

    inside = filled.is_filled(points)
    data = inside.reshape(resolution, resolution, resolution)

    return Voxels(data, [resolution] * 3, translate.tolist(), scale, "xyz")


def load_voxels(mesh_filename: str, platform: str | None = None, read_voxels=None) -> Voxels:
    """Get RigNet's occupancy grid using the native binary or the macOS implementation."""
    platform = sys.platform if platform is None else platform
    normalized_obj = mesh_filename.replace("_remesh.obj", "_normalized.obj")
    if platform == "darwin":
        return voxelize_normalized_mesh(normalized_obj, resolution=88)

    filename = mesh_filename.replace("_remesh.obj", "_normalized.binvox")
    if not os.path.exists(filename):
        if platform in ("linux", "linux2"):
            os.system("./binvox -d 88 -pb " + normalized_obj)
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + normalized_obj)
        else:
            raise Exception("Sorry, we currently only support windows, linux and macOS.")  # noqa: TRY002

    if read_voxels is None:
        from utils.binvox_rw import read_as_3d_array

        read_voxels = read_as_3d_array
    with open(filename, "rb") as grid:
        return read_voxels(grid)
