"""Check the voxel grid that RigNet consumes on macOS."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import trimesh

from scripts import patch_rignet_macos_compat as patch


def test_rignet_voxelization_uses_cell_centers_and_xyz_axes(tmp_path, monkeypatch):
    utils = ModuleType("utils")
    utils.__path__ = []
    binvox = ModuleType("utils.binvox_rw")
    binvox.Voxels = lambda data, dims, translate, scale, axis_order: SimpleNamespace(
        data=data, dims=dims, translate=translate, scale=scale, axis_order=axis_order
    )
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.binvox_rw", binvox)
    compat = importlib.import_module("scripts.rignet_voxelize_compat")

    mesh = trimesh.creation.box(extents=[4, 2, 1])
    mesh.apply_translation([3, -2, 1])
    obj = tmp_path / "box.obj"
    mesh.export(obj)

    voxels = compat.voxelize_normalized_mesh(str(obj), resolution=12)
    assert voxels.data.dtype == np.bool_
    assert voxels.dims == [12, 12, 12]
    assert voxels.translate == [1, -3, 0.5]
    assert voxels.scale == 4
    assert voxels.axis_order == "xyz"
    assert voxels.data[9, 1, 1]
    assert not voxels.data[9, 9, 1]
    assert not voxels.data[9, 1, 9]

    vendor = tmp_path / "rignet"
    (vendor / "utils").mkdir(parents=True)
    monkeypatch.setattr(patch, "RIGNET", vendor)
    assert patch.write_voxelize_compat_module()
    assert (vendor / "utils/voxelize_compat.py").read_bytes() == patch.VOXELIZE_COMPAT_SOURCE.read_bytes()
    assert not patch.write_voxelize_compat_module()
