"""Use the shipped sparse lookup directly, with real CPU tensors."""

import importlib
import sys

import pytest
import torch


def test_sparse_lookup_preserves_coordinate_order_misses_and_duplicate_keys():
    from scripts.trellis_sparse_hashmap_compat import CPUHashMap

    lookup = CPUHashMap((1024, 1024, 1024), device="cpu")
    lookup.insert(torch.tensor([[0, 2, 3, 4], [0, 10, 11, 12], [0, 2, 3, 4]]))
    queried = torch.tensor([[0, 10, 11, 12], [0, 2, 3, 4], [0, 0, 0, 0], [0, -1, 2, 3]])
    result = lookup.lookup_3d(queried)
    assert result.dtype == torch.int64
    assert result.device.type == "cpu"
    assert result.tolist() == [1, 2, 0xFFFFFFFF, 0xFFFFFFFF]
    lookup.insert(queried[:1])
    assert lookup.lookup_3d(queried[:2]).tolist() == [0, 0xFFFFFFFF]


def test_patch_installs_identical_helper_and_upgrades_previous_sparse_class(tmp_path, monkeypatch):
    from scripts import patch_trellis_metal_backends as patch
    from scripts import trellis_sparse_hashmap_compat as helper

    target = tmp_path / "deps/trellis2-apple/o-voxel/o_voxel/convert"
    target.mkdir(parents=True)
    (target / "__init__.py").touch()
    grid = target / "flexible_dual_grid.py"
    grid.write_text('''import torch

class _CPUHashMap:
    """Sparse coordinate lookup for non-CUDA decoders."""

    def __init__(self, grid_size, device='cpu'):
        self.device = device
        self._entries = {}

def following_function():
    return "kept"
''')
    assert patch.patch_ovoxel_sparse_hashmap(tmp_path)
    first = grid.read_text()
    copied = target / "trellis_sparse_hashmap_compat.py"
    assert copied.read_bytes() == patch.SPARSE_HASHMAP_SOURCE.read_bytes()
    assert not patch.patch_ovoxel_sparse_hashmap(tmp_path)
    copied.write_text("# stale copy\n")
    assert patch.patch_ovoxel_sparse_hashmap(tmp_path)
    assert copied.read_bytes() == patch.SPARSE_HASHMAP_SOURCE.read_bytes()
    assert grid.read_text() == first
    monkeypatch.syspath_prepend(str(target.parent))
    modules = ("convert.flexible_dual_grid", "convert.trellis_sparse_hashmap_compat", "convert")
    for name in modules:
        # Restore any previously imported generic package name when the test ends.
        monkeypatch.setitem(sys.modules, name, None)
        sys.modules.pop(name)
    try:
        vendor = importlib.import_module("convert.flexible_dual_grid")
        assert vendor.following_function() == "kept"
        instance = vendor._CPUHashMap((512, 512, 512))
        query = torch.tensor([[0, 13, 14, 15], [0, 1, 2, 3]])
        instance.insert(query[:1])
        assert instance.lookup_3d(query).tolist() == [0, 0xFFFFFFFF]
        assert helper.CPUHashMap.__name__ == vendor._CPUHashMap.__name__
    finally:
        for name in modules:
            sys.modules.pop(name, None)


def test_unknown_sparse_lookup_source_does_not_install_helper(tmp_path):
    from scripts import patch_trellis_metal_backends as patch

    target = tmp_path / "deps/trellis2-apple/o-voxel/o_voxel/convert"
    target.mkdir(parents=True)
    grid = target / "flexible_dual_grid.py"
    grid.write_text("class _CPUHashMap:\n    pass\n")
    with pytest.raises(RuntimeError, match="expected one source anchor"):
        patch.patch_ovoxel_sparse_hashmap(tmp_path)
    assert sorted(path.name for path in target.iterdir()) == [grid.name]
