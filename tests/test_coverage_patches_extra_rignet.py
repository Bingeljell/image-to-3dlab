"""Patch a small RigNet checkout and invoke its patched Python entry points."""

import runpy
import sys
from types import SimpleNamespace

import patch_rignet_macos_compat as patch
import pytest

QUICK_START = '''import os
from sys import platform

def create_single_data(mesh_filaname, mesh_normalized):
    o3d.io.write_triangle_mesh(mesh_filename.replace("_remesh.obj", "_normalized.obj"), mesh_normalized)
    # voxel
    if not os.path.exists(mesh_filaname.replace('_remesh.obj', '_normalized.binvox')):
        if platform == "linux" or platform == "linux2":
            os.system("./binvox -d 88 -pb " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        else:
            raise Exception('Sorry, we currently only support windows and linux.')

    with open(mesh_filaname.replace('_remesh.obj', '_normalized.binvox'), 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)
    return vox

def predict_skeleton(mesh_filename, pred_skel):
    try:
        img = show_obj_skel(mesh_filename, pred_skel.root)
    except:
        print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")
    return pred_skel
'''
POINT_CONV = "from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate\n"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(patch, "RIGNET", tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "utils").mkdir()
    (tmp_path / "quick_start.py").write_text(QUICK_START)
    (tmp_path / "models/PairCls_GCN.py").write_text(POINT_CONV)
    (tmp_path / "utils/binvox_rw.py").write_text('''import numpy as np
def dense_to_sparse(voxel_data, dtype=np.int):
    return dtype
def sparse_to_dense(voxel_data, dims, dtype=np.bool):
    return dtype
''')
    return tmp_path


def test_main_applies_every_rignet_fix_and_leaves_second_run_unchanged(tree, monkeypatch, capsys):
    patch.main()
    output = capsys.readouterr().out
    assert "PointConv -> PointNetConv" in output and "np.int/np.bool -> builtins" in output
    assert "blocking visualizer: disabled" in output
    assert (tree / "utils/voxelize_compat.py").read_bytes() == patch.VOXELIZE_COMPAT_SOURCE.read_bytes()
    assert "import PointNetConv as PointConv" in (tree / "models/PairCls_GCN.py").read_text()
    names = runpy.run_path(str(tree / "utils/binvox_rw.py"))
    assert names["dense_to_sparse"]([]) is int and names["sparse_to_dense"]([], []) is bool

    # The patched pipeline must use its argument even when called by a different module.
    events = []
    monkeypatch.setitem(sys.modules, "utils.voxelize_compat", SimpleNamespace(
        load_voxels=lambda name: events.append(name) or "grid",
    ))
    start = runpy.run_path(str(tree / "quick_start.py"))
    namespace = start["create_single_data"].__globals__
    namespace.update(platform="darwin", o3d=SimpleNamespace(io=SimpleNamespace(
        write_triangle_mesh=lambda path, mesh: events.append((path, mesh)),
    )))
    namespace["show_obj_skel"] = lambda *args: pytest.fail("viewer opened")
    assert start["create_single_data"]("fixture_remesh.obj", "normalized mesh") == "grid"
    assert start["predict_skeleton"]("fixture.obj", "skeleton") == "skeleton"
    assert events == [("fixture_normalized.obj", "normalized mesh"), "fixture_remesh.obj"]

    tracked = [p for p in tree.rglob("*.py")]
    snapshot = {p: p.read_bytes() for p in tracked}
    patch.main()
    assert "already" in capsys.readouterr().out
    assert {p: p.read_bytes() for p in tracked} == snapshot


@pytest.mark.parametrize(("method", "path", "message"), [
    ("patch_binvox_platform_check", "quick_start.py", "platform-check"),
    ("patch_create_single_data_typo", "quick_start.py", "typo line"),
    ("patch_disable_blocking_visualizer", "quick_start.py", "blocking-visualizer"),
    ("patch_pointconv_rename", "models/PairCls_GCN.py", "PointConv import"),
    ("patch_numpy_deprecated_aliases", "utils/binvox_rw.py", "numpy-alias"),
])
def test_rignet_existing_file_with_moved_anchor_is_untouched(tree, method, path, message):
    target = tree / path
    revised = "# upstream changed this file\n"
    target.write_text(revised)
    with pytest.raises(RuntimeError, match=message):
        getattr(patch, method)()
    assert target.read_text() == revised


def test_rignet_main_skips_absent_checkout(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(patch, "RIGNET", tmp_path / "absent")
    patch.main()
    assert "not present" in capsys.readouterr().out


@pytest.mark.parametrize(("platform", "cached", "command"), [
    ("linux", False, "./binvox -d 88 -pb "),
    ("linux2", False, "./binvox -d 88 -pb "),
    ("win32", False, "binvox.exe -d 88 "),
    ("linux", True, None),
    ("win32", True, None),
    ("other", True, None),
])
def test_voxel_loader_reuses_existing_grid_or_runs_the_platform_binary(
    monkeypatch, tmp_path, platform, cached, command,
):
    from scripts import rignet_voxelize_compat as compat

    mesh = str(tmp_path / "figure_remesh.obj")
    output = tmp_path / "figure_normalized.binvox"
    calls = []
    if cached:
        output.write_bytes(b"cached grid")

    def system(argv):
        calls.append(argv)
        output.write_bytes(b"new grid")
        return 7  # The existing path reads the file even if the tool returns a warning.

    monkeypatch.setattr(compat.os, "system", system)
    assert compat.load_voxels(mesh, platform, lambda file: file.read()) == (b"cached grid" if cached else b"new grid")
    expected = [command + str(tmp_path / "figure_normalized.obj")] if command else []
    assert calls == expected


def test_voxel_loader_missing_grid_refuses_unknown_platform(tmp_path, monkeypatch):
    from scripts import rignet_voxelize_compat as compat

    monkeypatch.setattr(compat.os, "system", lambda *a: pytest.fail("binary launch"))
    with pytest.raises(Exception, match="only support windows, linux and macOS"):
        compat.load_voxels(str(tmp_path / "figure_remesh.obj"), "other", lambda file: file.read())


def test_voxel_loader_uses_darwin_geometry_even_if_binvox_cache_exists(tmp_path, monkeypatch):
    from scripts import rignet_voxelize_compat as compat

    path = tmp_path / "figure_normalized.binvox"
    path.write_bytes(b"older cache")
    monkeypatch.setattr(compat, "voxelize_normalized_mesh", lambda name, *, resolution: (name, resolution))
    monkeypatch.setattr(compat.os, "system", lambda *a: pytest.fail("binary launch"))
    got = compat.load_voxels(str(tmp_path / "figure_remesh.obj"), "darwin", lambda _: pytest.fail("cache read"))
    assert got == (str(tmp_path / "figure_normalized.obj"), 88)
