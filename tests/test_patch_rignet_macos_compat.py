"""Regression tests for the RigNet macOS-compat patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_rignet_macos_compat.py"
SPEC = importlib.util.spec_from_file_location("patch_rignet_macos_compat", SCRIPT)
patch_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch_mod)

QUICK_START_BEFORE = '''    # voxel
    if not os.path.exists(mesh_filaname.replace('_remesh.obj', '_normalized.binvox')):
        if platform == "linux" or platform == "linux2":
            os.system("./binvox -d 88 -pb " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        elif platform == "win32":
            os.system("binvox.exe -d 88 " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))
        else:
            raise Exception('Sorry, we currently only support windows and linux.')

    with open(mesh_filaname.replace('_remesh.obj', '_normalized.binvox'), 'rb') as fvox:
        vox = binvox_rw.read_as_3d_array(fvox)
'''

POINTCONV_IMPORT_BEFORE = (
    "from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate\n"
)


BINVOX_RW_BEFORE = (
    "import numpy as np\n\n"
    "def dense_to_sparse(voxel_data, dtype=np.int):\n    pass\n\n"
    "def sparse_to_dense(voxel_data, dims, dtype=np.bool):\n    pass\n"
)

CLUSTER_UTILS_BEFORE = (
    "import numpy as np\n\n"
    "def nms_meanshift(sorted_ids):\n"
    "    unique = np.ones(len(sorted_ids), dtype=np.bool)\n"
    "    return unique\n"
)


def _fake_rignet(tmp_path, monkeypatch):
    root = tmp_path / "rignet"
    (root / "utils").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "quick_start.py").write_text(QUICK_START_BEFORE)
    (root / "models" / "PairCls_GCN.py").write_text(POINTCONV_IMPORT_BEFORE)
    (root / "models" / "ROOT_GCN.py").write_text(POINTCONV_IMPORT_BEFORE)
    (root / "utils" / "binvox_rw.py").write_text(BINVOX_RW_BEFORE)
    (root / "utils" / "cluster_utils.py").write_text(CLUSTER_UTILS_BEFORE)
    monkeypatch.setattr(patch_mod, "RIGNET", root)
    return root


def test_write_voxelize_compat_module_is_idempotent(tmp_path, monkeypatch):
    _fake_rignet(tmp_path, monkeypatch)

    assert patch_mod.write_voxelize_compat_module() is True
    written = (tmp_path / "rignet" / "utils" / "voxelize_compat.py").read_text()
    assert "def voxelize_normalized_mesh" in written
    assert '"xyz"' in written  # axis_order must match the coordinate formula used here

    assert patch_mod.write_voxelize_compat_module() is False


def test_patch_binvox_platform_check_adds_darwin_branch_once(tmp_path, monkeypatch):
    _fake_rignet(tmp_path, monkeypatch)

    assert patch_mod.patch_binvox_platform_check() is True
    patched = (tmp_path / "rignet" / "quick_start.py").read_text()
    assert 'platform == "darwin"' in patched
    assert "voxelize_normalized_mesh" in patched
    # the linux/windows branches must still be reachable, unmodified
    assert '"./binvox -d 88 -pb "' in patched
    assert '"binvox.exe -d 88 "' in patched

    assert patch_mod.patch_binvox_platform_check() is False
    assert (tmp_path / "rignet" / "quick_start.py").read_text() == patched


def test_patch_binvox_platform_check_missing_anchor_raises(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)
    (root / "quick_start.py").write_text("# not the real file\n")

    try:
        patch_mod.patch_binvox_platform_check()
    except RuntimeError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a missing anchor")


def test_patch_pointconv_rename_both_files_idempotent(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)

    changed = patch_mod.patch_pointconv_rename()
    assert {p.name for p in changed} == {"PairCls_GCN.py", "ROOT_GCN.py"}

    for name in ("PairCls_GCN.py", "ROOT_GCN.py"):
        text = (root / "models" / name).read_text()
        assert "PointNetConv as PointConv" in text
        assert text.count("PointConv") == 1  # only the "as PointConv" alias; "PointNetConv" doesn't match

    assert patch_mod.patch_pointconv_rename() == []


def test_patch_numpy_deprecated_aliases_idempotent(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)

    changed = patch_mod.patch_numpy_deprecated_aliases()
    assert {p.name for p in changed} == {"binvox_rw.py", "cluster_utils.py"}

    binvox_rw = (root / "utils" / "binvox_rw.py").read_text()
    assert "np.int" not in binvox_rw
    assert "np.bool" not in binvox_rw
    assert "dtype=int" in binvox_rw
    assert "dtype=bool" in binvox_rw

    cluster_utils = (root / "utils" / "cluster_utils.py").read_text()
    assert "np.bool" not in cluster_utils
    assert "dtype=bool" in cluster_utils

    assert patch_mod.patch_numpy_deprecated_aliases() == []


def test_patch_numpy_deprecated_aliases_missing_anchor_raises(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)
    (root / "utils" / "binvox_rw.py").write_text("# not the real file\n")

    try:
        patch_mod.patch_numpy_deprecated_aliases()
    except RuntimeError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for a missing anchor")


def test_patch_create_single_data_typo_idempotent(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)
    quick_start = root / "quick_start.py"
    quick_start.write_text(
        QUICK_START_BEFORE
        + '\n    o3d.io.write_triangle_mesh(mesh_filename.replace("_remesh.obj", "_normalized.obj"), mesh_normalized)\n'
    )

    assert patch_mod.patch_create_single_data_typo() is True
    patched = quick_start.read_text()
    assert 'mesh_filaname.replace("_remesh.obj", "_normalized.obj"), mesh_normalized' in patched

    assert patch_mod.patch_create_single_data_typo() is False
    assert quick_start.read_text() == patched


VIS_CALL_BEFORE = (
    "    try:\n"
    "        img = show_obj_skel(mesh_filename, pred_skel.root)\n"
    "    except:\n"
    '        print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")\n'
)


def test_patch_disable_blocking_visualizer_idempotent(tmp_path, monkeypatch):
    root = _fake_rignet(tmp_path, monkeypatch)
    quick_start = root / "quick_start.py"
    quick_start.write_text(QUICK_START_BEFORE + "\n" + VIS_CALL_BEFORE)

    assert patch_mod.patch_disable_blocking_visualizer() is True
    patched = quick_start.read_text()
    assert "#     img = show_obj_skel(mesh_filename, pred_skel.root)" in patched
    active_calls = [
        line for line in patched.splitlines()
        if "show_obj_skel(mesh_filename" in line and not line.strip().startswith("#")
    ]
    assert active_calls == []

    assert patch_mod.patch_disable_blocking_visualizer() is False
    assert quick_start.read_text() == patched


def test_main_skips_cleanly_when_vendor_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(patch_mod, "RIGNET", tmp_path / "does-not-exist")
    patch_mod.main()
    assert "skipped" in capsys.readouterr().out
