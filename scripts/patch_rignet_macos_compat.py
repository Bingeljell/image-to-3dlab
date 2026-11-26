#!/usr/bin/env python3
"""Make the vendored RigNet checkout run inference on macOS / Apple Silicon.

RigNet (github.com/zhan-xu/RigNet) was built for Ubuntu + CUDA and never anticipated
macOS. Two things block a straight `pip install` + run on this Mac:

1. **The bundled `binvox` binary is Linux x86-64 only** (and `binvox.exe` is
   Windows-only) -- `quick_start.py` raises `Exception('Sorry, we currently only
   support windows and linux.')` on any other platform. `binvox` builds a solid voxel
   occupancy grid used only for inside/outside bone checks, so this patch adds a
   macOS branch that builds the same grid with `trimesh`'s containment test instead
   of shelling out to a missing binary.
2. **`torch_geometric.nn.PointConv` was renamed to `PointNetConv`** somewhere between
   RigNet's 2020-era pin and the modern torch_geometric this environment installs
   (old pinned versions have no Apple Silicon wheels, so this project intentionally
   uses current torch_geometric instead of RigNet's exact pins).
3. **`np.int` / `np.bool` are gone** in modern numpy (removed, not just deprecated) --
   RigNet was written against numpy < 1.20. Two of these are function-default-argument
   values in `binvox_rw.py`, which are evaluated at import time, so that module failed
   to import at all before this patch touches it; a third, in `cluster_utils.py`, is
   only hit once inference actually runs joint clustering.
4. **`create_single_data`'s own parameter is misspelled** `mesh_filaname` everywhere in
   its body except one line, which references the correctly-spelled `mesh_filename`.
   That line only "works" in RigNet's own `quick_start.py` because it runs inside
   `if __name__ == '__main__':`, where a same-named global happens to exist by
   coincidence -- calling `create_single_data` from any other caller (as our own
   `scripts/rignet_infer.py` does) raises `NameError`. Fixes the typo, not a
   platform issue, but it was hit while getting inference running here.
5. **`predict_skeleton` always opens an interactive Open3D viewer window** and blocks
   on it (`vis.create_window()` + `vis.run()` inside `show_obj_skel`, wrapped in a bare
   `try/except` that does nothing for this since blocking isn't an exception). Cost a
   36-minute hang here -- 20 seconds of real CPU time against 36 minutes of wall clock,
   waiting for a human to close a debug window that nobody was there to close. This
   patch comments out that call; the return value (`img`) was never used downstream.

Best-effort, like every other patch script here: never fails a clone that doesn't
have `vendor/rignet` checked out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RIGNET = ROOT / "vendor" / "rignet"

VOXELIZE_COMPAT_SOURCE = Path(__file__).with_name("rignet_voxelize_compat.py")


def write_voxelize_compat_module() -> bool:
    """Drop utils/voxelize_compat.py into the vendored tree. Idempotent."""
    target = RIGNET / "utils" / "voxelize_compat.py"
    source = VOXELIZE_COMPAT_SOURCE.read_bytes()
    if target.exists() and target.read_bytes() == source:
        return False
    target.write_bytes(source)
    return True


def patch_binvox_platform_check() -> bool:
    """Route darwin through voxelize_compat instead of the missing binvox binary."""
    path = RIGNET / "quick_start.py"
    source = path.read_text()

    needle = (
        "    # voxel\n"
        "    if not os.path.exists(mesh_filaname.replace('_remesh.obj', '_normalized.binvox')):\n"
        '        if platform == "linux" or platform == "linux2":\n'
        '            os.system("./binvox -d 88 -pb " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))\n'
        '        elif platform == "win32":\n'
        '            os.system("binvox.exe -d 88 " + mesh_filaname.replace("_remesh.obj", "_normalized.obj"))\n'
        "        else:\n"
        "            raise Exception('Sorry, we currently only support windows and linux.')\n"
        "\n"
        "    with open(mesh_filaname.replace('_remesh.obj', '_normalized.binvox'), 'rb') as fvox:\n"
        "        vox = binvox_rw.read_as_3d_array(fvox)\n"
    )
    replacement = (
        "    # voxel\n"
        "    from utils.voxelize_compat import load_voxels\n"
        "    vox = load_voxels(mesh_filaname)\n"
    )
    if replacement in source:
        return False
    if '    if platform == "darwin":\n' in source:
        legacy_start = '    # voxel\n    normalized_obj = mesh_filaname.replace("_remesh.obj", "_normalized.obj")\n'
        legacy_end = "            vox = binvox_rw.read_as_3d_array(fvox)\n"
        start = source.find(legacy_start)
        end = source.find(legacy_end, start + len(legacy_start))
        if start >= 0 and end >= 0:
            needle = source[start:end + len(legacy_end)]
            if hashlib.sha256(needle.encode()).hexdigest() != (
                "f12d6a43b69ca97291e5682382120ed744ece5094786ffe0f4feee09ef2b0c63"
            ):
                raise RuntimeError(f"previous RigNet voxel patch has changed in {path}")
    if source.count(needle) != 1:
        raise RuntimeError(f"expected binvox platform-check block not found exactly once in {path}")
    path.write_text(source.replace(needle, replacement))
    return True


def patch_pointconv_rename() -> list[Path]:
    """torch_geometric renamed PointConv -> PointNetConv; alias it back locally."""
    needle = "from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate"
    replacement = "from torch_geometric.nn import PointNetConv as PointConv, fps, radius, global_max_pool, knn_interpolate"

    changed = []
    for name in ("models/PairCls_GCN.py", "models/ROOT_GCN.py"):
        path = RIGNET / name
        if not path.exists():
            continue
        source = path.read_text()
        if replacement in source:
            continue
        if needle not in source:
            raise RuntimeError(f"expected PointConv import not found in {path}")
        path.write_text(source.replace(needle, replacement))
        changed.append(path)
    return changed


NUMPY_ALIAS_FIXES = {
    "utils/binvox_rw.py": [
        ("def dense_to_sparse(voxel_data, dtype=np.int):", "def dense_to_sparse(voxel_data, dtype=int):"),
        ("def sparse_to_dense(voxel_data, dims, dtype=np.bool):", "def sparse_to_dense(voxel_data, dims, dtype=bool):"),
    ],
    "utils/cluster_utils.py": [
        ("unique = np.ones(len(sorted_ids), dtype=np.bool)", "unique = np.ones(len(sorted_ids), dtype=bool)"),
    ],
}


def patch_numpy_deprecated_aliases() -> list[Path]:
    """np.int / np.bool were removed (not just deprecated) in modern numpy."""
    changed = []
    for name, replacements in NUMPY_ALIAS_FIXES.items():
        path = RIGNET / name
        if not path.exists():
            continue
        source = path.read_text()
        original = source
        for needle, replacement in replacements:
            if replacement in source:
                continue
            if needle not in source:
                raise RuntimeError(f"expected numpy-alias line not found in {path}: {needle!r}")
            source = source.replace(needle, replacement)
        if source != original:
            path.write_text(source)
            changed.append(path)
    return changed


def patch_create_single_data_typo() -> bool:
    """One line in create_single_data references the misspelled param by its correct
    spelling, which only resolves because RigNet's own __main__ block happens to set a
    same-named global. Any other caller hits NameError."""
    path = RIGNET / "quick_start.py"
    source = path.read_text()

    needle = (
        '    o3d.io.write_triangle_mesh(mesh_filename.replace("_remesh.obj", '
        '"_normalized.obj"), mesh_normalized)'
    )
    replacement = (
        '    o3d.io.write_triangle_mesh(mesh_filaname.replace("_remesh.obj", '
        '"_normalized.obj"), mesh_normalized)'
    )
    if replacement in source:
        return False
    if needle not in source:
        raise RuntimeError(f"expected create_single_data typo line not found in {path}")
    path.write_text(source.replace(needle, replacement))
    return True


def patch_disable_blocking_visualizer() -> bool:
    """predict_skeleton opens an interactive Open3D window and blocks on vis.run()."""
    path = RIGNET / "quick_start.py"
    source = path.read_text()

    needle = (
        "    try:\n"
        "        img = show_obj_skel(mesh_filename, pred_skel.root)\n"
        "    except:\n"
        '        print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")\n'
    )
    replacement = (
        "    # image-to-3dlab: show_obj_skel blocks forever on vis.run() waiting for a\n"
        "    # human to close its window -- disabled for headless/batch inference.\n"
        "    # try:\n"
        "    #     img = show_obj_skel(mesh_filename, pred_skel.root)\n"
        "    # except:\n"
        '    #     print("Visualization is not supported on headless servers. Please consider other headless rendering methods.")\n'
    )
    if replacement in source:
        return False
    if needle not in source:
        raise RuntimeError(f"expected blocking-visualizer call not found in {path}")
    path.write_text(source.replace(needle, replacement))
    return True


def main() -> None:
    if not RIGNET.exists():
        print(f"  [patch_rignet_macos_compat] skipped: {RIGNET} not present")
        return

    wrote_module = write_voxelize_compat_module()
    print(f"  [patch_rignet_macos_compat] voxelize_compat.py: {'wrote' if wrote_module else 'already current'}")

    patched_quick_start = patch_binvox_platform_check()
    print(f"  [patch_rignet_macos_compat] quick_start.py binvox branch: {'patched' if patched_quick_start else 'already patched'}")

    renamed = patch_pointconv_rename()
    if renamed:
        for path in renamed:
            print(f"  [patch_rignet_macos_compat] PointConv -> PointNetConv in {path.relative_to(RIGNET)}")
    else:
        print("  [patch_rignet_macos_compat] PointConv rename: already applied")

    numpy_fixed = patch_numpy_deprecated_aliases()
    if numpy_fixed:
        for path in numpy_fixed:
            print(f"  [patch_rignet_macos_compat] np.int/np.bool -> builtins in {path.relative_to(RIGNET)}")
    else:
        print("  [patch_rignet_macos_compat] numpy alias fixes: already applied")

    typo_fixed = patch_create_single_data_typo()
    print(f"  [patch_rignet_macos_compat] create_single_data typo: {'fixed' if typo_fixed else 'already fixed'}")

    vis_disabled = patch_disable_blocking_visualizer()
    print(f"  [patch_rignet_macos_compat] blocking visualizer: {'disabled' if vis_disabled else 'already disabled'}")


if __name__ == "__main__":
    main()
