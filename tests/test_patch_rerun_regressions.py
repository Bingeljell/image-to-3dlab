"""Regressions found by re-checking the old review (e8dede4) against main.

Every test here uses synthetic source strings or tmp files. None needs the vendored
checkouts, none writes into vendor/, and none touches the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import patch_metal_hashmap_miss as hashmap
import patch_mtlbvh_production_traversal as mtlbvh
import patch_pixal3d_rembg as pixal_rembg
import patch_trellis_enable_cleanup as cleanup
import patch_trellis_face_cap as face_cap
import patch_trellis_mlx_attention as mlx
import patch_trellis_no_bria as no_bria
import patch_trellis_rocm as rocm


def _argv(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["patch", *args])


# --- 1. --check must fail when the patch is absent ------------------------------------


def test_hashmap_check_on_absent_patch_exits_nonzero(tmp_path, monkeypatch):
    path = tmp_path / "remesh.metal"
    path.write_text(
        "".join(sig + "\n" + original for sig, original, _ in hashmap.PATCHES)
    )
    _argv(monkeypatch, "--check", "--path", str(path))
    assert hashmap.main() != 0, "--check returned 0 although the patch is ABSENT"


def test_mtlbvh_check_on_absent_patch_exits_nonzero(tmp_path, monkeypatch):
    metal = tmp_path / "bvh.metal"
    native = tmp_path / "metal_bvh.mm"
    metal.write_text("stack based traversal only\n")
    native.write_text(mtlbvh.CPU_OLD + "\n")
    monkeypatch.setattr(mtlbvh, "METAL", metal)
    monkeypatch.setattr(mtlbvh, "NATIVE", native)
    _argv(monkeypatch, "--check")
    assert mtlbvh.main() != 0, "--check returned 0 although the patch is ABSENT"


def test_enable_cleanup_check_on_stubbed_source_exits_nonzero(tmp_path, monkeypatch):
    path = tmp_path / "base.py"
    path.write_text("".join(sig + stub for sig, stub in cleanup.STUBS))
    _argv(monkeypatch, "--check", "--path", str(path))
    assert cleanup.main() != 0, "--check returned 0 although cleanup is still stubbed"


def test_face_cap_check_on_unpatched_source_exits_nonzero(tmp_path, monkeypatch):
    path = tmp_path / "generate.py"
    path.write_text(
        "target_faces = min(args.bake_target_faces, 200000, len(faces_np))\n"
    )
    _argv(monkeypatch, "--check", "--target", str(path))
    assert face_cap.main() != 0, "--check returned 0 although the patch is NOT PATCHED"


def test_mlx_check_on_unpatched_tree_exits_nonzero(tmp_path):
    root = tmp_path / "TRELLIS.2"
    config = root / "trellis2" / "modules" / "sparse" / "config.py"
    attn = root / "trellis2" / "modules" / "sparse" / "attention" / "full_attn.py"
    attn.parent.mkdir(parents=True)
    config.write_text(mlx.BACKEND_LIST_OLD + "\n" + mlx.BACKEND_LIST_OLD + "\n")
    attn.write_text(mlx.DISPATCH_ANCHOR + "\n")
    before = config.read_text(), attn.read_text()
    assert mlx.main(["--check", "--root", str(root)]) != 0, (
        "--check returned 0 although files would change"
    )
    assert (config.read_text(), attn.read_text()) == before


# --- 2. BRIA: a note beside a live constructor is not "patched" -----------------------

BRIA_NEEDLE = (
    "        pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)
BRIA_NOTE = (
    "        # image-to-3dlab disables separately licensed background models.\n"
    "        # Inputs must already contain a non-opaque alpha channel.\n"
    "        pipeline.rembg_model = None"
)


@pytest.mark.parametrize(
    "text",
    [
        BRIA_NOTE + "\n" + BRIA_NEEDLE + "\n",
        BRIA_NEEDLE + "\n" + BRIA_NOTE + "\n",
    ],
    ids=["constructor-after-note", "constructor-before-note"],
)
def test_no_bria_removes_a_constructor_that_sits_beside_the_note(tmp_path, text):
    path = tmp_path / "trellis2_image_to_3d.py"
    path.write_text(text)
    try:
        no_bria.patch(path)
    except RuntimeError:
        return  # refusing loudly is acceptable; leaving it live is not
    assert BRIA_NEEDLE not in path.read_text(), (
        "patch() returned success with the BRIA constructor still in the file"
    )


# --- 5. Pixal3D rembg ----------------------------------------------------------------

_BIREFNET_NOTE_ONLY = (
    "class BiRefNet:\n"
    '    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):\n'
    f"        # {pixal_rembg.MARKER}: note kept, guard code lost\n"
    '        self.model = load("briaai/RMBG-2.0")\n'
)


def test_pixal3d_marker_without_the_guard_is_not_patched():
    assert not pixal_rembg.is_patched(_BIREFNET_NOTE_ONLY)


def test_pixal3d_apply_repairs_a_file_that_kept_only_the_marker():
    repaired = pixal_rembg.apply(_BIREFNET_NOTE_ONLY)
    assert repaired != _BIREFNET_NOTE_ONLY
    assert 'model_name.startswith("briaai/")' in repaired


def test_pixal3d_check_on_marker_only_file_exits_nonzero(tmp_path, monkeypatch):
    path = tmp_path / "BiRefNet.py"
    path.write_text(_BIREFNET_NOTE_ONLY)
    _argv(monkeypatch, "--check", str(path))
    assert pixal_rembg.main() != 0, "--check says patched; a run would load BRIA"


# --- 6. half-done states -------------------------------------------------------------

_HALF_CONFIG = (
    f"ok = x in {mlx.BACKEND_LIST_NEW}\n"
    f"def set_attn_backend(backend: Literal{mlx.BACKEND_LIST_OLD}):\n    pass\n"
)


def test_mlx_config_with_one_allow_list_done_is_finished_or_refused():
    try:
        patched, _ = mlx.patch_config(_HALF_CONFIG)
    except RuntimeError:
        return
    assert patched.count(mlx.BACKEND_LIST_NEW) == 2, (
        "patch_config reported 'already applied' with the Literal still old"
    )


def _half_cxx20_setup(tmp_path: Path) -> tuple[Path, rocm.Patch]:
    spec = rocm.CXX20_PATCHES[0]
    path = tmp_path / "setup.py"
    flags = ["-std=c++20"] + ["-std=c++17"] * (spec.expected - 1)
    path.write_text("".join(f'    "{flag}",\n' for flag in flags))
    return path, spec


def test_half_done_cxx20_setup_py_is_finished_on_rerun(tmp_path):
    path, spec = _half_cxx20_setup(tmp_path)
    try:
        rocm.patch_file(path, spec)
    except RuntimeError:
        pytest.fail("a re-run refuses to finish a half-patched setup.py")
    assert "-std=c++17" not in path.read_text(), "a C++17 flag survived the re-run"
