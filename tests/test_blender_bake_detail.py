"""Tests for the bake-detail step: normal map + metallic-roughness onto a finished mesh.

The bpy-free parts are the ones that silently ruin an asset: baking across two meshes
that do not line up (a garbage normal map that still "succeeds"), overwriting a repaint's
own metallic-roughness map with the source's, and arguments arriving in the wrong order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import compress_glb_textures as glb_tools
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_bake_detail.py"


def _load():
    spec = importlib.util.spec_from_file_location("bake_detail", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bd = _load()


def test_arguments_come_after_the_double_dash():
    argv = ["blender", "--background", "--python", "x.py", "--",
            "high.glb", "low.glb", "out.glb", "1024"]
    args = bd.parse_args(argv)
    assert (args.source, args.target, args.output) == ("high.glb", "low.glb", "out.glb")
    assert args.size == 1024


def test_the_map_size_defaults_to_2048():
    assert bd.parse_args(["--", "a.glb", "b.glb", "c.glb"]).size == 2048


def test_too_few_arguments_is_an_error_not_a_guess():
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb"])


def test_a_map_size_outside_the_sane_range_is_refused():
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb", "c.glb", "100"])
    with pytest.raises(SystemExit):
        bd.parse_args(["--", "a.glb", "b.glb", "c.glb", "16384"])


def test_the_same_shape_at_a_different_scale_fits():
    """A repaint may rescale the mesh; a uniform scale is fine and gets corrected."""
    fit = bd.fit(high_size=(0.9, 0.6, 0.88), low_size=(1.8, 1.2, 1.76))
    assert fit.ok
    assert fit.scale == pytest.approx(2.0)
    assert fit.spread == pytest.approx(1.0)


def test_a_rotated_or_different_mesh_is_refused():
    """The orc bake on 2026-09-27 read a spread of 2.06 and baked garbage without complaint."""
    fit = bd.fit(high_size=(0.906, 0.876, 0.612), low_size=(0.908, 0.611, 0.879))
    assert not fit.ok
    assert fit.spread > 1.4


def test_the_real_orc_pair_is_accepted():
    """Axis ratios measured on the aligned orc bake: 1.0021, 0.9986, 1.0035."""
    fit = bd.fit(high_size=(1.0, 1.0, 1.0), low_size=(1.0021, 0.9986, 1.0035))
    assert fit.ok


def test_a_degenerate_mesh_is_refused_rather_than_divided_by():
    fit = bd.fit(high_size=(1.0, 0.0, 1.0), low_size=(1.0, 1.0, 1.0))
    assert not fit.ok


def test_metallic_roughness_is_transferred_onto_a_bare_retopo():
    assert bd.transfer_metallic_roughness(source_has_map=True, target_has_map=False)


def test_a_repaint_keeps_its_own_metallic_roughness():
    """Hunyuan's PBR repaint makes its own map, matched to the new colours."""
    assert not bd.transfer_metallic_roughness(source_has_map=True, target_has_map=True)


def test_nothing_is_transferred_when_the_source_has_no_map():
    assert not bd.transfer_metallic_roughness(source_has_map=False, target_has_map=False)


def test_the_ray_reach_scales_with_the_asset():
    assert bd.ray_reach((2.0, 1.0, 0.5)) == pytest.approx(0.04)
    assert bd.ray_reach((0.5, 0.2, 0.1)) == pytest.approx(0.01)


def test_the_sign_fix_is_the_one_from_the_normal_bake_script():
    """One implementation of the fix, imported, not a re-derived copy (test rule 1)."""
    assert bd.resolve_tangent_sign.__module__ == "blender_bake_normals"


def test_a_texel_pointing_into_the_surface_is_flipped_back_out():
    np = pytest.importorskip("numpy")
    inward = np.array([[[128, 128, 0]]], dtype=np.uint8)   # z = -1
    fixed, fraction = bd.resolve_tangent_sign(inward)
    assert fixed[0, 0, 2] > 250
    assert fraction == pytest.approx(1.0)


def _glb(normals, tangents):
    """The smallest GLB carrying one primitive's normals and tangents."""
    import numpy as np

    normals = np.asarray(normals, dtype="<f4")
    tangents = np.asarray(tangents, dtype="<f4")
    binary = normals.tobytes() + tangents.tobytes()
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": normals.nbytes},
            {"buffer": 0, "byteOffset": normals.nbytes, "byteLength": tangents.nbytes},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": len(tangents), "type": "VEC4"},
        ],
        "meshes": [{"primitives": [{"attributes": {"NORMAL": 0, "TANGENT": 1}}]}],
    }
    return glb_tools.build_glb(document, binary)


def _tangents(glb, count):
    import numpy as np

    _document, binary = glb_tools.parse_glb(glb)
    return np.frombuffer(binary, dtype="<f4", count=count * 4,
                         offset=count * 12).reshape(count, 4)


def test_a_zero_tangent_gets_a_unit_direction_across_its_normal():
    import numpy as np

    normals = [[0, 0, 1], [0.6, 0, 0.8], [1, 0, 0]]
    tangents = [[1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
    repaired, count = bd.repair_zero_tangents(_glb(normals, tangents))
    assert count == 2
    fixed = _tangents(repaired, 3)
    assert fixed[0].tolist() == [1, 0, 0, 1]          # a good tangent is left alone
    for i in (1, 2):
        assert np.linalg.norm(fixed[i, :3]) == pytest.approx(1.0, abs=1e-6)
        assert np.dot(fixed[i, :3], normals[i]) == pytest.approx(0.0, abs=1e-6)
        assert fixed[i, 3] == 1.0


def test_a_zero_normal_still_gets_a_unit_tangent_not_nan():
    import numpy as np

    repaired, count = bd.repair_zero_tangents(_glb([[0, 0, 0]], [[0, 0, 0, 1]]))
    assert count == 1
    fixed = _tangents(repaired, 1)
    assert np.isfinite(fixed).all()
    assert np.linalg.norm(fixed[0, :3]) == pytest.approx(1.0, abs=1e-6)


def test_a_nan_tangent_counts_as_broken():
    import numpy as np

    nan = float("nan")
    repaired, count = bd.repair_zero_tangents(_glb([[0, 0, 1]], [[nan, nan, nan, 1]]))
    assert count == 1
    assert np.isfinite(_tangents(repaired, 1)).all()


def test_a_glb_without_zero_tangents_comes_back_untouched():
    glb = _glb([[0, 0, 1]], [[1, 0, 0, 1]])
    repaired, count = bd.repair_zero_tangents(glb)
    assert count == 0
    assert repaired is glb


def test_a_glb_with_no_binary_chunk_comes_back_untouched():
    glb = glb_tools.build_glb({"asset": {"version": "2.0"}}, b"")
    repaired, count = bd.repair_zero_tangents(glb)
    assert count == 0
    assert repaired is glb


def test_something_that_is_not_a_glb_is_refused():
    with pytest.raises(ValueError):
        bd.repair_zero_tangents(b"PK\x03\x04" + bytes(40))


def test_the_export_lands_at_out_only_once_repaired(tmp_path):
    """Finish's --resume keeps any OUT that is not empty, so until the repair is done the
    exporter writes somewhere else: a bake cut short leaves nothing for it to trust."""
    import numpy as np

    final = tmp_path / "4_baked.glb"
    partial = bd.partial_path(final)
    assert partial.parent == final.parent and partial != final
    assert partial.suffix == ".glb"          # the glTF exporter would append one otherwise
    partial.write_bytes(_glb([[0, 0, 1]], [[0, 0, 0, 1]]))
    assert not final.exists()
    assert bd.finish_export(partial, final) == 1
    assert not partial.exists()
    assert np.linalg.norm(_tangents(final.read_bytes(), 1)[0, :3]) == pytest.approx(1.0)


def test_a_clean_export_is_moved_into_place_as_it_is(tmp_path):
    final = tmp_path / "LOD0.glb"
    glb = _glb([[0, 0, 1]], [[1, 0, 0, 1]])
    bd.partial_path(final).write_bytes(glb)
    assert bd.finish_export(bd.partial_path(final), final) == 0
    assert final.read_bytes() == glb


def test_a_failed_repair_leaves_no_out(tmp_path):
    final = tmp_path / "LOD0.glb"
    bd.partial_path(final).write_bytes(b"PK\x03\x04" + bytes(40))
    with pytest.raises(ValueError):
        bd.finish_export(bd.partial_path(final), final)
    assert not final.exists()


def test_the_glb_reader_it_borrows_loads_inside_blender():
    """Blender's Python has numpy but not Pillow, so the reader must not need it to load."""
    import subprocess
    import sys

    probe = (f"import sys; sys.path.insert(0, {str(SCRIPT.parent)!r}); "
             "import compress_glb_textures; assert 'PIL' not in sys.modules")
    subprocess.run([sys.executable, "-c", probe], check=True)

