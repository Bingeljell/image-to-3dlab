"""Tests for staging assets in the live Blender viewport.

The bpy-free parts decide whether a comparison is readable: offsets that centre the group,
and collection names that make "the second one" unambiguous between the viewport and the
outliner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_stage.py"


def _load():
    spec = importlib.util.spec_from_file_location("blender_stage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = _load()


def test_single_asset_sits_at_the_origin():
    assert stage.spread_offsets(1) == [0.0]


def test_two_assets_straddle_the_origin():
    assert stage.spread_offsets(2, gap=2.0) == [-1.0, 1.0]


def test_three_assets_centre_the_middle_one():
    assert stage.spread_offsets(3, gap=1.0) == [-1.0, 0.0, 1.0]


def test_offsets_stay_centred_for_any_count():
    for count in range(1, 8):
        offsets = stage.spread_offsets(count)
        assert sum(offsets) == pytest.approx(0.0)


def test_no_assets_is_not_an_error():
    assert stage.spread_offsets(0) == []


# --- labels ---------------------------------------------------------------------------


def test_labels_default_to_file_stems():
    names = stage.labels_for(["/tmp/old.glb", "/tmp/new.glb"], None)
    assert "old" in names[0] and "new" in names[1]


def test_labels_are_index_prefixed_so_outliner_order_matches_the_viewport():
    names = stage.labels_for(["/a.glb", "/b.glb", "/c.glb"], ["zebra", "apple", "mango"])
    assert names[0].startswith("A_")
    assert names[1].startswith("B_")
    assert names[2].startswith("C_")
    assert sorted(names) == names   # alphabetical order == left-to-right order


def test_labels_sanitise_awkward_characters():
    names = stage.labels_for(["/a.glb"], ["ours: 300k (uncapped)"])
    assert all(ch.isalnum() or ch == "_" for ch in names[0])


def test_mismatched_label_count_is_rejected():
    """Silently mislabelling a comparison is worse than refusing to stage it."""
    with pytest.raises(SystemExit):
        stage.labels_for(["/a.glb", "/b.glb"], ["only-one"])


# --- generated code -------------------------------------------------------------------


def test_viewport_is_set_to_material_preview():
    """Otherwise the user sees untextured grey and asks where the texture went."""
    code = stage.build_code(["/a.glb"], ["A_x"], [0.0], culled=True)
    assert 'shading.type = "MATERIAL"' in code


def test_backface_culling_is_on_by_default():
    code = stage.build_code(["/a.glb"], ["A_x"], [0.0], culled=True)
    assert "culled = True" in code


def test_double_sided_mode_disables_culling():
    code = stage.build_code(["/a.glb"], ["A_x"], [0.0], culled=False)
    assert "culled = False" in code


def test_nothing_is_left_selected():
    """A selected object draws orange, which has been mistaken for a mesh property."""
    code = stage.build_code(["/a.glb"], ["A_x"], [0.0], culled=True)
    assert 'bpy.ops.object.select_all(action="DESELECT")' in code


def test_generated_code_compiles():
    code = stage.build_code(["/a.glb", "/b.glb"], ["A_x", "B_y"], [-1.0, 1.0], culled=True)
    compile(code, "<blender>", "exec")
