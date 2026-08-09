"""The culled-render flags must actually reach the Blender script.

A flag that is parsed but never threaded through produces a render that looks like
the honest test and is not one -- the exact failure mode this project has hit
repeatedly (`--bake-target-faces` was inert on every Apple Silicon run for the
repo's entire history, and the multi-view marker was swallowed by its own log
filter). So assert on the generated code, which is the real artifact sent to Blender.
"""

from __future__ import annotations

from pathlib import Path

from scripts.blender_render_asset import blender_code


def _code(**kwargs) -> str:
    return blender_code(Path("/tmp/a.glb"), Path("/tmp/out"), "fox", "dark", **kwargs)


def test_culled_off_by_default():
    code = _code()
    assert "if False:" in code
    assert "use_backface_culling" in code, "the branch should exist, just be disabled"


def test_culled_enables_backface_culling_and_grey():
    code = _code(culled=True)
    assert "if True:\n    # Plain grey" in code
    assert "grey.use_backface_culling = True" in code
    assert "obj.data.materials.clear()" in code, "textured material must be replaced"


def test_recalc_normals_is_independent_of_culling():
    """They are separate flags because Recalculate Outside has previously made a
    culled render WORSE on this asset. Coupling them would hide that."""
    culled_only = _code(culled=True)
    assert "if False:\n    # Culling WITHOUT fixing normals" in culled_only

    recalc_only = _code(recalc_normals=True)
    assert "normals_make_consistent(inside=False)" in recalc_only
    assert "if False:\n    # Plain grey" in recalc_only


def test_both_flags_together():
    code = _code(culled=True, recalc_normals=True)
    assert "if True:\n    # Culling WITHOUT fixing normals" in code
    assert "if True:\n    # Plain grey" in code


def test_recalc_runs_before_the_material_swap():
    """Order matters: normals_make_consistent needs the mesh, and the grey swap
    must not be undone by a later edit-mode round trip."""
    code = _code(culled=True, recalc_normals=True)
    assert code.index("normals_make_consistent") < code.index("HONEST_GREY")
