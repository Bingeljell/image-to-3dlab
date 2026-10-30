"""Live Blender helpers must not wipe the user's open file.

A turntable, a wind demo, a normal fix, and a solidify bake used to delete every
object and purge orphans. That emptied unsaved scenes. The generated code is the
artifact sent to Blender, so these tests import it and read it.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from blender_scene import isolated_scene_prelude, restore_previous_scene


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_private(code: str, scene_name: str) -> None:
    assert isolated_scene_prelude(scene_name) in code
    assert restore_previous_scene() in code
    assert "for obj in list(bpy.data.objects)" not in code
    assert "orphans_purge" not in code


def test_turntable_uses_a_private_scene():
    code = _load("blender_turntable").blender_code(
        Path("/tmp/a.glb"), Path("/tmp/frames"), "fox", "dark", 24, 256, 1.0, False
    )
    _assert_private(code, "IMG3D turntable")


def test_wind_demo_uses_a_private_scene():
    code = _load("blender_wind_demo").blender_code(
        Path("/tmp/m.glb"), Path("/tmp/l.glb"), Path("/tmp/frames"), "fox", "dark",
        256, 12, 1.0, 25.0, 1.0, 1.0, 0.2, 22.0, 1.0, 0.35, 0.2, 0.1, 0.05,
    )
    _assert_private(code, "IMG3D wind")


def test_fix_normals_uses_a_private_scene():
    code = _load("blender_fix_normals").blender_code(Path("/tmp/a.glb"), Path("/tmp/out.glb"))
    _assert_private(code, "IMG3D normals")


def test_solidify_bake_uses_a_private_scene():
    code = _load("blender_solidify_bake").blender_code(
        Path("/tmp/a.glb"), Path("/tmp/out.glb"), 0.01, 512, 0.002, 32
    )
    _assert_private(code, "IMG3D solidify")


def test_compare_renders_in_a_headless_blender():
    compare = _load("compare_to_source")
    argv = compare.headless_argv(Path("/tmp/job.py"))
    assert "--background" in argv
    assert "--python" in argv
    assert "send_to_blender" not in inspect.getsource(compare.main)
    code = compare.blender_code(
        Path("/tmp/a.glb"), Path("/tmp/out"), "cmp", [0.0], 8.0, "studio", False
    )
    assert "for obj in list(bpy.data.objects)" not in code
    assert "orphans_purge" not in code
    assert "read_factory_settings" in code


def test_stage_and_bakes_do_not_wipe_the_open_file():
    stage = _load("blender_stage").build_code(["/a.glb"], ["A_x"], [0.0], False)
    assert isolated_scene_prelude("IMG3D stage") in stage
    assert "for obj in list(bpy.data.objects)" not in stage
    assert "orphans_purge" not in stage

    bake = _load("blender_bake_normals").blender_code(
        Path("/tmp/low.glb"), Path("/tmp/high.ply"), Path("/tmp/out.png"),
        1024, 0.1, False, "CPU", 1.0, 0, 8,
    )
    _assert_private(bake, "IMG3D normal bake")

    orbit = _load("blender_light_orbit").blender_code(
        Path("/tmp/a.glb"), Path("/tmp/f_"), 4, 52.0, 64, "[]",
    )
    _assert_private(orbit, "IMG3D light orbit")

    regions = _load("blender_split_regions").blender_code(
        Path("/tmp/a.glb"), Path("/tmp/out"), 256, 0.25, -0.1,
    )
    _assert_private(regions, "IMG3D regions")


def test_marker_spawn_refuses_to_replace_markers_unless_confirmed():
    markers = _load("blender_joint_markers")
    code = markers.spawn_code(Path("/tmp/a.glb"))
    assert "orphans_purge" not in code
    assert "raise RuntimeError" in code
    assert "JOINT_" in code
    confirmed = markers.spawn_code(Path("/tmp/a.glb"), confirm=True)
    assert "orphans_purge" not in confirmed
    assert confirmed != code


def test_stage_bake_puts_the_open_file_back():
    code = _load("blender_stage_bake").build_code(
        "Walk", 0.0, 0.0, None, None, None, 1, 256, "/tmp/out", "fox",
    )
    assert "previous_action" in code
    assert "previous_camera" in code
    assert "previous_target_rotation" in code
    assert code.index("bpy.ops.render.render") < code.index("animation_data.action = previous_action")


def test_prelude_only_clears_its_own_scene():
    code = isolated_scene_prelude("IMG3D turntable")
    assert "bpy.data.scenes.new" in code or "bpy.data.scenes.get" in code
    assert "for obj in list(render_scene.objects)" in code
    assert "bpy.data.objects)" not in code.split("render_scene.objects")[0]
    assert restore_previous_scene() == "bpy.context.window.scene = previous_scene\n"
