"""The preview render must leave the user's own Blender scenes alone.

It runs inside a long-lived Blender session. It once removed every object in the file
and purged the orphans, which emptied a day of unsaved scenes. Assert on the generated
code, the real artifact sent to Blender.
"""

from __future__ import annotations

from pathlib import Path

from scripts.blender_render_asset import blender_code


def _code() -> str:
    return blender_code(Path("/tmp/a.glb"), Path("/tmp/out"), "fox", "dark")


def test_never_clears_every_object_in_the_file():
    code = _code()
    assert "for obj in list(bpy.data.objects)" not in code
    assert "orphans_purge" not in code


def test_clears_only_its_own_render_scene():
    code = _code()
    assert 'RENDER_SCENE = "IMG3D render"' in code
    assert "bpy.context.window.scene = render_scene" in code
    assert "for obj in list(render_scene.objects):" in code


def test_hands_the_window_back_after_rendering():
    code = _code()
    assert code.index("previous_scene = bpy.context.window.scene") < code.index("render_scene =")
    assert code.rindex("bpy.context.window.scene = previous_scene") > code.rindex("render.render(")


def test_world_is_created_when_the_scene_has_none():
    code = _code()
    assert 'scene.world = bpy.data.worlds.new("World")' in code
    assert code.index("if scene.world is None:") < code.index("scene.world.color =")
