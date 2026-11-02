"""Return the Blender text that renders in a private scene.

Run (internal): imported by live Blender scripts. Not a command.

A long-lived Blender session holds the user's scenes. Wiping every object, which
these helpers once did, emptied unsaved work. The text below switches to a scene
this repo owns, clears only that scene, and the caller puts the window back.
"""

from __future__ import annotations


def isolated_scene_prelude(scene_name: str) -> str:
    """Blender statements that enter `scene_name` and clear only that scene."""
    quoted = repr(scene_name)
    return (
        "previous_scene = bpy.context.window.scene\n"
        f"render_scene = bpy.data.scenes.get({quoted}) or bpy.data.scenes.new({quoted})\n"
        "bpy.context.window.scene = render_scene\n"
        "if render_scene.world is None:\n"
        '    render_scene.world = bpy.data.worlds.new(render_scene.name + " World")\n'
        "for obj in list(render_scene.objects):\n"
        "    bpy.data.objects.remove(obj, do_unlink=True)\n"
        "for prior in list(render_scene.collection.children):\n"
        "    bpy.data.collections.remove(prior)\n"
    )


def restore_previous_scene() -> str:
    """Blender statement that hands the window back to the scene we found."""
    return "bpy.context.window.scene = previous_scene\n"
