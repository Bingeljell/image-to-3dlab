"""Check shared scene links, transformed staging and bake settings in Blender."""

import builtins
import sys
from math import isclose
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from blender_light_orbit import blender_code as orbit_code
from blender_scene import isolated_scene_prelude, restore_previous_scene
from blender_scene_stage_worker import stage_assets, stage_bake


def check_shared_links(workspace):
    previous = bpy.context.window.scene
    original = bpy.data.scenes.new("Shared links fixture")
    bpy.context.window.scene = original
    shared = bpy.data.collections.new("Shared collection")
    nested = bpy.data.collections.new("Nested collection")
    original.collection.children.link(shared)
    shared.children.link(nested)
    shared_marker = bpy.data.objects.new("Shared marker", None)
    nested.objects.link(shared_marker)
    direct_marker = bpy.data.objects.new("Directly shared marker", None)
    original.collection.objects.link(direct_marker)
    isolated = bpy.data.collections.new("Private collection")
    stale = bpy.data.objects.new("Private marker", None)
    isolated.objects.link(stale)
    isolated.children.link(nested)
    private_parent = bpy.data.objects.new("Parent used outside the private scene", None)
    isolated.objects.link(private_parent)
    shared_marker.parent = private_parent
    private = bpy.data.scenes.get("IMG3D light orbit") or bpy.data.scenes.new(
        "IMG3D light orbit"
    )
    private.collection.children.link(shared)
    private.collection.children.link(isolated)
    private.collection.objects.link(direct_marker)

    code = orbit_code(
        workspace / "missing.glb", workspace / "out-", 1, 52, 64, "[[0,0,0]]"
    )
    import_module = builtins.__import__

    def import_without_checkout(name, *args, **kwargs):
        if name in {"blender_scene", "scripts.blender_scene", "scripts"}:
            raise ImportError(f"{name} is unavailable in remote Blender")
        return import_module(name, *args, **kwargs)

    try:
        # Make both checkout module names unavailable during remote execution.
        with patch("builtins.__import__", import_without_checkout):
            exec(compile(code, "<packaged orbit request>", "exec"), {})
    except RuntimeError as error:
        assert "Please select a file" in str(error) or "missing.glb" in str(error), str(
            error
        )
    else:
        raise AssertionError("Missing import was accepted")
    assert bpy.context.window.scene == original
    assert shared.name in original.collection.children
    assert nested.name in shared.children
    assert {shared_marker, direct_marker} <= set(original.objects)
    assert not list(private.objects)
    assert not list(private.collection.children)
    assert "Private marker" not in bpy.data.objects
    assert "Private collection" not in bpy.data.collections

    private.collection.children.link(shared)
    prelude = isolated_scene_prelude(private.name)
    exec(prelude + restore_previous_scene(), {"bpy": bpy})
    assert bpy.context.window.scene == original
    assert shared.name in original.collection.children
    assert shared_marker.name in nested.objects
    assert shared_marker.parent == private_parent
    assert not list(private.objects)
    bpy.context.window.scene = previous
    print("PASS failed orbit import and generated prelude preserve shared links")


def check_transformed_stage(workspace):
    original = bpy.context.window.scene
    fixture = bpy.data.scenes.new("Transformed stage fixture")
    bpy.context.window.scene = fixture
    bpy.ops.mesh.primitive_cube_add(location=(10, -4, 7), scale=(2, 1, 0.5))
    obj = bpy.context.object
    obj.rotation_euler.z = 0.35
    source = workspace / "transformed.glb"
    assert bpy.ops.export_scene.gltf(
        filepath=str(source),
        export_format="GLB",
        use_active_scene=True,
    ) == {"FINISHED"}
    bpy.context.window.scene = original

    try:
        stage_assets(
            bpy,
            [{"name": "Bad", "path": str(workspace / "missing.glb"), "dy": 0}],
            False,
        )
    except RuntimeError:
        assert bpy.context.window.scene == original
    else:
        raise AssertionError("Staging accepted a missing import")

    stage_assets(bpy, [{"name": "Transformed", "path": str(source), "dy": 3}], False)
    bpy.context.view_layer.update()
    collection = bpy.context.scene.collection.children["Transformed"]
    corners = [
        mesh.matrix_world @ Vector(corner)
        for mesh in collection.objects
        if mesh.type == "MESH"
        for corner in mesh.bound_box
    ]
    low = tuple(min(v[i] for v in corners) for i in range(3))
    high = tuple(max(v[i] for v in corners) for i in range(3))
    assert isclose(low[0], -0.5, abs_tol=1e-5), (low, high)
    assert isclose(high[0], 0.5, abs_tol=1e-5), (low, high)
    assert isclose((low[1] + high[1]) / 2, 3, abs_tol=1e-5), (low, high)
    assert isclose(low[2], 0, abs_tol=1e-5), (low, high)
    assert high[2] < high[1] - low[1], (low, high)
    bpy.context.window.scene = original
    print(
        "PASS staging restores the window on failure and normalizes a transformed root"
    )


def check_bake_restore(workspace):
    original = bpy.context.window.scene
    scene = bpy.data.scenes.new("Bake restoration")
    bpy.context.window.scene = scene
    scene.render.engine = "BLENDER_WORKBENCH"

    rig = bpy.data.objects.new("rig", bpy.data.armatures.new("Bake armature"))
    scene.collection.objects.link(rig)
    rig.animation_data_create()
    bpy.context.view_layer.update()

    def make_action(name):
        rig.location.x = 0
        rig.keyframe_insert(data_path="location", frame=1)
        action = rig.animation_data.action
        action.name = name
        return action

    action = make_action("Bake test action")
    action.use_fake_user = True
    previous_action = action.copy()
    previous_action.name = "Previous action"
    rig.animation_data.action = previous_action
    target = bpy.data.objects.new("CAM_TARGET", None)
    scene.collection.objects.link(target)
    target.rotation_euler = (0.2, 0.1, -0.3)
    camera = bpy.data.objects.new("QuickCam2", bpy.data.cameras.new("Bake camera"))
    scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 4
    camera.location.z = 4
    camera.parent = target
    prior_camera = bpy.data.objects.new(
        "Previous camera", bpy.data.cameras.new("Previous camera")
    )
    scene.collection.objects.link(prior_camera)
    scene.camera = prior_camera
    scene.render.resolution_x, scene.render.resolution_y = 43, 39
    scene.render.filepath = str(workspace / "original.jpg")
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "BW"
    scene.frame_set(17, subframe=0.25)

    def settings():
        return (
            rig.animation_data.action,
            rig.animation_data.action_slot,
            tuple(target.rotation_euler),
            camera.data.ortho_scale,
            scene.camera,
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.film_transparent,
            scene.view_settings.view_transform,
            getattr(scene.render.image_settings, "media_type", None),
            scene.render.image_settings.file_format,
            scene.render.image_settings.color_mode,
            scene.render.image_settings.color_depth,
            scene.render.filepath,
            scene.frame_current,
            scene.frame_subframe,
        )

    before = settings()
    assert before[0] == previous_action

    def bake(destination):
        return stage_bake(
            bpy, action.name, 21, 9, 9, 1, 2, 1, 32, str(destination), "shot"
        )

    success = workspace / "bake"
    result = bake(success)
    assert result["frame_count"] == 2
    assert all((success / frame).stat().st_size > 0 for frame in result["frames"])
    assert settings() == before, (settings(), before)

    blocked = workspace / "blocked"
    blocked.write_text("existing file", encoding="utf-8")
    try:
        bake(blocked)
    except FileExistsError:
        assert settings() == before, (settings(), before)
    else:
        raise AssertionError("Bake accepted a file as a directory")

    second_failure = workspace / "mid-bake"
    second_failure.mkdir()
    (second_failure / result["frames"][1]).mkdir()
    try:
        bake(second_failure)
    except RuntimeError:
        assert (second_failure / result["frames"][0]).stat().st_size > 0
        assert settings() == before, (settings(), before)
    else:
        raise AssertionError("Bake overwrote a directory as if it were a file")

    image_settings = scene.render.image_settings
    if hasattr(image_settings, "media_type"):
        image_settings.media_type = "VIDEO"
        video_before = settings()
        result = bake(workspace / "bake-from-video")
        assert result["frame_count"] == 2
        assert settings() == video_before, (settings(), video_before)

    bpy.context.window.scene = original
    print(
        "PASS bake restores settings after success, output setup failure and render failure"
    )


def check_scene_safety(workspace):
    check_shared_links(workspace)
    check_transformed_stage(workspace)
    check_bake_restore(workspace)
    return {
        "shared_links_preserved": True,
        "transformed_root_centered": True,
        "bake_settings_restored": True,
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="assetfurnace-scene-safety-") as directory:
        check_scene_safety(Path(directory))
