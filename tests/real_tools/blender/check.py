"""Exercise rendering, geometry export, and scene restoration in Blender."""

import argparse
import json
import sys
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from blender_scene_stage_worker import run

PRIVATE_RENDER_SCENE = "IMG3D light orbit"


def make_fixture(path):
    # All four triangles point inward so normal repair has something to fix.
    mesh = bpy.data.meshes.new("Inward tetrahedron")
    mesh.from_pydata(
        [(0.3, 0.3, 0.3), (-0.3, -0.3, 0.3), (-0.3, 0.3, -0.3), (0.3, -0.3, -0.3)],
        [],
        [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
    )
    obj = bpy.data.objects.new(mesh.name, mesh)
    fixture_scene = bpy.data.scenes.new("Fixture")
    bpy.context.window.scene = fixture_scene
    fixture_scene.collection.objects.link(obj)
    assert bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_active_scene=True,
    ) == {"FINISHED"}
    bpy.data.objects.remove(obj, do_unlink=True)


def assert_scene_restored(scene, camera, original_objects, active):
    assert bpy.context.window.scene == scene, "Blender did not restore the active scene"
    assert scene.camera == camera
    assert set(scene.objects) == original_objects
    assert bpy.context.view_layer.objects.active == active
    assert active.select_get()


def assert_not_blank(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        assert tuple(image.size) == (64, 64)
        pixels = image.pixels[:]
        background = pixels[:3]
        different = sum(
            max(
                abs(pixels[offset + channel] - background[channel])
                for channel in range(3)
            )
            > 0.05
            for offset in range(0, len(pixels), 4)
        )
        assert different > 64, "The render contains only background"
        assert different < 64 * 64 - 64, "The render has no visible background"
        return different
    finally:
        bpy.data.images.remove(image)


def check_render(workspace, source, assert_restored):
    private_scene = bpy.data.scenes.new(PRIVATE_RENDER_SCENE)
    stale_collection = bpy.data.collections.new("Stale render collection")
    private_scene.collection.children.link(stale_collection)
    stale = bpy.data.objects.new("Stale render object", None)
    stale_collection.objects.link(stale)
    parameters = {
        "asset": str(source),
        "out": str(workspace / "orbit-"),
        "resolution": 64,
        "rotations": [(0.0, 0.0, 0.0)],
    }
    result = run({"action": "light_orbit", "parameters": parameters})
    assert_restored()
    assert result["frames"] == 1
    subject_pixels = assert_not_blank(workspace / "orbit-0001.png")
    assert "Stale render object" not in bpy.data.objects
    assert "Stale render collection" not in bpy.data.collections
    assert any(obj.type == "MESH" for obj in private_scene.objects)
    print("PASS light orbit rendered a visible model and restored the original scene")
    return {
        "frames": result["frames"],
        "pixels_different_from_background": subject_pixels,
    }


def check_failed_import(workspace, assert_restored):
    request = {
        "action": "light_orbit",
        "parameters": {
            "asset": str(workspace / "missing.glb"),
            "out": str(workspace / "failed-"),
            "resolution": 64,
            "rotations": [(0.0, 0.0, 0.0)],
        },
    }
    try:
        run(request)
    except RuntimeError:
        pass
    else:
        raise AssertionError("The render worker accepted a missing asset")
    assert_restored()
    assert not list(bpy.data.scenes[PRIVATE_RENDER_SCENE].objects), (
        "Objects from the previous render survived cleanup"
    )
    assert not list(workspace.glob("failed-*"))
    print(
        "PASS failed import restored the original scene and cleared the previous render"
    )


def check_output_failure(action, parameters, blocked_path, assert_restored):
    try:
        run({"action": action, "parameters": parameters})
    except RuntimeError as error:
        assert str(blocked_path) in str(error), str(error)
    else:
        raise AssertionError(f"{action} accepted an unwritable output path")
    assert_restored()
    print(f"PASS {action} restored the original scene after an output error")


def check_export(workspace, source, assert_restored):
    output = workspace / "fixed.glb"
    result = run(
        {
            "action": "fix_normals",
            "parameters": {"asset": str(source), "output": str(output)},
        }
    )
    assert_restored()
    assert result["before"] > 99, result
    assert result["after"] < 1, result

    bpy.context.window.scene = bpy.data.scenes.new("Reopened export")
    assert bpy.ops.import_scene.gltf(filepath=str(output)) == {"FINISHED"}
    objects = list(bpy.context.scene.objects)
    assert len(objects) == 1, (
        f"Export contains extra objects: {[obj.name for obj in objects]}"
    )
    obj = objects[0]
    assert obj.type == "MESH"
    assert len(obj.data.polygons) == 4
    assert all(len(face.vertices) == 3 for face in obj.data.polygons)
    assert all(abs(dimension - 0.6) < 1e-5 for dimension in obj.dimensions)
    assert all(face.normal.dot(face.center) > 0 for face in obj.data.polygons)
    print(
        "PASS normal repair exported an outward-facing mesh that Blender could reopen"
    )
    return {
        "triangles": len(obj.data.polygons),
        "inward_area_before": float(result["before"]),
        "inward_area_after": float(result["after"]),
    }


def main():
    assert bpy.app.background, "Run this check in a separate background Blender process"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="save outputs and results in this directory"
    )
    args = parser.parse_args(
        sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    )
    with TemporaryDirectory(prefix="assetfurnace-blender-") as directory:
        workspace = Path(directory)
        source = workspace / "inward.glb"
        original = bpy.context.scene
        camera = original.camera
        original_objects = set(original.objects)
        active = bpy.context.view_layer.objects.active

        def assert_restored():
            assert_scene_restored(original, camera, original_objects, active)

        summary = {"blender": bpy.app.version_string, "passed": False}
        try:
            make_fixture(source)
            bpy.context.window.scene = original
            summary["render"] = check_render(workspace, source, assert_restored)
            check_failed_import(workspace, assert_restored)
            summary["failed_import_restored_scene"] = True
            blocker = workspace / "blocked-output"
            blocker.touch()
            check_output_failure(
                "light_orbit",
                {
                    "asset": str(source),
                    "out": str(blocker / "orbit-"),
                    "resolution": 64,
                    "rotations": [(0.0, 0.0, 0.0)],
                },
                blocker,
                assert_restored,
            )
            summary["failed_render_restored_scene"] = True
            check_output_failure(
                "fix_normals",
                {"asset": str(source), "output": str(blocker / "fixed.glb")},
                blocker,
                assert_restored,
            )
            summary["failed_export_restored_scene"] = True
            summary["export"] = check_export(workspace, source, assert_restored)
            summary["passed"] = True
        finally:
            if args.output is not None:
                args.output.mkdir(parents=True, exist_ok=True)
                for name in ("inward.glb", "orbit-0001.png", "fixed.glb"):
                    artifact = workspace / name
                    if artifact.is_file():
                        copy2(artifact, args.output / name)
                (args.output / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )


if __name__ == "__main__":
    main()
