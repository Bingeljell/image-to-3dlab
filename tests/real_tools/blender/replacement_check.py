"""Exercise scene-scoped replacement and rollback in Blender."""

import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import blender_agent_worker as agent
import blender_rig_weights_worker as rig
import blender_scene_query_worker as markers


def check_replacement(workspace):
    check_marker_load_scope()
    original_scene = bpy.context.window.scene
    existing_scenes = set(bpy.data.scenes)
    existing_objects = set(bpy.data.objects)
    scene = bpy.data.scenes.new("Replacement test")
    bpy.context.window.scene = scene
    try:
        return _check_replacement(workspace, scene)
    finally:
        bpy.context.window.scene = original_scene
        for obj in set(bpy.data.objects) - existing_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for created in set(bpy.data.scenes) - existing_scenes:
            bpy.data.scenes.remove(created)


def _check_replacement(workspace, scene):
    other = bpy.data.scenes.new("Other scene")
    bpy.ops.mesh.primitive_cube_add()
    original = bpy.context.object
    source = str(workspace / "replacement.glb")
    assert bpy.ops.export_scene.gltf(filepath=source, use_active_scene=True) == {"FINISHED"}
    sibling = bpy.data.objects.new("Other mesh", original.data.copy())
    other.collection.objects.link(sibling)
    sibling.parent = original
    shared = bpy.data.objects.new("Shared mesh", original.data.copy())
    scene.collection.objects.link(shared)
    other.collection.objects.link(shared)
    path = str(workspace / "absent.glb")
    try:
        agent.import_asset(bpy, path, clear_scene=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing asset was accepted")
    assert original in set(scene.objects)
    assert set(other.objects) == {sibling, shared}

    marker = bpy.data.objects.new("JOINT_old", None)
    scene.collection.objects.link(marker)
    try:
        markers.spawn_markers(bpy, path, {}, "JOINT_", confirm=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing marker asset was accepted")
    assert marker in set(scene.objects)
    assert set(other.objects) == {sibling, shared}
    before_invalid_markers = set(bpy.data.objects)
    try:
        markers.spawn_markers(bpy, source, {"bad": ("x", 0, 0)}, "JOINT_", confirm=True)
    except TypeError:
        pass
    else:
        raise AssertionError("invalid marker was accepted")
    assert set(bpy.data.objects) == before_invalid_markers

    conflicting = bpy.data.objects.new("JOINT_neck", None)
    other.collection.objects.link(conflicting)
    foreign_markers = bpy.data.collections.new("JOINT_MARKERS")
    other.collection.children.link(foreign_markers)
    assert markers.spawn_markers(
        bpy, source, {"neck": (0.5, 0.5, 0.5)}, "JOINT_", confirm=True,
    )["spawned"] == 1
    assert markers.read_markers(bpy, "JOINT_") == {"neck": [0, 0, 0]}
    assert "JOINT_old" not in scene.objects
    assert conflicting.name in other.objects
    active_neck = next(obj for obj in scene.objects if obj.name.startswith("JOINT_neck."))
    active_count = len(scene.objects)
    restored = markers.run({
        "action": "load_markers",
        "parameters": {"markers": {"neck": [1, 2, 3]}, "prefix": "JOINT_"},
    })
    assert restored == {"restored": 1}
    bpy.context.view_layer.update()
    assert tuple(conflicting.location) == (0, 0, 0)
    assert tuple(active_neck.location) == (1, 2, 3)
    assert len(scene.objects) == active_count
    assert markers.read_markers(bpy, "JOINT_") == {"neck": [1, 2, 3]}
    other.collection.objects.unlink(conflicting)

    instance_collection = bpy.data.collections.new("Instanced in another scene")
    scene.collection.children.link(instance_collection)
    instanced = bpy.data.objects.new("Instanced mesh", original.data.copy())
    instance_collection.objects.link(instanced)
    instance = bpy.data.objects.new("Collection instance", None)
    other.collection.objects.link(instance)
    instance.instance_type = "COLLECTION"
    instance.instance_collection = instance_collection

    original_name = original.name
    outcome = agent.import_asset(bpy, source, clear_scene=True)
    assert len(outcome["meshes"]) == 1
    assert original_name not in scene.objects
    assert sibling.parent == original
    assert shared.name not in scene.objects
    assert set(other.objects) == {sibling, shared, instance}
    assert instanced.name in instance_collection.objects
    assert instanced.name in scene.objects
    scene.collection.children.unlink(instance_collection)
    agent.clear_scene(bpy, keep_lights=True)
    assert original_name not in scene.objects
    assert set(other.objects) == {sibling, shared, instance}
    assert shared.name not in scene.objects

    bpy.ops.mesh.primitive_cube_add()
    target = bpy.context.object
    old_rig = bpy.data.objects.new("QuadRig", bpy.data.armatures.new("Old rig"))
    scene.collection.objects.link(old_rig)
    target.parent = old_rig
    old_mod = target.modifiers.new("Previous binding", "ARMATURE")
    old_mod.object = old_rig
    old_group = target.vertex_groups.new(name="Existing weight")
    old_group.add([0], 0.625, "REPLACE")
    old_mesh = target.data
    mesh_peer = bpy.data.objects.new("Other scene mesh with original weights", old_mesh)
    other.collection.objects.link(mesh_peer)
    unrelated = bpy.data.objects.new("Unrelated rig", bpy.data.armatures.new("Unrelated rig"))
    scene.collection.objects.link(unrelated)
    names = sorted({name for _, head, tail, _, _ in rig.SKELETON for name in (head, tail)} | {"ear_L", "ear_R"})
    for index, name in enumerate(names):
        joint = bpy.data.objects.new("JOINT_" + name, None)
        scene.collection.objects.link(joint)
        joint.location = (index * 0.1, 0, index * 0.03)
    bpy.context.view_layer.update()
    before = set(bpy.data.objects)

    class FailingBind:
        def __getattr__(self, name):
            return getattr(bpy.ops.object, name)

        def parent_set(self, **_kwargs):
            raise RuntimeError("injected bind failure")

    proxy = SimpleNamespace(
        context=bpy.context,
        data=bpy.data,
        ops=SimpleNamespace(object=FailingBind()),
    )
    result = rig.build_rig(proxy, Vector, ear_length=0.1, weight_threshold=0.15)
    assert result["heat_error"] == "injected bind failure"
    assert result["applied"] is False
    assert before == set(bpy.data.objects)
    assert target.parent == old_rig
    assert target.modifiers["Previous binding"].object == old_rig
    assert target.data == mesh_peer.data == old_mesh
    assert target.vertex_groups["Existing weight"].weight(0) == 0.625
    assert unrelated in set(scene.objects)

    class WorkingBind(FailingBind):
        def parent_set(self, **_kwargs):
            arm = bpy.context.view_layer.objects.active
            candidate = next(obj for obj in bpy.context.selected_objects if obj.type == "MESH")
            candidate.parent = arm
            candidate.modifiers.new("New binding", "ARMATURE").object = arm
            for bone in arm.data.bones:
                candidate.vertex_groups.new(name=bone.name).add([0], 0.75, "REPLACE")
            return {"FINISHED"}

    # Keep the old rig in use from another scene during a successful rebuild.
    sibling.parent = old_rig
    proxy.ops.object = WorkingBind()
    result = rig.build_rig(proxy, Vector, ear_length=0.1, weight_threshold=0.15)
    assert result["applied"] is True, result["verdict"]
    assert target.parent != old_rig
    assert target.modifiers["New binding"].object == target.parent
    assert target.data != old_mesh
    assert mesh_peer.data == old_mesh
    assert list(mesh_peer.vertex_groups.keys()) == ["Existing weight"]
    assert mesh_peer.vertex_groups["Existing weight"].weight(0) == 0.625
    expected_bones = set(target.parent.data.bones.keys())
    assert set(target.vertex_groups.keys()) == expected_bones
    weights = {
        target.vertex_groups[assignment.group].name: assignment.weight
        for assignment in target.data.vertices[0].groups
    }
    assert weights == dict.fromkeys(expected_bones, 0.75)
    assert sibling.parent == old_rig
    assert old_rig in set(scene.objects)
    assert unrelated in set(scene.objects)
    print("PASS scene-scoped replacement and bind rollback")
    return {"scene_scoped_replacement": True, "bind_rollback": True}


def check_marker_load_scope():
    original_scene = bpy.context.window.scene
    existing_objects = set(bpy.data.objects)
    existing_collections = set(bpy.data.collections)
    active = bpy.data.scenes.new("Load markers")
    foreign = bpy.data.scenes.new("Other markers")
    bpy.context.window.scene = active
    try:
        _check_marker_load_scope(active, foreign)
    finally:
        bpy.context.window.scene = original_scene
        bpy.data.scenes.remove(active)
        bpy.data.scenes.remove(foreign)
        for obj in set(bpy.data.objects) - existing_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for collection in set(bpy.data.collections) - existing_collections:
            bpy.data.collections.remove(collection)
    print("PASS scene-scoped marker loading")


def _check_marker_load_scope(active, foreign):
    foreign_collection = bpy.data.collections.new("JOINT_MARKERS")
    foreign.collection.children.link(foreign_collection)
    assert markers.load_markers(bpy, {"jaw": [0, 1, 2]}, "JOINT_") == {"restored": 1}
    local, = active.collection.children
    assert "JOINT_jaw" in local.objects
    assert not foreign_collection.objects

    shared_collection = bpy.data.collections.new("JOINT_MARKERS")
    active.collection.children.link(shared_collection)
    foreign.collection.children.link(shared_collection)
    shared = bpy.data.objects.new("JOINT_eye", None)
    shared["assetfurnace_marker_name"] = "JOINT_eye"
    shared_collection.objects.link(shared)
    directly_shared = bpy.data.objects.new("JOINT_ear", None)
    directly_shared["assetfurnace_marker_name"] = "JOINT_ear"
    active.collection.objects.link(directly_shared)
    foreign.collection.objects.link(directly_shared)

    instanced_collection = bpy.data.collections.new("JOINT_MARKERS")
    active.collection.children.link(instanced_collection)
    instanced = bpy.data.objects.new("JOINT_nose", None)
    instanced_collection.objects.link(instanced)
    instance = bpy.data.objects.new("Marker collection instance", None)
    foreign.collection.objects.link(instance)
    instance.instance_type = "COLLECTION"
    instance.instance_collection = instanced_collection
    referenced = bpy.data.objects.new("JOINT_chin", None)
    active.collection.objects.link(referenced)
    foreign_child = bpy.data.objects.new("Foreign child", None)
    foreign.collection.objects.link(foreign_child)
    foreign_child.parent = referenced
    local_marker = bpy.data.objects.new("JOINT_brow", None)
    active.collection.objects.link(local_marker)
    local_child = bpy.data.objects.new("Local child", None)
    active.collection.objects.link(local_child)
    local_child.parent = local_marker
    ancestor = bpy.data.objects.new("JOINT_root", None)
    active.collection.objects.link(ancestor)
    bridge = bpy.data.objects.new("Local bridge", None)
    active.collection.objects.link(bridge)
    bridge.parent = ancestor
    foreign_grandchild = bpy.data.objects.new("Foreign grandchild", None)
    foreign.collection.objects.link(foreign_grandchild)
    foreign_grandchild.parent = bridge
    foreign_constraint = bpy.data.objects.new("Foreign constrained object", None)
    foreign.collection.objects.link(foreign_constraint)
    constraint_target = bpy.data.objects.new("JOINT_lid", None)
    active.collection.objects.link(constraint_target)
    constraint = foreign_constraint.constraints.new("COPY_LOCATION")
    constraint.target = constraint_target
    bpy.context.window.scene = foreign
    bpy.context.view_layer.update()
    foreign_dependents = [foreign_child, foreign_grandchild, foreign_constraint]
    locations = [obj.matrix_world.translation.copy() for obj in foreign_dependents]
    bpy.context.window.scene = active

    # Put both unsafe marker collections ahead of the private collection.
    active.collection.children.unlink(local)
    active.collection.children.link(local)
    before = set(active.objects)
    for blocked in ("chin", "root", "lid"):
        try:
            markers.load_markers(
                bpy, {"brow": [8, 8, 8], "missing": [9, 9, 9], blocked: [4, 5, 6]}, "JOINT_",
            )
        except RuntimeError as exc:
            assert f"JOINT_{blocked}" in str(exc)
        else:
            raise AssertionError(f"Loading {blocked} accepted a dependency in another scene")
        assert set(active.objects) == before
        assert tuple(local_marker.location) == (0, 0, 0)
    bpy.context.window.scene = foreign
    bpy.context.view_layer.update()
    assert [obj.matrix_world.translation for obj in foreign_dependents] == locations
    bpy.context.window.scene = active
    assert markers.load_markers(
        bpy, {
            "eye": [1, 2, 3], "ear": [2, 3, 4], "nose": [3, 4, 5],
            "brow": [5, 6, 7],
        },
        "JOINT_",
    ) == {"restored": 4}
    bpy.context.view_layer.update()
    assert tuple(local_marker.location) == (5, 6, 7)
    assert tuple(local_child.matrix_world.translation) == (5, 6, 7)
    assert markers.load_markers(bpy, {"eye": [4, 5, 6]}, "JOINT_") == {"restored": 1}
    bpy.context.view_layer.update()
    assert markers.read_markers(bpy, "JOINT_") == {
        "brow": [5, 6, 7], "chin": [0, 0, 0], "eye": [4, 5, 6], "ear": [2, 3, 4],
        "jaw": [0, 1, 2], "lid": [0, 0, 0], "nose": [3, 4, 5], "root": [0, 0, 0],
    }
    assert len(local.objects) == 4
    assert set(foreign_collection.objects) == set()
    assert set(shared_collection.objects) == {shared}
    assert set(instanced_collection.objects) == {instanced}
    assert set(foreign.objects) == {shared, directly_shared, instance, *foreign_dependents}
    assert all(
        tuple(obj.location) == (0, 0, 0)
        for obj in (shared, directly_shared, instanced, referenced, ancestor)
    )


if __name__ == "__main__":
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        check_replacement(Path(directory))
