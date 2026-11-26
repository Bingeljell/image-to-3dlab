"""Exercise standalone live Blender bake and rig workers on small fixtures."""

import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace as Obj

import numpy as np
import pytest
from test_coverage_geometry_live_workers import Items, MeshObject
from test_coverage_geometry_live_workers import Vector as BaseVector
from test_coverage_geometry_live_workers import (
    private_scene as private_scene,  # noqa: PLC0414 - expose fixture
)


class Vector(BaseVector):
    @property
    def length(self):
        return np.linalg.norm(self)

    def normalized(self):
        return self / self.length


def test_normal_bake_fit_reports_uniform_and_degenerate_axes():
    from scripts import blender_bake_normals_worker as worker

    assert worker.axis_fit([1, 2, 3], [2, 4, 6]) == ([2, 2, 2], 1, 2)
    assert worker.axis_fit([1, 0, 3], [2, 4, 6]) == ([2, 0, 2], 999, 4 / 3)


def test_weight_report_ignores_weak_and_unknown_groups():
    from scripts import blender_rig_weights_worker as worker

    vertices = [
        Obj(co=Obj(z=0), groups=[Obj(group=0, weight=.8)]),
        Obj(co=Obj(z=1), groups=[
            Obj(group=1, weight=.15), Obj(group=0, weight=.14), Obj(group=99, weight=1),
        ]),
    ]
    report = worker.weight_report(vertices, {0: "back_paw", 1: "head"}, ["back_paw", "head", "jaw"], 1, .15)
    assert report == {
        "back_paw": {"verts": 1, "max_z": 0, "reaches_head": False},
        "head": {"verts": 1, "max_z": 1, "reaches_head": True},
        "jaw": {"verts": 0, "max_z": None, "reaches_head": False},
    }


@pytest.mark.parametrize("ear", [(2, 0, 1), (0, 0, 1)])
def test_ear_tip_extrapolates_from_head_or_uses_vertical_fallback(ear):
    from scripts import blender_rig_weights_worker as worker

    tip = worker.ear_tip(Vector(ear), Vector((0, 0, 1)), .2, Vector)
    expected = (2.2, 0, 1) if ear[0] else (0, 0, 1.2)
    np.testing.assert_allclose(tip, expected)


@pytest.mark.parametrize("enhanced,failed", [(True, False), (False, True)])
def test_normal_bake_centres_mesh_and_restores_window(request, tmp_path, monkeypatch, capsys, enhanced, failed):
    from scripts import blender_bake_normals_worker as worker

    bpy, user, events = request.getfixturevalue("private_scene")("IMG3D normal bake", existing=enhanced, fail=failed, shader=enhanced)
    library_image = bpy.data.images.new
    bpy.data.images.new = lambda *args, float_buffer, **kwargs: library_image(*args, **kwargs)
    images = bpy.data.images.new

    def new_image(*args, **kwargs):
        image = images(*args, **kwargs)
        image.colorspace_settings = Obj()
        return image

    bpy.data.images.new = new_image
    bpy.ops.object.shade_smooth = lambda: events.append(("smooth",))

    def make_mesh(name, corners, offset, materials, triangles):
        mesh = MeshObject(name, Obj(materials=materials, polygons=[None] * triangles))
        mesh.bound_box = corners
        mesh.scale = (1, 1, 1)
        mesh.location = Vector(offset)
        mesh.rotation_euler = (0, 0, 0)
        mesh.modifiers = Obj(new=lambda name, _kind: Obj(name=name))

        class Matrix:
            def __matmul__(self, point):
                if mesh.rotation_euler[0]:
                    point = Vector((point.x, -point.z, point.y))
                return point * mesh.scale + mesh.location

        mesh.matrix_world = Matrix()
        return mesh

    material = bpy.data.materials.new("original")
    low = make_mesh("low", list(product((0, 2), (0, 4), (0, 6))), (10, 0, 0), [] if enhanced else [material], 1)
    high_dims = ((0, 1), (0, 3), (0, 2)) if enhanced else ((0, 1), (0, 2), (0, 3))
    high = make_mesh("high", list(product(*high_dims)), (0, 10, 0), [], 5)
    bpy.ops.wm = Obj(ply_import=lambda filepath: bpy.context.scene.objects.append(high))
    bpy.ops.import_scene.gltf = lambda filepath: bpy.context.scene.objects.append(low)
    bpy.context.view_layer.update = lambda: None
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", Obj(Vector=Vector))
    output = tmp_path / "normals.png"
    args = {
        "low_path": "low.glb", "high_path": "high.ply", "out_path": str(output),
        "size": 8, "ray": .02, "smooth_source": enhanced, "device": "CPU",
        "max_ray": .2, "decimate_to": 3 if enhanced else 0, "samples": 2, "rot_x": enhanced,
    }
    report = worker.run(args) if enhanced else worker.run(args, bpy=bpy, vector=Vector)
    assert report == json.loads(capsys.readouterr().out)
    assert report["alignment_error"] == 0
    assert report["uniform_scale_applied"] == 2
    assert report["axis_ratios"] == [2, 2, 2]
    assert report["orientation_spread"] == 1
    assert output.exists() is not failed
    assert report["error"] == ("fixture bake failed" if failed else None)
    assert report["written"] == (None if failed else str(output))
    assert low.selected and high.selected
    assert not user.objects[0].selected
    assert bpy.context.window.scene is user
    assert ("bake", "NORMAL", "LOWPOLY") in events
    assert (("smooth",) in events) is enhanced
    assert any(event[0] == "apply" for event in events) is enhanced
    bake = bpy.data.scenes["IMG3D normal bake"].render.bake
    assert bake.use_selected_to_active and bake.cage_extrusion == .02 and bake.max_ray_distance == .2
    assert low.data.materials[0].node_tree.nodes.active.image.colorspace_settings.name == "Non-Color"


class Modifiers(Items):
    def new(self, name, kind):
        modifier = Obj(name=name, type=kind)
        self.append(modifier)
        return modifier


class Bones(dict):
    def new(self, name):
        bone = Obj(name=name, parent=None)
        self[name] = bone
        return bone


class ObjectLibrary(Items):
    def new(self, name, data):
        obj = MeshObject(name, data)
        obj.type = "ARMATURE"
        return obj

    def remove(self, obj, *, do_unlink=True):
        assert do_unlink
        super().remove(obj)


@pytest.fixture
def rig_scene(monkeypatch):
    objects = ObjectLibrary()
    active = Obj(active=None)
    modes = []
    operations = []

    def deselect(action):
        assert action == "DESELECT"
        for obj in objects:
            obj.select_set(False)

    def armature(name):
        bones = Bones()
        return Obj(name=name, edit_bones=bones, bones=bones)

    collection = type("Collection", (), {"children_recursive": ()})()
    collection.objects = Obj(link=objects.append, unlink=lambda _: None)
    scene = type("Scene", (), {})()
    scene.collection = collection
    scene.objects = objects
    bpy = Obj(
        data=Obj(
            objects=objects,
            scenes=[scene],
            armatures=Obj(new=armature, remove=lambda _: None),
            meshes=Obj(remove=lambda _: None),
            user_map=lambda *, subset: {obj: {collection} for obj in subset},
        ),
        context=Obj(
            scene=scene,
            selected_objects=[],

            view_layer=Obj(objects=active),
        ),
        ops=Obj(object=Obj(
            mode_set=lambda mode: modes.append(mode),
            select_all=deselect,
            modifier_apply=lambda modifier: operations.append(("apply", modifier)),
            datalayout_transfer=lambda modifier: operations.append(("layout", modifier)),
        )),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", Obj(Vector=Vector))
    return bpy, active, modes, operations


def point(z, *weights):
    return Obj(co=Vector((0, 0, z)), groups=[Obj(group=group, weight=weight) for group, weight in weights])


@pytest.mark.parametrize("heat,replacement", [
    ("fail", True), ("empty", True), ("clean", True), ("contaminated", True),
    ("fail", False), ("empty", False), ("contaminated", False),
])
def test_build_worker_links_bones_and_reports_heat_weighting(rig_scene, capsys, heat, replacement):
    from scripts import blender_rig_weights_worker as worker

    bpy, active, modes, _operations = rig_scene
    mesh = MeshObject("mesh", Obj(vertices=[point(0), point(1)]))
    old_armature = MeshObject("old rig")
    old_armature.type = "ARMATURE"
    mesh.parent = old_armature if replacement else None
    mesh.vertex_groups = [Obj(name="stale")] if replacement else []
    mesh.modifiers = Modifiers([Obj(name="mirror", type="MIRROR")])
    if replacement:
        mesh.modifiers.append(Obj(name="old armature", type="ARMATURE", object=old_armature))
    mesh.users_collection = old_armature.users_collection = [bpy.context.scene.collection]
    old_armature.users_scene = [bpy.context.scene]
    old_armature.users = 0
    mesh.matrix_world = Vector((0, 0, 0))
    mesh.matrix_parent_inverse = Vector((0, 0, 0))
    mesh.data.copy = lambda: Obj(vertices=deepcopy(mesh.data.vertices))
    mesh.copy = lambda: deepcopy(mesh)
    bpy.data.objects.extend([mesh, old_armature])
    marker_names = sorted({name for bone in worker.SKELETON for name in bone[1:3]})
    markers = {name: Vector((index / 10, 0, .2)) for index, name in enumerate(marker_names)}
    markers["head"] = Vector((0, 0, 1))
    markers["ear_L"] = markers["head"].copy()
    markers["ear_R"] = Vector((1, 0, 1))
    markers["pelvis"] = markers["spine_mid"].copy()
    for name, xyz in markers.items():
        obj = MeshObject(f"JOINT_{name}")
        obj.matrix_world = Obj(translation=xyz)
        obj.get = lambda _key: None
        bpy.data.objects.append(obj)

    created = []

    def parent_set(type):
        candidate = next(obj for obj in bpy.data.objects if obj.selected and obj.type == "MESH")
        assert type == "ARMATURE_AUTO" and active.active.selected and candidate is not mesh
        assert candidate.vertex_groups == []
        created.append(active.active)
        if heat == "fail":
            raise RuntimeError("heat unavailable")
        candidate.parent = active.active
        candidate.modifiers.new("New bind", "ARMATURE").object = active.active
        names = list(active.active.data.bones)
        if heat == "empty":
            names = ["head"]
        candidate.vertex_groups = [Obj(index=index, name=name) for index, name in enumerate(names)]
        for index, name in enumerate(names):
            vertex = candidate.data.vertices[int(name == "head" or (heat == "contaminated" and name == "frontL_paw"))]
            vertex.groups.append(Obj(group=index, weight=.9))
        return {"FINISHED"}

    bpy.ops.object.parent_set = parent_set
    args = {"operation": "build", "ear_length": .2, "weight_threshold": .15}
    if heat == "fail":
        result = worker.run(args, bpy=bpy, vector=Vector)
    else:
        result = worker.run(args)
    output = capsys.readouterr().out
    assert json.loads(output[output.index("{"):]) == result
    assert modes == ["EDIT", "OBJECT"]
    expected_applied = not replacement or heat == "clean"
    assert result["applied"] is expected_applied
    if replacement and not expected_applied:
        assert mesh.parent is old_armature
        assert [mod.name for mod in mesh.modifiers] == ["mirror", "old armature"]
        assert [group.name for group in mesh.vertex_groups] == ["stale"]
        assert active.active is None
    else:
        assert [mod.name for mod in mesh.modifiers] == (
            ["mirror", "Armature"] if heat == "fail" else ["mirror", "New bind"]
        )
    assert (old_armature in bpy.data.objects) is (not replacement or not expected_applied)
    rig = created[0]
    assert (rig in bpy.data.objects) is expected_applied
    bones = rig.data.bones
    assert rig.name == "QuadRig"
    assert result["bones"] == len(worker.SKELETON) + 2
    assert bones["spine_02"].parent is bones["spine_01"]
    assert bones["spine_02"].use_connect
    assert bones["ear_R"].parent is bones["head"]
    np.testing.assert_allclose(bones["ear_R"].tail, [1.2, 0, 1])
    np.testing.assert_allclose(bones["ear_L"].tail, [0, 0, 1.2])
    np.testing.assert_allclose(bones["spine_01"].tail - bones["spine_01"].head, [0, 0, .001])
    assert result["heat_error"] == ("heat unavailable" if heat == "fail" else None)
    expected = {"fail": "EMPTY GROUPS", "empty": "EMPTY GROUPS", "clean": "CLEAN",
                "contaminated": "ENVELOPE FALLBACK"}
    assert result["verdict"] == expected[heat]


def test_rig_worker_validates_scene_before_touching_meshes(rig_scene):
    from scripts import blender_rig_weights_worker as worker

    bpy, _active, _modes, _operations = rig_scene
    with pytest.raises(RuntimeError, match="missing markers: "):
        worker.run({"operation": "build", "ear_length": .1, "weight_threshold": .15})
    names = {name for bone in worker.SKELETON for name in bone[1:3]} | {"ear_L", "ear_R"}
    for name in names:
        obj = MeshObject(f"JOINT_{name}")
        obj.matrix_world = Obj(translation=Vector((0, 0, 0)))
        obj.get = lambda _key: None
        bpy.data.objects.append(obj)
    with pytest.raises(RuntimeError, match="expected exactly one mesh, found 0"):
        worker.run({"operation": "build", "ear_length": .1, "weight_threshold": .15})
    with pytest.raises(RuntimeError, match="need exactly one mesh and one armature, found 0 and 0"):
        worker.run({"operation": "voxel", "voxel_size": .01, "weight_threshold": .15, "keep_proxy": False})
    with pytest.raises(ValueError, match="unknown rig operation"):
        worker.run({"operation": "other"})


@pytest.mark.parametrize("keep", [False, True])
def test_voxel_worker_transfers_weights_and_keeps_proxy_only_on_request(rig_scene, capsys, keep):
    from scripts import blender_rig_weights_worker as worker

    bpy, active, _modes, operations = rig_scene
    source_vertices = [point(0), point(.5), point(1)]
    data = Obj(vertices=source_vertices, polygons=[None] * 4)
    data.copy = lambda: Obj(vertices=[point(0), point(1)], polygons=[None] * 2)
    mesh = MeshObject("mesh", data)
    armature = MeshObject("rig")
    armature.type = "ARMATURE"
    mesh.parent = armature
    mesh.vertex_groups = [Obj(index=0, name="stale")]
    mesh.modifiers = Modifiers([Obj(name="old bind", type="ARMATURE"), Obj(name="old transfer", type="DATA_TRANSFER"),
                               Obj(name="mirror", type="MIRROR")])
    stale_proxy = MeshObject("PROXY_stale", data)
    stale_proxy.users_collection = [bpy.context.scene.collection]
    stale_proxy.users = 0
    unrelated_rig = MeshObject("Unrelated rig")
    unrelated_rig.type = "ARMATURE"
    bpy.data.objects.extend([mesh, unrelated_rig, armature, stale_proxy])
    proxy = MeshObject("copy", data)
    proxy.vertex_groups = mesh.vertex_groups.copy()
    proxy.modifiers = Modifiers(mesh.modifiers)
    mesh.copy = lambda: proxy
    if keep:
        marker = MeshObject("JOINT_head")
        marker.matrix_world = Obj(translation=Vector((0, 0, .6)))
        marker.get = lambda _key: None
        bpy.data.objects.append(marker)

    def parent_set(type):
        assert active.active is armature and armature.selected
        operations.append(("parent", type))
        if type == "ARMATURE_AUTO":
            assert proxy.selected
            proxy.vertex_groups = [Obj(index=0, name="front_paw"), Obj(index=1, name="jaw")]
            proxy.data.vertices[0].groups = [Obj(group=0, weight=.9)]
            if keep:
                raise RuntimeError("proxy heat failed")
        else:
            assert mesh.selected and not proxy.selected
            mesh.parent = armature

    def apply(modifier):
        operations.append(("apply", modifier))
        if modifier == "Remesh":
            remesh = proxy.modifiers.get(modifier)
            assert remesh.mode == "VOXEL" and remesh.voxel_size == .02
            assert proxy.vertex_groups == []
        else:
            transfer = mesh.modifiers.get(modifier)
            assert transfer.object is proxy and transfer.vert_mapping == "POLYINTERP_NEAREST"
            assert transfer.data_types_verts == {"VGROUP_WEIGHTS"}
            mesh.vertex_groups = [Obj(index=0, name="front_paw")]
            if keep:
                mesh.vertex_groups.append(Obj(index=1, name="jaw"))
            source_vertices[int(keep) * 2].groups = [Obj(group=0, weight=.75)]

    bpy.ops.object.parent_set = parent_set
    bpy.ops.object.modifier_apply = apply
    result = worker.run({"operation": "voxel", "voxel_size": .02, "weight_threshold": .15, "keep_proxy": keep})
    assert result == json.loads(capsys.readouterr().out)
    assert operations == [
        ("apply", "Remesh"), ("parent", "ARMATURE_AUTO"), ("layout", "WeightTransfer"),
        ("apply", "WeightTransfer"), ("parent", "ARMATURE_NAME"),
    ]
    assert stale_proxy not in bpy.data.objects
    assert unrelated_rig in bpy.data.objects
    assert (proxy in bpy.data.objects) is keep
    assert mesh.parent is armature
    assert mesh.modifiers.get("mirror") is not None
    assert mesh.modifiers.get("old bind") is None
    assert result["proxy_faces"] == 2 and result["proxy_groups_with_weight"] == 1
    assert result["unweighted_vertices"] == 2
    assert result["proxy_heat_error"] == ("proxy heat failed" if keep else None)
    assert result["empty_groups"] == (["jaw"] if keep else [])
    assert result["legs_reaching_head"] == (["front_paw"] if keep else [])
    assert result["verdict"] == ("ENVELOPE-LIKE BLEED" if keep else "CLEAN")


@pytest.mark.parametrize("filename", ["blender_build_rig.py", "blender_voxel_weights.py"])
def test_rig_launchers_help_runs_without_connecting(filename, monkeypatch, tmp_path, capsys):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(sys, "argv", [filename, "--help"])
    with pytest.raises(SystemExit) as status:
        run_path(str(scripts / filename), run_name="__main__")
    assert status.value.code == 0
    assert "--weight-threshold" in capsys.readouterr().out
