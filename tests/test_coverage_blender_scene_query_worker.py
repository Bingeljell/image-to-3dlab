"""Exercise the Blender scene query worker's coordinate math."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace as NS

import blender_inspect as inspect_cli
import blender_scene_query_worker as worker
import numpy as np
import pytest

from tests.test_coverage_blender_agent_worker import Vector as NumericVector


@pytest.mark.parametrize(
    "view, expected",
    [
        ("front_xneg", (-2, 0, 5)),
        ("front_three_quarter", (-1.8, -0.9, 5)),
        ("rear_xpos", (2, 0, 5)),
        ("profile_yneg", (0, -2, 5)),
        ("profile_ypos", (0, 2, 5)),
    ],
)
def test_pick_camera_locations_match_existing_view_names(view, expected):
    assert worker.pick_location(view, 2, 5) == pytest.approx(expected)


def test_markers_use_world_bounds_and_fractional_joint_coordinates():
    assert worker.marker_position((2, 9, 3), (8, 2, 10), (0.25, 0.5, 1)) == (4, 10, 13)


class Vector(NumericVector):
    @property
    def w(self):
        return self[3]

    @property
    def xyz(self):
        return self[:3]

    def normalized(self):
        return self / self.length


class Matrix:
    def __init__(self, data=None):
        self.data = np.eye(4) if data is None else np.asarray(data, dtype=float)

    @classmethod
    def translated(cls, xyz):
        matrix = cls()
        matrix.data[:3, 3] = xyz
        return matrix

    @property
    def translation(self):
        return Vector(self.data[:3, 3])

    def inverted(self):
        return Matrix(np.linalg.inv(self.data))

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        if len(other) == 3:
            return Vector((self.data @ [*other, 1])[:3])
        return Vector(self.data @ other)


def test_scene_inspection_is_read_only_and_returns_scene_objects():
    parent = NS(name="rig")
    obj = NS(
        name="part",
        type="MESH",
        dimensions=(1.23456, 2, 3),
        scale=(1, 1, 1),
        location=(0, 0, 2),
        hide_render=True,
        parent=parent,
    )
    marker = NS(
        name="marker",
        type="EMPTY",
        dimensions=(0, 0, 0),
        scale=(1, 1, 1),
        location=(1, 2, 3),
        hide_render=False,
        parent=None,
    )
    bpy = NS(
        context=NS(scene=NS(objects=[obj, marker])),
        data=NS(
            filepath="/tmp/scene.blend",
            is_dirty=True,
            actions=[NS(name="Walk")],
            materials=[NS(name="Skin")],
        ),
    )
    result = worker.inspect_scene(bpy)
    assert result == {
        "filepath": "/tmp/scene.blend",
        "is_dirty": True,
        "object_count": 2,
        "objects": [
            {
                "name": "part",
                "type": "MESH",
                "dimensions": [1.2346, 2, 3],
                "scale": [1, 1, 1],
                "location": [0, 0, 2],
                "hide_render": True,
                "parent": "rig",
            },
            {
                "name": "marker",
                "type": "EMPTY",
                "dimensions": [0, 0, 0],
                "scale": [1, 1, 1],
                "location": [1, 2, 3],
                "hide_render": False,
                "parent": None,
            },
        ],
        "actions": ["Walk"],
        "materials": ["Skin"],
    }
    assert obj.dimensions == (1.23456, 2, 3)


def test_library_inspection_never_requests_an_append(monkeypatch, capsys):
    library = NS(
        objects=["rig"],
        actions=["Walk"],
        meshes=["Shape"],
        armatures=["Bones"],
        materials=["Skin"],
    )
    dest = NS()
    path = "/tmp/café's character.blend"

    @contextmanager
    def load(actual, *, link, assets_only):
        assert actual == path and not link and not assets_only
        yield library, dest
        assert vars(dest) == {}

    app = NS(data=NS(libraries=NS(load=load)))
    monkeypatch.setitem(sys.modules, "bpy", app)
    code = inspect_cli.build_inspect_code(path=path)
    bridge = {}
    exec(compile(code, "<Blender inspect request>", "exec"), bridge)
    output = capsys.readouterr().out.removeprefix("INSPECT_RESULT ")
    expected = {"path": path, **vars(library)}
    assert json.loads(output) == bridge["result"] == expected


def test_object_details_include_material_and_animation_when_present():
    obj = NS(
        name="arm",
        type="MESH",
        dimensions=(1, 2, 3),
        scale=(1, 1, 1),
        location=(0, 0, 1),
        rotation_euler=(np.pi / 2, 0, 0),
        parent=NS(name="body"),
        children=[NS(name="hand")],
        hide_render=False,
        modifiers=[NS(name="Armature", type="ARMATURE")],
        data=NS(materials=[NS(name="Skin")]),
        animation_data=NS(action=NS(name="Wave")),
    )
    bpy = NS(data=NS(objects={"arm": obj}))
    result = worker.inspect_object(bpy, "arm")
    assert result["name"] == "arm"
    assert result["rotation_euler_deg"] == pytest.approx([90, 0, 0])
    assert result["parent"] == "body" and result["children"] == ["hand"]
    assert result["modifiers"] == [("Armature", "ARMATURE")]
    assert result["materials"] == ["Skin"] and result["active_action"] == "Wave"
    obj.parent, obj.data, obj.animation_data.action = None, None, None
    assert worker.inspect_object(bpy, "arm")["active_action"] is None
    obj.animation_data = None
    del obj.modifiers
    assert worker.inspect_object(bpy, "arm")["materials"] == []
    assert worker.inspect_object(bpy, "absent") == {
        "error": "no object named 'absent' in the live scene"
    }


class MarkerObject:
    def __init__(self, name, data=None):
        self.name = name
        self.users = 0
        self.type = "EMPTY" if data is None else "MESH"
        self.location = (0, 0, 0)
        self.bound_box = [(-1, -2, 0), (1, 2, 8)]
        self.properties = {}

    def get(self, key):
        return self.properties.get(key)

    def __setitem__(self, key, value):
        self.properties[key] = value

    @property
    def matrix_world(self):
        return Matrix.translated(self.location)


class NamedItems(list):
    def __init__(self, factory, items=()):
        super().__init__(items)
        self.factory = factory

    def get(self, name):
        return next((item for item in self if item.name == name), None)

    def new(self, name, *args):
        item = self.factory(name, *args)
        self.append(item)
        return item

    def remove(self, obj, *, do_unlink=True):
        assert do_unlink
        super().remove(obj)


@pytest.fixture
def markers(monkeypatch):
    items = NamedItems(
        MarkerObject, [MarkerObject("JOINT_old"), MarkerObject("keep this")]
    )
    class Collection:
        children_recursive = ()

        def __init__(self, name):
            self.name = name
            self.objects = NS(link=lambda obj: setattr(obj, "users_collection", [self]))

    collections = NamedItems(Collection)
    view = NS(type="VIEW_3D", shading=NS(type="SOLID"))
    linked = []
    imported = []
    root = Collection("Scene collection")
    root.children_recursive = linked
    root.children = NS(link=linked.append)
    root.objects = NS(unlink=lambda _: None)
    scene = NS(collection=root, objects=items)
    for obj in items:
        obj.users_collection = [root]

    def gltf(*, filepath):
        imported.append(filepath)
        if filepath != "bad.glb":
            mesh = items.new("new mesh", object())
            mesh.location = (10, -5, 2)
        return {"FINISHED"}

    bpy = NS(
        data=NS(
            objects=items, collections=collections, scenes=[scene],
            user_map=lambda *, subset: {item: set() for item in subset},
        ),
        context=NS(
            scene=scene,
            screen=NS(
                areas=[
                    NS(type="PROPERTIES"),
                    NS(
                        type="VIEW_3D",
                        spaces=[
                            NS(type="PROPERTIES"),
                            view,
                        ],
                    ),
                ]
            ),
        ),
        ops=NS(import_scene=NS(gltf=gltf)),
    )
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    return bpy, items, linked, imported, view


def test_marker_spawn_requires_confirmation_and_preserves_non_marker_objects(markers):
    bpy, items, linked, imported, view = markers
    fractions = {"base": (0.5, 0.5, 0), "top": (1, 0, 1)}
    with pytest.raises(RuntimeError, match="replace the markers only"):
        worker.spawn_markers(bpy, "shape.glb", fractions, "JOINT_", False)
    assert not imported and items.get("JOINT_old")
    assert worker.spawn_markers(bpy, "shape.glb", fractions, "JOINT_", True) == {
        "spawned": 2,
        "bbox_min": [9, -7, 2],
        "bbox_max": [11, -3, 10],
    }
    assert items.get("JOINT_old") is None
    assert items.get("keep this") and items.get("new mesh").hide_select
    assert linked[0].name == "JOINT_MARKERS"
    assert view.shading.type == "MATERIAL"
    assert items.get("JOINT_top").empty_display_size == pytest.approx(0.144)
    assert worker.read_markers(bpy, "JOINT_") == {
        "base": [10, -5, 2],
        "top": [11, -7, 10],
    }


def test_markers_load_updates_existing_and_creates_missing(
    markers, monkeypatch, capsys
):
    bpy, items, linked, _, _ = markers
    old = items.get("JOINT_old")
    assert worker.load_markers(
        bpy, {"old": [5, 0, 0], "second": [2, 3, 4]}, "JOINT_"
    ) == {"restored": 2}
    assert old.location == [5, 0, 0] and items.get("JOINT_old") is old
    assert items.get("JOINT_second").show_name
    assert len(linked) == 1
    assert worker.load_markers(bpy, {"second": [4, 4, 4]}, "JOINT_") == {"restored": 1}
    assert len(linked) == 1
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    result = worker.run({"action": "read_markers", "parameters": {"prefix": "JOINT_"}})
    assert (
        capsys.readouterr().out == json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    assert result == {"old": [5, 0, 0], "second": [4, 4, 4]}


def test_read_markers_prefers_private_logical_marker_regardless_of_link_order(markers):
    bpy, items, _, _, _ = markers
    scene = bpy.context.scene
    active = items.get("JOINT_old")
    active.name = "JOINT_old.001"
    active.properties["assetfurnace_marker_name"] = "JOINT_old"
    active.location = (1, 2, 3)
    foreign_collection = type("Collection", (), {"children_recursive": ()})()
    bpy.data.scenes.append(NS(collection=foreign_collection))
    shared = items.new("JOINT_old.002")
    shared.properties["assetfurnace_marker_name"] = "JOINT_old"
    shared.users_collection = [scene.collection, foreign_collection]
    assert worker.read_markers(bpy, "JOINT_") == {"old": [1, 2, 3]}
    items.reverse()
    assert worker.read_markers(bpy, "JOINT_") == {"old": [1, 2, 3]}


def test_load_markers_rejects_external_dependencies_before_any_edit(markers, monkeypatch):
    bpy, items, linked, _, _ = markers
    referenced = items.get("JOINT_old")
    external_user = object()
    monkeypatch.setattr(bpy.data, "user_map", lambda *, subset: (
        {referenced: {external_user}} if referenced in subset else {}
    ))
    with pytest.raises(RuntimeError, match="JOINT_old"):
        worker.load_markers(bpy, {"missing": [4, 5, 6], "old": [1, 2, 3]}, "JOINT_")
    assert not linked and items.get("JOINT_missing") is None
    assert referenced.location == (0, 0, 0)


def test_marker_spawn_rejects_import_with_no_mesh(markers):
    bpy, items, linked, imported, _ = markers
    original = items.get("JOINT_old")
    with pytest.raises(RuntimeError, match="no mesh imported"):
        worker.spawn_markers(bpy, "bad.glb", {}, "JOINT_", True)
    assert items.get("JOINT_old") is original
    assert imported == ["bad.glb"] and not linked


@pytest.mark.parametrize("hit", [False, True])
def test_pixel_hit_unprojects_ray_and_converts_local_coordinates(monkeypatch, hit):
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    depsgraph = object()
    projection = Matrix(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -3, -4],
            [0, 0, -1, 0],
        ]
    )
    events = []

    class Camera:
        @property
        def matrix_world(self):
            return Matrix.translated(self.location)

        def calc_matrix_camera(self, deps, **settings):
            assert deps is depsgraph
            assert settings == {"x": 64, "y": 32, "scale_x": 1, "scale_y": 1}
            assert events == ["update"]
            return projection

    mesh = NS(
        name="body",
        type="MESH",
        matrix_world=Matrix.translated((1, 2, 3)),
        bound_box=[(-4, 0, 0), (4, 2, 3)],
    )
    obj_ignored = NS(name="lamp", type="LIGHT")
    camera = Camera()

    def ray_cast(deps, origin, direction):
        assert deps is depsgraph
        assert list(origin) == pytest.approx([0, -18, 4.74])
        assert list(direction) == pytest.approx(
            [1 / np.sqrt(6), -1 / np.sqrt(6), -2 / np.sqrt(6)]
        )
        return hit, Vector((2, 5, 7)), Vector((0, 1, 0)), 24, mesh, mesh.matrix_world

    scene = NS(
        camera=camera,
        objects=[mesh, obj_ignored],
        ray_cast=ray_cast,
        render=NS(pixel_aspect_x=1, pixel_aspect_y=1),
    )
    bpy = NS(
        context=NS(
            scene=scene,
            evaluated_depsgraph_get=lambda: depsgraph,
            view_layer=NS(update=lambda: events.append("update")),
        )
    )
    result = worker.pick_pixel(bpy, [12, 15], 64, 32, "profile_yneg", [0.5, -0.5])
    expected = {"hit": hit, "pixel": [12, 15], "view": "profile_yneg"}
    if hit:
        expected.update(
            {
                "object": "body",
                "face_index": 24,
                "world": [2, 5, 7],
                "mesh_local": [1, 3, 4],
                "gltf_local": [1, 4, -3],
                "normal_world": [0, 1, 0],
            }
        )
    assert result == expected


def test_orbit_camera_uses_vertical_fov_and_optional_render(monkeypatch):
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    objects = NamedItems(lambda name, data: NS(name=name, data=data))
    linked, rendered = [], []
    scene = NS(camera=None, render=NS())

    def render(*, write_still):
        assert write_still
        rendered.append(scene.render.filepath)

    bpy = NS(
        data=NS(objects=objects, cameras=NS(new=lambda name: NS(name=name))),
        context=NS(scene=scene, collection=NS(objects=NS(link=linked.append))),
        ops=NS(render=NS(render=render)),
    )
    assert worker.orbit_camera(
        bpy, "main", [3, 4, 5], [0, 0, -2], 30, None, [80, 60]
    ) == {
        "camera": "main",
        "location": [3, 4, 5],
    }
    camera = scene.camera
    assert camera.data.angle == pytest.approx(np.pi / 6)
    assert camera.data.lens_unit == "FOV" and camera.data.sensor_fit == "VERTICAL"
    assert camera.rotation_euler == (0, 0, -1)
    assert not rendered
    result = worker.orbit_camera(
        bpy, "main", [2, 4, 5], [0, 0, -2], 60, "shot.png", [80, 60]
    )
    assert result["rendered"] == "shot.png"
    assert scene.camera is camera and linked == [camera] and rendered == ["shot.png"]
    assert (scene.render.resolution_x, scene.render.resolution_y) == (80, 60)
