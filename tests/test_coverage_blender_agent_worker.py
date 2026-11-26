"""Exercise the agent's shipped actions without a live Blender session."""

import json
import sys
from itertools import product
from types import SimpleNamespace as NS

import blender_agent as agent
import blender_agent_worker as worker
import numpy as np
import pytest


def test_orbit_direction_faces_front_at_zero_and_rotates_upward():
    assert worker.orbit_direction(0, 0) == pytest.approx((0, -1, 0))
    assert worker.orbit_direction(90, 0) == pytest.approx((1, 0, 0))
    assert worker.orbit_direction(-90, 0) == pytest.approx((-1, 0, 0))
    assert worker.orbit_direction(20, 90) == pytest.approx((0, 0, 1))


@pytest.mark.parametrize(
    ("wanted", "available", "expected"),
    [
        (" eevee ", ["CYCLES", "BLENDER_EEVEE_NEXT"], "BLENDER_EEVEE_NEXT"),
        ("eevee", ["BLENDER_WORKBENCH", "BLENDER_EEVEE"], "BLENDER_EEVEE"),
        ("cycles", ["BLENDER_WORKBENCH", "CYCLES"], "CYCLES"),
        ("not present", ["CYCLES"], None),
    ],
)
def test_engine_resolution_matches_running_blender(wanted, available, expected):
    assert worker.resolve_engine(wanted, available) == expected


class Vector(np.ndarray):
    """Numeric triples with Blender's coordinate accessors."""

    def __new__(cls, coordinates):
        return np.asarray(coordinates, dtype=float).view(cls)

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        return self[2]

    @property
    def length(self):
        return float(np.linalg.norm(self))

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        # Return the requested direction so scene assertions can check the aim.
        return NS(to_euler=lambda: tuple(self))


class Mesh:
    def __init__(self, count=3):
        self.polygons = [object()] * count
        self.materials = []

    def copy(self):
        other = Mesh(len(self.polygons))
        other.materials = self.materials.copy()
        return other


class Object:
    def __init__(self, name, kind, data):
        self.name, self.type, self.data = name, kind, data
        self.location = (0, 0, 0)
        self.scale = (1, 1, 1)
        self.rotation_euler = (0, 0, 0)
        self.box_size = (2, 2, 2)
        self.orientation = np.eye(3)
        self._evaluated_scale = self.scale
        self.matrix_world = self

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, coordinates):
        self._location = Vector(coordinates)

    @property
    def translation(self):
        return self.location

    @property
    def bound_box(self):
        return list(product(*[(-size / 2, size / 2) for size in self.box_size]))

    @property
    def dimensions(self):
        return tuple(
            size * scale for size, scale in zip(self.box_size, self._evaluated_scale)
        )

    def __matmul__(self, point):
        return self.orientation @ (point * self._evaluated_scale) + self.location

    def copy(self):
        other = Object(self.name, self.type, self.data)
        other.location = self.location.copy()
        other.scale = self.scale
        other.rotation_euler = self.rotation_euler
        other.box_size = self.box_size
        return other


class Objects(list):
    def get(self, name):
        return next((obj for obj in self if obj.name == name), None)

    def keys(self):
        return [obj.name for obj in self]

    def __contains__(self, obj):
        return (
            self.get(obj) is not None
            if isinstance(obj, str)
            else super().__contains__(obj)
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            obj = self.get(key)
            if obj is None:
                raise KeyError(key)
            return obj
        return super().__getitem__(key)

    def new(self, name, data):
        return Object(name, data.kind, data)

    def remove(self, obj, *, do_unlink):
        assert do_unlink
        super().remove(obj)


def world():
    return NS(
        use_nodes=False,
        node_tree=NS(
            nodes={
                "Background": NS(inputs=[NS(default_value=None), NS(default_value=1)])
            }
        ),
    )


class Materials(dict):
    def new(self, name):
        bsdf = NS(inputs={key: NS() for key in ("Base Color", "Metallic", "Roughness")})
        result = NS(use_nodes=False, node_tree=NS(nodes={"Principled BSDF": bsdf}))
        self[name] = result
        return result


@pytest.fixture
def app(monkeypatch):
    objects = Objects()
    scene = NS(
        camera=None,
        world=None,
        cycles=NS(samples=1),
        view_settings=NS(exposure=0),
        render=NS(
            resolution_x=320,
            resolution_y=240,
            film_transparent=False,
            engine="CYCLES",
            image_settings=NS(file_format="JPEG"),
        ),
        collection=NS(objects=NS(link=objects.append)),
    )
    context = NS(scene=scene, active_object=None)
    audit = NS(renders=[], primitives=[], updates=[])

    def update():
        audit.updates.append(True)
        for obj in objects:
            obj._evaluated_scale = obj.scale

    context.view_layer = NS(update=update)

    def render(*, write_still):
        assert write_still
        audit.renders.append(scene.render.filepath)

    def primitive(kind):
        def add(**args):
            audit.primitives.append((kind, args))
            obj = Object(kind, "MESH", Mesh())
            obj.box_size = (1, 1, 1)
            objects.append(obj)
            context.active_object = obj

        return add

    engine = NS(
        enum_items=[NS(identifier=kind) for kind in ("CYCLES", "BLENDER_EEVEE_NEXT")]
    )
    app = NS(
        context=context,
        data=NS(
            objects=objects,
            materials=Materials(),
            cameras=NS(new=lambda _: NS(kind="CAMERA")),
            lights=NS(new=lambda _, *, type: NS(kind="LIGHT", type=type)),
            worlds=NS(new=lambda _: world()),
        ),
        ops=NS(
            render=NS(render=render),
            import_scene=NS(),
            mesh=NS(
                **{
                    f"primitive_{kind}_add": primitive(kind)
                    for kind in (
                        "cube",
                        "uv_sphere",
                        "cylinder",
                        "cone",
                        "torus",
                        "plane",
                    )
                }
            ),
        ),
        types=NS(RenderSettings=NS(bl_rna=NS(properties={"engine": engine}))),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))
    return app, scene, objects, audit


def test_shipped_request_prints_compatible_blender_reply(app, capsys):
    bpy, scene, objects, audit = app
    scene.camera = Object("Cam", "CAMERA", NS())
    objects.append(scene.camera)
    path = "/tmp/café's preview.png"
    request = agent.build_call(
        "render_view",
        {"resolution": 96, "samples": 2},
        {"output_path": path},
    )
    bridge_namespace = {"bpy": bpy, "previous_scene": scene}
    exec(compile(request, "<Blender bridge request>", "exec"), bridge_namespace)
    captured = capsys.readouterr().out
    assert captured.count(agent.RESULT_SENTINEL) == 1
    reply = json.dumps({"status": "success", "result": {"result": captured}})
    expected = {"rendered": path, "resolution": [96, 96]}
    assert agent.parse_response(reply) == expected
    assert bridge_namespace["result"] == expected
    assert bridge_namespace["previous_scene"] is scene
    assert "run" not in bridge_namespace
    assert audit.renders == [path]
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.cycles.samples == 2


def test_render_without_camera_reports_error_without_rendering(app):
    bpy, scene, _, audit = app
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    assert worker.render_view(bpy, "/tmp/no.png", 128, 3) == {
        "error": "no active camera; call set_camera first",
    }
    assert not audit.renders
    assert scene.cycles.samples == 1
    assert scene.render.resolution_percentage == 100


def test_describe_scene_refreshes_world_matrices_and_uses_world_mesh_bounds(app):
    bpy, scene, objects, audit = app
    mesh = Object("subject", "MESH", Mesh(8))
    mesh.location = (10, 20, 30)
    mesh.scale = (3, 2, 1)
    mesh.orientation = np.array(((0, 1, 0), (-1, 0, 0), (0, 0, 1)))
    light = Object("spot", "LIGHT", NS(type="SPOT", energy=12.34567))
    scene.camera = Object("camera", "CAMERA", NS(lens=35.125))
    objects.extend([mesh, light, scene.camera, Object("ignored", "CURVE", NS())])
    assert worker.describe_scene(bpy) == {
        "objects": [
            {
                "name": "subject",
                "type": "MESH",
                "location": [10, 20, 30],
                "bounds_min": [8, 17, 29],
                "bounds_max": [12, 23, 31],
                "dimensions": [4, 6, 2],
                "polygons": 8,
            },
            {
                "name": "spot",
                "type": "LIGHT",
                "location": [0, 0, 0],
                "light_type": "SPOT",
                "energy": 12.346,
            },
            {
                "name": "camera",
                "type": "CAMERA",
                "location": [0, 0, 0],
                "lens_mm": 35.12,
            },
        ],
        "active_camera": "camera",
        "resolution": [320, 240],
        "film_transparent": False,
        "engine": "CYCLES",
        "world_strength": None,
    }
    assert audit.updates == [True]
    scene.camera = None
    assert worker.describe_scene(bpy)["active_camera"] is None


@pytest.mark.parametrize("clear", [False, True])
def test_import_only_clears_meshes_and_reports_new_objects(app, clear):
    bpy, _, objects, _ = app
    objects.extend(
        [
            Object("old shell", "MESH", Mesh()),
            Object("past rig", "ARMATURE", NS()),
            Object("marker", "EMPTY", None),
        ]
    )

    def gltf(*, filepath):
        assert filepath == "creature.glb"
        assert ("old shell" in objects) is not clear
        objects.extend(
            [Object("skin", "MESH", Mesh()), Object("new rig", "ARMATURE", NS())]
        )

    bpy.ops.import_scene.gltf = gltf
    assert worker.import_asset(bpy, "creature.glb", clear) == {
        "imported": ["skin", "new rig"],
        "meshes": ["skin"],
    }
    assert "past rig" in objects and "marker" in objects


@pytest.mark.parametrize("keep_lights", [False, True])
def test_clear_scene_preserves_cameras_rigs_and_markers(app, keep_lights):
    bpy, _, objects, _ = app
    types = {
        "shell": "MESH",
        "key": "LIGHT",
        "camera": "CAMERA",
        "rig": "ARMATURE",
        "marker": "EMPTY",
    }
    objects.extend(Object(name, kind, None) for name, kind in types.items())
    removed = ["shell"] if keep_lights else ["shell", "key"]
    assert worker.clear_scene(bpy, keep_lights) == {
        "removed": removed,
        "count": len(removed),
    }
    assert objects.keys() == [key for key in types if key not in removed]


def test_camera_ignores_ground_unless_requested_and_reuses_scene_camera(app):
    bpy, scene, objects, _ = app
    floor = Object("floor", "MESH", Mesh())
    floor.box_size = (50, 50, 0)
    floor.location = (0, 0, -1)
    subject = Object("subject", "MESH", Mesh())
    subject.location = (2, 3, 4)
    objects.extend([floor, subject])
    result = worker.set_camera(bpy, 90, 0, 3, 55, None)
    assert result["aimed_at"] == [2, 3, 4]
    assert result["distance"] == pytest.approx(3 * np.sqrt(3), abs=1e-4)
    assert result["location"] == pytest.approx([2 + 3 * np.sqrt(3), 3, 4], abs=1e-4)
    camera = scene.camera
    assert camera.rotation_euler == pytest.approx((-3 * np.sqrt(3), 0, 0))
    result = worker.set_camera(bpy, 0, 0, 1, 35, "floor")
    assert result["aimed_at"] == [0, 0, -1]
    assert result["lens_mm"] == 35 and scene.camera is camera
    assert worker.set_camera(bpy, 0, 0, 1, 35, "absent") == {
        "error": "no mesh to aim at named 'absent'"
    }
    objects.remove(subject, do_unlink=True)
    assert worker.set_camera(bpy, 0, 0, 1, 35, None)["aimed_at"] == [0, 0, -1]
    objects.remove(floor, do_unlink=True)
    assert worker.set_camera(bpy, 0, 0, 1, 35, None) == {"error": "no mesh to aim at"}


def test_light_uses_subject_size_and_reuses_named_role(app):
    bpy, _, objects, _ = app
    params = dict(
        role="key", azimuth=0, elevation=90, energy=120, size=2, color=[1, 0.5, 0, 0.5]
    )
    assert worker.set_light(bpy, **params) == {"error": "no mesh in the scene to light"}
    objects.append(Object("subject", "MESH", Mesh()))
    assert worker.set_light(bpy, **params) == {
        "light": "AGENT_LIGHT_KEY",
        "energy": 120,
        "location": [0, 0, 5.1962],
    }
    light = objects["AGENT_LIGHT_KEY"]
    assert light.data.size == pytest.approx(2 * np.sqrt(3))
    assert light.data.color == (1, 0.5, 0)
    assert light.rotation_euler == pytest.approx((0, 0, -3 * np.sqrt(3)))
    params["energy"] = 200
    assert worker.set_light(bpy, **params)["energy"] == 200
    assert objects["AGENT_LIGHT_KEY"] is light and len(objects) == 2


def test_world_background_is_created_then_reused(app):
    bpy, scene, _, _ = app
    assert worker.set_world(bpy, 0.3, [0.2, 0.1, 0.4, 1]) == {
        "world_strength": 0.3,
        "world_color": [0.2, 0.1, 0.4],
    }
    initial = scene.world
    assert scene.world.node_tree.nodes["Background"].inputs[0].default_value == (
        0.2,
        0.1,
        0.4,
        1,
    )
    assert worker.describe_scene(bpy)["world_strength"] == 0.3
    worker.set_world(bpy, 0.15, [0, 0, 0])
    assert initial is scene.world
    initial.node_tree.nodes.clear()
    assert worker.set_world(bpy, 0.3, [0.2, 0.1, 0.4]) == {
        "error": "world has no Background node"
    }
    assert worker.describe_scene(bpy)["world_strength"] is None


def test_render_settings_update_only_for_available_engine(app):
    bpy, scene, _, _ = app
    assert worker.set_render_settings(bpy, "Eevee", True, 0.4444) == {
        "engine": "BLENDER_EEVEE_NEXT",
        "film_transparent": True,
        "exposure": 0.444,
        "available": ["CYCLES", "BLENDER_EEVEE_NEXT"],
    }
    result = worker.set_render_settings(bpy, "missing", False, 1)
    assert "MISSING" in result["error"] and "CYCLES" in result["error"]
    assert scene.render.film_transparent and scene.view_settings.exposure == 0.4444


@pytest.mark.parametrize(
    ("kind", "operator", "settings"),
    [
        ("cube", "cube", {"size": 6}),
        ("sphere", "uv_sphere", {"radius": 3}),
        ("cylinder", "cylinder", {"radius": 3, "depth": 6}),
        ("cone", "cone", {"radius1": 3, "depth": 6}),
        ("torus", "torus", {"major_radius": 3, "minor_radius": 1}),
        ("plane", "plane", {"size": 6}),
    ],
)
def test_add_primitive_forwards_shape_parameters_and_reports_fresh_dimensions(
    app, kind, operator, settings
):
    bpy, _, objects, audit = app
    result = worker.add_primitive(
        bpy, kind, "shape", [0, 1, 2], 6, [0, 45, 90], [2, 3, 4]
    )
    assert result == {
        "created": "shape",
        "kind": kind,
        "location": [0, 1, 2],
        "dimensions": [2, 3, 4],
    }
    assert audit.primitives == [(operator, settings)]
    assert objects["shape"].rotation_euler == pytest.approx((0, np.pi / 4, np.pi / 2))
    assert audit.updates == [True]


def test_add_primitive_does_not_mutate_scene_for_invalid_kind_or_taken_name(app):
    bpy, _, objects, audit = app
    objects.append(Object("taken", "MESH", Mesh()))
    kwargs = dict(location=[0, 0, 0], size=1, rotation=[0, 0, 0], scale=[1, 1, 1])
    assert (
        "unknown primitive"
        in worker.add_primitive(bpy, "capsule", "new", **kwargs)["error"]
    )
    assert (
        "already exists"
        in worker.add_primitive(bpy, "cube", "taken", **kwargs)["error"]
    )
    assert not audit.primitives


def test_absolute_and_relative_transforms_preserve_unset_channels(app):
    bpy, _, objects, _ = app
    obj = Object("body", "MESH", Mesh())
    obj.location = (0, 1, 2)
    obj.rotation_euler = (0, 0, np.pi / 2)
    objects.append(obj)
    assert worker.set_object_transform(bpy, "body", None, None, [1, 2, 3]) == {
        "name": "body",
        "location": [0, 1, 2],
        "rotation_deg": [0, 0, 90],
        "scale": [1, 2, 3],
        "dimensions": [2, 4, 6],
    }
    assert worker.set_object_transform(bpy, "body", [2, 4, 6], [0, 10, 15], None) == {
        "name": "body",
        "location": [2, 4, 6],
        "rotation_deg": [0, 10, 15],
        "scale": [1, 2, 3],
        "dimensions": [2, 4, 6],
    }
    assert worker.move_object(bpy, "body", 3, -5, 0.25) == {
        "moved": "body",
        "location": [5, -1, 6.25],
    }
    assert worker.set_object_transform(bpy, "missing", None, None, None) == {
        "error": "no object named missing",
    }
    assert worker.move_object(bpy, "missing", 0, 0, 0) == {
        "error": "no object named missing"
    }
    assert worker.delete_object(bpy, "body") == {"deleted": "body"}
    assert worker.delete_object(bpy, "body") == {"error": "no object named body"}
    assert not objects


def test_set_material_shares_one_material_but_only_on_named_meshes(app):
    bpy, _, objects, _ = app
    a, b, c = (Object(name, "MESH", Mesh()) for name in ("a", "b", "c"))
    a.data.materials.append("older paint")
    marker = Object("marker", "EMPTY", None)
    objects.extend([a, b, c, marker])
    result = worker.set_material(
        bpy, ["a", "b", "marker", "absent"], [0.1, 0.2, 0.9], 0.4, 0.7
    )
    assert result == {
        "coloured": ["a", "b"],
        "not_found": ["marker", "absent"],
        "material": "AGENT_MAT_0.100_0.200_0.900_0.40_0.70",
        "color": [0.1, 0.2, 0.9],
    }
    material = bpy.data.materials[result["material"]]
    inputs = material.node_tree.nodes["Principled BSDF"].inputs
    assert inputs["Base Color"].default_value == (0.1, 0.2, 0.9, 1)
    assert inputs["Roughness"].default_value == 0.4
    assert inputs["Metallic"].default_value == 0.7
    assert a.data.materials == b.data.materials == [material]
    assert not c.data.materials
    assert worker.set_material(bpy, ["c"], [0.1, 0.2, 0.9], 0.4, 0.7)["coloured"] == [
        "c"
    ]
    assert c.data.materials[0] is material
    assert len(bpy.data.materials) == 1
    assert worker.set_material(bpy, [], [], 0.5, 0) == {
        "error": "set_material needs name (one object) or names (several)",
    }


def test_material_without_principled_node_is_still_assigned(app):
    bpy, _, objects, _ = app
    obj = Object("surface", "MESH", Mesh())
    objects.append(obj)

    def new_material(name):
        node_tree = NS(nodes={})
        return NS(use_nodes=False, node_tree=node_tree)

    bpy.data.materials.new = new_material
    assert worker.set_material(bpy, ["surface"], [0, 0, 0], 0.4, 0.7)["coloured"] == [
        "surface"
    ]
    assert obj.data.materials[0].use_nodes


def test_duplicate_grid_keeps_source_location_and_independent_meshes(app):
    bpy, _, objects, _ = app
    source = Object("part", "MESH", Mesh())
    source.location = (1, 2, 3)
    objects.append(source)
    result = worker.duplicate_grid(bpy, "part", [2, 1, 2], [4, 0, -2], None)
    assert result == {
        "created": ["part_0_0_0", "part_0_0_1", "part_1_0_0", "part_1_0_1"],
        "count": 4,
    }
    assert objects[0] is source
    assert [list(o.location) for o in objects] == [
        [1, 2, 3],
        [1, 2, 1],
        [5, 2, 3],
        [5, 2, 1],
    ]
    assert len({o.data for o in objects}) == len(objects)
    objects[-1].data.materials.append("blue")
    assert not source.data.materials
    assert worker.duplicate_grid(bpy, "unknown", [2, 1, 1], [4, 0, 0], None) == {
        "error": "no object named unknown",
    }
    assert worker.duplicate_grid(bpy, source.name, [10, 20, 2], [1, 1, 1], "copy_") == {
        "error": "that is 400 copies; the limit is 200",
    }
    assert len(objects) == 4


def test_mirror_negates_location_and_scale_on_just_one_world_axis(app):
    bpy, _, objects, audit = app
    source = Object("left", "MESH", Mesh())
    source.location = (1, 2, 3)
    source.scale = (2, 3, 4)
    objects.append(source)
    assert worker.mirror_object(bpy, "left", "right", " y ") == {
        "created": "right",
        "mirrored": "left",
        "axis": "y",
        "location": [1, -2, 3],
    }
    mirrored = objects["right"]
    assert mirrored.scale == (2, -3, 4)
    assert source.scale == (2, 3, 4)
    assert list(source.location) == [1, 2, 3]
    assert mirrored.data is not source.data
    assert worker.mirror_object(bpy, "left", "right", "x") == {
        "error": "an object named right already exists",
    }
    assert worker.mirror_object(bpy, "left", "extra", "q") == {
        "error": "axis must be x, y or z, not q"
    }
    assert worker.mirror_object(bpy, "unknown", "extra", "z") == {
        "error": "no object named unknown"
    }
    assert audit.updates == [True] and len(objects) == 2
