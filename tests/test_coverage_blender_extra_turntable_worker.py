"""Exercise the shipped Blender turntable worker through an in-memory scene."""

import math
import sys
from types import SimpleNamespace as NS

import blender_turntable as turntable
import blender_turntable_worker as worker
import numpy as np
import pytest


class Vector:
    def __init__(self, xyz):
        self.xyz = np.array(tuple(xyz), dtype=float)

    def __iter__(self):
        return iter(self.xyz)

    def __add__(self, other):
        return Vector(self.xyz + other.xyz)

    def __sub__(self, other):
        return Vector(self.xyz - other.xyz)

    def __mul__(self, other):
        return Vector(self.xyz * other)

    x = property(lambda self: self.xyz[0])
    y = property(lambda self: self.xyz[1])
    z = property(lambda self: self.xyz[2])

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return NS(to_euler=lambda: tuple(self.xyz))


class SceneObject:
    def __init__(self, name, data=None, *, kind="EMPTY", parent=None):
        self.name = name
        self.type = kind
        self.data = data
        self.parent = parent
        self.location = Vector((0, 0, 0))
        self.users_collection = []
        self.bound_box = [(0, -2, 3), (2, 2, 5)]
        self.matrix_world = self
        self.material_slots = []

    def __matmul__(self, point):
        return point + self.parent.location if self.parent else point


class CollectionObjects:
    def __init__(self, collection, scene):
        self.collection = collection
        self.scene = scene

    def link(self, obj):
        if obj not in self.scene.objects:
            self.scene.objects.append(obj)
        obj.users_collection.append(self.collection)

    def unlink(self, obj):
        obj.users_collection.remove(self.collection)


class RenderSettings(NS):
    def __setattr__(self, name, value):
        if self.legacy and name == "engine" and value == "BLENDER_EEVEE_NEXT":
            raise TypeError("engine unavailable")
        super().__setattr__(name, value)


class ViewSettings(NS):
    def __setattr__(self, name, value):
        if self.legacy and value in ("AgX", "AgX - Medium High Contrast"):
            raise TypeError("setting unavailable")
        super().__setattr__(name, value)


class ShaderNodes(list):
    def new(self, kind):
        ports = {
            "ShaderNodeNewGeometry": ({}, {"Backfacing": object()}),
            "ShaderNodeBsdfDiffuse": ({"Color": NS()}, {"BSDF": object()}),
            "ShaderNodeMixShader": (
                {"Fac": object(), 1: object(), 2: object()},
                {"Shader": object()},
            ),
        }
        inputs, outputs = ports[kind]
        node = NS(type=kind, inputs=inputs, outputs=outputs)
        self.append(node)
        return node


def material_slots():
    front = object()
    surface = NS(is_linked=True, links=[NS(from_socket=front)])
    output = NS(
        type="OUTPUT_MATERIAL", is_active_output=True, inputs={"Surface": surface}
    )
    inactive = NS(type="OUTPUT_MATERIAL", is_active_output=False)
    graph_links = []
    graph = NS(
        nodes=ShaderNodes([inactive, output]),
        links=NS(new=lambda *args: graph_links.append(args)),
    )
    unlinked = NS(
        type="OUTPUT_MATERIAL",
        is_active_output=True,
        inputs={"Surface": NS(is_linked=False)},
    )
    slots = [
        NS(material=None),
        NS(material=NS(use_nodes=False)),
        NS(material=NS(use_nodes=True, node_tree=NS(nodes=[inactive]))),
        NS(material=NS(use_nodes=True, node_tree=NS(nodes=[unlinked]))),
        NS(material=NS(use_nodes=True, node_tree=graph)),
    ]
    return slots, graph, graph_links, front, surface


class TurntableScene:
    def __init__(self, *, legacy=False, existing=False, fail=False):
        self.original = NS(objects=[SceneObject("keep")])
        self.window = NS(scene=self.original)
        self.scenes = {}
        self.frames = []
        self.removed_objects = []
        self.removed_collections = []
        self.updates = 0
        self.fail = fail
        self.existing_world = NS()
        self.legacy = legacy
        self.mesh = SceneObject("Mesh", kind="MESH")
        self.child = SceneObject("Mesh child", parent=self.mesh)
        self.slots, self.graph, self.graph_links, self.front, self.surface = (
            material_slots()
        )
        self.mesh.material_slots = self.slots
        if existing:
            scene = self.new_scene("IMG3D turntable")
            scene.world = self.existing_world
            scene.objects.append(SceneObject("stale"))
            scene.collection.children.link(NS(name="prior"))
        self.bpy = NS(
            context=self,
            data=NS(
                scenes=NS(get=self.scenes.get, new=self.new_scene),
                worlds=NS(new=lambda name: NS(name=name)),
                collections=NS(new=self.new_collection, remove=self.remove_collection),
                objects=self,
                cameras=NS(new=lambda name: NS(name=name)),
                lights=NS(new=lambda name, kind: NS(name=name, type=kind)),
            ),
            ops=NS(import_scene=NS(gltf=self.load), render=NS(render=self.render)),
        )
        self.view_layer = NS(update=self.update)

    @property
    def scene(self):
        return self.window.scene

    def __iter__(self):
        return iter([*self.original.objects, *self.scene.objects])

    def new(self, name, data):
        return SceneObject(name, data)

    def remove(self, obj, *, do_unlink):
        assert do_unlink
        self.removed_objects.append(obj.name)
        self.scene.objects.remove(obj)

    def new_scene(self, name):
        scene = NS(
            name=name,
            objects=[],
            world=None,
            render=RenderSettings(legacy=self.legacy, image_settings=NS()),
            view_settings=ViewSettings(legacy=self.legacy),
            collection=NS(children=[]),
        )
        children = []

        class Children(list):
            link = list.append

        scene.collection.children = Children(children)
        if not self.legacy:
            scene.eevee = NS()
        self.scenes[name] = scene
        return scene

    def new_collection(self, name):
        collection = NS(name=name)
        collection.objects = CollectionObjects(collection, self.scene)
        return collection

    def remove_collection(self, collection):
        self.removed_collections.append(collection.name)
        self.scene.collection.children.remove(collection)

    def load(self, *, filepath):
        assert filepath == "/remote input/creature.glb"
        source = self.new_collection("import owner")
        source.objects.link(self.mesh)
        source.objects.link(self.child)

    def update(self):
        self.updates += 1

    def render(self, *, write_still):
        assert write_still
        assert self.scene.name == "IMG3D turntable"
        if self.fail:
            raise RuntimeError("render failed")
        spin = self.mesh.parent
        self.frames.append((self.scene.render.filepath, spin.rotation_euler))


@pytest.fixture
def arguments(tmp_path):
    return {
        "asset_path": "/remote input/creature.glb",
        "frame_dir": str(tmp_path / "frames"),
        "label": "model",
        "frames": 4,
        "resolution": 256,
        "zoom": 2.0,
        "darken_backfaces": True,
        "world_color": [0.1, 0.2, 0.3],
        "raytracing": True,
    }


@pytest.mark.parametrize("legacy", [False, True])
def test_turntable_worker_renders_and_restores_private_scene(
    legacy, arguments, monkeypatch
):
    app = TurntableScene(legacy=legacy, existing=legacy)
    arguments["darken_backfaces"] = not legacy
    monkeypatch.setitem(sys.modules, "bpy", app.bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))

    result = worker.run(arguments)

    assert app.window.scene is app.original
    assert [obj.name for obj in app.original.objects] == ["keep"]
    assert app.removed_objects == (["stale"] if legacy else [])
    assert app.removed_collections == (["prior"] if legacy else [])
    scene = app.scenes["IMG3D turntable"]
    if legacy:
        assert scene.world is app.existing_world
        assert scene.view_settings.look == "Medium High Contrast"
    else:
        assert scene.world.name == "IMG3D turntable World"
        assert scene.eevee.use_raytracing is True
        assert scene.view_settings.view_transform == "AgX"
        assert scene.view_settings.look == "AgX - Medium High Contrast"
    assert scene.world.color == arguments["world_color"]
    assert scene.render.engine == ("BLENDER_EEVEE" if legacy else "BLENDER_EEVEE_NEXT")
    assert (scene.render.resolution_x, scene.render.resolution_y) == (256, 256)
    assert scene.render.resolution_percentage == 100
    assert scene.render.film_transparent is False
    assert vars(scene.render.image_settings) == {
        "file_format": "PNG",
        "color_mode": "RGBA",
    }
    assert scene.camera.data.lens == 58
    assert scene.camera.location.xyz == pytest.approx([0, -4.5, 1.12])
    assert scene.camera.rotation_euler == pytest.approx([0, 4.5, 0])

    (collection,) = scene.collection.children
    assert collection.name == "IMG3D_TT_model"
    assert app.mesh.users_collection == app.child.users_collection == [collection]
    assert app.child.parent is app.mesh
    assert app.mesh.parent.location.xyz == pytest.approx([-1, 0, -3])
    lights = [
        obj
        for obj in scene.objects
        if obj.data and getattr(obj.data, "type", None) == "AREA"
    ]
    assert [(obj.name.rsplit("_", 1)[1], obj.data.energy) for obj in lights] == [
        ("KEY", 700),
        ("FILL", 420),
        ("RIM", 600),
    ]
    assert app.updates == 6
    assert [rotation[2] for _, rotation in app.frames] == pytest.approx(
        [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    )
    expected_paths = [
        f"{arguments['frame_dir']}/model_{index:04d}.png" for index in range(4)
    ]
    assert [path for path, _ in app.frames] == expected_paths
    assert result == {
        "frames": 4,
        "size": 4,
        "first": expected_paths[0],
        "last": expected_paths[-1],
    }
    if legacy:
        assert not app.graph_links
    else:
        geometry, dark, mix = app.graph.nodes[-3:]
        assert dark.inputs["Color"].default_value == (0.012, 0.012, 0.012, 1.0)
        assert app.graph_links == [
            (geometry.outputs["Backfacing"], mix.inputs["Fac"]),
            (app.front, mix.inputs[1]),
            (dark.outputs["BSDF"], mix.inputs[2]),
            (mix.outputs["Shader"], app.surface),
        ]


def test_turntable_restores_previous_scene_after_render_error(arguments, monkeypatch):
    app = TurntableScene(fail=True)
    monkeypatch.setitem(sys.modules, "bpy", app.bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))

    with pytest.raises(RuntimeError, match="render failed"):
        worker.run(arguments)
    assert app.window.scene is app.original


def test_turntable_bundles_shipped_worker_with_remote_arguments(arguments, monkeypatch):
    from pathlib import Path

    def bundle(path, payload):
        assert path.samefile(worker.__file__)
        assert payload == {
            **arguments,
            **turntable.ENVIRONMENTS["dark"],
        }
        return "packed worker"

    monkeypatch.setattr(turntable, "build_worker_code", bundle)
    assert (
        turntable.blender_code(
            Path(arguments["asset_path"]),
            Path(arguments["frame_dir"]),
            "model",
            "dark",
            4,
            256,
            2.0,
            True,
        )
        == "packed worker"
    )
