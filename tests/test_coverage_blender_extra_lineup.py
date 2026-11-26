"""Exercise lineup import and environment construction at the Blender boundary."""

import math
import sys
from types import SimpleNamespace as NS

import blender_lineup_sweep as lineup
import pytest


class Model:
    def __init__(self, height):
        self.type = "MESH"
        self.dimensions = NS(z=height)
        self.location = NS(x=0, z=0)
        self.rotation_euler = [0, 0, 0]
        self.bound_box = [(0, 0, -1), (0, 0, 1)]
        self.matrix_world = self

    def __matmul__(self, corner):
        return NS(z=corner[2] * self.scale[2] + self.location.z)


@pytest.mark.parametrize("height, scale", [(2, 0.5), (0, 1)])
def test_import_asset_scales_and_grounds_only_new_meshes(height, scale, monkeypatch):
    existing, import_mesh = Model(3), Model(height)
    objects = [existing]
    updates = []

    def load(*, filepath):
        assert filepath == "asset.glb"
        empty = Model(0)
        empty.type = "EMPTY"
        objects.extend([empty, import_mesh])

    app = NS(
        context=NS(view_layer=NS(update=lambda: updates.append(import_mesh.scale))),
        data=NS(objects=objects),
        ops=NS(import_scene=NS(gltf=load)),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=tuple))

    assert lineup.import_asset("asset.glb", "Label", 2, 3) == [import_mesh]
    assert import_mesh.name == "Label"
    assert import_mesh.scale == (scale,) * 3
    assert import_mesh.location.x == pytest.approx(1.7)
    assert import_mesh.location.z == pytest.approx(scale)
    assert import_mesh.rotation_euler[2] == pytest.approx(math.radians(-25))
    assert updates == [(scale,) * 3]
    assert not hasattr(existing, "scale")


def test_environment_creates_floor_world_and_lights(monkeypatch):
    scene = NS()
    floor = NS(data=NS(materials=[]))
    material = NS(
        node_tree=NS(
            nodes={
                "Principled BSDF": NS(
                    inputs={
                        "Base Color": NS(),
                        "Roughness": NS(),
                    }
                )
            }
        )
    )
    world = NS(node_tree=NS(nodes={"Background": NS(inputs=[NS(), NS()])}))
    lights = []
    planes = []

    def add_plane(**settings):
        planes.append(settings)
        app.context.active_object = floor

    app = NS(
        context=NS(scene=scene, collection=NS(objects=NS(link=lights.append))),
        data=NS(
            materials=NS(new=lambda name: material),
            worlds=NS(new=lambda name: world),
            lights=NS(new=lambda name, kind: NS(name=name, kind=kind)),
            objects=NS(new=lambda name, data: NS(name=name, data=data)),
        ),
        ops=NS(mesh=NS(primitive_plane_add=add_plane)),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)

    lineup.build_environment()

    assert planes == [{"size": 400, "location": (0, 0, 0)}]
    assert floor.name == "Floor" and floor.data.materials == [material]
    assert scene.world is world and world.use_nodes and material.use_nodes
    assert (
        material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value
        == 0.55
    )
    assert world.node_tree.nodes["Background"].inputs[1].default_value == 1
    assert [(light.name, light.data.kind, light.data.energy) for light in lights] == [
        ("Key", "AREA", 1400),
        ("Rim", "AREA", 800),
    ]
    assert lights[0].location[1] < 0 < lights[1].location[1]
    assert all(light.data.size == 10 for light in lights)


def test_lineup_dispatches_labeled_and_unlabeled_models(monkeypatch, capsys):
    # The other lineup tests exercise each setup function against its own Blender boundary.
    calls = []
    for name in (
        "clear_scene",
        "import_asset",
        "build_environment",
        "build_camera",
        "configure_render",
    ):
        monkeypatch.setattr(
            lineup, name, lambda *args, name=name: calls.append((name, args))
        )
    app = NS(
        context=NS(scene=NS(render=NS(engine="BLENDER_EEVEE"))),
        ops=NS(render=NS(render=lambda **options: calls.append(("render", options)))),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "frames/f_", "a.glb:First", "b.glb"]
    )

    lineup.main()

    assert calls == [
        ("clear_scene", ()),
        ("import_asset", ("a.glb", "First", 0, 2)),
        ("import_asset", ("b.glb", "asset_1", 1, 2)),
        ("build_environment", ()),
        ("build_camera", (2,)),
        ("configure_render", ("frames/f_",)),
        ("render", {"animation": True}),
    ]
    assert "wrote frames/f_" in capsys.readouterr().out
