"""Exercise the source comparison worker and its headless command."""

import json
import sys
from itertools import product
from pathlib import Path
from runpy import run_path
from subprocess import CompletedProcess
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import blender_compare_source_worker as worker
from scripts import compare_to_source


def test_launcher_passes_render_request_to_tracked_worker(tmp_path, monkeypatch):
    asset = tmp_path / "model's source.glb"
    output = tmp_path / "rendered frames"
    monkeypatch.setenv("BLENDER", "/fixture path/blender")

    def run(command, *, capture_output, text, check):
        assert capture_output and text and not check
        assert command[:5] == ["/fixture path/blender", "--background", "--python-exit-code", "1", "--python"]
        worker = Path(command[5])
        assert worker.name == "blender_compare_source_worker.py"
        assert worker.parent == Path(compare_to_source.__file__).parent
        assert worker.is_file()
        assert command[6] == "--"
        assert json.loads(command[7]) == {
            "asset_path": str(asset), "output_dir": str(output), "label": "quoted 'asset'",
            "angles": [-27, 0, 90], "elevation_deg": 8, "env": "dark", "grey_mode": True,
        }
        return CompletedProcess(command, 0, "render output", "diagnostic")

    monkeypatch.setattr(compare_to_source.subprocess, "run", run)
    assert compare_to_source.render_headless(
        asset, output, "quoted 'asset'", [-27, 0, 90], 8, "dark", grey=True
    ) == "render outputdiagnostic"


class Vector(np.ndarray):
    """Vector arithmetic for the Blender calls made by the worker."""

    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    x = property(lambda self: self[0])
    y = property(lambda self: self[1])
    z = property(lambda self: self[2])

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return SimpleNamespace(to_euler=lambda: tuple(self))


class Objects(list):
    def new(self, name, data):
        obj = SceneObject(name, data)
        self.append(obj)
        return obj

    def link(self, obj):
        self.append(obj)

    def unlink(self, obj):
        self.remove(obj)


class SceneObject:
    def __init__(self, name, data=None, parent=None):
        self.name = name
        self.data = data
        self.parent = parent
        self.location = Vector((0, 0, 0))


class Settings:
    def __init__(self, legacy):
        object.__setattr__(self, "legacy", legacy)

    def __setattr__(self, name, value):
        if self.legacy and value in ("BLENDER_EEVEE_NEXT", "AgX", "AgX - Medium High Contrast"):
            raise TypeError(value)
        object.__setattr__(self, name, value)


@pytest.fixture
def scene_fixture():
    def build(legacy):
        obj = SimpleNamespace
        rendered = []
        setup = []
        materials = []
        objects = Objects()
        prior = SceneObject("unrelated")
        objects.append(prior)
        original_collection = obj(objects=Objects())
        children = Objects()
        scene = obj(
            collection=obj(children=children),
            render=Settings(legacy),
            view_settings=Settings(legacy),
            world=obj(),
        )
        scene.render.image_settings = obj()
        if not legacy:
            scene.eevee = obj()

        imported_parent = SceneObject("Imported root")
        mesh = SceneObject("Offset mesh", obj(materials=["original"]), imported_parent)
        mesh.type = "MESH"
        imported_parent.type = "EMPTY"
        mesh.bound_box = list(product((2, 6), (4, 6), (1, 4)))
        mesh.users_collection = [original_collection]
        imported_parent.users_collection = [original_collection]
        original_collection.objects.extend([imported_parent, mesh])

        class WorldMatrix:
            def __matmul__(self, corner):
                offset = mesh.parent.parent.location if mesh.parent.parent else Vector((0, 0, 0))
                return Vector(corner + offset)

        mesh.matrix_world = WorldMatrix()

        def reset(*, use_empty):
            assert use_empty
            setup.append("reset")

        def import_gltf(*, filepath):
            setup.append(Path(filepath).name)
            objects.extend([imported_parent, mesh])

        def new_collection(name):
            return obj(name=name, objects=Objects())

        def new_material(name):
            inputs = {key: obj() for key in ("Base Color", "Roughness", "Metallic")}
            material = obj(name=name, node_tree=obj(nodes={"Principled BSDF": obj(inputs=inputs)}))
            materials.append(material)
            return material

        def render(*, write_still):
            assert write_still
            rendered.append((scene.render.filepath, tuple(scene.camera.location)))

        bpy = obj(
            context=obj(scene=scene, view_layer=obj(update=lambda: setup.append("update"))),
            data=obj(
                objects=objects, collections=obj(new=new_collection), materials=obj(new=new_material),
                cameras=obj(new=lambda name: obj(name=name)),
                lights=obj(new=lambda name, type: obj(name=name, type=type)),
            ),
            ops=obj(wm=obj(read_factory_settings=reset), import_scene=obj(gltf=import_gltf),
                    render=obj(render=render)),
        )
        return bpy, mesh, imported_parent, original_collection, prior, materials, rendered, setup

    return build


@pytest.mark.parametrize("grey,legacy", [(False, False), (True, True)])
def test_worker_centres_model_and_renders_each_angle(scene_fixture, tmp_path, grey, legacy):
    bpy, mesh, parent, original, prior, materials, rendered, setup = scene_fixture(legacy)
    mode = "grey" if grey else "tex"
    result = worker.run("input.glb", str(tmp_path), "label", [0, 90], 0, "dark", grey, bpy=bpy, vector=Vector)
    assert setup == ["reset", "input.glb", "update", "update"]
    scene = bpy.context.scene
    collection, = scene.collection.children
    assert collection.name == "IMG3D_CMP_label"
    assert original.objects == []
    assert prior in bpy.data.objects and prior not in collection.objects
    assert mesh.parent is parent
    assert parent.parent.name == "IMG3D_CMP_label_ROOT"
    np.testing.assert_allclose(parent.parent.location, (-4, -5, -1))
    assert scene.camera.data.lens == 85
    assert [obj.data.energy for obj in collection.objects if obj.name.endswith(("_KEY", "_FILL", "_RIM"))] == [
        700, 420, 600,
    ]
    assert scene.render.film_transparent
    assert scene.render.image_settings.color_mode == "RGBA"
    assert scene.render.image_settings.file_format == "PNG"
    assert (scene.render.resolution_x, scene.render.resolution_y) == (1100, 1100)
    assert scene.world.color == (.05, .05, .06)
    if legacy:
        assert scene.render.engine == "BLENDER_EEVEE"
        assert scene.view_settings.look == "Medium High Contrast"
    else:
        assert scene.render.engine == "BLENDER_EEVEE_NEXT"
        assert scene.view_settings.look == "AgX - Medium High Contrast"
        assert scene.eevee.use_raytracing
    if grey:
        material, = materials
        assert mesh.data.materials == [material]
        assert material.use_backface_culling
        assert material.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value == .55
    else:
        assert mesh.data.materials == ["original"]
        assert materials == []
    paths = [str(tmp_path / f"label_{mode}_az{angle}.png") for angle in (0, 90)]
    assert result == {"renders": paths, "size": 4}
    assert [path for path, _ in rendered] == paths
    np.testing.assert_allclose([position for _, position in rendered], [(-9.6, 0, 1.62), (0, -9.6, 1.62)], atol=1e-10)


def test_worker_entrypoint_loads_matching_blender_arguments(scene_fixture, tmp_path, monkeypatch):
    bpy, _mesh, _parent, _original, _prior, _materials, rendered, _setup = scene_fixture(False)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=Vector))
    request = {
        "asset_path": "input.glb", "output_dir": str(tmp_path), "label": "fixture",
        "angles": [0], "elevation_deg": 0, "env": "studio", "grey_mode": False,
    }
    monkeypatch.setattr(sys, "argv", ["blender", "--background", "--", json.dumps(request)])
    run_path(worker.__file__, run_name="__main__")
    assert [Path(path).name for path, _ in rendered] == ["fixture_tex_az0.png"]
    assert bpy.context.scene.world.color == (.32, .32, .34)


def test_worker_error_reaches_comparison_command(monkeypatch, tmp_path):
    def run(command, **_kwargs):
        return CompletedProcess(command, 1, "render failed: ", "invalid mesh")

    monkeypatch.setattr(compare_to_source.subprocess, "run", run)
    with pytest.raises(SystemExit, match="render failed: invalid mesh"):
        compare_to_source.render_headless(Path("asset.glb"), tmp_path, "fixture", [0], 0, "studio", False)


def test_comparison_script_runs_directly_from_outside_repo(tmp_path, monkeypatch, capsys):
    script = Path(compare_to_source.__file__)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(script.parent))
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as exit_status:
        run_path(str(script), run_name="__main__")
    assert exit_status.value.code == 0
    assert "--sweep" in capsys.readouterr().out


def test_comparison_treats_opaque_panels_as_full_subjects():
    from PIL import Image

    opaque = Image.new("RGB", (3, 4), "black")
    overlay = compare_to_source.silhouette_overlay(opaque, opaque, height=4)
    assert overlay.size == (3, 4)
    assert np.all(np.asarray(overlay) == compare_to_source.BOTH)


def test_comparison_font_falls_back_on_older_pillow(monkeypatch):
    from PIL import ImageFont

    font = object()

    def load_default(**kwargs):
        if kwargs:
            raise TypeError("size unsupported")
        return font

    monkeypatch.setattr(ImageFont, "load_default", load_default)
    assert compare_to_source._label_font(12) is font
