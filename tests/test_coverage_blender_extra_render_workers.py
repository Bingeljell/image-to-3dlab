"""Exercise live previews and headless comparisons through a small scene boundary."""

import math
import runpy
import socket
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import blender_render_asset as preview
import blender_render_asset_worker as preview_worker
import blender_render_glb_comparison_worker as comparison_worker
import numpy as np
import pytest
import render_glb_comparison as comparison


class Vector:
    def __init__(self, coordinates):
        self.xyz = np.array(tuple(coordinates), dtype=float)

    def __iter__(self):
        return iter(self.xyz)

    def __getitem__(self, index):
        return self.xyz[index]

    def __add__(self, other):
        return Vector(self.xyz + other.xyz)

    def __sub__(self, other):
        return Vector(self.xyz - other.xyz)

    def __mul__(self, factor):
        return Vector(self.xyz * factor)

    x = property(lambda self: self[0])
    y = property(lambda self: self[1])
    z = property(lambda self: self[2])

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return NS(to_euler=lambda: tuple(self.xyz))


class Object:
    def __init__(self, name, data=None, *, kind="EMPTY", parent=None):
        self.name, self.data, self.type, self.parent = name, data, kind, parent
        self.location = (0, 0, 0)
        self.bound_box = [(0, -2, 3), (2, 2, 5)]
        self.matrix_world = self
        self.selected = False
        self.users_collection = []

    @property
    def users(self):
        return len(self.users_collection)

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, coordinates):
        self._location = Vector(coordinates)

    @property
    def material_slots(self):
        return [NS(material=material) for material in self.data.materials]

    def __matmul__(self, vector):
        return (
            vector
            + self.location
            + (self.parent.location if self.parent else Vector((0, 0, 0)))
        )

    def select_set(self, selected):
        self.selected = selected


class Names(list):
    link = list.append
    unlink = list.remove


class Collection:
    def __init__(self, name, scene):
        self.name, self.scene = name, scene
        self.objects = self
        self.children = Names()
        self.users = 0

    def __iter__(self):
        return (obj for obj in self.scene.objects if self in obj.users_collection)

    def link(self, obj):
        assert self not in obj.users_collection
        if obj not in self.scene.objects:
            self.scene.objects.append(obj)
        obj.users_collection.append(self)

    def unlink(self, obj):
        obj.users_collection.remove(self)
        if not obj.users_collection:
            self.scene.objects.remove(obj)


class Render(NS):
    def __setattr__(self, key, value):
        if key == "engine" and value not in self.available:
            raise TypeError("engine unavailable")
        super().__setattr__(key, value)


class View(NS):
    def __setattr__(self, key, value):
        if self.legacy and value in ("AgX", "AgX - Medium High Contrast"):
            raise TypeError("view setting unavailable")
        super().__setattr__(key, value)


class Materials(list):
    def clear(self):
        self.events.append("material swap")
        super().clear()


class App:
    def __init__(
        self, monkeypatch, *, engines=("BLENDER_EEVEE_NEXT",), empty=False, fail=False
    ):
        self.engines = engines
        self.legacy = "BLENDER_EEVEE_NEXT" not in engines
        self.empty = empty
        self.fail = fail
        self.events = []
        self.scenes = {}
        self.window = NS(scene=self.new_scene("user"))
        self.original = self.scene
        self.original.objects.append(Object("user mesh", kind="MESH"))
        self.frames = []
        self.removed = []
        self.mode = "OBJECT"
        self.view_layer = NS(update=lambda: None, objects=NS(active=None))
        self.original_material = NS(name="paint")
        textured = Materials([self.original_material, None])
        textured.events = self.events
        self.mesh = Object("mesh", NS(materials=textured), kind="MESH")
        self.child = Object("child", parent=self.mesh)

        bpy = NS(
            context=self,
            data=NS(
                scenes=NS(new=self.new_scene, get=self.scenes.get),
                user_map=lambda subset: {},
                collections=NS(new=self.new_collection, remove=self.remove_collection),
                objects=self,
                cameras=NS(new=lambda name: NS(name=name, angle_x=1.0, angle_y=0.6)),
                lights=NS(new=lambda name, type: NS(name=name, type=type)),
                materials=NS(new=self.new_material),
                worlds=NS(new=self.new_world),
            ),
            ops=NS(
                wm=NS(read_factory_settings=self.reset),
                import_scene=NS(gltf=self.load),
                object=NS(select_all=self.deselect, mode_set=self.set_mode),
                mesh=NS(
                    select_all=self.select_vertices, normals_make_consistent=self.recalc
                ),
                render=NS(render=self.render),
            ),
        )
        monkeypatch.setitem(sys.modules, "bpy", bpy)
        monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector))

    @property
    def scene(self):
        return self.window.scene

    @property
    def collection(self):
        return self.scene.collection

    def __iter__(self):
        return iter(obj for scene in self.scenes.values() for obj in scene.objects)

    def new_scene(self, name):
        render = type(
            "EngineSettings",
            (Render,),
            {
                "bl_rna": NS(
                    properties={
                        "engine": NS(
                            enum_items=[
                                NS(identifier=engine) for engine in self.engines
                            ]
                        )
                    }
                ),
            },
        )
        scene = NS(name=name, objects=[], world=None, collection=NS(children=Names()))
        scene.render = render(available=self.engines, image_settings=NS())
        scene.view_settings = View(legacy=self.legacy)
        scene.collection.objects = Collection(name, scene)
        if not self.legacy:
            scene.eevee = NS()
        self.scenes[name] = scene
        return scene

    def new_collection(self, name):
        return Collection(name, self.scene)

    def remove_collection(self, collection, *, do_unlink):
        assert not do_unlink
        self.removed.append(collection.name)

    def new(self, name, data):
        return Object(name, data)

    def remove(self, obj, *, do_unlink):
        assert not do_unlink and not obj.users
        self.removed.append(obj.name)

    def new_material(self, name):
        sockets = {name: NS() for name in ("Base Color", "Roughness", "Metallic")}
        return NS(
            name=name, node_tree=NS(nodes={"Principled BSDF": NS(inputs=sockets)})
        )

    def new_world(self, name):
        return NS(name=name, node_tree=NS(nodes={"Background": NS(inputs=[NS()])}))

    def reset(self, *, use_empty):
        assert use_empty
        self.events.append("reset")
        self.window.scene = self.new_scene("headless")

    def load(self, *, filepath):
        assert filepath == "/remote/input.glb"
        self.events.append("import")
        if not self.empty:
            self.collection.objects.link(self.mesh)
            self.collection.objects.link(self.child)

    def deselect(self, *, action):
        assert action == "DESELECT"
        for obj in self.scene.objects:
            obj.selected = False

    def set_mode(self, *, mode):
        assert self.view_layer.objects.active is self.mesh and self.mesh.selected
        self.mode = mode

    def select_vertices(self, *, action):
        assert action == "SELECT" and self.mode == "EDIT"
        self.events.append("vertices")

    def recalc(self, *, inside):
        assert self.mode == "EDIT" and not inside
        self.events.append("recalc")

    def render(self, *, write_still):
        assert write_still
        if self.fail:
            raise RuntimeError("renderer crashed")
        self.frames.append(
            (self.scene.render.filepath, tuple(self.scene.camera.location))
        )


def preview_arguments(culled, recalc):
    return {
        "asset_path": "/remote/input.glb",
        "output_dir": "/remote/output",
        "label": "preview",
        "culled": culled,
        "recalc_normals": recalc,
        **preview.ENVIRONMENTS["studio"],
    }


@pytest.mark.parametrize(
    "culled,recalc", [(False, False), (True, False), (False, True), (True, True)]
)
def test_preview_camera_material_and_normals_are_applied(culled, recalc, monkeypatch):
    engine = "BLENDER_EEVEE" if culled else "BLENDER_EEVEE_NEXT"
    app = App(monkeypatch, engines=(engine,))
    old_render = app.new_scene("IMG3D render") if recalc else None
    if old_render:
        old_render.world = app.new_world("existing")
        old_render.collection.objects.link(Object("stale render mesh"))
        shared = Collection("keep collection", old_render)
        shared.users = 1  # Still linked elsewhere after removal from this scene.
        old_render.collection.children.extend([Collection("IMG3D_old", old_render), shared])

    result = preview_worker.run(preview_arguments(culled, recalc))

    scene = app.scenes["IMG3D render"]
    assert app.window.scene is app.original
    assert [obj.name for obj in app.original.objects] == ["user mesh"]
    assert app.removed == (["IMG3D_old", "stale render mesh"] if recalc else [])
    assert scene.world.name == ("existing" if recalc else "World")
    assert scene.world.color == pytest.approx(
        preview.ENVIRONMENTS["studio"]["world_color"]
    )
    assert scene.render.engine == engine
    assert (scene.render.resolution_x, scene.render.resolution_y) == (900, 900)
    assert vars(scene.render.image_settings) == {
        "file_format": "PNG",
        "color_mode": "RGBA",
    }
    assert scene.view_settings.look == (
        "Medium High Contrast" if culled else "AgX - Medium High Contrast"
    )
    if not culled:
        assert (
            scene.eevee.use_raytracing is True
            and scene.view_settings.view_transform == "AgX"
        )
    assert app.mesh.parent.name == "IMG3D_preview_ROOT"
    assert (
        app.child.parent is app.mesh and not app.mesh.selected and app.mode == "OBJECT"
    )
    names = [
        "front_xneg",
        "front_three_quarter",
        "rear_xpos",
        "profile_yneg",
        "profile_ypos",
    ]
    paths = [f"/remote/output/preview_{name}.png" for name in names]
    assert result["objects"] == ["mesh"]
    assert result["bounds"] == [[-1, -2, 0], [1, 2, 2]]
    assert result["renders"] == paths == [path for path, _ in app.frames]
    assert [location[:2] for _, location in app.frames] == [
        (-9, 0),
        (-8.1, -4.05),
        (9, 0),
        (0, -9),
        (0, 9),
    ]
    assert [location[2] for _, location in app.frames] == pytest.approx([1.12] * 5)
    assert app.events == ["import"] + (["vertices", "recalc"] if recalc else []) + (
        ["material swap"] if culled else []
    )
    if culled:
        (grey,) = app.mesh.data.materials
        assert grey.use_backface_culling and grey.use_nodes
        assert result["materials"] == ["HONEST_GREY_preview"]
    else:
        assert app.mesh.data.materials == [app.original_material, None]
        assert result["materials"] == ["paint"]


def test_preview_restores_window_on_render_failure(monkeypatch):
    app = App(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="renderer crashed"):
        preview_worker.run(preview_arguments(False, False))
    assert app.window.scene is app.original


def test_preview_bundles_flags_and_environment_for_remote_blender(monkeypatch):
    def bundle(path, arguments):
        assert path.samefile(preview_worker.__file__)
        assert arguments == preview_arguments(True, False)
        return "bundled"

    monkeypatch.setattr(preview, "build_worker_code", bundle)
    assert (
        preview.blender_code(
            Path("/remote/input.glb"),
            Path("/remote/output"),
            "preview",
            "studio",
            culled=True,
        )
        == "bundled"
    )


@pytest.mark.parametrize(
    ("engines", "expected", "clay"),
    [
        (("BLENDER_EEVEE_NEXT", "CYCLES"), "BLENDER_EEVEE_NEXT", False),
        (("BLENDER_EEVEE", "CYCLES"), "BLENDER_EEVEE", True),
        (("CYCLES",), "CYCLES", True),
        ((), None, False),
    ],
)
def test_comparison_worker_frames_box_and_preserves_or_replaces_paint(
    engines, expected, clay, monkeypatch
):
    app = App(monkeypatch, engines=engines)
    comparison_worker.render_panel(
        "/remote/input.glb", "/tmp/panel.png", 90.0, 0.0, 160, 120, clay
    )

    assert app.events == ["reset", "import"] + (["material swap"] if clay else [])
    scene = app.scene
    assert scene.name == "headless"
    assert (scene.render.resolution_x, scene.render.resolution_y) == (160, 120)
    assert getattr(scene.render, "engine", None) == expected
    assert scene.render.image_settings.file_format == "PNG"
    radius = math.sqrt(6)
    distance = radius / math.sin(0.3) * 1.02
    assert tuple(scene.camera.location) == pytest.approx([1 + distance, 0, 4])
    assert scene.camera.rotation_euler == pytest.approx([-distance, 0, 0])
    assert [path for path, _ in app.frames] == ["/tmp/panel.png"]
    assert scene.world.node_tree.nodes["Background"].inputs[0].default_value == (
        0.04,
        0.04,
        0.045,
        1,
    )
    lights = [obj for obj in scene.objects if getattr(obj.data, "type", None) == "AREA"]
    assert [obj.name for obj in lights] == ["key", "fill", "rim"]
    assert [obj.data.energy for obj in lights] == pytest.approx([1440, 528, 768])
    if clay:
        (material,) = app.mesh.data.materials
        assert material.name == "clay" and material.use_nodes
        assert (
            material.node_tree.nodes["Principled BSDF"]
            .inputs["Roughness"]
            .default_value
            == 0.62
        )
    else:
        assert app.mesh.data.materials == [app.original_material, None]


def test_headless_worker_entry_accepts_blender_arguments_and_rejects_empty_import(
    monkeypatch,
):
    app = App(monkeypatch, empty=True)
    path = comparison_worker.__file__
    monkeypatch.setattr(
        sys,
        "argv",
        [
            path,
            "--background",
            "--",
            "/remote/input.glb",
            "/tmp/panel.png",
            "215",
            "14",
            "1200",
            "820",
            "1",
        ],
    )
    with pytest.raises(SystemExit, match="no mesh in /remote/input.glb"):
        runpy.run_path(path, run_name="__main__")
    assert app.events == ["reset", "import"]
    assert not app.frames


@pytest.mark.parametrize(
    ("module", "option"),
    [(preview, "--recalc-normals"), (comparison, "--clay")],
)
def test_render_command_help_exits_before_contacting_blender(
    module, option, monkeypatch, capsys
):
    script = Path(module.__file__)
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    monkeypatch.setitem(sys.modules, "bpy", None)

    def blocked(*args, **kwargs):
        pytest.fail("Showing help must not contact Blender")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(comparison.subprocess, "run", blocked)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")

    assert exc.value.code == 0
    assert option in capsys.readouterr().out


def test_comparison_detects_when_discovered_blender_has_disappeared(
    tmp_path, monkeypatch
):
    missing = tmp_path / "missing"
    monkeypatch.setattr(comparison.shutil, "which", lambda name: str(missing / name))

    with pytest.raises(SystemExit, match="Blender not found"):
        comparison.main(
            [str(tmp_path / "out.jpg"), "a.glb", "--blender", str(missing / "custom")]
        )


def test_comparison_refuses_to_render_when_any_input_is_missing(tmp_path):
    blender = tmp_path / "blender"
    blender.touch()
    valid = tmp_path / "one.glb"
    valid.touch()
    missing = tmp_path / "two.glb"

    with pytest.raises(SystemExit, match=r"missing GLB\(s\).*two.glb"):
        comparison.main(
            [
                str(tmp_path / "out.jpg"),
                str(valid),
                str(missing),
                "--blender",
                str(blender),
            ]
        )


@pytest.mark.skipif(
    sys.platform == "win32", reason="Windows filenames cannot contain a colon"
)
def test_comparison_can_open_an_existing_colon_separated_filename(tmp_path):
    asset = tmp_path / "preview:coat.glb"
    asset.touch()

    assert comparison.parse_asset(str(asset)) == (asset, "preview:coat")
