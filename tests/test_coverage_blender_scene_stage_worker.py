"""Exercise the Blender staging worker's independent measurements."""

import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import blender_fix_normals as normals_cli
import blender_scene_stage_worker as worker
import numpy as np
import pytest

from tests.test_coverage_blender_scene_query_worker import Matrix, NamedItems, Vector


def test_inward_area_uses_polygon_areas_instead_of_polygon_count():
    normals = [(1, 0, 0), (1, 0, 0), (0, -1, 0)]
    centres = [(-3, 1, 0), (2, -1, 0), (1, 0, 0)]
    assert worker.inward_area_percentage(normals, centres, [6, 2, 2]) == pytest.approx(
        60
    )


class MutableVector(Vector):
    @Vector.y.setter
    def y(self, value):
        self[1] = value


class SceneObject:
    def __init__(self, name, data, kind="MESH"):
        self.name, self.data, self.type = name, data, kind
        self.parent = None
        self.location = MutableVector((0, 0, 0))
        self.scale = (1, 1, 1)
        self.rotation_euler = (0, 0, 0)
        self.users_collection = []
        self.selected = False
        self.bound_box = [(-2, -1, -3), (2, 1, 5)]
        self.matrix_world = Matrix()
        self.material_slots = [
            NS(material=NS(use_backface_culling=None)),
            NS(material=None),
        ]

    def select_set(self, value):
        self.selected = value


class Links(list):
    def link(self, item):
        self.append(item)


class Collection:
    def __init__(self, name):
        self.name = name
        self.children = Links()
        self.members = []
        self.objects = NS(link=self.link, unlink=self.unlink)

    def link(self, obj):
        self.members.append(obj)
        obj.users_collection.append(self)

    def unlink(self, obj):
        self.members.remove(obj)
        obj.users_collection.remove(self)


class Scene:
    def __init__(self, name):
        self.name = name
        self.world = None
        self.camera = None
        self.collection = Collection(name + " objects")
        self.render = NS(
            resolution_x=640,
            resolution_y=480,
            film_transparent=False,
            image_settings=NS(file_format="JPEG", color_mode="RGB"),
        )
        self.view_settings = NS(view_transform="Filmic")
        self.frames = []

    @property
    def objects(self):
        return self.collection.members + [
            obj for child in self.collection.children for obj in child.members
        ]

    def frame_set(self, frame):
        self.frames.append(frame)


class Collections(NamedItems):
    def __init__(self, scenes):
        super().__init__(Collection)
        self.scenes = scenes

    def remove(self, collection):
        for scene in self.scenes:
            if collection in scene.collection.children:
                scene.collection.children.remove(collection)
        super().remove(collection, do_unlink=True)


class Context:
    @property
    def scene(self):
        return self.window.scene


class Datablocks(NamedItems):
    def remove(self, obj, *, do_unlink):
        for collection in list(obj.users_collection):
            collection.unlink(obj)
        super().remove(obj, do_unlink=do_unlink)


@pytest.fixture
def scene_state(monkeypatch):
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=MutableVector))
    origin = Scene("user work")
    world = NS(use_nodes=False, node_tree=NS(nodes={"Background": NS(inputs=[NS()])}))
    scenes = NamedItems(Scene, [origin])
    objects = Datablocks(lambda name, data: SceneObject(name, data, data.kind))
    keep = SceneObject("unsaved work", object())
    objects.append(keep)
    origin.collection.link(keep)
    context = Context()
    context.window = NS(scene=origin)
    context.view_layer = NS(objects=NS(active=keep))
    area = NS(
        type="VIEW_3D",
        spaces=[NS(shading=NS(type="SOLID"))],
        regions=[NS(type="UI"), NS(type="WINDOW")],
    )
    context.screen = NS(areas=[NS(type="OUTLINER"), area])
    audit = NS(renders=[], selection=[], framed=[], worlds=[])

    @contextmanager
    def override(**attrs):
        assert attrs == {"area": area, "region": area.regions[1]}
        yield

    context.temp_override = override

    def create_world(name):
        audit.worlds.append(name)
        return world

    def deselect(*, action):
        assert action == "DESELECT"
        audit.selection.append(action)
        for obj in context.scene.objects:
            obj.select_set(False)

    def render(*, write_still):
        assert write_still
        sc = context.scene
        key = objects.get("KEY")
        audit.renders.append((sc.render.filepath, key.rotation_euler if key else None))

    bpy = NS(
        context=context,
        data=NS(
            scenes=scenes,
            objects=objects,
            collections=Collections(scenes),
            cameras=NS(new=lambda name: NS(name=name, kind="CAMERA")),
            lights=NS(new=lambda name, type: NS(name=name, type=type, kind="LIGHT")),
            worlds=NS(new=create_world),
            actions={},
        ),
        ops=NS(
            import_scene=NS(),
            object=NS(select_all=deselect),
            render=NS(render=render),
            view3d=NS(view_all=lambda: audit.framed.append(True)),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    def add_object(name, data=None, kind="MESH"):
        obj = SceneObject(name, data, kind)
        objects.append(obj)
        context.scene.collection.link(obj)
        return obj

    return bpy, origin, keep, area, audit, add_object


@pytest.mark.parametrize("culled", [True, False])
def test_stage_clears_only_its_scene_and_centres_the_mesh(scene_state, culled, capsys):
    bpy, origin, keep, area, audit, add = scene_state
    private = bpy.data.scenes.new("IMG3D stage")
    previous_world = NS() if culled else None
    private.world = previous_world
    bpy.context.window.scene = private
    stale = add("stale")
    child = bpy.data.collections.new("old collection")
    private.collection.children.link(child)
    bpy.context.window.scene = origin
    imported = []

    def gltf(*, filepath):
        assert bpy.context.scene is private
        assert stale not in bpy.data.objects
        if filepath == "shape.glb":
            body = add("body")
            extra = add("child")
            extra.parent = body
            body.selected = True
            imported.extend([body, extra])
        else:
            assert filepath == "no meshes.glb"
            add("marker", kind="EMPTY")

    bpy.ops.import_scene.gltf = gltf
    items = [
        {"name": "A_body", "path": "shape.glb", "dy": 2.0},
        {"name": "B_marker", "path": "no meshes.glb", "dy": 0},
    ]
    assert worker.stage_assets(bpy, items, culled) == {"staged": ["A_body", "B_marker"]}
    assert capsys.readouterr().out == "STAGED: A_body, B_marker\n"
    assert bpy.context.scene is private and origin.objects == [keep]
    if culled:
        assert private.world is previous_world
    assert [collection.name for collection in private.collection.children] == [
        "A_body",
        "B_marker",
    ]
    assert imported[0].scale == (0.125,) * 3
    assert list(imported[0].location) == [0, 2, 0.375]
    assert imported[1].scale == (1, 1, 1)
    assert all(
        o.material_slots[0].material.use_backface_culling is culled for o in imported
    )
    assert {obj.name for obj in private.collection.members} == {"KEY", "FILL", "RIM"}
    assert all(not obj.selected for obj in private.objects)
    assert audit.selection == ["DESELECT"] and audit.framed == [True]
    assert bpy.context.view_layer.objects.active is None
    assert area.spaces[0].shading.type == "MATERIAL"


def test_light_orbit_keeps_the_camera_fixed_and_restores_window_scene(
    scene_state, capsys
):
    bpy, origin, keep, _, audit, add = scene_state
    rotations = [[0.6, 0, 0], [0.6, 0, 1.2]]

    def gltf(*, filepath):
        assert filepath == "shape.glb"
        add("body")

    bpy.ops.import_scene.gltf = gltf
    assert worker.light_orbit(bpy, "shape.glb", "frames/f_", 80, rotations) == {
        "asset": "shape.glb",
        "frames": 2,
        "written_to": "frames/f_",
    }
    assert json.loads(capsys.readouterr().out) == {
        "asset": "shape.glb",
        "frames": 2,
        "written_to": "frames/f_",
    }
    rendered = bpy.data.scenes.get("IMG3D light orbit")
    assert audit.renders == list(zip(["frames/f_0001", "frames/f_0002"], rotations))
    assert rendered.frames == [1, 2]
    assert rendered.camera.location == (2.15, -1.75, 0.10)
    assert rendered.camera.data.lens == 70
    assert rendered.render.resolution_x == rendered.render.resolution_y == 80
    assert bpy.context.scene is origin and origin.objects == [keep]


def test_light_orbit_restores_scene_after_failed_import(scene_state):
    bpy, origin, keep, _, audit, add = scene_state
    bpy.ops.import_scene.gltf = lambda *, filepath: add("marker", kind="EMPTY")
    with pytest.raises(RuntimeError, match="no mesh imported"):
        worker.light_orbit(bpy, "bad.glb", "out", 80, [])
    assert bpy.context.scene is origin and origin.objects == [keep]
    assert not audit.renders


@pytest.fixture
def bake_scene(scene_state):
    bpy, origin, _, _, audit, add = scene_state
    arm = add("rig", kind="ARMATURE")
    arm.animation_data = NS(action=NS(name="Idle"))
    action = NS(name="Walk", frame_range=(2, 8))
    bpy.data.actions = {"Walk": action}
    target = add("CAM_TARGET", kind="EMPTY")
    target.rotation_euler = (0, 0, 2)
    camera = add("QuickCam2", NS(type="ORTHO", ortho_scale=10), kind="CAMERA")
    previous = add("original", NS(type="PERSP"), kind="CAMERA")
    origin.camera = previous
    return bpy, origin, arm, action, target, camera, previous, audit


@pytest.mark.parametrize("mode", ["native", "override", "failure"])
def test_stage_bake_restores_the_live_scene_after_rendering(
    bake_scene, tmp_path, capsys, mode
):
    bpy, origin, arm, action, target, camera, previous, audit = bake_scene
    old_action = arm.animation_data.action
    directory = tmp_path / "frames"
    args = dict(
        action_name="Walk",
        az=33,
        el=-10,
        ortho_scale=None,
        frame_start=None,
        frame_end=None,
        frame_step=3,
        resolution=128,
        output_dir=str(directory),
        label="small",
    )
    wanted = [2, 5, 8]
    if mode == "override":
        args.update(ortho_scale=20, frame_start=3, frame_end=6)
        camera.name = "Camera"
        wanted = [3, 6]
    original_render = bpy.ops.render.render

    def render(*, write_still):
        assert origin.camera is camera and arm.animation_data.action is action
        assert (
            origin.render.film_transparent
            and origin.view_settings.view_transform == "Standard"
        )
        assert origin.render.resolution_x == origin.render.resolution_y == 128
        assert origin.render.image_settings.color_mode == "RGBA"
        assert target.rotation_euler == pytest.approx(
            (-math.pi / 18, 0, 33 * math.pi / 180)
        )
        if mode == "failure":
            raise RuntimeError("render failed")
        original_render(write_still=write_still)

    bpy.ops.render.render = render
    if mode == "failure":
        with pytest.raises(RuntimeError, match="render failed"):
            worker.stage_bake(bpy, **args)
        assert not audit.renders
    else:
        frames = [f"small_Walk_az33_f{frame:03d}.png" for frame in wanted]
        expected = {
            "label": "small",
            "action": "Walk",
            "az": 33,
            "el": -10,
            "native_range": [2, 8],
            "step": 3,
            "frame_count": len(wanted),
            "frames": frames,
        }
        assert worker.stage_bake(bpy, **args) == expected
        assert origin.frames == wanted
        assert audit.renders == [(str(directory / file), None) for file in frames]
        assert (
            json.loads(capsys.readouterr().out.removeprefix("STAGE_BAKE ")) == expected
        )
        assert camera.data.ortho_scale == (20 if mode == "override" else 10)
    assert directory.is_dir()
    assert arm.animation_data.action is old_action
    assert origin.camera is previous and target.rotation_euler == (0, 0, 2)
    assert (
        not origin.render.film_transparent
        and origin.view_settings.view_transform == "Filmic"
    )
    assert (origin.render.resolution_x, origin.render.resolution_y) == (640, 480)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("rig", "no object named"),
        ("animation", "no animation_data"),
        ("action", "no action named"),
        ("target", "no CAM_TARGET"),
        ("camera", "no ortho camera"),
    ],
)
def test_stage_bake_rejects_incomplete_scene_before_editing(
    bake_scene, tmp_path, failure, message
):
    bpy, origin, arm, _, target, camera, previous, _ = bake_scene
    old_action = arm.animation_data.action
    if failure == "rig":
        arm.name = "other rig"
    elif failure == "animation":
        arm.animation_data = None
    elif failure == "action":
        bpy.data.actions.clear()
    elif failure == "target":
        target.name = "other target"
    else:
        camera.data.type = "PERSP"
    destination = tmp_path / "shots"
    with pytest.raises(RuntimeError, match=message):
        worker.stage_bake(
            bpy, "Walk", 30, 15, None, None, None, 1, 256, str(destination), "preview"
        )
    assert not destination.exists()
    assert origin.camera is previous and target.rotation_euler == (0, 0, 2)
    if arm.animation_data is not None:
        assert arm.animation_data.action is old_action


class Polygons:
    def __init__(self):
        self.normals = [1, 0, 0, 1, 0, 0]

    def __len__(self):
        return 2

    def foreach_get(self, name, dest):
        values = {"normal": self.normals, "center": [-2, 0, 0, 2, 0, 0], "area": [3, 1]}
        assert dest.dtype == np.float32
        assert dest.flags.c_contiguous
        assert len(dest) == len(values[name])
        for i, value in enumerate(values[name]):
            dest[i] = value


@pytest.mark.parametrize("join", [False, True])
def test_normal_fix_measures_before_and_after_and_exports_only_private_scene(
    scene_state, capsys, join
):
    bpy, origin, keep, _, _, add = scene_state
    polygons = Polygons()
    events, bodies = [], []

    def gltf(*, filepath):
        assert filepath == "shape.glb"
        bodies.append(add("body", NS(polygons=polygons)))
        if join:
            bodies.append(add("other", NS(polygons=polygons)))
        add("ignored lamp", kind="LIGHT").selected = True

    def join_meshes():
        assert all(obj.selected for obj in bodies)
        assert bpy.context.view_layer.objects.active is bodies[0]
        events.append("join")

    def recalc(*, inside):
        assert not inside and bpy.context.scene.name == "IMG3D normals"
        events.append("recalc")
        polygons.normals[0] = -1

    def export(**options):
        assert options == {
            "filepath": "fixed.glb",
            "export_format": "GLB",
            "use_selection": False,
        }
        assert not any(
            obj is keep or (obj.type == "LIGHT" and obj.selected)
            for obj in bpy.context.scene.objects
        )
        events.append("export")

    bpy.ops.import_scene.gltf = gltf
    bpy.ops.object.join = join_meshes
    bpy.ops.object.mode_set = lambda *, mode: events.append(mode)
    bpy.ops.mesh = NS(
        select_all=lambda *, action: events.append(action),
        normals_make_consistent=recalc,
    )
    bpy.ops.export_scene = NS(gltf=export)
    if join:
        code = normals_cli.blender_code(Path("shape.glb"), Path("fixed.glb"))
        bridge = {}
        exec(compile(code, "<Blender normal fix request>", "exec"), bridge)
        result = bridge["result"]
    else:
        result = worker.fix_normals(bpy, "shape.glb", "fixed.glb")
    assert result == {"before": 75, "after": 0}
    assert capsys.readouterr().out == "NORMALS:: inward area 75.0% -> 0.0%\n"
    assert events == [
        *(["join"] if join else []),
        "EDIT",
        "SELECT",
        "recalc",
        "OBJECT",
        "export",
    ]
    assert bpy.context.scene is origin and origin.objects == [keep]


def test_normal_area_of_empty_geometry_is_not_a_percentage():
    with pytest.warns(RuntimeWarning, match="Mean of empty slice|invalid value"):
        assert math.isnan(worker.inward_area_percentage([], [], []))


def test_normal_area_with_zero_total_area_is_not_a_percentage():
    with pytest.warns(RuntimeWarning, match="invalid value"):
        result = worker.inward_area_percentage([(1, 0, 0)], [(0, 0, 0)], [0])
    assert math.isnan(result)


def test_normal_area_uses_float32_centroid_at_facing_threshold():
    # The average X is 16,777,217 in double precision, but rounds to
    # the first face's X in float32. Its facing dot is then zero.
    normals = [(1, 0, 0), (1, 0, 0)]
    centres = [(16_777_216, 0, 0), (16_777_218, 0, 0)]
    assert worker.inward_area_percentage(normals, centres, [3, 1]) == 0
