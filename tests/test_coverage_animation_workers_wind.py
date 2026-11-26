"""Exercise wind labelling, deformation and rendering with tiny imported meshes."""

import json
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree


class Collection:
    def __init__(self, field, values):
        self.field = field
        self.values = np.array(values)

    def __len__(self):
        return len(self.values)

    def foreach_get(self, field, out):
        assert field == self.field
        out[:] = self.values.ravel()

    def foreach_set(self, field, values):
        assert field == self.field
        self.values[:] = values.reshape(self.values.shape)


class KDTree:
    def __init__(self, size):
        self.points = [None] * size

    def insert(self, point, index):
        self.points[index] = point

    def balance(self):
        self.tree = cKDTree(self.points)

    def find(self, point):
        distance, index = self.tree.query(point)
        return self.points[index], index, distance


class Vector(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return SimpleNamespace(to_euler=lambda: tuple(self))


class Scene:
    def __init__(self, name, world=None, render=None, view_settings=None):
        self.name = name
        self.objects = []
        self.world = world
        self.collection = SimpleNamespace(
            children=[],
            objects=SimpleNamespace(link=self.objects.append),
        )
        self.render = render or SimpleNamespace(image_settings=SimpleNamespace())
        self.view_settings = view_settings or SimpleNamespace()
        self.eevee = SimpleNamespace(use_raytracing=False)


class LegacySettings:
    def __init__(self, **values):
        self.__dict__.update(values)

    def __setattr__(self, key, value):
        if value in ("BLENDER_EEVEE_NEXT", "AgX", "AgX - Medium High Contrast"):
            raise TypeError(value)
        setattr_in_dict = self.__dict__
        setattr_in_dict[key] = value


class MeshObject:
    type = "MESH"

    def __init__(self, coords):
        self.data = SimpleNamespace(
            vertices=Collection("co", coords), update=self.update
        )
        self.updates = 0
        self.selected = False
        self.hide_render = False

    def update(self):
        self.updates += 1

    def select_set(self, value):
        self.selected = value


@pytest.fixture
def wind_scene(monkeypatch):
    def setup(
        *, domain="POINT", legacy=False, failure=None, reuse=False, rigid_only=False
    ):
        original = Scene("user")
        previous_object = MeshObject([[8, 8, 8]])
        original.objects.append(previous_object)
        window = SimpleNamespace(scene=original)
        active = SimpleNamespace(active=None)
        render = LegacySettings(image_settings=SimpleNamespace()) if legacy else None
        views = LegacySettings(view_transform="Standard") if legacy else None
        private = Scene(
            "IMG3D wind", SimpleNamespace() if reuse else None, render, views
        )
        if legacy:
            del private.eevee
        stale = MeshObject([[9, 9, 9]])
        child = object()
        private.objects.append(stale)
        private.collection.children.append(child)
        objects = [previous_object, stale]
        body = MeshObject([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
        labels = MeshObject([[2.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]])
        if failure == "empty labels":
            labels.data.vertices = Collection("co", [])
        if failure == "empty body":
            body.data.vertices = Collection("co", [])
        colors = [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 1, 1]]
        if failure == "no rigid labels":
            colors[1] = [0, 1, 0, 1]
        if rigid_only:
            colors = [[1, 0, 0, 1]] * 3
        if domain == "CORNER":
            vertex_order = [2, 0, 1]
            colors = [colors[index] for index in vertex_order]
            labels.data.loops = Collection("vertex_index", vertex_order)
        attributes = (
            []
            if failure == "no color labels"
            else [
                SimpleNamespace(domain=domain, data=Collection("color", colors)),
            ]
        )
        labels.data.color_attributes = attributes
        rendered, joins, created = [], [], []

        def new_scene(name):
            assert name == private.name
            created.append(name)
            return private

        def remove(obj, *, do_unlink):
            assert do_unlink and window.scene is private
            private.objects.remove(obj)
            objects.remove(obj)

        def import_asset(*, filepath):
            assert window.scene is private
            if filepath == "body.glb" and failure == "no mesh":
                return
            added = (
                [body, MeshObject([[10, 1, 1]])] if filepath == "body.glb" else [labels]
            )
            private.objects.extend(added)
            objects.extend(added)

        def join():
            joined = [
                obj
                for obj in private.objects
                if obj.selected and obj is not active.active
            ]
            joins.append(joined)
            for obj in joined:
                remove(obj, do_unlink=True)

        def render_frame(*, write_still):
            assert window.scene is private and labels.hide_render and write_still
            rendered.append((private.render.filepath, body.data.vertices.values.copy()))
            if failure == "render":
                raise RuntimeError("render stopped")

        object_store = SimpleNamespace(
            new=lambda name, data: SimpleNamespace(name=name, data=data), remove=remove
        )

        class Objects:
            def __iter__(self):
                return iter(objects)

            new = staticmethod(object_store.new)
            remove = staticmethod(object_store.remove)

        class Context:
            @property
            def scene(self):
                return window.scene

        context = Context()
        context.window = window
        context.view_layer = SimpleNamespace(objects=active)
        bpy = SimpleNamespace(
            context=context,
            data=SimpleNamespace(
                objects=Objects(),
                scenes=SimpleNamespace(
                    get=lambda _: private if reuse else None, new=new_scene
                ),
                worlds=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
                collections=SimpleNamespace(remove=private.collection.children.remove),
                cameras=SimpleNamespace(new=lambda name: SimpleNamespace(name=name)),
                lights=SimpleNamespace(
                    new=lambda name, kind: SimpleNamespace(name=name, type=kind)
                ),
            ),
            ops=SimpleNamespace(
                import_scene=SimpleNamespace(gltf=import_asset),
                object=SimpleNamespace(join=join),
                render=SimpleNamespace(render=render_frame),
            ),
        )
        mathutils = ModuleType("mathutils")
        mathutils.Vector = Vector
        monkeypatch.setitem(sys.modules, "bpy", bpy)
        monkeypatch.setitem(sys.modules, "mathutils", mathutils)
        monkeypatch.setitem(
            sys.modules, "mathutils.kdtree", SimpleNamespace(KDTree=KDTree)
        )
        return SimpleNamespace(
            window=window,
            original=original,
            private=private,
            body=body,
            rendered=rendered,
            joins=joins,
            created=created,
            stale=stale,
            previous_object=previous_object,
        )

    return setup


ARGS = {
    "mesh_path": "body.glb",
    "body_name": "Test Plant",
    "labels_name": "Test Labels",
    "labels_path": "labels.glb",
    "frame_dir": "/frames",
    "label": "leaves",
    "resolution": 64,
    "frames": 4,
    "zoom": 1.0,
    "azimuth": 25.0,
    "frequency": 1.0,
    "wave": 0.0,
    "gust": 0.0,
    "bend": 90.0,
    "axis_y": 0.0,
    "axis_z": 1.0,
    "cluster_size": 0.2,
    "jitter": 0.0,
    "flutter": 0.0,
    "world_color": [0.1, 0.2, 0.3],
    "raytracing": True,
}


@pytest.mark.parametrize(("domain", "legacy"), [("POINT", False), ("CORNER", True)])
def test_wind_moves_matching_vertices_and_keeps_anchors(
    wind_scene, domain, legacy, capsys
):
    from blender_wind_demo_worker import run

    case = wind_scene(domain=domain, legacy=legacy, reuse=legacy)
    assert run(ARGS) == {"frames": 4}

    assert case.window.scene is case.original
    assert case.original.objects == [case.previous_object]
    assert not case.previous_object.selected
    assert case.stale not in case.private.objects
    assert case.private.collection.children == []
    assert case.created == ([] if legacy else ["IMG3D wind"])
    assert len(case.joins) == 1 and len(case.joins[0]) == 1
    assert case.private.world.color == [0.1, 0.2, 0.3]
    assert case.private.render.engine == (
        "BLENDER_EEVEE" if legacy else "BLENDER_EEVEE_NEXT"
    )
    assert case.private.view_settings.look == (
        "Medium High Contrast" if legacy else "AgX - Medium High Contrast"
    )
    if not legacy:
        assert case.private.eevee.use_raytracing
    assert (case.private.render.resolution_x, case.private.render.resolution_y) == (
        64,
        64,
    )
    assert case.private.camera.data.lens == 60
    assert [path for path, _ in case.rendered] == [
        f"/frames/leaves_{i:04d}.png" for i in range(4)
    ]
    assert case.body.name == "Test Plant"
    assert any(obj.name == "Test Labels" for obj in case.private.objects)
    assert case.body.updates == 4
    assert "moving 2 of 3 vertices, max stiffness 1.000" in capsys.readouterr().out
    np.testing.assert_allclose(case.rendered[0][1], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    quarter = case.rendered[1][1]
    tip_angle = (0.65 + 0.35 * math.sin(0.85 * math.pi)) * math.pi / 2
    np.testing.assert_allclose(
        quarter,
        [
            [0, 0, 0],
            [math.cos(tip_angle / 2), math.sin(tip_angle / 2), 0],
            [2 * math.cos(tip_angle), 2 * math.sin(tip_angle), 0],
        ],
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("no mesh", "no mesh"),
        ("no color labels", "no color"),
        ("no rigid labels", "rigid"),
        ("empty labels", "label mesh has no vertices"),
        ("empty body", "render mesh has no vertices"),
        ("render", "render stopped"),
    ],
)
def test_wind_restores_the_original_scene_on_failure(wind_scene, failure, message):
    from blender_wind_demo_worker import run

    case = wind_scene(failure=failure)
    with pytest.raises(RuntimeError, match=message):
        run(ARGS)
    assert case.window.scene is case.original
    assert case.original.objects == [case.previous_object]
    assert len(case.rendered) == (1 if failure == "render" else 0)


def test_wind_can_render_rigid_vertices_without_movement(wind_scene):
    from blender_wind_demo_worker import run

    case = wind_scene(rigid_only=True)
    assert run({**ARGS, "frames": 2}) == {"frames": 2}
    for _, positions in case.rendered:
        np.testing.assert_array_equal(positions, [[0, 0, 0], [1, 0, 0], [2, 0, 0]])


@pytest.mark.parametrize("control", ["wave", "jitter"])
def test_wind_phase_offsets_move_clumps_at_frame_zero(wind_scene, control):
    from blender_wind_demo_worker import run

    case = wind_scene()
    run({**ARGS, control: 0.5, "frames": 1})
    positions = case.rendered[0][1]
    np.testing.assert_allclose(np.linalg.norm(positions, axis=1), [0, 1, 2], atol=1e-6)
    assert np.all(np.abs(positions[1:, 1]) > 0.01)


def test_wind_gust_scales_tip_flutter(wind_scene):
    from blender_wind_demo_worker import run

    case = wind_scene()
    run({**ARGS, "bend": 0, "flutter": 0.1, "gust": 0.5})
    positions = case.rendered[1][1]
    flutter = 0.3 * math.sin(3.1 * math.pi / 2)
    np.testing.assert_allclose(
        positions, [[0, 0, 0], [1, flutter / 2, 0], [2, flutter, 0]], atol=1e-6
    )


@pytest.mark.parametrize("ending", [b'"success"}', b"", TimeoutError(), b"bad"])
def test_wind_client_sends_the_request_and_handles_split_replies(monkeypatch, ending):
    import blender_wind_demo as cli

    class Connection:
        def __init__(self):
            self.chunks = iter([b'{"status":', ending, b""])
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True

        def settimeout(self, timeout):
            assert timeout == 321

        def sendall(self, data):
            assert json.loads(data) == {
                "type": "execute_code",
                "params": {"code": "pass"},
            }

        def recv(self, amount):
            assert amount == 65536
            chunk = next(self.chunks)
            if isinstance(chunk, Exception):
                raise chunk
            return chunk

    connection = Connection()

    def connect(address, *, timeout):
        assert address == ("studio.local", 4444)
        assert timeout == 10
        return connection

    monkeypatch.setattr(cli.socket, "create_connection", connect)
    result = cli.send("studio.local", 4444, "pass", 321)
    assert connection.closed
    assert result == '{"status":' + (
        ending.decode("utf-8", errors="replace") if isinstance(ending, bytes) else ""
    )
