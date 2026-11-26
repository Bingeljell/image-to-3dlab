"""Exercise Blender command entry points against small in-memory scenes."""

import ast
import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import hero_blender as hero
import numpy as np
import pytest

from image_to_3dlab.glb_turn import build_glb, parse_glb
from scripts import blender_joint_markers as markers
from scripts import blender_upright as upright


class Vector(np.ndarray):
    def __new__(cls, values=(0, 0, 0)):
        return np.asarray(values, dtype=float).view(cls)

    @property
    def z(self):
        return self[2]

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return NS(to_euler=lambda: tuple(self))


class Matrix(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values, dtype=float).view(cls)

    @classmethod
    def Identity(cls, size):
        return cls(np.eye(size))

    def __matmul__(self, rhs):
        if rhs.shape == (3,):
            return Vector(np.asarray(self) @ np.r_[rhs, 1.0])[:3]
        return Matrix(np.asarray(self) @ np.asarray(rhs))


class Mesh:
    type = "MESH"

    def __init__(self, points, *, faces=1):
        self.name = "Imported mesh"
        self.parent = object()
        self.matrix_world = Matrix.Identity(4)
        self.data = MeshData(points, faces)
        self.selected = False

    def select_set(self, selected):
        self.selected = selected


class MeshData:
    def __init__(self, points, faces):
        self.vertices = [NS(co=Vector(point)) for point in points]
        self.polygons = [None] * faces
        self.materials = ["material"]
        self.updated = False

    def copy(self):
        return MeshData([v.co for v in self.vertices], len(self.polygons))

    def transform(self, matrix):
        for vertex in self.vertices:
            vertex.co = matrix @ vertex.co

    def update(self):
        self.updated = True


class LegacyRender:
    def __init__(self, accepted):
        self.accepted = accepted
        self.attempted = []
        self._engine = "DEFAULT"

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, engine):
        self.attempted.append(engine)
        if engine != self.accepted:
            raise TypeError(engine)
        self._engine = engine


@pytest.fixture
def scene(monkeypatch):
    def setup(objects, *, engine="BLENDER_EEVEE_NEXT"):
        calls = []
        active = NS(active=None)
        render = LegacyRender(engine)
        scene = NS(objects=list(objects), render=render)
        scene.collection = NS(objects=NS(link=scene.objects.append))

        def remove_object(obj, *, do_unlink):
            assert do_unlink
            scene.objects.remove(obj)

        def new_object(name, data=None):
            return NS(name=name, data=data)

        def record(name):
            def operation(**kwargs):
                calls.append((name, kwargs))

            return operation

        def export(**kwargs):
            record("export")(**kwargs)
            Path(kwargs["filepath"]).write_bytes(
                build_glb(
                    {
                        "asset": {"version": "2.0", "generator": "Blender"},
                        "meshes": [{}],
                    },
                    b"mesh",
                )
            )

        def render_frame(*, write_still):
            assert write_still
            calls.append(
                ("render", (Path(render.filepath).name, scene.camera.location.copy()))
            )

        class Objects:
            def __iter__(self):
                return iter(scene.objects)

            new = staticmethod(new_object)
            remove = staticmethod(remove_object)

        bpy = NS(
            context=NS(scene=scene, view_layer=NS(objects=active)),
            data=NS(
                objects=Objects(),
                worlds=NS(new=new_object),
                cameras=NS(new=lambda name: NS(name=name)),
                lights=NS(new=lambda name, kind: NS(name=name, kind=kind)),
            ),
            ops=NS(
                wm=NS(read_factory_settings=record("factory")),
                import_scene=NS(gltf=record("import")),
                object=NS(
                    transform_apply=record("apply"),
                    join=record("join"),
                    select_all=record("select_all"),
                ),
                export_scene=NS(gltf=export),
                render=NS(render=render_frame),
            ),
        )
        monkeypatch.setitem(sys.modules, "bpy", bpy)
        monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=Vector, Matrix=Matrix))
        return bpy, calls

    return setup


@pytest.mark.parametrize(
    "engine", ["BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH", "DEFAULT"]
)
def test_hero_check_uses_largest_mesh_and_renders_six_framings(
    scene, monkeypatch, tmp_path, engine
):
    large = Mesh([[-1, -1, 0], [1, 1, 4], [-1, -1, 3]], faces=4)
    decoy = Mesh([[0, 0, 90], [3, 0, 120]])
    bpy, calls = scene([decoy, large, NS(type="EMPTY")], engine=engine)
    output = tmp_path / "checks"
    monkeypatch.setattr(
        sys,
        "argv",
        ["blender", "--", "check", "asset.glb", str(output), "--head-share", "0.5"],
    )

    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(hero.__file__, run_name="__main__")
    assert stopped.value.code == 0
    assert output.is_dir()
    assert calls[:2] == [
        ("factory", {"use_empty": True}),
        ("import", {"filepath": "asset.glb"}),
    ]
    shots = dict(value for op, value in calls if op == "render")
    assert tuple(shots) == (
        "head_front.png",
        "head_three_quarter.png",
        "head_side.png",
        "head_back.png",
        "body_front.png",
        "body_three_quarter.png",
    )
    assert shots["head_front.png"] == pytest.approx([0, -4.4, 3.5])
    assert shots["head_side.png"] == pytest.approx([4.4, 0, 3.5])
    assert shots["body_front.png"][1] == pytest.approx(-2.5)
    assert bpy.context.scene.render.engine == engine
    assert len(bpy.context.scene.render.attempted) == (
        1 if engine.endswith("NEXT") else 3
    )


@pytest.mark.parametrize("with_mesh", [False, True])
def test_hero_join_exports_one_scene_object_without_animations(
    scene, monkeypatch, tmp_path, capsys, with_mesh
):
    head = Mesh([[0, 0, 0]])
    body = Mesh([[0, 0, 0]])
    empty = NS(type="EMPTY", select_set=lambda selected: None)
    bpy, calls = scene([head, body, empty] if with_mesh else [empty])
    output = tmp_path / "joined.glb"
    monkeypatch.setattr(sys, "argv", ["hero", "join", "asset.glb", str(output)])

    assert hero.main() == (0 if with_mesh else 1)
    assert output.exists() is with_mesh
    if with_mesh:
        assert [op for op, _ in calls] == [
            "factory",
            "import",
            "apply",
            "join",
            "export",
        ]
        assert bpy.context.scene.objects == [head]
        assert head.name == "Character"
        assert bpy.context.view_layer.objects.active is head
        assert calls[-1][1]["export_animations"] is False
        assert calls[-1][1]["use_active_scene"] is True
    else:
        assert "no mesh" in capsys.readouterr().err


@pytest.mark.parametrize("points", [[], [[0, 0]], np.empty((0, 3)), [[0, math.inf, 0]]])
def test_upright_rejects_invalid_coordinates(points):
    with pytest.raises(ValueError, match="finite 3D points"):
        upright.upright_matrix(points)


def test_upright_transforms_world_vertices_and_preserves_source_metadata(
    scene, monkeypatch, tmp_path
):
    box = [[x, y, z] for x in (-1, 1) for y in (-0.5, 0.5) for z in (0, 3)]
    mesh = Mesh(box)
    mesh.matrix_world[:3, :3] = upright.tilt_matrix(12, 7)
    mesh.matrix_world[:3, 3] = [4, 8, 2]
    original_data = mesh.data
    scene([mesh, NS(type="EMPTY")])
    glb = tmp_path / "source.glb"
    glb.write_bytes(
        build_glb(
            {
                "asset": {
                    "version": "2.0",
                    "generator": "Source",
                    "extras": {"custom": 7},
                }
            },
            b"data",
        )
    )
    output = tmp_path / "nested" / "upright.glb"
    monkeypatch.setattr(sys, "argv", ["blender", "--", str(glb), str(output)])

    runpy.run_path(upright.__file__, run_name="__main__")

    points = np.array([v.co for v in mesh.data.vertices])
    assert mesh.data is not original_data and mesh.data.updated and mesh.selected
    assert mesh.parent is None
    assert mesh.matrix_world == pytest.approx(np.eye(4))
    assert np.sort(points[:, 2]) == pytest.approx([0] * 4 + [3] * 4, abs=0.01)
    assert (points.max(0) + points.min(0))[:2] == pytest.approx([0, 0])
    doc, binary = parse_glb(output.read_bytes())
    assert doc == {
        "asset": {"version": "2.0", "generator": "Source", "extras": {"custom": 7}},
        "meshes": [{}],
    }
    assert binary == b"mesh"


def test_upright_refuses_rigged_asset_before_export(scene, monkeypatch, tmp_path):
    _, calls = scene([Mesh([[0, 0, 0]]), NS(type="ARMATURE")])
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "rigged.glb", str(tmp_path / "out.glb")]
    )
    with pytest.raises(ValueError, match="before rigging"):
        upright.main()
    assert [op for op, _ in calls] == ["factory", "import"]


@pytest.mark.parametrize(
    "mode, action",
    [("spawn", "spawn_markers"), ("load", "load_markers"), ("read", "read_markers")],
)
def test_joint_marker_cli_sends_the_correct_operation(
    monkeypatch, tmp_path, capsys, mode, action
):
    path = tmp_path / "points.json"
    path.write_text('{"neck": [1, 2, 3]}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "markers",
            mode,
            "--file",
            str(path),
            "--asset",
            "rig.glb",
            "--yes",
            "--host",
            "studio",
            "--port",
            "5678",
        ],
    )

    def send(code, host, port):
        assert (host, port) == ("studio", 5678)
        tree = ast.parse(code)
        (args,) = (
            json.loads(ast.literal_eval(n.args[0]))
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "loads"
        )
        assert args["action"] == action
        assert args["parameters"]["prefix"] == "JOINT_"
        if mode == "spawn":
            assert args["parameters"]["confirm"] and args["parameters"][
                "asset_path"
            ] == str(Path("rig.glb").resolve())
        if mode == "load":
            assert args["parameters"]["markers"] == {"neck": [1, 2, 3]}
        return '{"status": "sent"}'

    monkeypatch.setattr(markers, "send", send)
    assert markers.main() == 0
    assert json.loads(capsys.readouterr().out) == {"status": "sent"}


@pytest.mark.parametrize(
    "mode, flag", [("spawn", "--asset"), ("load", "--file"), ("save", "--file")]
)
def test_joint_marker_cli_requires_input_or_destination(monkeypatch, mode, flag):
    monkeypatch.setattr(sys, "argv", ["markers", mode])
    monkeypatch.setattr(markers, "send", lambda *args: '{"status": "success"}')
    with pytest.raises(SystemExit, match=f"{mode} requires {flag}"):
        markers.main()


def test_joint_marker_save_refuses_to_write_empty_scene(monkeypatch, tmp_path):
    target = tmp_path / "markers.json"
    target.write_text('{"existing": [0, 0, 0]}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["markers", "save", "--file", str(target)])
    monkeypatch.setattr(
        markers, "send", lambda *args: 'log\n{"result": {"result": "{}"}}'
    )
    with pytest.raises(SystemExit, match="nothing to save"):
        markers.main()
    assert json.loads(target.read_text()) == {"existing": [0, 0, 0]}


def test_marker_direct_entrypoint_describes_save_and_load(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["markers", "--help"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(markers.__file__, run_name="__main__")
    assert stopped.value.code == 0
    assert "{spawn,read,save,load}" in capsys.readouterr().out
