"""Test the Blender preview worker with an in-memory scene at the Blender boundary."""

import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import blender_preview_worker as worker
import pytest


class Vector(tuple):
    def __neg__(self):
        return Vector(-value for value in self)

    def __sub__(self, other):
        return Vector(a - b for a, b in zip(self, other, strict=True))

    def to_track_quat(self, axis, up):
        assert (axis, up) == ("-Z", "Y")
        return self

    def to_euler(self):
        return tuple(self)


def test_preview_camera_fits_shorter_side_and_orbits_the_subject():
    centre, radius = worker.bounding_sphere([(-4, -1, 2), (0, 3, 2)])
    assert centre == (-2, 1, 2)
    assert radius == pytest.approx(math.sqrt(8))
    assert worker.bounding_sphere([(1, 2, 3)]) == ((1, 2, 3), 1)
    with pytest.raises(ValueError, match="no mesh vertices"):
        worker.bounding_sphere(iter(()))
    square = worker.camera_distance(radius, math.pi / 2, 100, 100)
    portrait = worker.camera_distance(radius, math.pi / 2, 100, 200)
    assert square == pytest.approx(4.2)
    assert portrait > square
    front = worker.camera_location(centre, square, 0)
    side = worker.camera_location(centre, square, 90)
    assert front[0] == centre[0]
    assert front[1] < centre[1] and side[0] > centre[0]
    assert side[1] == pytest.approx(centre[1])
    assert math.dist(centre, front) == pytest.approx(square)
    assert math.dist(centre, side) == pytest.approx(square)
    assert front[2] == side[2] > centre[2]


def test_preview_shot_encoding_preserves_zero_and_unsampled_frames():
    shots = worker.parse_shots("Still=180=;Start=35=0;Half=35=0.5")
    assert shots == [("Still", 180, None), ("Start", 35, 0), ("Half", 35, 0.5)]
    assert worker.shot_frames(shots, 10, 30) == {"Still": None, "Start": 10, "Half": 20}


class Render:
    def __init__(self, engines):
        self.fps = 20
        self.fps_base = 2
        self.image_settings = SimpleNamespace()
        self.bl_rna.properties["engine"].enum_items = [SimpleNamespace(identifier=name) for name in engines]

    class Engine:
        enum_items = ()

    bl_rna = SimpleNamespace(properties={"engine": Engine()})


@pytest.fixture
def blender_scene(monkeypatch):
    renders, frame_sets, collection = [], [], []
    scene = SimpleNamespace(frame_start=0, frame_end=12, frame=0, collection=SimpleNamespace(
        objects=SimpleNamespace(link=collection.append),
    ))
    scene.render = Render(["BLENDER_EEVEE_NEXT", "CYCLES"])

    def frame_set(frame, subframe):
        scene.frame = frame + subframe
        frame_sets.append((frame, subframe))

    scene.frame_set = frame_set
    depsgraph = object()
    mesh_clear = Mock()

    def evaluated_get(graph):
        assert graph is depsgraph

        class WorldMatrix:
            def __matmul__(self, point):
                return (point[0] + scene.frame, point[1], point[2])

        mesh = SimpleNamespace(vertices=[SimpleNamespace(co=position) for position in [(-1, -1, -1), (1, 1, 1)]])
        return SimpleNamespace(matrix_world=WorldMatrix(), to_mesh=lambda: mesh, to_mesh_clear=mesh_clear)

    mesh = SimpleNamespace(type="MESH", hide_render=False, visible_get=lambda: True,
                           evaluated_get=evaluated_get, data=SimpleNamespace(materials=["original paint"]))
    hidden = SimpleNamespace(type="MESH", hide_render=False, visible_get=lambda: False,
                             evaluated_get=Mock(side_effect=AssertionError("hidden mesh evaluated")))
    already_hidden = SimpleNamespace(type="MESH", hide_render=True, visible_get=lambda: True)
    scene.objects = [mesh, hidden, already_hidden, SimpleNamespace(type="ARMATURE")]

    def render(*, write_still):
        assert write_still
        path = Path(scene.render.filepath)
        path.write_bytes(b"fixture PNG")
        lights = next(obj for obj in collection if obj.name == "Lights")
        renders.append((path.name, scene.frame, scene.camera.location, lights.rotation_euler))

    def make_world(name):
        return SimpleNamespace(node_tree=SimpleNamespace(nodes={"Background": SimpleNamespace(
            inputs=[SimpleNamespace(), SimpleNamespace()],
        )}))

    def make_material(name):
        return SimpleNamespace(name=name, node_tree=SimpleNamespace(nodes={"Principled BSDF": SimpleNamespace(
            inputs={"Base Color": SimpleNamespace()},
        )}))

    reset, read_gltf = Mock(), Mock()
    bpy = SimpleNamespace(
        context=SimpleNamespace(scene=scene, evaluated_depsgraph_get=lambda: depsgraph),
        ops=SimpleNamespace(wm=SimpleNamespace(read_factory_settings=reset),
                            import_scene=SimpleNamespace(gltf=read_gltf),
                            render=SimpleNamespace(render=render)),
        data=SimpleNamespace(actions=[],
                             worlds=SimpleNamespace(new=make_world),
                             materials=SimpleNamespace(new=make_material),
                             cameras=SimpleNamespace(new=lambda name: SimpleNamespace(angle=math.pi / 2)),
                             lights=SimpleNamespace(new=lambda name, type: SimpleNamespace(type=type)),
                             objects=SimpleNamespace(new=lambda name, data: SimpleNamespace(name=name, data=data))),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Vector=Vector))
    return SimpleNamespace(bpy=bpy, scene=scene, mesh=mesh, hidden=hidden, renders=renders, frames=frame_sets,
                           collection=collection, reset=reset, read_gltf=read_gltf, mesh_clear=mesh_clear)


@pytest.mark.parametrize("animated", [False, True])
def test_preview_worker_renders_angles_with_common_framing_and_turning_lights(
        tmp_path, monkeypatch, blender_scene, animated):
    work = blender_scene
    source = tmp_path / "model.glb"
    if animated:
        work.bpy.data.actions = [SimpleNamespace(frame_range=(0, 8)), SimpleNamespace(frame_range=(3, 10))]
        shots = "Early=0=0.25;Late=90=0.75"
    else:
        shots = "Front=0=;Back=180="
        work.scene.render.bl_rna.properties["engine"].enum_items = [SimpleNamespace(identifier="CYCLES")]
    args = [str(source), str(tmp_path), "0", "512", "640", "1" if animated else "0", shots]
    monkeypatch.setattr(sys, "argv", ["blender", "--background", "--", *args])
    runpy.run_path(worker.__file__, run_name="__main__")

    work.reset.assert_called_once_with(use_empty=True)
    work.read_gltf.assert_called_once_with(filepath=str(source))
    assert work.hidden.hide_render is True
    assert work.scene.render.image_settings.file_format == "PNG"
    assert (work.scene.render.resolution_x, work.scene.render.resolution_y) == (512, 640)
    assert work.scene.render.resolution_percentage == 100
    assert work.scene.render.engine == ("BLENDER_EEVEE_NEXT" if animated else "CYCLES")
    assert work.scene.world.use_nodes
    assert work.scene.world.node_tree.nodes["Background"].inputs[1].default_value == 0.6

    times = json.loads((tmp_path / "times.json").read_text(encoding="utf-8"))
    if animated:
        [material] = work.mesh.data.materials
        assert material.name == "clay"
        assert material.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value == (
            0.62, 0.62, 0.60, 1.0,
        )
        assert work.frames == [(2, 0.5), (7, 0.5), (2, 0.5), (7, 0.5)]
        assert times == {"Early": 0.25, "Late": 0.75}
        assert [row[0] for row in work.renders] == ["Early.png", "Late.png"]
        assert work.mesh_clear.call_count == 2
        centre = (5, 0, 0)
        assert work.renders[0][2][0] == centre[0]
        assert work.renders[1][2][0] > centre[0]
    else:
        assert work.mesh.data.materials == ["original paint"]
        assert work.frames == []
        assert times == {}
        assert [row[0] for row in work.renders] == ["Front.png", "Back.png"]
        work.mesh_clear.assert_called_once()
        centre = (0, 0, 0)
        assert work.renders[0][2][1] < 0 < work.renders[1][2][1]

    assert all((tmp_path / row[0]).is_file() for row in work.renders)
    first, second = work.renders
    assert math.dist(centre, first[2]) == pytest.approx(math.dist(centre, second[2]))
    assert first[3][2] == pytest.approx(0)
    assert second[3][2] == pytest.approx(math.pi / 2 if animated else math.pi)
    lights = [obj for obj in work.collection if obj.name in {"key", "fill", "rim"}]
    assert [obj.name for obj in lights] == ["key", "fill", "rim"]
    assert lights[0].data.energy > lights[-1].data.energy > lights[1].data.energy


def test_preview_worker_exits_before_rendering_when_no_mesh_is_renderable(tmp_path, blender_scene):
    blender_scene.mesh.hide_render = True
    with pytest.raises(SystemExit, match="no mesh in"):
        worker.main(["missing.glb", str(tmp_path), "0", "20", "20", "0", "Front=0="])
    assert blender_scene.renders == []
    assert not (tmp_path / "times.json").exists()


def test_preview_worker_help_runs_without_blender(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "bpy", None)
    monkeypatch.setattr(sys, "argv", [worker.__file__, "--help"])
    with pytest.raises(SystemExit) as caught:
        worker.main()
    assert caught.value.code == 0
    assert "source" in capsys.readouterr().out
