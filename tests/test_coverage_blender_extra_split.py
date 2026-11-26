"""Exercise prop splitting with real vertex arrays and a small Blender boundary."""

import json
import sys
from itertools import product
from types import SimpleNamespace as NS

import blender_split_props as split
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from image_to_3dlab.glb_turn import build_glb, mark_front


class Matrix:
    def __init__(self, rows):
        self.values = np.asarray(rows)

    def to_4x4(self):
        result = np.eye(4)
        result[:3, :3] = self.values
        return result

    @staticmethod
    def Rotation(angle, size, axis):
        assert (size, axis) == (4, "Z")
        return Matrix(Rotation.from_euler(axis, angle).as_matrix()).to_4x4()

    @staticmethod
    def Translation(offset):
        result = np.eye(4)
        result[:3, 3] = offset
        return result


class Mesh:
    def __init__(self, points, faces):
        self.points = np.asarray(points, dtype=float)
        self.polygons = list(range(faces))
        self.updated = False
        self.vertices = self

    def __len__(self):
        return len(self.points)

    def foreach_get(self, attribute, out):
        assert attribute == "co"
        out[:] = self.points.ravel()

    def transform(self, matrix):
        self.points = self.points @ matrix[:3, :3].T + matrix[:3, 3]

    def update(self):
        self.updated = True


class Object:
    type = "MESH"

    def __init__(self, name, points, faces):
        self.name = name
        self.data = Mesh(points, faces)
        self.selected = False
        self.location = (0, 0, 0)

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, value):
        self._location = np.asarray(value)

    def select_set(self, selected):
        self.selected = selected


def box(half_extents, centre):
    return np.array(list(product(*[(-x, x) for x in half_extents]))) + centre


class PropScene:
    def __init__(self, source, parts, *, empty=False):
        self.objects = []
        self.active = NS(active=None)
        self.sheet = Object("sheet", [(0, 0, 0)], 1)
        self.parts = parts
        self.exports = []
        self.packed = False
        self.saved = None

        def load(*, filepath, merge_vertices):
            assert filepath == str(source)
            assert merge_vertices
            if not empty:
                self.objects.append(self.sheet)

        self.bpy = NS(
            data=NS(objects=self.objects),
            context=NS(view_layer=NS(objects=self.active)),
            ops=NS(
                wm=NS(read_factory_settings=self.reset, save_as_mainfile=self.save),
                import_scene=NS(gltf=load),
                object=NS(
                    select_all=self.deselect,
                    transform_apply=self.apply_transform,
                    mode_set=self.switch_mode,
                    join=self.join,
                ),
                mesh=NS(select_all=self.select_vertices, separate=self.separate),
                export_scene=NS(gltf=self.export),
                file=NS(pack_all=self.pack),
            ),
        )
        self.mode = "OBJECT"
        self.selected_vertices = False
        self.applied = False

    def reset(self, *, use_empty):
        assert use_empty
        self.objects.clear()

    def deselect(self, *, action):
        assert action == "DESELECT"
        for obj in self.objects:
            obj.select_set(False)

    def apply_transform(self, *, location, rotation, scale):
        assert all((location, rotation, scale))
        assert self.active.active is self.sheet and self.sheet.selected
        self.applied = True

    def switch_mode(self, *, mode):
        self.mode = mode

    def select_vertices(self, *, action):
        assert self.mode == "EDIT" and action == "SELECT"
        self.selected_vertices = True

    def separate(self, *, type):
        assert type == "LOOSE" and self.mode == "EDIT"
        assert self.selected_vertices and self.applied
        self.objects[:] = self.parts

    def join(self):
        active = self.active.active
        for other in list(self.objects):
            if other.selected and other is not active:
                active.data.points = np.vstack([active.data.points, other.data.points])
                active.data.polygons.extend(other.data.polygons)
                self.objects.remove(other)

    def export(self, *, filepath, export_format, use_selection):
        obj = self.active.active
        assert [part for part in self.objects if part.selected] == [obj]
        assert obj.data.updated
        assert export_format == "GLB" and use_selection
        assert obj.location.tolist() == [0, 0, 0]
        self.exports.append((filepath, obj.data.points.copy()))

    def pack(self):
        self.packed = True

    def save(self, *, filepath, compress):
        assert compress and self.packed
        self.saved = filepath


@pytest.mark.parametrize("front", [False, True])
def test_split_places_origins_and_exports_one_file_per_prop(
    front, tmp_path, monkeypatch, capsys
):
    source = tmp_path / "sheet.glb"
    glb = build_glb({"asset": {"version": "2.0"}}, b"")
    source.write_bytes(mark_front(glb) if front else glb)
    out = tmp_path / "props"
    centre = np.array([2, 1, 1])
    tilted = box((0.5, 0.2, 0.7), (0, 0, 0))
    tilted = Rotation.from_euler("zx", [42, 19], degrees=True).apply(tilted) + centre
    parts = [
        Object("rotated", tilted, 1000),
        Object("lid", box((0.08, 0.08, 0.08), centre), 4),
        Object("plain", box((0.5, 0.2, 0.7), (-2, 1, 1)), 1000),
        Object("crumb", [(9, 9, 9)], 1),
    ]
    scene = PropScene(source, parts)
    monkeypatch.setitem(sys.modules, "bpy", scene.bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Matrix=Matrix, Vector=np.asarray))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    argv = [
        "blender",
        "--",
        str(source),
        str(out),
        "--names",
        "first",
        "--turn",
        "first=90",
    ]
    second_name = "prop_02" if front else "second"
    if not front:
        argv.insert(argv.index("--turn"), second_name)
        argv += ["--no-straighten"]
    monkeypatch.setattr(sys, "argv", argv)

    assert split.main() == 0

    record = json.loads((out / "props.json").read_text())
    assert record["source"] == str(source)
    assert record["straighten"] == front
    first, second = record["props"]
    assert [first["name"], second["name"]] == ["first", second_name]
    assert (first["faces"], second["faces"]) == (
        (1000, 1004) if front else (1004, 1000)
    )
    assert first["extra_turn_degrees"] == 90
    assert [path for path, _ in scene.exports] == [
        str(out / "first.glb"),
        str(out / f"{second_name}.glb"),
    ]
    for entry, (_, vertices) in zip(record["props"], scene.exports, strict=True):
        lo = vertices.min(axis=0)
        hi = vertices.max(axis=0)
        assert lo[2] == pytest.approx(0)
        assert (lo + hi)[:2] == pytest.approx([0, 0])
        assert hi - lo == pytest.approx(entry["size"], abs=0.0001)
    if front:
        assert first["yaw_degrees"] == 0
        assert second["yaw_degrees"] == pytest.approx(-42, abs=0.5)
        assert second["yaw_tie"] is True
        assert second["size"] == pytest.approx([1, 0.4, 1.4], abs=0.02)
    else:
        assert "tilt_degrees" not in first
    assert len(scene.objects) == 2
    by_name = {obj.name: obj for obj in scene.objects}
    assert by_name["first"].location[0] == pytest.approx(
        first["size"][0] / 2, abs=0.0001
    )
    assert by_name[second_name].location[0] > by_name["first"].location[0]
    assert scene.saved == str(out / "props.blend")
    log = capsys.readouterr().out
    assert "1 crumbs dropped" in log
    assert ("names for 2 props" in log) is front


def test_split_stops_when_the_import_has_no_mesh(tmp_path, monkeypatch):
    source = tmp_path / "empty.glb"
    source.write_bytes(build_glb({"asset": {"version": "2.0"}}, b""))
    out = tmp_path / "props"
    scene = PropScene(source, [], empty=True)
    monkeypatch.setitem(sys.modules, "bpy", scene.bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Matrix=Matrix, Vector=np.asarray))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(sys, "argv", ["blender", "--", str(source), str(out)])

    with pytest.raises(SystemExit, match="no mesh found"):
        split.main()
    assert not scene.exports and scene.saved is None
    assert not (out / "props.json").exists()


def test_large_vertex_sample_is_reproducible_without_duplicates():
    points = np.arange((split.TILT_SAMPLE + 3) * 3).reshape(-1, 3)
    sample = split.sample(points)
    assert len(sample) == split.TILT_SAMPLE
    assert len(np.unique(sample[:, 0])) == split.TILT_SAMPLE
    assert np.array_equal(sample, split.sample(points))


def test_single_point_has_no_yaw_gain():
    assert split.best_yaw([[2, 5]])[1] == 0
    assert split.reading_order([], []) == []


@pytest.mark.parametrize("invalid_share", ["-0.1", "1", "nan"])
def test_split_rejects_invalid_minimum_face_share(invalid_share, capsys):
    with pytest.raises(SystemExit) as exc:
        split.parse_args(
            ["blender", "--", "in.glb", "out", "--min-share", invalid_share]
        )
    assert exc.value.code == 2
    assert "--min-share is a fraction" in capsys.readouterr().err
