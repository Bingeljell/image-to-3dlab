"""A normal bake bakes every mesh in the low-poly file, and a bake that wrote nothing fails.

Blender is not needed: the generated program runs against a tiny fake `bpy`, and `main`
runs with the Blender socket replaced.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _load():
    spec = importlib.util.spec_from_file_location(
        "bake_normals_under_test", SCRIPTS / "blender_bake_normals.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Stop(Exception):
    """Raised by the fake PLY import: everything we care about happens before it."""


class _Objects(list):
    def remove(self, obj, do_unlink=True):
        if obj in self:
            super().remove(obj)


def _fake_blender(piece_count: int):
    joined = []
    scene = types.SimpleNamespace(
        objects=_Objects(),
        world=object(),
        collection=types.SimpleNamespace(children=[]),
        name="IMG3D normal bake",
    )

    class Piece:
        def __init__(self, index):
            self.type = "MESH"
            self.name = f"piece{index}"

        def select_set(self, _state):
            pass

    def import_gltf(filepath):
        for index in range(piece_count):
            scene.objects.append(Piece(index))

    def join():
        pieces = [o for o in scene.objects if isinstance(o, Piece)]
        joined.append(len(pieces))
        for extra in pieces[1:]:
            scene.objects.remove(extra)

    def import_ply(filepath):
        raise _Stop

    noop = lambda *a, **k: None
    bpy = types.SimpleNamespace(
        context=types.SimpleNamespace(
            window=types.SimpleNamespace(scene=scene),
            view_layer=types.SimpleNamespace(
                objects=types.SimpleNamespace(active=None)
            ),
        ),
        data=types.SimpleNamespace(
            scenes=types.SimpleNamespace(
                get=lambda name: scene, new=lambda name: scene
            ),
            objects=scene.objects,
            collections=types.SimpleNamespace(remove=noop),
            worlds=types.SimpleNamespace(new=lambda name: object()),
        ),
        ops=types.SimpleNamespace(
            import_scene=types.SimpleNamespace(gltf=import_gltf),
            wm=types.SimpleNamespace(ply_import=import_ply),
            object=types.SimpleNamespace(select_all=noop, join=join),
        ),
    )
    return bpy, joined


def _run_generated(monkeypatch, piece_count: int) -> list[int]:
    bake = _load()
    code = bake.blender_code(
        Path("/tmp/low.glb"),
        Path("/tmp/high.ply"),
        Path("/tmp/n.png"),
        64,
        0.02,
        True,
        "CPU",
        0.0,
        0,
        1,
    )
    bpy, joined = _fake_blender(piece_count)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", types.SimpleNamespace(Vector=tuple))
    with pytest.raises(_Stop):
        exec(compile(code, "<normal-bake>", "exec"), {"__name__": "blender"})
    return joined


def test_a_file_with_several_mesh_pieces_is_joined_before_baking(monkeypatch):
    joined = _run_generated(monkeypatch, piece_count=3)
    assert joined == [3], (
        f"3 mesh pieces were imported but join calls were {joined}: only the first piece "
        "would be baked"
    )


def test_a_single_mesh_needs_no_join(monkeypatch):
    assert _run_generated(monkeypatch, piece_count=1) == []


def _run_main(
    monkeypatch, tmp_path, reply: dict, *extra: str, existing_png=False
) -> int:
    bake = _load()
    out = tmp_path / "normal.png"
    if existing_png:
        out.write_bytes(b"stale")
    monkeypatch.setattr(bake, "send", lambda *a, **k: json.dumps(reply))
    monkeypatch.setattr(sys, "argv", ["bake", "low.glb", "high.ply", str(out), *extra])
    return bake.main()


NOTHING_WRITTEN = {"error": "Bake failed", "written": None}


def test_exits_nonzero_when_blender_reports_nothing_written(
    monkeypatch, tmp_path, capsys
):
    code = _run_main(monkeypatch, tmp_path, NOTHING_WRITTEN)
    assert code != 0, "no picture was written but the bake exited 0"


def test_exits_nonzero_when_nothing_written_even_with_keep_sign(monkeypatch, tmp_path):
    code = _run_main(monkeypatch, tmp_path, NOTHING_WRITTEN, "--keep-sign")
    assert code != 0, "--keep-sign with no picture written exited 0"


def test_a_stale_png_does_not_hide_a_failed_bake(monkeypatch, tmp_path):
    code = _run_main(
        monkeypatch, tmp_path, NOTHING_WRITTEN, "--keep-sign", existing_png=True
    )
    assert code != 0, "an old PNG on disk made a failed bake look fine"
