"""The Finish's UVs do not paint a texel twice: Smart UV Project at the Finish's wide angle
folds islands onto themselves; `blender_retopo_bake.unwrap_islands` unwraps them without that."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from image_to_3dlab.blender import find_blender

REPO = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO / "scripts" / "blender_retopo_bake.py"


def _load():
    spec = importlib.util.spec_from_file_location("retopo_uv", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = '''
import sys, math, json, bpy
sys.path.insert(0, {scripts!r})
import blender_retopo_bake as rb
bpy.ops.wm.read_factory_settings(use_empty=True)
for name, fixed in (("plain", False), ("fixed", True)):
    # a ramp turning twice round: one connected surface whose faces all face up, so the
    # wide angle keeps it one island stacked on itself
    verts, faces = [], []
    turns, steps, rings = 2, 64, 6
    for i in range(steps * turns + 1):
        a = 2 * math.pi * i / steps
        for j in range(rings + 1):
            r = 0.3 + 0.7 * j / rings
            verts.append((r * math.cos(a), r * math.sin(a), 0.02 * a))
    for i in range(steps * turns):
        for j in range(rings):
            k = i * (rings + 1) + j
            faces.append((k, k + 1, k + rings + 2, k + rings + 1))
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    angle = math.radians(89.0)
    if fixed:
        bpy.context.scene.tool_settings.mesh_select_mode = (True, False, False)   # as Blender starts
        print("LEFT", rb.unwrap_islands(bpy, obj.data, angle))
        print("MODE", tuple(bpy.context.scene.tool_settings.mesh_select_mode))
    else:
        bpy.ops.uv.smart_project(angle_limit=angle, island_margin=0.0002)
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data
    tris = [[list(uv[loop].uv) for loop in tri.loops] for tri in mesh.loop_triangles]
    json.dump(tris, open({out!r} + name + ".json", "w"))
    obj.hide_set(True)
'''


def overlap_percent(triangles, size: int = 512) -> float:
    """Share of the painted texels that more than one UV triangle covers."""
    hits = np.zeros((size, size), dtype=np.int32)
    ys, xs = np.mgrid[0:size, 0:size]
    px, py = (xs + 0.5) / size, (ys + 0.5) / size
    for (ax, ay), (bx, by), (cx, cy) in triangles:
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(area) < 1e-12:
            continue
        x0, x1 = (int(max(0, np.floor(min(ax, bx, cx) * size))), int(min(size, np.ceil(max(ax, bx, cx) * size))))
        y0, y1 = (int(max(0, np.floor(min(ay, by, cy) * size))), int(min(size, np.ceil(max(ay, by, cy) * size))))
        qx, qy = px[y0:y1, x0:x1], py[y0:y1, x0:x1]
        w0 = ((bx - qx) * (cy - qy) - (cx - qx) * (by - qy)) / area
        w1 = ((cx - qx) * (ay - qy) - (ax - qx) * (cy - qy)) / area
        inside = (w0 > 0) & (w1 > 0) & (w0 + w1 < 1)   # strictly: a shared edge is no overlap
        hits[y0:y1, x0:x1] += inside
    used = int((hits > 0).sum())
    return 100 * int((hits > 1).sum()) / max(used, 1)


def test_the_overlap_measure_sees_a_folded_island():
    square = [[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]
    assert overlap_percent(square, 64) == 0
    assert overlap_percent(square + square[:1], 64) == pytest.approx(50, abs=3)


def _full_version(blender: Path) -> tuple[int, ...] | None:
    """Blender's "5.2.0" from `--version`, which opens no scene, so it is safe on a Blender that crashes."""
    out = subprocess.run([str(blender), "--version"], capture_output=True, text=True, timeout=60).stdout
    found = re.search(r"Blender (\d+)\.(\d+)\.(\d+)", out or "")
    return tuple(int(n) for n in found.groups()) if found else None


@pytest.mark.skipif(find_blender() is None, reason="needs Blender")
def test_the_finish_unwraps_without_painting_a_texel_twice(tmp_path):
    version = _full_version(find_blender())
    if version is None or _load().unwrap_crashes(version):
        pytest.skip(f"Blender {version} crashes in the unwrap's overlap check (or its version is unknown)")
    script = tmp_path / "unwrap.py"
    script.write_text(SCRIPT.format(scripts=str(REPO / "scripts"), out=str(tmp_path / "uv_")), encoding="utf-8")
    done = subprocess.run([str(find_blender()), "--background", "--factory-startup", "--python", str(script)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "MODE (True, False, False)" in done.stdout   # the person's select mode is given back
    plain = overlap_percent(json.loads((tmp_path / "uv_plain.json").read_text(encoding="utf-8")))
    fixed = overlap_percent(json.loads((tmp_path / "uv_fixed.json").read_text(encoding="utf-8")))
    assert plain > 4.0, plain          # the wide-angle projection folds
    assert fixed < 1.0, fixed          # unwrapped, almost nothing is painted twice


def test_the_select_mode_and_sync_are_given_back_even_when_an_operator_fails(monkeypatch):
    rb = _load()
    settings = types.SimpleNamespace(use_uv_select_sync=False, mesh_select_mode=(True, False, False))

    def broken(**kwargs):
        raise RuntimeError("operator failed")

    bpy = types.SimpleNamespace(context=types.SimpleNamespace(scene=types.SimpleNamespace(tool_settings=settings)),
                                ops=types.SimpleNamespace(uv=types.SimpleNamespace(smart_project=broken)))
    monkeypatch.setitem(sys.modules, "bmesh", types.SimpleNamespace())
    with pytest.raises(RuntimeError):
        rb.unwrap_islands(bpy, object(), 1.5)
    assert settings.use_uv_select_sync is False and settings.mesh_select_mode == (True, False, False)


@pytest.mark.parametrize("version", [(4, 2, 23), (5, 0, 1), (5, 1, 2), (5, 2, 1), (5, 2, 2), (5, 3, 0)])
def test_the_unwrap_runs_on_the_blender_versions_it_passed_on(version):
    assert _load().unwrap_crashes(version) is False


def test_blender_5_2_0_falls_back_to_the_projection():
    """5.2.0 crashes inside uv.select_overlap (fixed in 5.2.1); bpy.app.version is a 3-tuple."""
    assert _load().unwrap_crashes((5, 2, 0)) is True
    assert _load().unwrap_crashes([5, 2, 0, "extra"]) is True


def _versioned_bpy(version, calls):
    def record(name):
        return lambda **kwargs: calls.append((name, kwargs))

    uv = types.SimpleNamespace(smart_project=record("smart_project"), select_overlap=record("select_overlap"))
    return types.SimpleNamespace(app=types.SimpleNamespace(version=version), ops=types.SimpleNamespace(uv=uv))


def test_the_finish_only_projects_on_blender_5_2_0(monkeypatch):
    rb, calls = _load(), []
    monkeypatch.setattr(rb, "unwrap_islands", lambda *args: pytest.fail("unwrap ran on 5.2.0"))
    assert rb.unwrap_finish(_versioned_bpy((5, 2, 0), calls), object(), 1.5) is None
    assert calls == [("smart_project", {"angle_limit": 1.5, "island_margin": 0.0002})]


def test_the_finish_unwraps_on_blender_5_2_1(monkeypatch):
    rb, calls = _load(), []
    monkeypatch.setattr(rb, "unwrap_islands", lambda bpy, mesh, angle: 7)
    assert rb.unwrap_finish(_versioned_bpy((5, 2, 1), calls), object(), 1.5) == 7
    assert calls == []


class _Face:
    def __init__(self, select=False, uv_select=None):
        self.select = select
        if uv_select is not None:
            self.uv_select = uv_select

    def select_set(self, value):
        self.select = value


def test_overlapping_faces_are_read_from_the_mesh_selection():
    """Blender 4.x: with UV sync select on, uv.select_overlap marks mesh faces."""
    rb = _load()
    faces = [_Face(True), _Face(False), _Face(True)]
    assert rb.overlap_selection(types.SimpleNamespace(faces=faces)) == (2, 0)


def test_a_face_picked_only_in_the_uv_selection_is_selected_on_the_mesh():
    """Blender 5.0 keeps a UV selection of its own while it is in sync; stale UV flags of a
    selection that is not in use are ignored."""
    rb = _load()
    faces = [_Face(True, False), _Face(False, True), _Face(False, False)]
    assert rb.overlap_selection(types.SimpleNamespace(faces=faces, uv_select_sync_valid=True)) == (2, 1)
    assert [face.select for face in faces] == [True, True, False]
    stale = [_Face(False, True)]
    assert rb.overlap_selection(types.SimpleNamespace(faces=stale, uv_select_sync_valid=False)) == (0, 0)
    assert stale[0].select is False
