"""Rig and prop commands with real small fixtures and external process boundaries."""

import json
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import test_finish_props
import trimesh
from test_bone_names import _humanoid, _rigged
from test_coverage_utilities_mesh_edges import invoke

from scripts import (
    arm_clearance,
    bake_stiffness,
    export_roblox,
    finish_props,
    name_bones,
)

sheet = test_finish_props.sheet


@pytest.mark.parametrize("all_moves", [False, True])
def test_stiffness_requires_an_anchor_and_writes_zero_on_a_rigid_mesh(tmp_path, monkeypatch, capsys, all_moves):
    mesh = trimesh.creation.box()
    mesh.visual.vertex_colors = [0, 255, 0, 255] if all_moves else [255, 0, 0, 255]
    src, dst = tmp_path / "labels.glb", tmp_path / "nested" / "wind.glb"
    mesh.export(src)
    if all_moves:
        monkeypatch.setattr(sys, "argv", [bake_stiffness.__file__, str(src), str(dst)])
        with pytest.raises(SystemExit, match="no rigid vertices"):
            bake_stiffness.main()
        assert not dst.exists()
    else:
        invoke(bake_stiffness, monkeypatch, src, dst)
        assert "0 moving" in capsys.readouterr().out
        baked = trimesh.load(dst, force="mesh", process=False)
        np.testing.assert_array_equal(baked.visual.vertex_colors[:, :2], np.zeros((8, 2)))


@pytest.mark.parametrize("same_output", [False, True])
def test_unity_names_command_supports_copy_and_in_place(tmp_path, monkeypatch, capsys, same_output):
    parents, heads, _ = _humanoid()
    source = _rigged(tmp_path, parents, heads)
    original = source.read_bytes()
    dest = source if same_output else tmp_path / "named.glb"
    invoke(name_bones, monkeypatch, source, dest)
    assert "wrote" in capsys.readouterr().out
    assert b'"Hips"' in dest.read_bytes()
    if not same_output:
        assert source.read_bytes() == original


def test_unity_names_command_reports_invalid_rig_filename(tmp_path, monkeypatch):
    source = tmp_path / "prop.glb"
    trimesh.creation.box().export(source)
    before = source.read_bytes()
    monkeypatch.setattr(sys, "argv", [name_bones.__file__, str(source), str(source)])
    with pytest.raises(SystemExit, match="prop.glb:"):
        name_bones.main()
    assert source.read_bytes() == before


def test_arm_proxy_defaults_handle_missing_torso_and_arm():
    radii = {"head": 3.0}
    assert arm_clearance.limit_head(radii, {"head"}, {"spine"}) == radii
    assert arm_clearance.build_proxy({"head": "Head"}, np.zeros((0, 3)), [], {}, height=1) is None


def test_arm_swing_finds_no_sideways_direction_along_a_parallel_capsule():
    def point(x):
        return np.array([x, 0., 0.])

    segments = {"torso": (point(0), point(1)), "upper": (point(0), point(1.2)),
                "fore": (point(1.2), point(1.5)), "hand": (point(1.5), point(1.6))}
    proxy = arm_clearance.Proxy(
        body={"torso": 0.3},
        arms={"Left": {"upper": "upper", "fore": "fore", "hand": "hand",
                       "radius": {"fore": 0.2, "hand": 0.2}}},
        side_shoulder={"Left": None}, margin=0.01,
    )
    np.testing.assert_array_equal(arm_clearance.arm_swing(segments, proxy, "Left"), np.eye(3))


def test_prop_resume_keeps_existing_packed_lods_if_detail_log_was_removed(sheet):
    output, runs, launch = sheet
    launch()
    for path in output.rglob("*.detail.log"):
        path.unlink()
    runs.labels.clear()
    launch("--resume")
    assert runs.labels == []
    record = json.loads((output / "finish_props.json").read_text())
    assert all("web_glb" in lod and "detail" not in lod for lod in record["props"][0]["lods"])


def test_props_cli_reports_missing_blender_before_creating_output(tmp_path, monkeypatch):
    source, out = tmp_path / "part.glb", tmp_path / "out"
    source.touch()
    monkeypatch.setattr(sys, "argv", [finish_props.__file__, str(source), str(out),
                                     "--blender", str(tmp_path / "absent")])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(SystemExit, match="Blender|blender"):
        runpy.run_path(finish_props.__file__, run_name="__main__")
    assert not out.exists()


def test_roblox_cli_reports_bad_glb_through_script_entry_point(tmp_path, monkeypatch, capsys):
    source = tmp_path / "bad.glb"
    source.write_bytes(b"not a glTF asset")
    invoke(export_roblox, monkeypatch, source, status=2)
    assert "Can't export bad.glb" in capsys.readouterr().err


def test_roblox_cli_surfaces_notes_when_export_cannot_measure_size(tmp_path, monkeypatch, capsys):
    source = tmp_path / "rig.glb"
    source.touch()
    result = SimpleNamespace(
        notes=["missing vertex positions"], size_studs=None,
        checks=[{"level": "fail", "text": "height unavailable"}], passed=False,
    )
    monkeypatch.setattr(export_roblox, "export", lambda *args, **kwargs: result)
    monkeypatch.setattr(export_roblox, "write_folder", lambda args, root: root)
    assert export_roblox.main([str(source)]) == 1
    out = capsys.readouterr().out
    assert "missing vertex positions" in out and "height unavailable" in out and "studs" not in out
