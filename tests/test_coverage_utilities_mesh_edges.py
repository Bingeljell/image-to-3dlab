"""Mesh report and export commands on tiny real or deliberately malformed inputs."""

import json
import runpy
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import (
    compress_glb_textures,
    crop_mesh,
    glb_forensics,
    mesh_health,
    ribbon_metric,
    rig_check,
)


def invoke(module, monkeypatch, *args, status=0):
    monkeypatch.setattr(sys, "argv", [module.__file__, *(str(arg) for arg in args)])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(module.__file__, run_name="__main__")
    assert exited.value.code == status


@pytest.mark.parametrize("points, detail", [
    (np.zeros((4, 3)), "too few points"),
    (np.zeros((50, 3)), "flat"),
    (np.repeat([[0, 0, 0], [0, 1, 0]], 25, axis=0), "nothing near the floor"),
])
def test_foot_detector_reports_why_it_cannot_find_two_feet(points, detail):
    assert rig_check.stands_on_two_legs(points) == {"two_legs": False, "feet": 0, "detail": detail}


def test_rig_check_composes_column_major_matrix_and_reports_disconnected_roots(tmp_path, monkeypatch, capsys):
    path = tmp_path / "rig.glb"
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    doc = {"nodes": [{"name": "parent", "matrix": transform.T.ravel().tolist()},
                     {"name": "independent", "translation": [4, 5, 6]}], "skins": [{"joints": [0, 1]}]}
    path.write_bytes(compress_glb_textures.build_glb(doc, b""))
    _, heads = rig_check.skeleton(path)
    np.testing.assert_array_equal(heads["parent"], [1, 2, 3])
    np.testing.assert_array_equal(rig_check.mesh_points(path), np.empty((0, 3)))
    invoke(rig_check, monkeypatch, path, status=1)
    assert json.loads(capsys.readouterr().out)["detail"] == "expected one root bone, found 2"


def test_rig_check_ignores_unrelated_chunks_and_nodes_without_geometry(tmp_path):
    path = tmp_path / "group.glb"
    doc = {"asset": {"version": "2.0"}, "nodes": [{"children": [1]}, {"name": "node"}]}
    data = compress_glb_textures.build_glb(doc, b"")
    payload = data[12:] + struct.pack("<II4s", 4, 0x11111111, b"info")
    path.write_bytes(b"glTF" + struct.pack("<II", 2, len(payload) + 12) + payload)
    assert rig_check.mesh_points(path).shape == (0, 3)


def test_glb_forensics_rejects_first_binary_chunk_and_defines_empty_volume(tmp_path):
    path = tmp_path / "invalid.glb"
    path.write_bytes(b"glTF" + struct.pack("<IIII4s", 2, 24, 4, 0x004E4942, b"data"))
    with pytest.raises(ValueError, match="first chunk is not JSON"):
        glb_forensics.read_gltf_json(path)
    assert glb_forensics.signed_volume(trimesh.Trimesh()) == 0


def test_forensics_reports_texture_slots_extensions_and_degenerate_face(tmp_path, monkeypatch, capsys):
    mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                           faces=[[0, 1, 2]], process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1]],
        material=trimesh.visual.material.PBRMaterial(
            normalTexture=Image.new("RGB", (2, 3), (128, 128, 255)),
        ))
    raw = mesh.export(file_type="glb")
    doc, data = compress_glb_textures.parse_glb(raw)
    doc["extensionsUsed"] = ["VENDOR_example"]
    path = tmp_path / "shape.glb"
    path.write_bytes(compress_glb_textures.build_glb(doc, data))
    monkeypatch.setattr(sys, "argv", [glb_forensics.__file__, str(path)])
    runpy.run_path(glb_forensics.__file__, run_name="__main__")
    output = capsys.readouterr().out
    assert "VENDOR_example" in output and "textures=['normal']" in output
    assert "2x3" in output and "face area max/median : n/a" in output


def test_ribbon_reports_only_largest_scene_geometry_and_exits_on_open_mesh(tmp_path, monkeypatch, capsys):
    closed = trimesh.creation.box()
    triangle = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    scene = tmp_path / "joined.glb"
    trimesh.Scene([closed, triangle]).export(scene)
    opened = tmp_path / "single.ply"
    triangle.export(opened)
    assert ribbon_metric.measure(scene)["faces"] == 12
    assert ribbon_metric.measure(opened)["pct_boundary_edges"] == 100
    invoke(ribbon_metric, monkeypatch, scene)
    assert "PASS - safe" in capsys.readouterr().out
    invoke(ribbon_metric, monkeypatch, scene, opened, status=1)
    assert "FAIL - a mesh of ribbons" in capsys.readouterr().out


def test_crop_command_and_optional_bounds_validation(tmp_path, monkeypatch):
    source, output = tmp_path / "box.ply", tmp_path / "half.ply"
    trimesh.creation.box().export(source)
    invoke(crop_mesh, monkeypatch, source, output, "--axis", "x", "--from", 0.51)
    assert 0 < len(trimesh.load(output).faces) < 12
    with pytest.raises(ValueError, match="unknown region"):
        crop_mesh.resolve_bounds(SimpleNamespace(region="wrong"))
    with pytest.raises(ValueError, match="give --region or --axis"):
        crop_mesh.resolve_bounds(SimpleNamespace(region=None, axis=None))


def test_mesh_health_measure_only_leaves_source_untouched(tmp_path, monkeypatch, capsys):
    source = tmp_path / "box.ply"
    trimesh.creation.box().export(source)
    raw = source.read_bytes()
    invoke(mesh_health, monkeypatch, source)
    output = capsys.readouterr().out
    assert "watertight" in output and "True" in output
    assert "repaired" not in output and "weld sweep" not in output
    assert source.read_bytes() == raw
