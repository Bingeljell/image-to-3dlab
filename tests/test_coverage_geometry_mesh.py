"""Check geometry tools on small meshes with known measurements."""

import sys

import numpy as np
import pytest
import trimesh

from scripts import classify_thickness, crop_mesh, mesh_health, remove_loose_parts


@pytest.fixture
def mesh_with_island():
    box = trimesh.creation.box()
    island = trimesh.Trimesh(
        vertices=[[4, 0, 0], [5, 0, 0], [4, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    return trimesh.util.concatenate([box, island])


@pytest.mark.parametrize("weld", [0, .001])
def test_health_cli_measures_then_closes_box_hole(tmp_path, monkeypatch, capsys, weld):
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(1, 12))
    source, output = tmp_path / "open.glb", tmp_path / "repair" / "closed.glb"
    mesh.export(source)
    assert mesh_health.measure(mesh)["boundary_edges"] == 3
    monkeypatch.setattr(
        sys, "argv", ["mesh_health", str(source), "--weld-sweep", "--repair", str(output), "--weld", str(weld)]
    )
    assert mesh_health.main() == 0
    repaired = trimesh.load(output, force="mesh")
    stats = mesh_health.measure(repaired)
    assert stats["watertight"] and stats["winding_consistent"]
    assert stats["components"] == 1
    assert stats["volume"] == pytest.approx(1)
    record = capsys.readouterr().out
    assert "=== weld sweep" in record
    assert "=== repaired" in record


@pytest.mark.parametrize("min_faces,expected", [(1, 13), (2, 12)])
def test_prune_cli_reports_outlying_component(mesh_with_island, tmp_path, monkeypatch, capsys, min_faces, expected):
    source, output = tmp_path / "mesh.glb", tmp_path / "output" / "pruned.glb"
    mesh_with_island.export(source)
    monkeypatch.setattr(
        sys, "argv", ["remove_loose_parts", str(source), str(output), "--min-faces", str(min_faces)]
    )
    assert remove_loose_parts.main() == 0
    pruned = trimesh.load(output, force="mesh")
    assert len(pruned.faces) == expected
    record = capsys.readouterr().out
    if min_faces == 2:
        assert "dropped 1 components, 1 faces" in record
        assert "at (+4.333, +0.333, +0.000)" in record
        assert pruned.is_watertight
    else:
        assert "dropped 0 components, 0 faces" in record


@pytest.mark.parametrize("samples", [1, 5])
def test_thickness_detects_open_sheet_and_solid_box(mesh_with_island, samples):
    thickness = classify_thickness.local_thickness(mesh_with_island, cone_samples=samples)
    assert thickness[-1] == 0
    np.testing.assert_allclose(thickness[:-1], 1, atol=.025)
    mesh_with_island.invert()
    np.testing.assert_allclose(
        classify_thickness.local_thickness(mesh_with_island, cone_samples=samples), thickness, atol=.025
    )


@pytest.mark.parametrize("with_island,expected", [(False, {"solid"}), (True, {"solid", "foliage"})])
def test_thickness_cli_exports_each_nonempty_class(mesh_with_island, tmp_path, monkeypatch, with_island, expected):
    source, output = tmp_path / "mesh.glb", tmp_path / "split" / "classes.glb"
    mesh = mesh_with_island if with_island else trimesh.creation.box()
    mesh.export(source)
    monkeypatch.setattr(sys, "argv", ["classify_thickness", str(source), "--output", str(output)])
    assert classify_thickness.main() == 0
    parts = trimesh.load(output).geometry
    assert parts.keys() == expected
    assert len(parts["solid"].faces) == 12
    if with_island:
        assert len(parts["foliage"].faces) == 1


@pytest.mark.parametrize(
    "suffix,options,expected_faces",
    [
        (".glb", ["--region", "head"], 2),
        (".ply", ["--axis", "x", "--from", ".45", "--to", ".55"], None),
    ],
)
def test_crop_cli_keeps_whole_faces_in_requested_region(
    tmp_path, monkeypatch, suffix, options, expected_faces
):
    source, output = tmp_path / f"mesh{suffix}", tmp_path / "slice" / "part.glb"
    mesh = trimesh.creation.box()
    mesh.export(source)
    monkeypatch.setattr(sys, "argv", ["crop_mesh", str(source), str(output), *options])
    if expected_faces is None:
        with pytest.raises(SystemExit, match="contains no faces"):
            crop_mesh.main()
        assert not output.exists()
    else:
        assert crop_mesh.main() == 0
        cropped = trimesh.load(output, force="mesh")
        assert len(cropped.faces) == expected_faces
        assert (cropped.triangles_center[:, 1] > 0).all()


def test_winding_cli_restores_outward_faces_in_place(tmp_path, monkeypatch):
    from scripts import fix_winding

    path = tmp_path / "inward.glb"
    mesh = trimesh.creation.box()
    mesh.invert()
    assert mesh.volume == pytest.approx(-1)
    mesh.export(path)
    monkeypatch.setattr(sys, "argv", ["fix_winding", str(path)])
    assert fix_winding.main() == 0
    corrected = trimesh.load(path, force="mesh", process=False)
    assert corrected.is_winding_consistent
    assert corrected.volume == pytest.approx(1)
    np.testing.assert_allclose(corrected.vertices, mesh.vertices)


@pytest.mark.parametrize("json_output", [False, True])
def test_glb_forensics_cli_reports_inside_out_mesh(tmp_path, monkeypatch, capsys, json_output):
    import json

    from scripts import glb_forensics

    model = tmp_path / "inward.glb"
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    mesh.invert()
    mesh.export(model)
    args = ["glb_forensics", str(model)]
    if json_output:
        args.append("--json")
    monkeypatch.setattr(sys, "argv", args)
    glb_forensics.main()
    record = capsys.readouterr().out
    if json_output:
        report, = json.loads(record)
        stats, = report["geometry"].values()
        assert stats["volume"] == pytest.approx(-6)
        assert stats["boundary_edges"] == 0
        assert stats["watertight"]
        assert stats["faces"] == 12
        assert report["materials"] == []
    else:
        assert "NEGATIVE" in record
        assert "-6.000000" in record


@pytest.mark.parametrize("export", [False, True])
def test_checkpoint_forensics_deduplicates_vertices_and_triangles(tmp_path, monkeypatch, capsys, export):
    import torch

    from scripts import mesh_checkpoint_forensics
    path, output = tmp_path / "mesh.pt", tmp_path / "unique" / "mesh.glb"
    torch.save(
        {
            "vertices": torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 0., 0.]]),
            "faces": torch.tensor([[0, 1, 2], [2, 3, 0]]),
        },
        path,
    )
    args = ["mesh_checkpoint_forensics", str(path)]
    if export:
        args.extend(["--export-glb", str(output)])
    monkeypatch.setattr(sys, "argv", args)
    assert mesh_checkpoint_forensics.main() == 0
    record = capsys.readouterr().out
    assert "stored faces: 2" in record
    assert "unique positions: 3" in record
    assert "duplicate geometric face groups: 1" in record
    assert "boundary/open edges: 3" in record
    if export:
        mesh = trimesh.load(output, force="mesh", process=False)
        assert len(mesh.vertices) == 3
        assert len(mesh.faces) == 1
        assert mesh.area == pytest.approx(.5)
