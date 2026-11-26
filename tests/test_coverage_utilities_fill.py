"""Exercise hole-filling CLI behavior on a simple local mesh."""

import runpy
import sys

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import fill_holes


@pytest.mark.parametrize("textured", [True, False])
def test_fill_holes_cli_closes_small_opening_and_preserves_texture(
    monkeypatch, tmp_path, capsys, textured,
):
    mesh = trimesh.creation.box()
    mesh.update_faces(mesh.face_normals[:, 2] < 1)
    if textured:
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=(mesh.vertices[:, :2] + 0.5),
            material=trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (255, 0, 0)),
            ),
        )
    source, output = tmp_path / "box.glb", tmp_path / "new" / "filled.glb"
    mesh.export(source)
    monkeypatch.setattr(sys, "argv", ["fill_holes.py", str(source), str(output), "--max-perimeter", "4.1"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(fill_holes.__file__, run_name="__main__")
    assert status.value.code == 0
    printed = capsys.readouterr().out
    assert "filled 1 holes (4 new faces)" in printed
    assert "boundary edges: 4 -> 0" in printed
    saved = trimesh.load(output, force="mesh", process=False)
    assert len(saved.faces) == len(mesh.faces) + 4
    if textured:
        original = trimesh.load(source, force="mesh", process=False)
        np.testing.assert_array_equal(saved.visual.uv[:len(original.vertices)], original.visual.uv)
        assert saved.visual.material.baseColorTexture.getpixel((0, 0))[:3] == (255, 0, 0)


def test_weld_removes_faces_collapsed_by_shared_positions():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    )
    vertices, faces = fill_holes.weld(mesh)
    assert len(vertices) == 3
    assert len(faces) == 1
    np.testing.assert_array_equal(vertices[faces[0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
