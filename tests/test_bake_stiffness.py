"""Baking stiffness must add COLOR_0 without throwing away the texture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bake_stiffness
import numpy as np
import trimesh
from PIL import Image


def _gltf(path: Path) -> dict:
    data = path.read_bytes()
    chunk_length = int.from_bytes(data[12:16], "little")
    return json.loads(data[20 : 20 + chunk_length])


def _textured_triangle(path: Path) -> trimesh.Trimesh:
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [(10, 20, 30, 255), (40, 50, 60, 255), (70, 80, 90, 255), (100, 110, 120, 255)]
    )
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        image=image,
    )
    mesh.export(path)
    return mesh


def _labels(path: Path, vertices) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2]], process=False)
    mesh.visual.vertex_colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 255, 0, 255]]
    mesh.export(path)


def test_bake_keeps_the_texture_and_adds_color(tmp_path, monkeypatch):
    textured = tmp_path / "textured.glb"
    labels = tmp_path / "labels.glb"
    output = tmp_path / "baked.glb"
    source_mesh = _textured_triangle(textured)
    _labels(labels, source_mesh.vertices)

    monkeypatch.setattr(
        sys, "argv", ["bake_stiffness.py", str(labels), str(output), "--mesh", str(textured)]
    )
    assert bake_stiffness.main() == 0

    source = trimesh.load(textured, force="mesh")
    baked = trimesh.load(output, force="mesh")
    assert np.allclose(baked.visual.uv, source.visual.uv)
    baked_image = baked.visual.material.baseColorTexture or baked.visual.material.image
    source_image = source.visual.material.baseColorTexture or source.visual.material.image
    assert np.array(baked_image).tobytes() == np.array(source_image).tobytes()

    primitive = _gltf(output)["meshes"][0]["primitives"][0]
    assert "COLOR_0" in primitive["attributes"]
    assert "TEXCOORD_0" in primitive["attributes"]
