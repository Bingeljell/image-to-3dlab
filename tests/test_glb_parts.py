"""Models in parts (a Hero's body and head) through tools written for one part.

A Hero model failed Fix a spot outright ("expected one primitive, found 2"), and the error
dropped the connection. Parts are cut into files of their own, fixed, and joined again;
these pin that the round trip loses nothing and each part keeps its own material.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import glb_parts as gp
from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix as sf


def _textured(mesh, colour):
    image = Image.new("RGB", (8, 8), colour)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.random.default_rng(0).random((len(mesh.vertices), 2)),
                                                material=trimesh.visual.material.PBRMaterial(baseColorTexture=image))
    return mesh


@pytest.fixture
def two_parts(tmp_path):
    body = _textured(trimesh.creation.box((1, 2, 1)), (200, 30, 30))
    head = _textured(trimesh.creation.icosphere(2, 0.4), (30, 30, 200))
    head.apply_translation((0, 1.4, 0))
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="Body")
    scene.add_geometry(head, node_name="Head")
    path = tmp_path / "hero.glb"
    path.write_bytes(scene.export(file_type="glb"))
    # one mesh with two primitives, the way Blender's join writes a Hero model
    doc, binary = pp._split_glb(path.read_bytes())
    doc["meshes"] = [{"primitives": [p for m in doc["meshes"] for p in m["primitives"]]}]
    doc["nodes"] = [{"mesh": 0}]
    doc["scenes"] = [{"nodes": [0]}]
    path.write_bytes(pp._join_glb(doc, binary))
    return path


def test_split_gives_each_part_only_what_it_uses(two_parts, tmp_path):
    parts = gp.split(two_parts, tmp_path / "parts")
    assert gp.count(two_parts) == 2 and len(parts) == 2
    colours = []
    for part in parts:
        image = pp.read_glb(part)[3]      # a one-part reader accepts each
        colours.append(tuple(int(c) for c in image[0, 0, :3]))
        doc, _ = pp._split_glb(part.read_bytes())
        assert len(doc["images"]) == 1 and len(doc["materials"]) == 1
    assert colours == [(200, 30, 30), (30, 30, 200)]


def test_join_puts_the_parts_back_unchanged(two_parts, tmp_path):
    parts = gp.split(two_parts, tmp_path / "parts")
    gp.join(parts, tmp_path / "joined.glb")
    assert gp.count(tmp_path / "joined.glb") == 2
    for before, after in zip(parts, gp.split(tmp_path / "joined.glb", tmp_path / "again")):
        a, b = pp.read_glb(before), pp.read_glb(after)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_model_height_spans_every_part(two_parts):
    assert sf.model_height(two_parts) == pytest.approx(2.8, abs=1e-3)   # box from -1 to the sphere top at 1.8


def test_a_spot_on_one_part_leaves_the_other_alone(two_parts, tmp_path):
    out = tmp_path / "fixed.glb"
    reports = sf.apply_spots(two_parts, out, [{"point": np.array([0.0, 1.8, 0.0]), "radius": 0.15, "mode": "fill"}], None)
    assert reports[0]["cut"] > 0 and gp.count(out) == 2
    body_before = pp.read_glb(gp.split(two_parts, tmp_path / "b")[0])
    body_after = pp.read_glb(gp.split(out, tmp_path / "a")[0])
    assert all(np.array_equal(x, y) for x, y in zip(body_before, body_after))


def test_a_failing_patch_answers_instead_of_dropping_the_connection():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("fixspot_api", Path(__file__).resolve().parents[1] / "viewer" / "fixspot_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def broken(output, payload):
        raise ValueError("expected one primitive, found 2")
    status, body = module.answer(broken, Path("/tmp"), {})
    assert status == 500 and "expected one primitive" in body["error"]
