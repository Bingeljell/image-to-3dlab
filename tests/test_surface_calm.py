"""Calming a finished mesh's surface where the picture paints (Finish's High detail)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import surface_calm as sc


def _grid(n=21, bump=0.0):
    """A flat n x n grid in the XY plane facing +Z, with an optional one-vertex bump."""
    xs, ys = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], 1)
    centre = (n // 2) * n + n // 2
    pos[centre, 2] = bump
    faces = []
    for r in range(n - 1):
        for c in range(n - 1):
            a, b, d = r * n + c, r * n + c + 1, (r + 1) * n + c
            faces += [(a, b, d), (b, d + 1, d)]
    return pos, np.array(faces), centre


def test_zero_rounds_changes_nothing():
    pos, faces, _ = _grid(bump=0.3)
    assert np.array_equal(sc.taubin(pos, faces, np.ones(len(pos)), rounds=0), pos)


def test_a_small_bump_is_calmed():
    pos, faces, centre = _grid(bump=0.3)
    out = sc.taubin(pos, faces, np.ones(len(pos)), rounds=10)
    assert out[centre, 2] < 0.1


def test_the_surface_keeps_its_size():
    pos, faces, _ = _grid(bump=0.3)
    out = sc.taubin(pos, faces, np.ones(len(pos)), rounds=10)
    assert np.ptp(out[:, 0]) > 0.97 * np.ptp(pos[:, 0])


def test_zero_weight_leaves_a_vertex_alone():
    pos, faces, centre = _grid(bump=0.3)
    weights = np.ones(len(pos))
    weights[centre] = 0.0
    out = sc.taubin(pos, faces, weights, rounds=10)
    assert out[centre, 2] == 0.3


def test_vertices_split_at_a_uv_seam_move_together():
    pos, faces, centre = _grid(bump=0.3)
    # duplicate the bump vertex, as a UV seam would, and give one face the copy
    pos = np.vstack([pos, pos[centre]])
    copy = len(pos) - 1
    faces = faces.copy()
    first = np.flatnonzero((faces == centre).any(1))[0]
    faces[first][faces[first] == centre] = copy
    out = sc.taubin(pos, faces, np.ones(len(pos)), rounds=10)
    assert np.allclose(out[centre], out[copy])


def test_only_surfaces_facing_the_camera_are_weighted():
    normals = np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0]])
    pos = np.zeros((3, 3))
    w = sc.facing_weights(pos, normals, camera=np.array([0, 0, 5.0]))
    assert w[0] == 1.0 and w[1] == 0.0 and w[2] == 0.0


# --- the GLB round trip -----------------------------------------------------------------


def _write_glb(path: Path, pos, faces, uvs):
    import io

    from PIL import Image

    png = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 100, 50)).save(png, "PNG")
    chunks = [pos.astype("<f4").tobytes(), uvs.astype("<f4").tobytes(),
              faces.astype("<u4").ravel().tobytes(), png.getvalue()]
    views, binary = [], b""
    for chunk in chunks:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(chunk)})
        binary += chunk
    doc = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
             "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
            {"bufferView": 1, "componentType": 5126, "count": len(pos), "type": "VEC2"},
            {"bufferView": 2, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
        ],
        "images": [{"bufferView": 3, "mimeType": "image/png"}], "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2,
                                    "material": 0}]}],
        "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}],
    }
    path.write_bytes(pp._join_glb(doc, binary))


def test_calming_a_glb_moves_positions_only(tmp_path):
    pos, faces, centre = _grid(bump=0.3)
    # glTF is +Y up and faces +Z; the grid already faces +Z
    uvs = (pos[:, :2] + 1) / 2
    _write_glb(tmp_path / "in.glb", pos, faces, uvs)
    weights = np.ones(len(pos))
    record = sc.calm_glb(tmp_path / "in.glb", tmp_path / "out.glb", weights, rounds=10)
    new_pos, new_uvs, new_faces, _tex = pp.read_glb(tmp_path / "out.glb")
    assert new_pos[centre, 2] < 0.1
    assert np.allclose(new_uvs, uvs) and np.array_equal(new_faces.reshape(-1, 3), faces)
    assert record["rounds"] == 10 and record["vertices"] == len(pos)
    doc, _ = pp._split_glb((tmp_path / "out.glb").read_bytes())
    assert np.allclose(doc["accessors"][0]["max"], new_pos.max(0))
