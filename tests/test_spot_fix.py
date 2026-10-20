"""The studio's "Fix a spot" brush: cut around a point, cap the opening, keep everything else."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix as sf


def _sphere_with_pocket(tmp_path, n=24):
    """A UV sphere (facing +Z at the front) with a pocket pushed into its front."""
    lat, lon = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n, endpoint=False), indexing="ij")
    pos = np.stack([np.sin(lat) * np.sin(lon), np.cos(lat), np.sin(lat) * np.cos(lon)], -1).reshape(-1, 3)
    front = np.argmax(pos[:, 2])
    pos[front] *= 0.4                                  # the pocket: one vertex sunk deep
    faces = []
    for i in range(n - 1):
        for j in range(n):
            a, b, c, d = i * n + j, i * n + (j + 1) % n, (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            faces += [(a, c, b), (b, c, d)]
    faces = np.array(faces)
    uvs = np.stack([lon.ravel() / (2 * np.pi), lat.ravel() / np.pi], 1)
    png = io.BytesIO()
    Image.new("RGB", (4, 4), (180, 120, 90)).save(png, "PNG")
    chunks = [pos.astype("<f4").tobytes(), uvs.astype("<f4").tobytes(), faces.astype("<u4").ravel().tobytes(),
              png.getvalue()]
    views, binary = [], b""
    for chunk in chunks:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(chunk)})
        binary += chunk
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
           "accessors": [
               {"bufferView": 0, "componentType": 5126, "count": len(pos), "type": "VEC3",
                "min": pos.min(0).tolist(), "max": pos.max(0).tolist()},
               {"bufferView": 1, "componentType": 5126, "count": len(pos), "type": "VEC2"},
               {"bufferView": 2, "componentType": 5125, "count": faces.size, "type": "SCALAR"}],
           "images": [{"bufferView": 3, "mimeType": "image/png"}], "textures": [{"source": 0}],
           "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
           "nodes": [{"mesh": 0}], "scenes": [{"nodes": [0]}]}
    path = tmp_path / "in.glb"
    path.write_bytes(pp._join_glb(doc, binary))
    return path, pos, front


def _open_edges(faces, positions):
    from image_to_3dlab import surface_calm as sc

    key = sc.weld(positions)
    welded = np.sort(np.concatenate([key[faces][:, [0, 1]], key[faces][:, [1, 2]], key[faces][:, [2, 0]]]), 1)
    welded = welded[welded[:, 0] != welded[:, 1]]
    return int((np.unique(welded, axis=0, return_counts=True)[1] == 1).sum())


def test_the_pocket_is_lifted_and_the_mesh_stays_closed(tmp_path):
    source, _pos, _front = _sphere_with_pocket(tmp_path)
    report = sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    assert report["cut"] > 0 and report["capped"] == 1
    out_pos, out_uv, out_faces, texture = pp.read_glb(tmp_path / "out.glb")
    out_faces = out_faces.reshape(-1, 3)
    assert _open_edges(out_faces, out_pos) == 0
    # the new patch sits near the surface, not down in the pocket
    assert out_pos[:, 2].max() > 0.8
    assert len(out_uv) == len(out_pos) and texture.shape[:2] == (4, 4)


def test_triangles_far_from_the_spot_are_untouched(tmp_path):
    source, pos, _front = _sphere_with_pocket(tmp_path)
    sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    out_pos = pp.read_glb(tmp_path / "out.glb")[0]
    back = pos[:, 2] < -0.5
    assert np.allclose(out_pos[: len(pos)][back], pos[back])


def test_a_point_off_the_model_changes_nothing(tmp_path):
    source, _pos, _front = _sphere_with_pocket(tmp_path)
    report = sf.fix_spot(source, tmp_path / "out.glb", np.array([5.0, 5.0, 5.0]), radius=0.1)
    assert report == {"cut": 0, "capped": 0}
    assert not (tmp_path / "out.glb").exists()
