"""The studio's "Fix a spot" brush: cut around a point, cap the opening, keep everything else."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix as sf


def _sphere_with_pocket(tmp_path, n=24, texture=4):
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
    picture = Image.new("RGB", (texture, texture), (0, 0, 0))     # black = unused texture space
    picture.paste((180, 120, 90), (0, 0, texture, texture // 2))   # the sphere's UVs use the top half...
    picture.paste((180, 120, 90), (0, 0, texture, texture))        # ...painted all one colour here
    picture.save(png, "PNG")
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


def test_the_patch_is_one_clean_colour_not_stretched_paint(tmp_path):
    source, pos, _front = _sphere_with_pocket(tmp_path)
    sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    _pos, out_uv, _out_faces, _tex = pp.read_glb(tmp_path / "out.glb")
    new_vertices = np.arange(len(pos), len(out_uv))
    assert len(new_vertices) > 3
    assert np.allclose(out_uv[new_vertices], out_uv[new_vertices[0]])


def test_the_rim_does_not_move(tmp_path):
    source, pos, _front = _sphere_with_pocket(tmp_path)
    sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    out_pos = pp.read_glb(tmp_path / "out.glb")[0]
    assert np.allclose(out_pos[: len(pos)], pos)


def test_the_studio_brush_ring_uses_the_same_sizes_as_the_fix():
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "viewer" / "studio" / "viewer.js").read_text()
    table = re.search(r"const BRUSH = \{([^}]*)\}", js).group(1)
    studio = {k: float(v) for k, v in re.findall(r"(\w+): ([0-9.]+)", table)}
    assert studio == sf.RADII


def test_a_rim_crossing_a_uv_seam_is_capped(tmp_path):
    """Real models split vertices at UV seams; the rim then passes through two vertices at one place."""
    source, pos, _front = _sphere_with_pocket(tmp_path)
    doc, binary = pp._split_glb(source.read_bytes())
    faces = sf._read(doc, binary, 2).reshape(-1, 3).astype(np.int64)
    uvs = sf._read(doc, binary, 1)
    # split one vertex near the pocket into two at the same place (a seam), half its faces each
    target = int(np.argmin(np.linalg.norm(pos - np.array([0.25, 0.0, 0.95]), axis=1)))
    users = np.flatnonzero((faces == target).any(1))
    pos2 = np.vstack([pos, pos[target]]).astype("<f4")
    uvs2 = np.vstack([uvs, uvs[target] + 0.01]).astype("<f4")
    for f in users[: len(users) // 2]:
        faces[f][faces[f] == target] = len(pos)
    binary = bytearray(binary)
    attributes = {"POSITION": pos2, "TEXCOORD_0": uvs2}
    primitive = doc["meshes"][0]["primitives"][0]
    sf._write(doc, bytes(binary), primitive, attributes, faces, tmp_path / "seam.glb")
    report = sf.fix_spot(tmp_path / "seam.glb", tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.35)
    assert report["capped"] >= 1
    out_pos, _uv, out_faces, _tex = pp.read_glb(tmp_path / "out.glb")
    from test_spot_fix import _open_edges
    assert _open_edges(out_faces.reshape(-1, 3), out_pos) == 0


def test_with_room_in_the_texture_each_patch_gets_its_own_square(tmp_path):
    source, pos, _front = _sphere_with_pocket(tmp_path, texture=512)
    # squash the sphere's UVs into the top half so the bottom half is free
    doc, binary = pp._split_glb(source.read_bytes())
    faces = sf._read(doc, binary, 2).reshape(-1, 3).astype(np.int64)
    uvs = sf._read(doc, binary, 1) * np.array([1.0, 0.5], dtype=np.float32)
    sf._write(doc, binary, doc["meshes"][0]["primitives"][0], {"POSITION": pos.astype("<f4"), "TEXCOORD_0": uvs},
              faces, tmp_path / "half.glb")
    report = sf.fix_spot(tmp_path / "half.glb", tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    _pos, out_uv, _faces, texture = pp.read_glb(tmp_path / "out.glb")
    patch = out_uv[len(pos):]
    assert report["capped"] == 1 and not report["repainted"]
    assert (patch[:, 1] > 0.5).all(), "the patch moved into the free half of the texture"
    assert len(np.unique(np.round(patch, 4), axis=0)) > 3, "spread over its square, not one texel"
    # its square was filled with the patch's colour, not left black
    px = (patch * 512).astype(int).clip(0, 511)
    assert texture[px[:, 1], px[:, 0], :3].min() > 50
