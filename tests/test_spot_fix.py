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


def test_the_studio_brush_range_matches_the_server():
    import importlib.util
    import re
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    js = (repo / "viewer" / "studio" / "viewer.js").read_text()
    low, high = map(float, re.search(r"const BRUSH_MIN = ([0-9.]+), BRUSH_MAX = ([0-9.]+)", js).groups())
    spec = importlib.util.spec_from_file_location("fx", repo / "viewer" / "fixspot_api.py")
    fx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx)
    assert (low, high) == (fx.BRUSH_MIN, fx.BRUSH_MAX)


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


def test_a_round_hole_gets_an_even_curved_patch_not_a_fan(tmp_path):
    source, pos, _front = _sphere_with_pocket(tmp_path, n=48)
    report = sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.95]), radius=0.25)
    assert report["even"] == report["capped"] == 1
    out_pos, _uv, out_faces, _tex = pp.read_glb(tmp_path / "out.glb")
    out_faces = out_faces.reshape(-1, 3)
    assert _open_edges(out_faces, out_pos) == 0
    patch = out_pos[len(pos):]
    # curved like the sphere it fills (radius 1), not a flat lid across the hole
    assert np.abs(np.linalg.norm(patch, axis=1) - 1).max() < 0.05


def test_a_full_texture_gains_a_strip_and_the_paint_still_lines_up(tmp_path):
    """The 2026-10-20 camouflage bug: UVs moved for a strip that was then never added."""
    source, _pos, _front = _sphere_with_pocket(tmp_path, n=48, texture=256)   # UVs fill the whole texture
    _bpos, before_uv, _bfaces, before_tex = pp.read_glb(source)
    sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.95]), radius=0.25)
    _pos, out_uv, _faces, out_tex = pp.read_glb(tmp_path / "out.glb")
    assert out_tex.shape[0] > before_tex.shape[0], "the texture grew a strip"
    # every old vertex still reads the same pixel colour as before
    h0, h1 = before_tex.shape[0], out_tex.shape[0]
    old = before_tex[(before_uv[:, 1] * h0).astype(int).clip(0, h0 - 1), (before_uv[:, 0] * 256).astype(int).clip(0, 255)]
    new_uv = out_uv[: len(before_uv)]
    now = out_tex[(new_uv[:, 1] * h1).astype(int).clip(0, h1 - 1), (new_uv[:, 0] * 256).astype(int).clip(0, 255)]
    assert np.array_equal(old[:, :3], now[:, :3])


def test_repaint_fades_in_from_the_edge_of_the_spot(tmp_path):
    from test_winding import _views

    views = _views(tmp_path, colour=(255, 255, 255))            # the picture: all white
    source, _pos, _front = _sphere_with_pocket(tmp_path, n=48, texture=256)
    report = sf.repaint_area(source, tmp_path / "out.glb", [((0.0, 0.0, 1.0), 0.6)], views)
    assert report["repainted_texels"] > 0
    red = pp.read_glb(tmp_path / "out.glb")[3][..., 0].astype(int)
    assert (red >= 250).any(), "inside the spot the picture wins"
    blended = ((red > 190) & (red < 240)).sum()
    assert blended > 0.15 * (red >= 250).sum(), "past the ring, new paint fades into the old (a hard edge: ~4%)"


def test_a_rim_that_does_not_close_is_still_patched(tmp_path, monkeypatch):
    """Wrong-way triangles beside a cut can leave its rim as an open chain whose last point
    starts no edge (2026-10-20, the elf); Patch crashed looking up that point."""
    source, _pos, _front = _sphere_with_pocket(tmp_path)
    real = sf._boundary_loops

    def opened(faces, key):
        return [loop[:-1] for loop in real(faces, key)]      # drop each loop's closing edge
    monkeypatch.setattr(sf, "_boundary_loops", opened)
    monkeypatch.setattr(sf, "_even_patch", lambda _rim: None)   # the fan, which reads each edge's end
    report = sf.fix_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.9]), radius=0.3)
    assert report["cut"] > 0 and report["capped"] == 1


def test_smooth_lifts_a_dent_without_cutting_anything(tmp_path):
    source, _pos, front = _sphere_with_pocket(tmp_path)
    report = sf.smooth_spot(source, tmp_path / "out.glb", np.array([0, 0, 0.4]), 0.5, strength=1.0)
    assert report["smoothed"] > 0
    before, after = pp.read_glb(source), pp.read_glb(tmp_path / "out.glb")
    assert np.array_equal(before[2], after[2]), "same triangles: nothing cut or added"
    assert np.array_equal(before[1], after[1]), "same UVs: the paint stays put"
    assert after[0][front, 2] > 0.8, "the dent rose to the surface round it"
    far = np.linalg.norm(before[0] - [0, 0, 0.4], axis=1) > 0.5
    assert np.allclose(before[0][far], after[0][far]), "outside the ring nothing moves"


def test_smooth_strength_sets_how_far_it_goes(tmp_path):
    source, _pos, front = _sphere_with_pocket(tmp_path)
    lifts = []
    for strength in (0.2, 1.0):
        sf.smooth_spot(source, tmp_path / f"{strength}.glb", np.array([0, 0, 0.4]), 0.5, strength=strength)
        lifts.append(pp.read_glb(tmp_path / f"{strength}.glb")[0][front, 2] - 0.4)
    assert 0 < lifts[0] < lifts[1]


def test_smooth_marks_are_applied_and_repainted_with_the_rest(tmp_path):
    from test_winding import _views

    source, _pos, front = _sphere_with_pocket(tmp_path, n=48, texture=256)
    spots = [{"point": [0, 0, 0.4], "radius": 0.5, "mode": "smooth", "strength": 1.0}]
    reports = sf.apply_spots(source, tmp_path / "out.glb", spots, _views(tmp_path, colour=(255, 255, 255)))
    assert reports[0]["smoothed"] > 0 and reports[0]["repainted"]
    assert pp.read_glb(tmp_path / "out.glb")[0][front, 2] > 0.8


def test_the_studio_smooth_strength_range_matches_the_server():
    import re
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "viewer" / "studio" / "viewer.js").read_text()
    low, high = map(float, re.search(r"const STRENGTH_MIN = ([0-9.]+), STRENGTH_MAX = ([0-9.]+)", js).groups())
    assert (low, high) == (sf.STRENGTH_MIN, sf.STRENGTH_MAX)
    assert re.search(r"brushStrength = ([0-9.]+)", js).group(1) == str(sf.STRENGTH_DEFAULT)
    for mode in sf.MODES:
        assert f'data-mode="{mode}"' in js and f"{mode}:" in js, f"the studio offers {mode}"
