"""Tests for painting a model with the real pixels of its source photos.

Small synthetic scenes where the right answer is obvious: a square facing a camera, a
second one hidden behind it, one seen edge-on. The GLB tests round-trip a real file.
"""

from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import photo_paint as pp

IDENTITY = ((0, 1, 2), (1, 1, 1))


def _camera(z: float = 3.0, size: int = 64, fov: float = 0.6, image=None) -> pp.View:
    c2w = np.eye(4)
    c2w[2, 3] = z  # at +z, looking down -z towards the origin
    if image is None:
        image = np.zeros((size, size, 4), dtype=np.uint8)
        image[..., 3] = 255
    return pp.View(image, c2w, fov, "front")


def _quad(z: float = 0.0, half: float = 0.5):
    positions = np.array([[-half, -half, z], [half, -half, z], [half, half, z],
                          [-half, half, z]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    # glTF UVs: v grows downward, so the top of the quad (y = +half) is v = 0.
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    return positions, faces, uvs


# --- geometry --------------------------------------------------------------------------------


def test_a_point_on_the_camera_axis_lands_at_the_image_centre():
    view = _camera(size=64)
    x, y, depth = pp.project(np.array([[0.0, 0.0, 0.0]]), view)
    assert x[0] == pytest.approx(32.0) and y[0] == pytest.approx(32.0)
    assert depth[0] == pytest.approx(3.0)


def test_up_in_the_world_is_up_in_the_image():
    view = _camera(size=64)
    _, y_high, _ = pp.project(np.array([[0.0, 0.5, 0.0]]), view)
    _, y_low, _ = pp.project(np.array([[0.0, -0.5, 0.0]]), view)
    assert y_high[0] < 32 < y_low[0]


def test_the_frame_permutes_and_flips_axes():
    points = np.array([[1.0, 2.0, 3.0]])
    assert pp.to_view_space(points, ((0, 2, 1), (-1, 1, 1))).tolist() == [[-1.0, 3.0, 2.0]]
    assert pp.to_view_space(points, IDENTITY, mesh_scale=2.0).tolist() == [[0.5, 1.0, 1.5]]


def test_rasterize_covers_pixel_centres_inside_a_square():
    tri_xy = np.array([[[2, 2], [6, 2], [6, 6]], [[2, 2], [6, 6], [2, 6]]], dtype=float)
    face_of, _ = pp.rasterize(tri_xy, np.zeros((2, 3)), (8, 8))
    covered = face_of >= 0
    assert covered[2:6, 2:6].all()
    assert covered.sum() == 16


def test_rasterize_keeps_the_nearest_triangle():
    near = [[0, 0], [8, 0], [0, 8]]
    tri_xy = np.array([near, near], dtype=float)
    face_of, _ = pp.rasterize(tri_xy, np.array([[5.0] * 3, [2.0] * 3]), (8, 8))
    assert set(np.unique(face_of[face_of >= 0])) == {1}


def test_rasterize_barycentrics_sum_to_one():
    tri_xy = np.array([[[0, 0], [8, 0], [0, 8]]], dtype=float)
    face_of, bary = pp.rasterize(tri_xy, np.zeros((1, 3)), (8, 8))
    assert np.allclose(bary[face_of >= 0].sum(axis=1), 1.0)


def test_erode_counts_pixels_in_from_the_edge():
    mask = np.zeros((9, 9), dtype=bool)
    mask[1:8, 1:8] = True
    depth = pp.erode(mask, 5)
    assert depth[0, 0] == 0
    assert depth[1, 4] == 1
    assert depth[4, 4] == 4


# --- painting --------------------------------------------------------------------------------


def _split_photo(size: int = 64) -> np.ndarray:
    """Left half red, right half blue, fully opaque."""
    photo = np.zeros((size, size, 4), dtype=np.uint8)
    photo[:, : size // 2, 0] = 255
    photo[:, size // 2:, 2] = 255
    photo[..., 3] = 255
    return photo


def test_a_square_facing_the_camera_takes_the_photo_colours():
    positions, faces, uvs = _quad()
    texture = np.full((32, 32, 3), 128, dtype=np.uint8)
    view = _camera(size=64, image=_split_photo(64))
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    # Well inside the square: left of centre is red, right is blue, fully trusted.
    assert weight[16, 8] == pytest.approx(1.0)
    assert tuple(out[16, 8]) == (255, 0, 0)
    assert tuple(out[16, 24]) == (0, 0, 255)


def test_nothing_is_painted_where_the_camera_cannot_see():
    front = _quad(z=0.2)
    back = _quad(z=-0.2)
    positions = np.vstack([front[0], back[0]])
    faces = np.vstack([front[1], back[1] + 4])
    # Front quad on the left half of the texture, back quad on the right half.
    uvs = np.vstack([front[2] * [0.5, 1], back[2] * [0.5, 1] + [0.5, 0]])
    texture = np.full((32, 64, 3), 128, dtype=np.uint8)
    view = _camera(size=64, image=_split_photo(64))
    # Visibility only: colour matching would (rightly) nudge the hidden paint's palette.
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                   settings=pp.Settings(match_colour=False))
    assert weight[16, 16] == pytest.approx(1.0)
    assert weight[16, 48] == 0.0
    assert tuple(out[16, 48]) == (128, 128, 128)


def test_a_surface_seen_edge_on_keeps_its_own_paint():
    positions = np.array([[0, -0.5, -0.5], [0, 0.5, -0.5], [0, 0.5, 0.5], [0, -0.5, 0.5]],
                         dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64)
    texture = np.full((16, 16, 3), 128, dtype=np.uint8)
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [_camera(image=_split_photo())],
                                   frame=IDENTITY)
    assert weight.max() == 0.0
    assert (out == 128).all()


def test_the_matte_decides_where_the_photo_counts():
    positions, faces, uvs = _quad()
    photo = _split_photo(64)
    photo[:, 32:, 3] = 0  # right half is outside the cut-out
    texture = np.full((32, 32, 3), 128, dtype=np.uint8)
    out, weight = pp.paint_texture(texture, positions, uvs, faces,
                                   [_camera(size=64, image=photo)], frame=IDENTITY)
    assert tuple(out[16, 8]) == (255, 0, 0)
    assert weight[16, 24] == 0.0 and tuple(out[16, 24]) == (128, 128, 128)


def test_the_silhouette_edge_fades_in_rather_than_cutting_hard():
    positions, faces, uvs = _quad()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    view = _camera(size=128, image=_split_photo(128))
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    row = weight[32]
    assert row[0] < row[4] < row[16] == pytest.approx(1.0)


def test_two_views_that_agree_give_the_same_colour_as_one():
    positions, faces, uvs = _quad()
    texture = np.full((32, 32, 3), 128, dtype=np.uint8)
    view = _camera(size=64, image=_split_photo(64))
    one, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    two, _ = pp.paint_texture(texture, positions, uvs, faces, [view, view], frame=IDENTITY)
    assert np.array_equal(one, two)


def test_paint_grows_into_the_uv_gutter():
    positions, faces, _ = _quad()
    uvs = np.array([[0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25]])
    texture = np.full((32, 32, 3), 128, dtype=np.uint8)
    view = _camera(size=64, image=_split_photo(64))
    out, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                              settings=pp.Settings(gutter_px=3))
    # The island starts at x = 8. Gutter texels copy the island's edge texel (itself faded,
    # since that edge is also the silhouette); beyond gutter_px nothing changes.
    edge = tuple(out[16, 8])
    assert edge != (128, 128, 128)
    assert tuple(out[16, 7]) == edge and tuple(out[16, 5]) == edge
    assert tuple(out[16, 4]) == (128, 128, 128)


# --- views on disk -----------------------------------------------------------------------------


def test_views_load_from_a_transforms_directory(tmp_path):
    Image.fromarray(_split_photo(32)).save(tmp_path / "input.png")
    (tmp_path / "transforms.json").write_text(json.dumps({
        "camera_angle_x": 0.349, "mesh_scale": 1.0,
        "frames": [{"file_path": "input.png", "transform_matrix": np.eye(4).tolist()}],
    }))
    views, scale = pp.load_views(tmp_path)
    assert scale == 1.0 and len(views) == 1
    assert views[0].image.shape == (32, 32, 4)
    assert views[0].fov_x == pytest.approx(0.349)


# --- GLB round trip ----------------------------------------------------------------------------


def _textured_glb(colour=(128, 128, 128)) -> bytes:
    positions, faces, uvs = _quad()
    image = Image.new("RGB", (8, 8), colour)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    # trimesh stores UVs with v up and flips on export, so hand it the flipped copy.
    visual = trimesh.visual.TextureVisuals(uv=uvs * [1, -1] + [0, 1], material=material)
    mesh = trimesh.Trimesh(positions, faces, visual=visual, process=False)
    return mesh.export(file_type="glb")


def test_reading_a_glb_gets_geometry_and_texture(tmp_path):
    path = tmp_path / "quad.glb"
    path.write_bytes(_textured_glb())
    positions, uvs, faces, texture = pp.read_glb(path)
    assert positions.shape == (4, 3) and faces.shape == (2, 3)
    assert np.allclose(uvs, _quad()[2])
    assert texture.shape[:2] == (8, 8) and tuple(texture[0, 0, :3]) == (128, 128, 128)


def test_swapping_the_texture_changes_nothing_else(tmp_path):
    original = _textured_glb()
    new_png = pp.encode_png(np.full((16, 16, 3), 200, dtype=np.uint8))
    swapped = pp.replace_base_colour(original, new_png)
    path = tmp_path / "swapped.glb"
    path.write_bytes(swapped)
    positions, uvs, _, texture = pp.read_glb(path)
    assert texture.shape[:2] == (16, 16) and tuple(texture[0, 0, :3]) == (200, 200, 200)
    assert np.allclose(positions, _quad()[0]) and np.allclose(uvs, _quad()[2])
    # Still a valid GLB for another reader.
    assert len(trimesh.load(io.BytesIO(swapped), file_type="glb").geometry) == 1
    length = struct.unpack_from("<I", swapped, 8)[0]
    assert length == len(swapped)


def _webp_only_glb(colour=(90, 60, 30)) -> bytes:
    """A GLB whose base colour is WebP through EXT_texture_webp, with no PNG fallback:
    how Pixal3D's raw output arrives."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="WEBP", lossless=True)
    doc, binary = pp._split_glb(pp.replace_base_colour(_textured_glb(), buffer.getvalue()))
    doc["images"][0]["mimeType"] = "image/webp"
    texture = doc["textures"][0]
    texture["extensions"] = {"EXT_texture_webp": {"source": texture.pop("source")}}
    doc["extensionsUsed"] = ["EXT_texture_webp"]
    doc["extensionsRequired"] = ["EXT_texture_webp"]
    return pp._join_glb(doc, binary)


def test_a_webp_only_glb_can_be_read(tmp_path):
    path = tmp_path / "raw.glb"
    path.write_bytes(_webp_only_glb())
    _, _, _, texture = pp.read_glb(path)
    assert tuple(texture[0, 0, :3]) == (90, 60, 30)


def test_painting_a_webp_only_glb_writes_a_plain_png_texture(tmp_path):
    new_png = pp.encode_png(np.full((16, 16, 3), 200, dtype=np.uint8))
    swapped = pp.replace_base_colour(_webp_only_glb(), new_png)
    doc, _ = pp._split_glb(swapped)
    # A PNG behind a WebP extension would be a lie to every reader, so the texture points
    # at it directly and the extension is gone.
    assert doc["textures"][0]["source"] == 0 and "extensions" not in doc["textures"][0]
    assert "EXT_texture_webp" not in doc.get("extensionsRequired", [])
    assert "EXT_texture_webp" not in doc.get("extensionsUsed", [])
    path = tmp_path / "painted.glb"
    path.write_bytes(swapped)
    assert tuple(pp.read_glb(path)[3][0, 0, :3]) == (200, 200, 200)


def test_a_glb_with_a_node_transform_is_refused(tmp_path):
    doc, binary = pp._split_glb(_textured_glb())
    doc["nodes"][0]["translation"] = [1.0, 0.0, 0.0]
    path = tmp_path / "moved.glb"
    path.write_bytes(pp._join_glb(doc, binary))
    with pytest.raises(ValueError, match="node transforms"):
        pp.read_glb(path)


# --- colour matching ---------------------------------------------------------------------------


def test_colour_match_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    photo = rng.uniform(40, 200, size=(2000, 3))
    own = (photo - 10) / 1.25  # a paint that is duller and darker than the photo
    gain, offset = pp.fit_colour_match(own, photo, np.ones(2000))
    assert np.allclose(own * gain + offset, photo, atol=1e-6)


def test_colour_match_needs_enough_trusted_overlap():
    gain, offset = pp.fit_colour_match(np.zeros((100, 3)), np.ones((100, 3)), np.ones(100))
    assert gain.tolist() == [1, 1, 1] and offset.tolist() == [0, 0, 0]


def test_colour_match_is_clamped():
    own = np.full((1000, 3), 100.0) + np.arange(1000)[:, None] * 0.01
    photo = own * 10
    gain, _ = pp.fit_colour_match(own, photo, np.ones(1000))
    assert gain.max() == 2.0


def test_unseen_paint_takes_on_the_photo_palette():
    front = _quad(z=0.2)
    back = _quad(z=-0.2)
    positions = np.vstack([front[0], back[0]])
    faces = np.vstack([front[1], back[1] + 4])
    uvs = np.vstack([front[2] * [0.5, 1], back[2] * [0.5, 1] + [0.5, 0]])
    # Own paint: a two-tone pattern on both quads, duller than the photo's red/blue.
    texture = np.zeros((64, 128, 3), dtype=np.uint8)
    texture[:, :, :] = 60
    texture[:, 0:32, 0] = texture[:, 64:96, 0] = 120
    texture[:, 32:64, 2] = texture[:, 96:128, 2] = 120
    view = _camera(size=128, image=_split_photo(128))
    matched, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    plain, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                settings=pp.Settings(match_colour=False))
    # The hidden back quad is untouched without matching, and shifted towards the photo with it.
    assert tuple(plain[32, 80]) == (120, 60, 60)
    assert matched[32, 80, 0] > 120 and matched[32, 80, 1] < 60


# --- thin sheets: only surfaces that face the camera take its paint -------------------------


def _slab(thickness: float = 0.002, flip: bool = False):
    """A closed, very thin box: front face on the left half of the texture, back face on the
    right half. Thinner than the visibility tolerance, so depth alone cannot tell them apart."""
    t, h = thickness / 2, 0.5
    corners = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], dtype=np.float64)
    front = np.column_stack([corners, np.full(4, t)])
    back = np.column_stack([corners, np.full(4, -t)])
    quad_uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float64) * [0.5, 1]
    positions = [front, back]
    uvs = [quad_uv, quad_uv + [0.5, 0]]
    faces = [np.array([[0, 1, 2], [0, 2, 3]]), np.array([[4, 6, 5], [4, 7, 6]])]
    # Four side walls, outward, on their own vertices with a collapsed UV (cover no texels).
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        base = 8 + 4 * (len(faces) - 2)
        positions.append(np.array([front[a], back[a], back[b], front[b]]))
        uvs.append(np.full((4, 2), 0.999))
        faces.append(np.array([[base, base + 1, base + 2], [base, base + 2, base + 3]]))
    positions, uvs, faces = np.vstack(positions), np.vstack(uvs), np.vstack(faces)
    if flip:
        faces = faces[:, ::-1]
    return positions, faces, uvs


def test_outward_sign_reads_the_winding_of_a_closed_mesh():
    positions, faces, _ = _slab(thickness=0.5)
    assert pp.outward_sign(positions, faces) == 1.0
    assert pp.outward_sign(positions, faces[:, ::-1]) == -1.0


def test_outward_sign_is_unknown_for_an_open_sheet():
    positions, faces, _ = _quad()
    assert pp.outward_sign(positions, faces) == 0.0


def test_outward_sign_is_unknown_for_open_sheets_at_different_depths():
    # Two parallel squares enclose no volume even though their signed sum is not zero.
    positions, faces, _ = _occluded_scene()
    assert pp.outward_sign(positions, faces) == 0.0


def test_outward_sign_survives_vertices_split_at_a_uv_seam():
    positions, faces, _ = _slab(thickness=0.5)
    # The slab's walls already sit on their own vertices: a split mesh, welded back here.
    assert pp.outward_sign(positions, faces) == 1.0


@pytest.mark.parametrize("flip", [False, True])
def test_the_far_side_of_a_thin_sheet_keeps_its_own_paint(flip):
    positions, faces, uvs = _slab(flip=flip)
    texture = np.full((32, 64, 3), 128, dtype=np.uint8)
    view = _camera(size=64, image=_split_photo(64))
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                   settings=pp.Settings(match_colour=False))
    assert weight[16, 8] == pytest.approx(1.0) and tuple(out[16, 8]) == (255, 0, 0)
    assert weight[16, 40] == 0.0 and tuple(out[16, 40]) == (128, 128, 128)


# --- the picture's own cut-out edge ---------------------------------------------------------


def _half_cut_photo(size: int) -> np.ndarray:
    photo = _split_photo(size)
    photo[:, size // 2:, 3] = 0  # right half is outside the cut-out
    return photo


def test_paint_fades_out_towards_the_pictures_cut_out_edge():
    positions, faces, uvs = _quad()
    texture = np.full((256, 256, 3), 128, dtype=np.uint8)  # ~1.6 picture pixels per texel
    view = _camera(size=768, image=_half_cut_photo(768))
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    row = weight[128]
    # The cut-out edge is near texel 128: texels right beside it are distrusted, ones well
    # inside fully trusted, and the fade is monotonic in between.
    assert row[127] < 0.5
    assert row[110] == pytest.approx(1.0)
    assert row[110] >= row[124] >= row[125] >= row[126] >= row[127]
    assert 0.0 < row[125] < 1.0


def test_the_cut_out_fade_scales_with_the_picture():
    # Same scene at two picture sizes: the faded band covers the same share of the model.
    positions, faces, uvs = _quad()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    bands = []
    for size in (384, 1536):
        view = _camera(size=size, image=_half_cut_photo(size))
        _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
        bands.append(int((weight[32, 16:32] < 0.99).sum()))
    assert bands[0] >= 1 and abs(bands[0] - bands[1]) <= 1


def test_the_cut_out_fade_can_be_switched_off():
    positions, faces, uvs = _quad()
    texture = np.full((64, 64, 3), 128, dtype=np.uint8)
    view = _camera(size=768, image=_half_cut_photo(768))
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                 settings=pp.Settings(cutout_edge=0.0))
    assert weight[32, 31] == pytest.approx(1.0)


# --- inner depth edges: an ear in front of a head, a collar in front of a neck ---------------


def _occluded_scene():
    """A big square at z=0 (left half of the texture) with a small one floating in front of
    its middle at z=0.3 (right half). Their shared outline is an inner depth edge."""
    back = _quad(z=0.0, half=0.5)
    front = _quad(z=0.3, half=0.25)
    positions = np.vstack([back[0], front[0]])
    faces = np.vstack([back[1], front[1] + 4])
    uvs = np.vstack([back[2] * [0.5, 1], front[2] * [0.5, 1] + [0.5, 0]])
    return positions, faces, uvs


def _full_photo(size: int) -> np.ndarray:
    photo = np.full((size, size, 4), 200, dtype=np.uint8)
    photo[..., 3] = 255
    return photo


def test_paint_fades_out_beside_an_inner_depth_edge():
    positions, faces, uvs = _occluded_scene()
    texture = np.full((256, 512, 3), 128, dtype=np.uint8)
    view = _camera(size=768, image=_full_photo(768))
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                 settings=pp.Settings(match_colour=False))
    row = weight[128]
    # Back square: the occluder (closer, so it looks ~11% bigger) hides texels from 57 on.
    # Right beside that edge the picture is distrusted; well away from it, fully trusted.
    assert row[57] == 0.0
    assert row[55] < 0.5
    assert row[30] == pytest.approx(1.0)
    # Front square: its own outline borders the back square, so its rim fades too.
    assert row[256 + 2] < 0.5
    assert row[256 + 128] == pytest.approx(1.0)


def test_a_smoothly_tilted_surface_is_not_an_inner_edge():
    # A square tilted 60 degrees: depth changes steadily, never jumps.
    angle = np.radians(60)
    positions, faces, uvs = _quad()
    positions = positions @ np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                      [-np.sin(angle), 0, np.cos(angle)]]).T
    texture = np.full((128, 128, 3), 128, dtype=np.uint8)
    view = _camera(size=768, image=_full_photo(768))
    on = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)[1]
    off = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                           settings=pp.Settings(inner_edge=0.0))[1]
    assert np.allclose(on, off)


def test_the_inner_edge_fade_can_be_switched_off():
    positions, faces, uvs = _occluded_scene()
    texture = np.full((256, 512, 3), 128, dtype=np.uint8)
    view = _camera(size=768, image=_full_photo(768))
    _, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                                 settings=pp.Settings(match_colour=False, inner_edge=0.0))
    assert weight[128, 55] == pytest.approx(1.0)


# --- local colour: the unseen sides meet each picture's colours where they join --------------


def _grid(n: int = 40):
    """A flat n x n grid facing the camera, sharing vertices, UVs straight across."""
    ticks = np.linspace(-0.5, 0.5, n + 1)
    xs, ys = np.meshgrid(ticks, ticks[::-1])
    positions = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    uvs = np.column_stack([(xs.ravel() + 0.5), (0.5 - ys.ravel())])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = np.concatenate([np.stack([d, c, b], -1).reshape(-1, 3),
                            np.stack([d, b, a], -1).reshape(-1, 3)])
    return positions, faces, uvs


def _side_strip_photo(size: int, top: int, bottom: int) -> np.ndarray:
    """Grey photo seen on both sides with an unseen strip down the middle, like the side of
    a model between its front and back pictures. Top half one shade, bottom half another."""
    photo = np.zeros((size, size, 4), dtype=np.uint8)
    photo[: size // 2, :, :3] = top
    photo[size // 2:, :, :3] = bottom
    photo[:, : int(size * 0.42), 3] = 255
    photo[:, int(size * 0.58):, 3] = 255
    return photo


def test_local_colour_carries_each_regions_gap_into_the_unseen_strip():
    positions, faces, uvs = _grid()
    texture = np.full((128, 128, 3), 100, dtype=np.uint8)
    view = _camera(size=768, image=_side_strip_photo(768, top=140, bottom=60))
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    # The middle strip is outside the cut-out: no picture there, but each half should carry
    # on its neighbours' colour (one global shift would leave both near 100).
    assert weight[20, 64] == 0.0 and weight[108, 64] == 0.0
    assert out[20, 64, 0] >= 125
    assert out[108, 64, 0] <= 75


def test_one_mismatched_patch_does_not_smear_into_the_unseen_part():
    positions, faces, uvs = _grid()
    texture = np.full((128, 128, 3), 100, dtype=np.uint8)
    photo = _side_strip_photo(768, top=120, bottom=120)
    photo[360:410, 230:310, :3] = 250  # a small feature the model painted differently
    view = _camera(size=768, image=photo)
    out, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY)
    # Unseen texels beside the patch follow the typical +20 gap, not the patch's +150.
    assert abs(int(out[64, 64, 0]) - 120) <= 10


def test_local_colour_can_fall_back_to_the_global_match():
    positions, faces, uvs = _grid()
    texture = np.full((128, 128, 3), 100, dtype=np.uint8)
    view = _camera(size=768, image=_side_strip_photo(768, top=140, bottom=60))
    out, _ = pp.paint_texture(texture, positions, uvs, faces, [view], frame=IDENTITY,
                              settings=pp.Settings(local_colour=False))
    # The global fit sees a zero average gap, so the unseen halves stay alike.
    assert abs(int(out[20, 64, 0]) - int(out[108, 64, 0])) < 20
