"""Exercise texture and surface editing with small, locally built GLB files."""

import json

import numpy as np
import pytest
from PIL import Image
from test_photo_paint import _textured_glb, _webp_only_glb
from test_spot_fix import _sphere_with_pocket
from test_surface_calm import _grid, _write_glb

from image_to_3dlab import photo_paint as pp
from image_to_3dlab import spot_fix, surface_calm


def test_all_triangles_outside_render_frame_leave_it_empty():
    xy = np.array([[[12, 12], [12, 14], [14, 14]]])
    face, barycentric = pp.rasterize(xy, np.ones((1, 3)), (4, 4))
    assert np.all(face == -1)
    assert not barycentric.any()


def test_read_glb_refuses_non_binary_and_empty_geometry(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(b"not a GLB")
    with pytest.raises(ValueError, match="not a GLB file"):
        pp.read_glb(path)
    path.write_bytes(pp._join_glb({"meshes": []}, b""))
    with pytest.raises(ValueError, match="expected one primitive, found 0"):
        pp.read_glb(path)


def test_read_glb_declines_interleaved_attributes(tmp_path):
    doc, binary = pp._split_glb(_textured_glb())
    position = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    doc["bufferViews"][doc["accessors"][position]["bufferView"]]["byteStride"] = 16
    model = tmp_path / "model.glb"
    model.write_bytes(pp._join_glb(doc, binary))
    with pytest.raises(ValueError, match="interleaved buffers"):
        pp.read_glb(model)


@pytest.mark.parametrize(("material", "textures", "error"), [
    ([], [], "expected one material, found 0"),
    ([{}], [], "no base colour texture"),
    ([{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}], [{}], "no image this reader understands"),
])
def test_missing_base_colour_data_reports_source(material, textures, error):
    with pytest.raises(ValueError, match=error):
        pp.base_colour_image_index({"materials": material, "textures": textures})


@pytest.mark.parametrize("other_webp", [False, True])
def test_repaint_preserves_unrelated_extensions(other_webp):
    doc, binary = pp._split_glb(_webp_only_glb())
    source = pp.base_colour_image_index(doc)
    texture = doc["textures"][0]
    texture["extensions"]["ARTIST_extra"] = {"name": "test"}
    doc.setdefault("extensionsUsed", []).append("ARTIST_extra")
    if other_webp:
        doc["textures"].append({"extensions": {"EXT_texture_webp": {"source": source + 1}}})
    changed, _ = pp._split_glb(pp.replace_base_colour(pp._join_glb(doc, binary),
                                                     pp.encode_png(np.zeros((4, 4, 3), np.uint8))))
    assert changed["textures"][0] == {"source": source, "extensions": {"ARTIST_extra": {"name": "test"}}}
    assert ("EXT_texture_webp" in changed["extensionsUsed"]) is other_webp
    assert "ARTIST_extra" in changed["extensionsUsed"]


def test_front_weights_follow_the_camera_and_calm_rewrites_normals(tmp_path):
    positions, faces, centre = _grid(n=5, bump=.2)
    model = tmp_path / "model.glb"
    _write_glb(model, positions, faces, (positions[:, :2] + 1) / 2)
    doc, binary = pp._split_glb(model.read_bytes())
    normals = np.zeros_like(positions, dtype="<f4").tobytes()
    offset = len(binary)
    doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(normals)})
    doc["accessors"].append({"bufferView": len(doc["bufferViews"]) - 1, "count": len(positions),
                             "type": "VEC3", "componentType": 5126})
    doc["meshes"][0]["primitives"][0]["attributes"]["NORMAL"] = len(doc["accessors"]) - 1
    doc["buffers"][0]["byteLength"] = offset + len(normals)
    model.write_bytes(pp._join_glb(doc, binary + normals))

    Image.new("RGBA", (8, 8), "red").save(tmp_path / "photo.png")
    camera = np.eye(4)
    camera[:3, 3] = pp.to_view_space(np.array([[0., 0., 5.]]))[0]
    (tmp_path / "transforms.json").write_text(json.dumps({
        "camera_angle_x": .5, "frames": [{"file_path": "photo.png", "transform_matrix": camera.tolist()}],
    }))
    weights = surface_calm.front_weights(model, tmp_path)
    assert weights.shape == (len(positions),)
    assert weights[centre] == pytest.approx(1)
    output = tmp_path / "calmed.glb"
    surface_calm.calm_glb(model, output, weights)
    assert pp.read_glb(output)[0][centre, 2] < .15
    out, data = pp._split_glb(output.read_bytes())
    normal = pp._accessor(out, data, len(doc["accessors"]) - 1)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1, atol=1e-6)


def test_smooth_far_outside_mesh_creates_no_output(tmp_path):
    source, *_ = _sphere_with_pocket(tmp_path)
    output = tmp_path / "smooth.glb"
    assert spot_fix.smooth_spot(source, output, np.array([10, 10, 10]), .1, .5) == {"smoothed": 0}
    assert not output.exists()


def test_height_measures_positions_when_bounds_are_absent(tmp_path):
    path = tmp_path / "model.glb"
    doc, binary = pp._split_glb(_textured_glb())
    position = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    points = pp._accessor(doc, binary, position)
    accessor = doc["accessors"][position]
    accessor.pop("min", None)
    accessor.pop("max", None)
    path.write_bytes(pp._join_glb(doc, binary))
    assert spot_fix.model_height(path) == pytest.approx(np.ptp(points[:, 1]))


def test_repaint_only_mark_preserves_geometry_with_missing_views(tmp_path):
    source, *_ = _sphere_with_pocket(tmp_path)
    output = tmp_path / "fixed.glb"
    mark = {"mode": "repaint", "point": np.array([0, 0, 1]), "radius": .4}
    report = spot_fix.apply_spots(source, output, [mark], tmp_path)
    assert output.read_bytes() == source.read_bytes()
    assert report == [{"mode": "repaint", "repainted": False, "repainted_faces": 0}]


def test_repaint_only_mark_changes_texels_with_real_views(tmp_path):
    from test_winding import _views

    source, *_ = _sphere_with_pocket(tmp_path, n=24, texture=64)
    views = _views(tmp_path, colour=(255, 255, 255))
    output = tmp_path / "repaint.glb"
    point = np.array([0, 0, 1])
    report = spot_fix.apply_spots(source, output, [{"mode": "repaint", "point": point, "radius": .5}], views)
    assert report[0]["repainted"]
    assert report[0]["repainted_faces"] > 0
    before, after = pp.read_glb(source), pp.read_glb(output)
    for old, new in zip(before[:3], after[:3], strict=True):
        np.testing.assert_array_equal(old, new)
    assert after[3].max() > before[3].max()
    single = tmp_path / "single.glb"
    assert spot_fix.repaint_spot(source, single, point, .5, views)["repainted_texels"] > 0
    assert single.read_bytes() == output.read_bytes()


def test_front_layout_and_profile_mask_handle_empty_cutouts():
    from image_to_3dlab import side_paint

    alpha = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="cut-out is empty"):
        side_paint.front_layout(alpha)
    assert not side_paint.profile_face_mask(alpha, 0, 7, True).any()


def test_front_layout_uses_middle_when_side_components_are_not_arms():
    from image_to_3dlab import side_paint

    alpha = np.zeros((60, 60), dtype=np.uint8)
    alpha[:15, 23:37] = 255
    alpha[15:22, 26:34] = 255
    alpha[22:, 18:42] = 255
    alpha[2:11, 4:8] = 255  # optional hair sits above possible arms
    alpha[4, 54] = 255      # speck too small to be a hand
    alpha[46:58, 47:52] = 255  # a separate leg starts too low
    result = side_paint.front_layout(alpha)
    assert not result.arm.any()
    assert result.hands == -1
    assert 15 <= result.neck < 22


def test_pixel_match_collects_alignment_evidence():
    from test_photo_paint import _camera, _quad

    positions, faces, uvs = _quad()
    texture = np.full((16, 16, 3), 128, dtype=np.uint8)
    reports = []
    out, weight = pp.paint_texture(texture, positions, uvs, faces, [_camera()],
                                   frame=((0, 1, 2), (1, 1, 1)),
                                   settings=pp.Settings(align=True), align_reports=reports)
    assert out.shape == texture.shape and weight.shape == texture.shape[:2]
    assert len(reports) == 1
