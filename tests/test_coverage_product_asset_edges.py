"""Mesh edits and export edge cases using small embedded buffers."""

import io
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import (
    bone_names,
    leg_check,
    matte,
    roblox_export,
    spot_fix,
)
from image_to_3dlab.glb_turn import build_glb, parse_glb
from tests.test_bone_names import _skinned
from tests.test_glb_skins import RIGGED, SKIN
from tests.test_leg_check import side_mask
from tests.test_roblox_export import _model, _picture
from tests.test_spot_fix import _sphere_with_pocket
from tests.test_winding import _views


@pytest.mark.parametrize("rim", [
    [(0, 0, 0), (2, 0, 0)],
    [(0, 0, 0), (1, 0, 0), (2, 0, 0)],
    [(0, 0, 0), (2, 2, 0), (2, 0, 0), (0, 2, 0)],
])
def test_even_patch_rejects_incomplete_collinear_and_crossed_boundaries(rim):
    assert spot_fix._even_patch(np.array(rim, dtype=float)) is None


def test_unclosed_two_edge_walk_does_not_become_a_cap():
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    loops = spot_fix._boundary_loops(faces, np.arange(4))
    assert loops == []


@pytest.fixture
def painted_mesh(tmp_path):
    source, pos, front = _sphere_with_pocket(tmp_path, n=36, texture=256)
    doc, binary = parse_glb(source.read_bytes())
    prim = doc["meshes"][0]["primitives"][0]
    attributes = {name: spot_fix._read(doc, binary, accessor) for name, accessor in prim["attributes"].items()}
    attributes["NORMAL"] = np.zeros_like(attributes["POSITION"])
    attributes["NORMAL"][:, 1] = 1
    attributes["TANGENT"] = np.tile(np.array([0, 0, 1, -1], dtype=np.float32), (len(pos), 1))
    for name, width in (("NORMAL", "VEC3"), ("TANGENT", "VEC4")):
        prim["attributes"][name] = len(doc["accessors"])
        doc["accessors"].append({"bufferView": 0, "componentType": 5126, "count": len(pos), "type": width})
    png = io.BytesIO()
    Image.new("RGB", (256, 256), (115, 177, 90)).save(png, "PNG")
    blob = png.getvalue()
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
    binary += blob
    doc["images"].append({"bufferView": len(doc["bufferViews"]) - 1, "mimeType": "image/png"})
    doc["textures"].append({"source": 1})
    doc["materials"][0]["normalTexture"] = {"index": 1}
    faces = spot_fix._read(doc, binary, prim["indices"]).reshape(-1, 3)
    baked = tmp_path / "normals.glb"
    spot_fix._write(doc, binary, prim, attributes, faces, baked)
    return baked, pos, front


def test_patch_recomputes_normal_and_repaints_only_its_texture_square(tmp_path, painted_mesh):
    source, before, _ = painted_mesh
    views = _views(tmp_path, colour=(255, 255, 255))
    fixed = tmp_path / "fixed.glb"
    report = spot_fix.fix_spot(source, fixed, np.array([0, 0, 0.94]), 0.3, views_dir=views)
    assert report["capped"] == 1
    assert report["repainted"]
    doc, binary = parse_glb(fixed.read_bytes())
    prim = doc["meshes"][0]["primitives"][0]
    attributes = {name: spot_fix._read(doc, binary, i) for name, i in prim["attributes"].items()}
    normals = attributes["NORMAL"][len(before):]
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
    assert np.any(np.abs(normals[:, 2]) > 0.3)
    np.testing.assert_array_equal(attributes["TANGENT"], np.tile([0, 0, 1, -1], (len(attributes["POSITION"]), 1)))
    normal_map = spot_fix._decode(doc, binary, 1)
    base_map = spot_fix._decode(doc, binary, 0)
    assert ((normal_map[..., :3] == (128, 128, 255)).all(2)).any()
    assert ((normal_map[..., :3] == (115, 177, 90)).all(2)).any()
    assert ((base_map[..., :3] == (180, 120, 90)).all(2)).any()
    assert (base_map[..., :3] > 245).all(2).any()


def test_smooth_updates_nearby_normals_but_keeps_distant_normals(tmp_path, painted_mesh):
    source, pos, front = painted_mesh
    output = tmp_path / "smooth.glb"
    report = spot_fix.smooth_spot(source, output, pos[front], 0.65, 0.6)
    assert report["smoothed"] > 0 and report["moved"] > 0.01
    doc, binary = parse_glb(output.read_bytes())
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    normals = spot_fix._read(doc, binary, attrs["NORMAL"])
    far = np.linalg.norm(pos - pos[front], axis=1) > 0.65 * 1.2
    np.testing.assert_array_equal(normals[far], np.tile([0, 1, 0], (far.sum(), 1)))
    assert np.linalg.norm(normals[front]) == pytest.approx(1.0, abs=1e-5)
    assert abs(normals[front, 2]) > 0.2


def test_cap_accepts_no_base_map_and_preserves_uvs():
    uvs = np.array([[0.1, 0.2]])
    assert spot_fix._own_squares({"materials": [{}]}, b"", {"TEXCOORD_0": uvs}, [], [], None) == (None, False)
    np.testing.assert_array_equal(uvs, [[0.1, 0.2]])


def test_clipped_foreground_row_has_no_floor_span_and_tiny_hole_merges():
    covered = np.ones((100, 100), dtype=bool)
    covered[96] = False
    covered[91:93, 50] = False
    measured = leg_check.measure(covered)
    assert measured["open_under_body"] == 0
    assert measured["parts"][0.08] == 1
    assert measured["gaps"][0.08] == 0


@pytest.mark.parametrize("use_default", [False, True])
def test_opaque_leg_picture_is_matted_before_outline_measurement(tmp_path, monkeypatch, use_default):
    path = tmp_path / "opaque.png"
    Image.new("RGBA", (128, 128), (0, 0, 0, 255)).save(path)
    calls = []

    def cut(image):
        assert image.mode == "RGB"
        calls.append(image.size)
        outline = side_mask(30)
        rgba = Image.new("RGBA", outline.shape[::-1])
        rgba.putalpha(Image.fromarray(outline.astype(np.uint8) * 255))
        return rgba

    monkeypatch.setattr(matte, "cut_out", lambda image: (cut(image), "fixture"))
    measured, problems = leg_check.check_picture(path, "side", cut_out=None if use_default else cut)
    assert calls == [(128, 128)]
    assert measured["open_under_body"] > 0.12
    assert problems == []


def test_uncompressed_jpeg_texture_is_converted_without_resize():
    original = _model(pictures=[("image/jpeg", _picture("JPEG", size=16))])
    doc, binary = parse_glb(original)
    packed, notes = roblox_export.convert_pictures(doc, binary, 32)
    assert notes == ["picture 1 converted to PNG"]
    assert doc["images"][0]["mimeType"] == "image/png"
    image = Image.open(io.BytesIO(roblox_export._view_bytes(doc, packed, doc["images"][0]["bufferView"])))
    assert image.size == (16, 16) and image.format == "PNG"


def test_skin_binary_validates_chunk_and_accessor_encoding():
    source = _skinned([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], [[[0.1] * 4], [[0.15] * 4]])
    bad_chunk = bytearray(source)
    struct.pack_into("<I", bad_chunk, 16, 0)
    with pytest.raises(ValueError, match="no JSON chunk"):
        bone_names.rename(bytes(bad_chunk), {})
    doc, binary = parse_glb(source)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    doc["accessors"][attrs["JOINTS_0"]]["type"] = "VEC3"
    with pytest.raises(ValueError, match="plain VEC4"):
        bone_names.limit_influences(build_glb(doc, binary))


def test_skin_without_binary_can_be_renamed_without_new_binary_chunk():
    model = build_glb({"asset": {"version": "2.0"}, "nodes": [{"name": "pelvis"}], "skins": [{"joints": [0]}]}, b"")
    result = bone_names.rename(model, {"pelvis": "Hips"})
    doc, binary = parse_glb(result)
    assert binary == b""
    assert doc["nodes"][0]["name"] == "Hips"
    assert "3 vertices trimmed to 4 weights" in bone_names.describe({
        "preset": "unity", "named": 1, "bones": 1, "capped": 3, "kept": [], "missing": [],
    })


def test_skin_rejects_wide_joint_that_cannot_be_moved_into_first_buffer():
    source = _skinned([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], [[[0.1] * 4], [[0.15] * 4]])
    doc, binary = parse_glb(source)
    index = doc["meshes"][0]["primitives"][0]["attributes"]["JOINTS_1"]
    accessor = doc["accessors"][index]
    view = doc["bufferViews"][accessor["bufferView"]]
    view.update(byteOffset=len(binary), byteLength=8)
    accessor["componentType"] = 5123
    binary += struct.pack("<4H", 260, 261, 262, 263)
    with pytest.raises(ValueError, match="JOINTS_0 is too narrow"):
        bone_names.limit_influences(build_glb(doc, binary))


def test_resize_scales_matrix_translation_without_touching_basis_or_normal_target():
    doc, binary = parse_glb(_model())
    matrix = np.eye(4)
    matrix[0, 0] = 3
    matrix[:3, 3] = [1, 2, 3]
    doc["nodes"][0]["matrix"] = matrix.T.ravel().tolist()
    primitive = doc["meshes"][0]["primitives"][0]
    primitive["targets"] = [{"NORMAL": primitive["attributes"]["POSITION"]}]
    result = roblox_export.scale_to_studs(doc, binary, 2)
    assert len(result) == len(binary)
    after = np.array(doc["nodes"][0]["matrix"]).reshape(4, 4).T
    np.testing.assert_array_equal(after[:3, :3], matrix[:3, :3])
    np.testing.assert_array_equal(after[:3, 3], [2, 4, 6])


def test_resize_refuses_bufferless_positions():
    doc, binary = parse_glb(_model())
    position = doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
    del doc["accessors"][position]["bufferView"]
    with pytest.raises(roblox_export.RobloxExportError, match="can't be resized"):
        roblox_export.scale_to_studs(doc, binary, 2)


def test_empty_picture_repacking_does_not_add_a_buffer_declaration():
    doc = {"asset": {"version": "2.0"}, "extensionsUsed": ["EXT_texture_webp"]}
    packed, notes = roblox_export.convert_pictures(doc, b"", 32)
    assert packed == b"" and not notes
    assert doc == {"asset": {"version": "2.0"}}


def test_skins_preserve_external_normal_and_ignore_absent_metallic_slot():
    from image_to_3dlab import glb_skins

    doc, binary = parse_glb(RIGGED)
    doc["images"][0] = {"uri": "normals.png"}
    doc["accessors"].append({"componentType": 5126, "count": 3, "type": "VEC3"})
    for texture in doc["textures"]:
        texture.pop("sampler")
    doc["materials"][0]["pbrMetallicRoughness"].pop("metallicRoughnessTexture")
    model = build_glb(doc, binary)
    dressed = glb_skins.dress(model, SKIN)
    assert b"GOLD-SKIRT-COLOUR" in dressed and b"GOLD-SHINE" not in dressed
    after, _ = parse_glb(dressed)
    assert {"uri": "normals.png"} in after["images"]
    assert "bufferView" not in after["accessors"][-1]
    assert all("sampler" not in texture for texture in after["textures"])
    variants = glb_skins.with_variants(model, [("gold", SKIN)])
    assert glb_skins.variant_names(variants) == ["Original", "gold"]
    described = glb_skins.describe(variants)
    assert described["materials"] == 2
    assert described["paint"][1] == {"baseColorTexture": len(b"GOLD-SKIRT-COLOUR")}


def test_straightening_splits_shared_tracks_without_rotating_unrelated_nodes():
    from tests.test_roblox_export import _skinned as sample_rig

    quaternion = (0, 0, np.sqrt(0.5), np.sqrt(0.5))
    doc, binary = parse_glb(sample_rig(rest=quaternion))
    doc["nodes"] += [{"name": "hat"}, {"name": "marker"}]
    doc["nodes"][1]["children"] = [2]
    animation = doc["animations"][0]
    animation["channels"] += [
        {"sampler": 0, "target": {"node": 2, "path": "translation"}},
        {"sampler": 0, "target": {"node": 3, "path": "translation"}},
        {"sampler": 0, "target": {"node": 2, "path": "weights"}},
    ]
    patched, note = roblox_export.straighten_bones(doc, binary)
    assert note and "rest straight" in note
    samplers = animation["samplers"]
    hip, hat, marker, shape = animation["channels"]
    assert hip["sampler"] != hat["sampler"] and marker["sampler"] == shape["sampler"] == 0
    moved = roblox_export._float_rows(doc, bytearray(patched), samplers[hat["sampler"]]["output"])
    np.testing.assert_allclose(moved, [[-1, 0, 0], [-2, 0, 0]], atol=1e-6)
    unchanged = roblox_export._float_rows(doc, bytearray(patched), samplers[0]["output"])
    np.testing.assert_array_equal(unchanged, [[0, 1, 0], [0, 2, 0]])


def test_invalid_native_geometry_is_refused_without_writing_mesh(tmp_path):
    from image_to_3dlab import pixal3d_native

    doc, binary = parse_glb(_model())
    path = tmp_path / "external.glb"
    primitive = doc["meshes"][0]["primitives"][0]
    accessor = doc["accessors"][primitive["attributes"]["POSITION"]]
    original_kind = accessor["componentType"]
    accessor["componentType"] = 5123
    path.write_bytes(build_glb(doc, binary))
    with pytest.raises(ValueError, match="float"):
        pixal3d_native.convert_replay_glb(path.read_bytes())
    accessor["componentType"] = original_kind
    del primitive["indices"]
    path.write_bytes(build_glb(doc, binary))
    with pytest.raises(ValueError, match="indexed mesh with positions"):
        pixal3d_native.convert_replay_glb(path.read_bytes())


def test_local_weights_take_precedence_without_traversing_hub(tmp_path, monkeypatch):
    from image_to_3dlab import weight_gate

    monkeypatch.chdir(tmp_path)
    local = tmp_path / "folder" / "weights"
    local.mkdir(parents=True)
    (local / "shape.gguf").write_bytes(b"placeholder")
    assert weight_gate.snapshot_dir("folder/weights", hub=tmp_path / "missing") == local.relative_to(tmp_path)


def test_matte_material_rewrite_is_idempotent_and_leaves_albedo_binary(tmp_path):
    from image_to_3dlab import trellis_backend

    path = tmp_path / "painted.glb"
    original = _model(pictures=[("image/png", _picture("PNG"))])
    path.write_bytes(original)
    _, binary = parse_glb(original)
    assert trellis_backend._normalize_glb_material(path, "matte") >= 1
    assert trellis_backend._normalize_glb_material(path, "matte") == 0
    material, result = parse_glb(path.read_bytes())
    assert material["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0
    assert result == binary
