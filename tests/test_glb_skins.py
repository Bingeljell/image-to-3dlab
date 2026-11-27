"""Dressing a model in a skin's paint, and one GLB holding every skin (image_to_3dlab/glb_skins.py).

What matters: only the colour and metal/shine pictures move across, the normal map and the mesh
stay; swapped-out pictures leave the file instead of bloating it; Finish's WebP pictures keep
working; the all-skins file lists every skin by name and maps each to its own material.
"""

from __future__ import annotations

import struct

import pytest

from image_to_3dlab import glb_skins as gs
from image_to_3dlab.glb_turn import build_glb, parse_glb


def _glb(base: bytes, shine: bytes, normal: bytes = b"NORMAL", webp_shine: bool = False) -> bytes:
    """A one-triangle model with a normal, colour and metal/shine picture, as Finish writes them."""
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    blobs = [positions, normal, base, shine]
    binary, views = b"", []
    for blob in blobs:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob
    shine_texture = ({"extensions": {gs.WEBP: {"source": 2}}, "sampler": 0} if webp_shine
                     else {"source": 2, "sampler": 0})
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        "materials": [{"name": "RETOPO_MAT", "normalTexture": {"index": 0},
                       "pbrMetallicRoughness": {"baseColorTexture": {"index": 1},
                                                "metallicRoughnessTexture": {"index": 2}}}],
        "samplers": [{}],
        "textures": [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 0}, shine_texture],
        "images": [{"bufferView": 1, "mimeType": "image/jpeg"}, {"bufferView": 2, "mimeType": "image/jpeg"},
                   {"bufferView": 3, "mimeType": "image/webp" if webp_shine else "image/png"}],
        **({"extensionsUsed": [gs.WEBP], "extensionsRequired": [gs.WEBP]} if webp_shine else {}),
    }
    return build_glb(document, binary)


def _pictures(data: bytes) -> dict[str, bytes]:
    document, binary = parse_glb(data)
    out = {}
    for role, image in gs.paint_images(document).items():
        out[role] = gs._image_bytes(document, binary, image)[0]
    normal = document["textures"][document["materials"][0]["normalTexture"]["index"]]
    out["normal"] = gs._image_bytes(document, binary, gs._source(normal))[0]
    return out


RIGGED = _glb(b"MODEL-OWN-PAINT", b"MODEL-OWN-SHINE")
SKIN = _glb(b"GOLD-SKIRT-COLOUR", b"GOLD-SHINE", normal=b"SKIN-NORMAL")


def test_dressing_swaps_the_paint_and_keeps_the_normal_map():
    dressed = gs.dress(RIGGED, SKIN)
    assert _pictures(dressed) == {"baseColorTexture": b"GOLD-SKIRT-COLOUR",
                                  "metallicRoughnessTexture": b"GOLD-SHINE", "normal": b"NORMAL"}


def test_the_swapped_out_pictures_leave_the_file():
    dressed = gs.dress(RIGGED, SKIN)
    assert b"MODEL-OWN-PAINT" not in dressed and b"MODEL-OWN-SHINE" not in dressed
    document, _ = parse_glb(dressed)
    assert len(document["images"]) == 3 and len(document["textures"]) == 3


def test_the_mesh_is_untouched():
    before, after = parse_glb(RIGGED), parse_glb(gs.dress(RIGGED, SKIN))
    view = lambda doc, binary: binary[doc["bufferViews"][doc["accessors"][0]["bufferView"]]["byteOffset"]:][:36]
    assert view(*before) == view(*after)
    assert after[0]["meshes"] == before[0]["meshes"]


def test_webp_pictures_from_finish_survive_both_ways():
    webp_model, webp_skin = _glb(b"OLD", b"OLD-W", webp_shine=True), _glb(b"NEW", b"NEW-W", webp_shine=True)
    dressed = gs.dress(webp_model, webp_skin)
    assert _pictures(dressed)["metallicRoughnessTexture"] == b"NEW-W"
    document, _ = parse_glb(dressed)
    assert gs.WEBP in document["extensionsRequired"]
    # a plain model dressed in a WebP skin starts requiring WebP, or viewers would skip the picture
    plain = gs.dress(RIGGED, webp_skin)
    assert gs.WEBP in parse_glb(plain)[0]["extensionsRequired"]


def test_one_file_holds_every_skin_as_a_named_variant():
    purple = _glb(b"PURPLE", b"PURPLE-SHINE")
    data = gs.with_variants(RIGGED, [("gold skirt", SKIN), ("purple", purple)])
    assert gs.variant_names(data) == ["Original", "gold skirt", "purple"]
    document, binary = parse_glb(data)
    mappings = document["meshes"][0]["primitives"][0]["extensions"][gs.VARIANTS]["mappings"]
    by_variant = {v: m["material"] for m in mappings for v in m["variants"]}
    paint = lambda material: gs._image_bytes(document, binary, gs.paint_images(document, material)["baseColorTexture"])[0]
    assert [paint(by_variant[i]) for i in range(3)] == [b"MODEL-OWN-PAINT", b"GOLD-SKIRT-COLOUR", b"PURPLE"]
    assert document["meshes"][0]["primitives"][0]["material"] == 0       # opens in its own paint
    assert gs.VARIANTS in document["extensionsUsed"]


def test_a_skin_without_paint_is_refused():
    bare = parse_glb(SKIN)
    bare[0]["materials"][0]["pbrMetallicRoughness"] = {}
    with pytest.raises(ValueError, match="no paint"):
        gs.dress(RIGGED, build_glb(*bare))


def test_each_skin_adds_only_its_own_two_pictures():
    """2026-10-19: the second skin also copied the first skin's material, so the knight's
    all-skins file carried the gold pictures twice (5 MB too many)."""
    skins = [("a", _glb(b"A-PAINT", b"A-SHINE")), ("b", _glb(b"B-PAINT", b"B-SHINE")),
             ("c", _glb(b"C-PAINT", b"C-SHINE"))]
    document, _ = parse_glb(gs.with_variants(RIGGED, skins))
    assert len(document["materials"]) == 4                     # the original + one per skin
    assert len(document["images"]) == 3 + 2 * 3                # its own three + two per skin


def _two_part(body: bytes, head: bytes) -> bytes:
    """Body (material 0) and head (material 1), each with its own colour and shine picture."""
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    blobs = [positions, b"BODY-NORMAL", body, b"BODY-SHINE", b"HEAD-NORMAL", head, b"HEAD-SHINE"]
    binary, views = b"", []
    for blob in blobs:
        binary += b"\0" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0},
                                   {"attributes": {"POSITION": 0}, "material": 1}]}],
        "materials": [
            {"name": "body", "normalTexture": {"index": 0},
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 1},
                                      "metallicRoughnessTexture": {"index": 2}}},
            {"name": "head", "normalTexture": {"index": 3},
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 4},
                                      "metallicRoughnessTexture": {"index": 5}}},
        ],
        "samplers": [{}],
        "textures": [{"source": i, "sampler": 0} for i in range(6)],
        "images": [{"bufferView": i + 1, "mimeType": "image/png"} for i in range(6)],
    }
    return build_glb(document, binary)


def _colour(data: bytes, material: int) -> bytes:
    document, binary = parse_glb(data)
    return gs._image_bytes(document, binary, gs.paint_images(document, material)["baseColorTexture"])[0]


def test_a_body_skin_does_not_repaint_the_head_when_dressing():
    """A skin made for the body only must leave the head's own paint alone."""
    dressed = gs.dress(_two_part(b"BODY-COLOUR", b"HEAD-COLOUR"), _glb(b"GOLD-BODY", b"GOLD-SHINE"))
    assert _colour(dressed, 0) == b"GOLD-BODY"
    assert _colour(dressed, 1) == b"HEAD-COLOUR"


def test_a_body_skin_does_not_repaint_the_head_in_the_all_skins_file():
    model = _two_part(b"BODY-COLOUR", b"HEAD-COLOUR")
    packed, binary = parse_glb(gs.with_variants(model, [("gold", _glb(b"GOLD-BODY", b"GOLD-SHINE"))]))
    head_copy = 3   # variant 1's materials are the copies after the two originals: body 2, head 3
    assert packed["materials"][head_copy]["name"].startswith("head")
    image = gs.paint_images(packed, head_copy)["baseColorTexture"]
    assert gs._image_bytes(packed, binary, image)[0] == b"HEAD-COLOUR"
