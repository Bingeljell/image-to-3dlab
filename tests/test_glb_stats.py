from __future__ import annotations

import io
import struct

import pytest
from PIL import Image

from image_to_3dlab.glb_stats import glb_stats, image_size
from image_to_3dlab.glb_turn import build_glb


def _png(width: int, height: int) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height)).save(out, "PNG")
    return out.getvalue()


def _model(*, indexed: bool = True, instances: int = 1, texture: bytes | None = None,
           joints: int = 0, moves: tuple[str, ...] = ()) -> bytes:
    """A GLB with one mesh of 4 vertices: 2 triangles if indexed, 1 (from 3 of 4) if not."""
    binary = b""
    views: list[dict] = []
    accessors = [{"count": 4 if indexed else 3, "componentType": 5126, "type": "VEC3"}]
    primitive: dict = {"attributes": {"POSITION": 0}}
    if indexed:
        accessors.append({"count": 6, "componentType": 5123, "type": "SCALAR"})
        primitive["indices"] = 1
    document: dict = {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [primitive]}],
        "nodes": [{"mesh": 0} for _ in range(instances)],
        "accessors": accessors,
        "materials": [{"name": "skin"}],
    }
    if texture is not None:
        views.append({"buffer": 0, "byteOffset": 0, "byteLength": len(texture)})
        binary += texture
        document["images"] = [{"bufferView": 0, "mimeType": "image/png"}]
        document["textures"] = [{"source": 0}]
    if joints:
        document["nodes"] += [{"name": f"bone{i}"} for i in range(joints)]
        first = instances
        document["skins"] = [{"joints": list(range(first, first + joints))}]
    if moves:
        document["animations"] = [{"name": name, "channels": [], "samplers": []} for name in moves]
    if views:
        document["bufferViews"] = views
        document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, binary)


def test_an_indexed_mesh_counts_triangles_from_its_indices():
    stats = glb_stats(_model())
    assert stats["triangles"] == 2
    assert stats["vertices"] == 4


def test_an_unindexed_mesh_counts_every_three_vertices_as_one_triangle():
    assert glb_stats(_model(indexed=False))["triangles"] == 1


def test_a_mesh_used_twice_counts_twice():
    """What a game engine draws, which is the number a game dev cares about."""
    stats = glb_stats(_model(instances=2))
    assert stats["triangles"] == 4 and stats["vertices"] == 8


def test_texture_resolution_is_read_from_the_embedded_image():
    stats = glb_stats(_model(texture=_png(64, 32)))
    assert stats["textures"] == [{"width": 64, "height": 32, "format": "png"}]


def test_bones_and_moves_are_listed():
    stats = glb_stats(_model(joints=3, moves=("idle", "walk")))
    assert stats["bones"] == 3
    assert stats["moves"] == ["idle", "walk"]


def test_a_plain_model_has_no_bones_and_no_moves():
    stats = glb_stats(_model())
    assert stats["bones"] == 0 and stats["moves"] == [] and stats["materials"] == 1


def test_file_size_is_the_whole_file():
    data = _model()
    assert glb_stats(data)["bytes"] == len(data)


def test_a_ktx2_texture_reports_its_size_without_decoding_it():
    """gltfpack can write KTX2, which Pillow cannot open; its header still says the size."""
    header = b"\xabKTX 20\xbb\r\n\x1a\n" + struct.pack("<IIII", 0, 1, 512, 256)
    assert image_size(header + b"\0" * 32, "image/ktx2") == (512, 256, "ktx2")


def test_an_unreadable_texture_is_reported_without_a_size():
    assert image_size(b"not an image", "image/png") == (None, None, "png")


def test_a_file_that_is_not_a_glb_is_refused():
    with pytest.raises(ValueError):
        glb_stats(b"hello")
