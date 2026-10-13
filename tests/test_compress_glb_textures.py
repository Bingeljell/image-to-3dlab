"""Tests for re-encoding a GLB's textures.

The size problem this solves was hiding behind the wrong number. Retopology cut the
Snag's geometry from 11.8 MB to 1.6 MB and the file still weighed 32 MB: the paint stage
writes a 4096x4096 albedo and a 4096x4096 metallic-roughness map as uncompressed PNG.

Two things must hold or this quietly breaks assets:

* **The container has to survive.** Every bufferView is rebuilt, so accessors, meshes and
  materials must still point at the right bytes afterwards.
* **A data map is not a colour map.** Metallic-roughness, normal and occlusion textures
  are read as numbers, so they are treated separately from the albedo.
"""

from __future__ import annotations

import importlib.util
import io
import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compress_glb_textures.py"


def _load():
    spec = importlib.util.spec_from_file_location("compress_glb_textures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cg = _load()


def _png(size=(64, 64), seed=0, mode="RGB"):
    """A noisy image, so PNG cannot trivially crush it and sizes stay meaningful."""
    rng = np.random.default_rng(seed)
    channels = 4 if mode == "RGBA" else 3
    pixels = rng.integers(0, 255, size=(size[1], size[0], channels), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode=mode).save(buffer, "PNG")
    return buffer.getvalue()


def _glb(images, *, material=None, extra_view=b"MESHDATA"):
    """Build a minimal but structurally real GLB carrying the given image blobs."""
    binary = bytearray()
    views = []
    for blob in images:
        binary += b"\x00" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob
    binary += b"\x00" * (-len(binary) % 4)
    mesh_view = len(views)
    views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(extra_view)})
    binary += extra_view

    document = {
        "asset": {"version": "2.0"},
        "images": [{"bufferView": i, "mimeType": "image/png"} for i in range(len(images))],
        "textures": [{"source": i} for i in range(len(images))],
        "materials": [material or {"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}}}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": mesh_view, "componentType": 5126, "count": 1,
                       "type": "VEC3"}],
    }
    return cg.build_glb(document, bytes(binary)), mesh_view


def test_round_trips_a_glb_unchanged_when_nothing_is_re_encoded():
    data, _ = _glb([])
    out, report = cg.compress(data)
    assert report == []
    assert out == data


def test_a_png_texture_gets_smaller():
    data, _ = _glb([_png((256, 256))])
    out, report = cg.compress(data, fmt="webp", quality=80)
    assert len(out) < len(data)
    assert report[0]["after"] < report[0]["before"]

    document, _binary = cg.parse_glb(out)
    assert document["images"][0]["mimeType"] == "image/webp"


def test_the_mesh_bytes_survive_the_rebuild():
    """The bug that would matter most: shifting image bytes must not move mesh data."""
    data, mesh_view = _glb([_png((128, 128)), _png((128, 128), seed=1)])
    out, _report = cg.compress(data)

    document, binary = cg.parse_glb(out)
    view = document["bufferViews"][mesh_view]
    start = view["byteOffset"]
    assert binary[start : start + view["byteLength"]] == b"MESHDATA"
    assert document["buffers"][0]["byteLength"] == len(binary)


def test_every_buffer_view_stays_inside_the_buffer():
    data, _ = _glb([_png((96, 96)), _png((96, 96), seed=2)])
    out, _report = cg.compress(data)

    document, binary = cg.parse_glb(out)
    for view in document["bufferViews"]:
        assert view["byteOffset"] + view["byteLength"] <= len(binary)
        assert view["byteOffset"] % 4 == 0


def test_a_metallic_roughness_map_is_treated_as_data():
    material = {"pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1},
    }}
    data, _ = _glb([_png((512, 512)), _png((512, 512), seed=3)], material=material)

    document, _binary = cg.parse_glb(data)
    assert cg.data_texture_images(document) == {1}

    out, report = cg.compress(data, map_size=64, max_size=None)
    assert report[0]["kind"] == "colour"
    assert report[1]["kind"] == "data"
    # The data map was capped at 64px and the albedo was not touched in resolution.
    document, binary = cg.parse_glb(out)
    view = document["bufferViews"][1]
    with Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])) as m:
        assert max(m.size) == 64
    view = document["bufferViews"][0]
    with Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])) as a:
        assert max(a.size) == 512


def test_the_default_format_is_core_gltf():
    """JPEG needs no extension, so it survives every converter on the way to an engine."""
    data, _ = _glb([_png((256, 256))])
    out, _report = cg.compress(data)

    document, _binary = cg.parse_glb(out)
    assert document["images"][0]["mimeType"] == "image/jpeg"
    assert "extensionsUsed" not in document


def test_colour_textures_are_capped_at_2048_by_default():
    """4096 is what the paint stage emits and more than these assets carry."""
    data, _ = _glb([_png((4096, 4096))])
    out, _report = cg.compress(data)

    document, binary = cg.parse_glb(out)
    view = document["bufferViews"][0]
    blob = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    with Image.open(io.BytesIO(blob)) as image:
        assert max(image.size) == 2048


def test_normal_and_occlusion_slots_count_as_data_too():
    material = {
        "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
        "normalTexture": {"index": 1},
        "occlusionTexture": {"index": 2},
    }
    data, _ = _glb([_png(), _png(seed=4), _png(seed=5)], material=material)
    document, _binary = cg.parse_glb(data)
    assert cg.data_texture_images(document) == {1, 2}


def test_webp_is_declared_used_but_never_required():
    """extensionsRequired would let a viewer refuse the file outright."""
    data, _ = _glb([_png((128, 128))])
    out, _report = cg.compress(data, fmt="webp")

    document, _binary = cg.parse_glb(out)
    assert "EXT_texture_webp" in document["extensionsUsed"]
    assert "EXT_texture_webp" not in document.get("extensionsRequired", [])
    texture = document["textures"][0]
    # The fallback source stays, so an unaware viewer still has an image to load.
    assert texture["source"] == 0
    assert texture["extensions"]["EXT_texture_webp"]["source"] == 0


def test_jpeg_keeps_alpha_by_falling_back_to_png():
    """Silently dropping alpha would turn a cutout opaque."""
    with Image.open(io.BytesIO(_png((64, 64), mode="RGBA"))) as image:
        encoded = cg.encode(image, "jpeg", 90, None)
    with Image.open(io.BytesIO(encoded)) as result:
        assert result.format == "PNG"
        assert result.mode in ("RGBA", "LA", "P")


def test_a_texture_that_would_grow_is_left_alone():
    """Re-encoding can inflate an already-compact texture; growing an asset is never the job."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    compact = io.BytesIO()
    Image.fromarray(noise, mode="RGB").save(compact, "WEBP", quality=60, method=6)
    data, _ = _glb([compact.getvalue()])

    # PNG of noise is far larger than a lossy WebP of it, so the re-encode must be refused.
    out, report = cg.compress(data, fmt="png")
    assert report[0]["skipped"] is True
    assert out == data


def test_the_json_chunk_stays_valid_json_and_padded():
    data, _ = _glb([_png((128, 128))])
    out, _report = cg.compress(data)

    length, kind = struct.unpack_from("<II", out, 12)
    assert kind == cg.JSON_CHUNK
    assert length % 4 == 0
    json.loads(out[20 : 20 + length])


def test_the_normal_map_keeps_more_resolution_than_the_other_data_maps():
    """A normal map carries the edges of every block and bevel, so it gets its own, larger cap."""
    material = {"pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1},
    }, "normalTexture": {"index": 2}}
    data, _ = _glb([_png((512, 512)), _png((512, 512), seed=3), _png((512, 512), seed=4)], material=material)

    document, _binary = cg.parse_glb(data)
    assert cg.normal_texture_images(document) == {2}

    out, _report = cg.compress(data, map_size=64, normal_size=256, max_size=None)
    document, binary = cg.parse_glb(out)

    def side(index):
        view = document["bufferViews"][index]
        with Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])) as image:
            return max(image.size)

    assert (side(0), side(1), side(2)) == (512, 64, 256)      # colour untouched, roughness small, normal bigger


def test_without_a_normal_size_the_normal_map_follows_the_other_data_maps():
    material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}, "normalTexture": {"index": 1}}
    data, _ = _glb([_png((512, 512)), _png((512, 512), seed=3)], material=material)
    out, _ = cg.compress(data, map_size=64, normal_size=None, max_size=None)
    document, binary = cg.parse_glb(out)
    view = document["bufferViews"][1]
    with Image.open(io.BytesIO(binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]])) as image:
        assert max(image.size) == 64
