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


def _glb(images, *, material=None, extra_view=b"MESHDATA", textures=None, samplers=None):
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
        "textures": textures if textures is not None
                    else [{"source": i} for i in range(len(images))],
        "materials": [material or {"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0}}}],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": mesh_view, "componentType": 5126, "count": 1,
                       "type": "VEC3"}],
    }
    if samplers is not None:
        document["samplers"] = samplers
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
        encoded, used = cg.encode(image, "jpeg", 90, None)
    assert used == "png"
    with Image.open(io.BytesIO(encoded)) as result:
        assert result.format == "PNG"
        assert result.mode in ("RGBA", "LA", "P")


def test_an_alpha_albedo_stays_labelled_png_when_jpeg_was_asked():
    image = Image.new("RGBA", (128, 128), (10, 20, 30, 128))
    blob = io.BytesIO()
    image.save(blob, "PNG", compress_level=0)
    data, _ = _glb([blob.getvalue()])
    out, report = cg.compress(data, fmt="jpeg")
    assert report[0]["skipped"] is False
    document, binary = cg.parse_glb(out)
    assert document["images"][0]["mimeType"] == "image/png"
    view = document["bufferViews"][0]
    payload = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    with Image.open(io.BytesIO(payload)) as result:
        assert result.format == "PNG"


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


# --- a game budget at one size, and dropping the surface maps (#239) ------------------------

def _three_maps(*, samplers=None, textures=None):
    """A finished model's material: colour, metal-roughness and normal, at mixed sizes."""
    material = {"pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1},
    }, "normalTexture": {"index": 2}}
    return _glb([_png((512, 512)), _png((1024, 1024), seed=3), _png((1024, 1024), seed=4)],
                material=material, textures=textures, samplers=samplers)


def _size_of(document, binary, view_index):
    view = document["bufferViews"][view_index]
    blob = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    with Image.open(io.BytesIO(blob)) as image:
        return max(image.size)


def test_the_tool_caps_every_map_at_the_size_it_is_given():
    """The size policy is Finish's (retopo_repaint.texture_arguments); the tool obeys it.

    Called with one number for all three, all three come out at it, so a caller that means
    "512 everywhere" gets exactly that.
    """
    data, _ = _three_maps()
    out, _report = cg.compress(data, max_size=256, map_size=256, normal_size=256)
    document, binary = cg.parse_glb(out)
    assert [_size_of(document, binary, i) for i in (0, 1, 2)] == [256, 256, 256]


def test_dropping_the_surface_maps_leaves_one_colour_atlas():
    data, _ = _three_maps()
    out, report = cg.compress(data, max_size=512, map_size=512, normal_size=512,
                              drop_surface_maps=True)
    document, binary = cg.parse_glb(out)

    assert len(document["images"]) == 1
    assert len(document["textures"]) == 1
    assert len(document["bufferViews"]) == 2          # the colour atlas and the mesh
    assert [row["kind"] for row in report] == ["colour"]
    assert _size_of(document, binary, 0) == 512


def test_a_dropped_map_leaves_a_valid_flat_material():
    data, _ = _three_maps()
    out, _ = cg.compress(data, drop_surface_maps=True, roughness=0.4)
    document, _binary = cg.parse_glb(out)

    material = document["materials"][0]
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert "metallicRoughnessTexture" not in material["pbrMetallicRoughness"]
    assert material["pbrMetallicRoughness"]["metallicFactor"] == 0
    assert material["pbrMetallicRoughness"]["roughnessFactor"] == pytest.approx(0.4)
    assert "normalTexture" not in material and "occlusionTexture" not in material


def test_a_dropped_map_leaves_no_orphan_texture_image_sampler_or_bytes():
    """Every index in the file must still point at something, or readers drop the asset."""
    samplers = [{"magFilter": 9729}, {"magFilter": 9728}, {"minFilter": 9986}]
    textures = [{"source": 0, "sampler": 0}, {"source": 1, "sampler": 1},
                {"source": 2, "sampler": 2}]
    data, _mesh_view = _three_maps(samplers=samplers, textures=textures)
    out, _ = cg.compress(data, drop_surface_maps=True)
    document, binary = cg.parse_glb(out)

    assert len(document["textures"]) == len(document["images"]) == len(document["samplers"]) == 1
    assert document["samplers"] == [{"magFilter": 9729}]
    assert document["textures"] == [{"source": 0, "sampler": 0}]
    for view in document["bufferViews"]:
        assert view["byteOffset"] + view["byteLength"] <= len(binary)
    mesh = document["bufferViews"][document["accessors"][0]["bufferView"]]
    assert binary[mesh["byteOffset"]:mesh["byteOffset"] + mesh["byteLength"]] == b"MESHDATA"


def test_an_image_shared_by_colour_and_a_dropped_map_stays():
    """Packing the normal into the albedo's image is legal; dropping the slot must not take it."""
    material = {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1}}
    # Both textures point at image 0: the albedo's.
    data, _ = _glb([_png((128, 128))], material=material,
                   textures=[{"source": 0}, {"source": 0}])
    out, _ = cg.compress(data, drop_surface_maps=True)
    document, _binary = cg.parse_glb(out)

    assert len(document["images"]) == 1
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert "normalTexture" not in document["materials"][0]


def test_dropping_still_re_encodes_the_colour_map():
    data, _ = _three_maps()
    out, report = cg.compress(data, max_size=128, map_size=128, normal_size=128,
                              drop_surface_maps=True)
    assert report[0]["skipped"] is False
    document, binary = cg.parse_glb(out)
    assert _size_of(document, binary, 0) == 128


def test_keeping_the_surface_maps_leaves_them_alone():
    data, _ = _three_maps()
    out, _ = cg.compress(data, max_size=512, map_size=512, normal_size=512,
                         drop_surface_maps=False)
    document, _binary = cg.parse_glb(out)
    assert len(document["images"]) == 3
    assert "normalTexture" in document["materials"][0]


def _real_glb(material):
    """A GLB a real reader can open: one triangle, and images where the material says."""
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
    binary = bytearray(positions)
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}]
    for blob in (_png((128, 128)), _png((256, 256), seed=5), _png((256, 256), seed=6)):
        binary += b"\x00" * (-len(binary) % 4)
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)})
        binary += blob
    document = {
        "asset": {"version": "2.0"},
        "images": [{"bufferView": i + 1, "mimeType": "image/png"} for i in range(3)],
        "textures": [{"source": i} for i in range(3)],
        "materials": [material],
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
    }
    return cg.build_glb(document, bytes(binary))


def test_a_real_gltf_reader_loads_the_dropped_model():
    """The whole point is a file a game engine opens: trimesh must find one valid material."""
    trimesh = pytest.importorskip("trimesh")

    material = {"pbrMetallicRoughness": {
        "baseColorTexture": {"index": 0},
        "metallicRoughnessTexture": {"index": 1},
    }, "normalTexture": {"index": 2}}
    data = _real_glb(material)
    out, _ = cg.compress(data, max_size=128, map_size=128, normal_size=128,
                         drop_surface_maps=True, roughness=0.5)
    scene = trimesh.load(io.BytesIO(out), file_type="glb", force="scene")

    (mesh,) = scene.geometry.values()
    loaded = mesh.visual.material
    assert loaded.baseColorTexture is not None
    assert getattr(loaded, "normalTexture", None) is None
    assert getattr(loaded, "metallicRoughnessTexture", None) is None
    assert loaded.metallicFactor == 0
    assert loaded.roughnessFactor == pytest.approx(0.5)
