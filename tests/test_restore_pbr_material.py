"""Tests for re-attaching the metallicRoughness map that matte mode orphaned.

The subtle one is `metallicFactor`. In glTF the factor multiplies the texture, so restoring
the map while leaving the factor at matte's 0.0 scales it to nothing — the file would look
fixed under inspection and render identically. That is the failure this suite exists for.

The other is `find_orphan_texture` refusing to guess. Wiring a base-colour image into the
metalness channel is invisible in a viewport and wrong in an engine, so "several orphans"
must raise rather than pick one.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "restore_pbr_material.py"


def _load():
    spec = importlib.util.spec_from_file_location("restore_pbr_material", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rpm = _load()


def matte_gltf() -> dict:
    """Exactly what `--material-mode matte` leaves: two textures, one referenced."""
    return {
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "OPAQUE",
                "doubleSided": True,
            }
        ],
        "textures": [{"source": 0}, {"source": 1}],
        "images": [{"mimeType": "image/png"}, {"mimeType": "image/png"}],
    }


# --- orphan detection --------------------------------------------------------------

def test_finds_the_single_orphaned_texture():
    assert rpm.find_orphan_texture(matte_gltf()) == 1


def test_no_orphan_when_everything_is_referenced():
    gltf = matte_gltf()
    gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 1}
    assert rpm.find_orphan_texture(gltf) is None


def test_refuses_to_guess_between_multiple_orphans():
    gltf = matte_gltf()
    gltf["textures"].append({"source": 2})
    assert rpm.find_orphan_texture(gltf) is None
    with pytest.raises(ValueError, match="--mr-texture"):
        rpm.restore_material(gltf)


def test_referenced_textures_covers_non_pbr_slots():
    gltf = {
        "materials": [
            {
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}},
                "normalTexture": {"index": 1},
                "occlusionTexture": {"index": 2},
                "emissiveTexture": {"index": 3},
            }
        ]
    }
    assert rpm.referenced_textures(gltf) == {0, 1, 2, 3}


# --- the restore itself ------------------------------------------------------------

def test_restores_map_and_raises_metallic_factor():
    gltf = matte_gltf()
    changes = rpm.restore_material(gltf)
    pbr = gltf["materials"][0]["pbrMetallicRoughness"]

    assert pbr["metallicRoughnessTexture"] == {"index": 1}
    # The whole point: a restored map multiplied by 0.0 is still flat.
    assert pbr["metallicFactor"] == 1.0
    assert pbr["roughnessFactor"] == 1.0
    assert gltf["materials"][0]["doubleSided"] is False
    assert len(changes) == 3


def test_keeps_double_sided_when_asked():
    gltf = matte_gltf()
    rpm.restore_material(gltf, single_sided=False)
    assert gltf["materials"][0]["doubleSided"] is True


def test_is_idempotent():
    gltf = matte_gltf()
    rpm.restore_material(gltf)
    assert rpm.restore_material(gltf, mr_texture=1) == []


def test_explicit_index_overrides_detection():
    gltf = matte_gltf()
    gltf["textures"].append({"source": 2})
    rpm.restore_material(gltf, mr_texture=2)
    assert gltf["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] == 2


def test_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        rpm.restore_material(matte_gltf(), mr_texture=99)


def test_rejects_a_gltf_with_no_materials():
    with pytest.raises(ValueError, match="no materials"):
        rpm.restore_material({"textures": [{}, {}]})


# --- the real artifact -------------------------------------------------------------

def _build_glb(gltf: dict, binary: bytes = b"\x00\x01\x02\x03") -> bytes:
    """A minimal but valid GLB, so the container round-trip is tested for real."""
    json_chunk = json.dumps(gltf).encode()
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    bin_chunk = binary + b"\x00" * ((4 - len(binary) % 4) % 4)
    body = (
        struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
        + struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body


def test_roundtrip_preserves_the_binary_chunk(tmp_path):
    """The texture images live in the binary chunk and must survive byte-for-byte.

    This is the property the whole approach rests on: matte only ever rewrote JSON, so the
    orphaned image is still in the file and a JSON edit is enough to get it back.
    """
    from image_to_3dlab.trellis_backend import read_glb, write_glb

    payload = bytes(range(256)) * 4
    path = tmp_path / "asset.glb"
    path.write_bytes(_build_glb(matte_gltf(), payload))

    gltf, chunks, json_index, version = read_glb(path)
    rpm.restore_material(gltf)
    write_glb(path, gltf, chunks, json_index, version)

    gltf2, chunks2, json_index2, _ = read_glb(path)
    binary = next(c[1] for i, c in enumerate(chunks2) if i != json_index2)
    assert binary.startswith(payload)
    assert gltf2["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] == {
        "index": 1
    }


# --- roughness scaling -------------------------------------------------------------

def test_roughness_scale_sets_the_factor():
    """The factor multiplies the map, so it is the gloss knob without touching texels."""
    gltf = matte_gltf()
    rpm.restore_material(gltf, roughness_scale=0.6)
    assert gltf["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 0.6


def test_roughness_scale_defaults_to_unity():
    gltf = matte_gltf()
    rpm.restore_material(gltf)
    assert gltf["materials"][0]["pbrMetallicRoughness"]["roughnessFactor"] == 1.0


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_roughness_scale_out_of_range_is_refused(bad):
    """Zero would make a mirror of the whole asset; above 1 pushes past fully rough."""
    with pytest.raises(ValueError, match="roughness_scale"):
        rpm.restore_material(matte_gltf(), roughness_scale=bad)


def test_idempotent_at_a_scaled_roughness():
    gltf = matte_gltf()
    rpm.restore_material(gltf, roughness_scale=0.6)
    assert rpm.restore_material(gltf, mr_texture=1, roughness_scale=0.6) == []
