from __future__ import annotations

import json
import struct

from image_to_3dlab.trellis_backend import _normalize_glb_material

_GLB_MAGIC = 0x46546C67
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942


def _make_glb(path, materials):
    gltf = {"asset": {"version": "2.0"}, "materials": materials}
    payload = json.dumps(gltf).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    # A trailing (empty) BIN chunk exercises the multi-chunk write path.
    body = (
        struct.pack("<II", len(payload), _JSON_CHUNK)
        + payload
        + struct.pack("<II", 0, _BIN_CHUNK)
    )
    path.write_bytes(struct.pack("<III", _GLB_MAGIC, 2, 12 + len(body)) + body)
    return path


def _read_materials(path):
    data = path.read_bytes()
    length = struct.unpack_from("<III", data, 0)[2]
    offset = 12
    while offset < length:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == _JSON_CHUNK:
            return json.loads(chunk.decode("utf-8"))["materials"]
        offset += 8 + chunk_len
    raise AssertionError("no JSON chunk")


def test_normalize_material_forces_opaque_matte(tmp_path):
    glb = _make_glb(
        tmp_path / "m.glb",
        [
            {
                "alphaMode": "BLEND",
                "pbrMetallicRoughness": {
                    "metallicFactor": 1.0,
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {"index": 1},
                },
            }
        ],
    )
    changed = _normalize_glb_material(glb)
    assert changed == 3

    material = _read_materials(glb)[0]
    assert material["alphaMode"] == "OPAQUE"
    pbr = material["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 0.0
    assert pbr["roughnessFactor"] == 1.0
    assert "metallicRoughnessTexture" not in pbr
    # The base color texture that carries the baked albedo must survive untouched.
    assert pbr["baseColorTexture"] == {"index": 0}


def test_normalize_material_is_idempotent(tmp_path):
    glb = _make_glb(
        tmp_path / "m.glb",
        [
            {
                "alphaMode": "OPAQUE",
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
            }
        ],
    )
    assert _normalize_glb_material(glb) == 0
