from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

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
    changed = _normalize_glb_material(glb, "matte")
    assert changed == 3

    material = _read_materials(glb)[0]
    assert material["alphaMode"] == "OPAQUE"
    pbr = material["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 0.0
    assert pbr["roughnessFactor"] == 1.0
    assert "metallicRoughnessTexture" not in pbr
    # The base color texture that carries the baked albedo must survive untouched.
    assert pbr["baseColorTexture"] == {"index": 0}


def test_default_mode_keeps_the_metallic_roughness_map(tmp_path):
    """The default must be `pbr` — matte silently discarded a map TRELLIS had produced.

    Pinned as its own test because the default is the whole defect: our maps match the
    reference implementation's values closely, so shipping flat was purely self-inflicted.
    """
    glb = _make_glb(
        tmp_path / "d.glb",
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
    changed = _normalize_glb_material(glb)  # no mode: exercise the default
    assert changed == 1  # only alphaMode, the actual glass-look bug

    pbr = _read_materials(glb)[0]["pbrMetallicRoughness"]
    assert pbr["metallicRoughnessTexture"] == {"index": 1}
    assert pbr["metallicFactor"] == 1.0


def test_normalize_material_pbr_mode_keeps_metalness(tmp_path):
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
    changed = _normalize_glb_material(glb, mode="pbr")
    assert changed == 1  # only alphaMode

    material = _read_materials(glb)[0]
    assert material["alphaMode"] == "OPAQUE"
    pbr = material["pbrMetallicRoughness"]
    assert pbr["metallicFactor"] == 1.0
    assert pbr["metallicRoughnessTexture"] == {"index": 1}


def test_normalize_material_rejects_unknown_mode(tmp_path):
    glb = _make_glb(tmp_path / "m.glb", [{"alphaMode": "OPAQUE"}])
    with pytest.raises(ValueError):
        _normalize_glb_material(glb, mode="glossy")


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


# --- the generate.py argv ----------------------------------------------------------

def _opts(**kw):
    from image_to_3dlab.trellis_backend import TrellisOptions
    return TrellisOptions(repo=Path("/repo"), **kw)


def test_command_omits_remesh_by_default():
    from image_to_3dlab.trellis_backend import build_generate_command

    cmd = build_generate_command(
        Path("/py"), Path("/gen.py"), [Path("a.png")], Path("/out"), _opts()
    )
    assert "--remesh" not in cmd


def test_command_passes_remesh_and_an_explicit_project():
    """The port defaults remesh_project to 0.9; we must never inherit that silently.

    At 0.9 the Forest Variant went from 6,008 to 16,954 boundary edges, so the value has
    to travel with the flag rather than being left to the vendored default.
    """
    from image_to_3dlab.trellis_backend import build_generate_command

    cmd = build_generate_command(
        Path("/py"), Path("/gen.py"), [Path("a.png")], Path("/out"),
        _opts(remesh=True, remesh_project=0.0),
    )
    assert "--remesh" in cmd
    assert cmd[cmd.index("--remesh-project") + 1] == "0.0"
    assert cmd[cmd.index("--remesh-band") + 1] == "1.0"


def test_command_carries_the_core_settings():
    from image_to_3dlab.trellis_backend import build_generate_command

    cmd = build_generate_command(
        Path("/py"), Path("/gen.py"), [Path("a.png"), Path("b.png")], Path("/out"),
        _opts(seed=42, pipeline_type="1024_cascade", texture_size=3072,
              bake_target_faces=300000, steps=12),
    )
    assert cmd[cmd.index("--seed") + 1] == "42"
    assert cmd[cmd.index("--pipeline-type") + 1] == "1024_cascade"
    assert cmd[cmd.index("--bake-target-faces") + 1] == "300000"
    assert cmd[cmd.index("--steps") + 1] == "12"
    # Both views must reach the generator; a dropped view silently halves the conditioning.
    assert "a.png" in cmd and "b.png" in cmd


def test_steps_omitted_when_unset():
    from image_to_3dlab.trellis_backend import build_generate_command

    cmd = build_generate_command(
        Path("/py"), Path("/gen.py"), [Path("a.png")], Path("/out"), _opts(steps=None)
    )
    assert "--steps" not in cmd


# --- pre_simplify_cap: a half-wired knob that crashed every manifest run -------------
#
# cli.py read args.trellis_pre_simplify_cap and passed it to TrellisOptions(...)
# unconditionally, but neither an argparse argument nor the dataclass field existed.
# Any manifest lacking 'pre_simplify_cap' aborted with AttributeError before generation,
# and adding only the arg would then have raised TypeError on TrellisOptions. Both
# surfaces are pinned so the crash cannot come back.

def test_trellis_options_accepts_pre_simplify_cap():
    assert _opts().pre_simplify_cap is None
    assert _opts(pre_simplify_cap=4_000_000).pre_simplify_cap == 4_000_000


def test_cli_defines_pre_simplify_cap_so_the_attr_always_exists():
    from image_to_3dlab.cli import build_parser

    # The exact failing shape: a run with no explicit cap must still define the attribute.
    args = build_parser().parse_args(["img.png", "--trellis"])
    assert args.trellis_pre_simplify_cap is None

    args = build_parser().parse_args(
        ["img.png", "--trellis", "--trellis-pre-simplify-cap", "4000000"]
    )
    assert args.trellis_pre_simplify_cap == 4_000_000
