from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import roblox_export as rx
from image_to_3dlab.glb_turn import build_glb, parse_glb


def _picture(fmt: str, size: int = 8, colour=(10, 200, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(out, fmt)
    return out.getvalue()


def _model(*, triangles: int = 2, meshes: int = 1, pictures: list[tuple[str, bytes]] | None = None,
           webp_texture: bool = False, packed: str | None = None, joints_1: bool = False,
           height: float = 1.8) -> bytes:
    """A GLB of `meshes` meshes, each `triangles` indexed triangles, `height` metres tall.

    pictures: (mimeType, bytes) in order; texture 0 is colour, 1 normal, 2 metal-roughness."""
    binary = b""
    views: list[dict] = []
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, height, 0)
    views.append({"buffer": 0, "byteOffset": 0, "byteLength": len(positions)})
    binary += positions
    accessors = [{"bufferView": 0, "count": 3, "componentType": 5126, "type": "VEC3",
                  "min": [0, 0, 0], "max": [1, height, 0]},
                 {"count": triangles * 3, "componentType": 5123, "type": "SCALAR"}]
    attributes = {"POSITION": 0} | ({"JOINTS_1": 0} if joints_1 else {})
    document: dict = {
        "asset": {"version": "2.0"},
        "accessors": accessors,
        "meshes": [{"name": f"part{i}", "primitives": [{"attributes": attributes, "indices": 1,
                                                        "material": 0}]} for i in range(meshes)],
        "nodes": [{"mesh": i} for i in range(meshes)],
        "scenes": [{"nodes": list(range(meshes))}],
        "materials": [{"name": "skin", "pbrMetallicRoughness": {}}],
    }
    if pictures:
        document["images"] = []
        for mime, data in pictures:
            binary += b"\0" * (-len(binary) % 4)
            views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
            binary += data
            document["images"].append({"bufferView": len(views) - 1, "mimeType": mime})
        document["textures"] = [{"source": i} for i in range(len(pictures))]
        material = document["materials"][0]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        if len(pictures) > 1:
            material["normalTexture"] = {"index": 1}
        if len(pictures) > 2:
            material["pbrMetallicRoughness"]["metallicRoughnessTexture"] = {"index": 2}
        if webp_texture:
            document["textures"][0] = {"extensions": {"EXT_texture_webp": {"source": 0}}}
            document["extensionsUsed"] = ["EXT_texture_webp"]
            document["extensionsRequired"] = ["EXT_texture_webp"]
    if packed:
        document.setdefault("extensionsUsed", []).append(packed)
    document["bufferViews"] = views
    document["buffers"] = [{"byteLength": len(binary)}]
    return build_glb(document, binary)


def _levels(checks, level):
    return [c["text"] for c in checks if c["level"] == level]


# ---------------------------------------------------------------- checks

def test_a_small_prop_passes():
    result = rx.export(_model(triangles=500))
    assert result.passed
    assert _levels(result.checks, "pass")


@pytest.mark.parametrize("extension", ["EXT_meshopt_compression", "KHR_draco_mesh_compression"])
def test_packed_geometry_is_refused_with_a_way_out(extension):
    with pytest.raises(rx.RobloxExportError, match="uncompressed file"):
        rx.export(_model(packed=extension))


def test_a_mesh_over_twenty_thousand_triangles_fails():
    checks = rx.check(parse_glb(_model(triangles=20_001))[0])
    assert any("20,000" in text for text in _levels(checks, "fail"))


def test_twenty_thousand_exactly_passes():
    checks = rx.check(parse_glb(_model(triangles=20_000))[0])
    assert not _levels(checks, "fail")


def test_the_limit_is_per_mesh_not_per_file():
    checks = rx.check(parse_glb(_model(triangles=15_000, meshes=2))[0])
    assert not _levels(checks, "fail")


def test_a_detailed_prop_gets_no_wearable_line_at_all():
    """It read as a fault on a character (user, 2026-11-16): a prop over 4,000 is simply a prop."""
    checks = rx.check(parse_glb(_model(triangles=5_000))[0], kind="prop")
    assert not any("wearable" in c["text"] for c in checks)
    assert not _levels(checks, "warn") and not _levels(checks, "fail")


def test_a_light_prop_is_told_it_could_be_a_wearable_item_too():
    checks = rx.check(parse_glb(_model(triangles=900))[0], kind="prop")
    assert any("wearable" in text for text in _levels(checks, "pass"))


def test_a_character_never_hears_about_wearable_items():
    checks = rx.check(parse_glb(_model(triangles=900))[0], kind="character")
    assert not any("wearable" in c["text"] for c in checks)


def test_a_model_with_a_skeleton_is_exported_as_a_character():
    result = rx.export(_skinned())
    assert not any("wearable" in c["text"] for c in result.checks)


def test_an_accessory_over_four_thousand_triangles_fails():
    checks = rx.check(parse_glb(_model(triangles=4_001))[0], kind="accessory")
    assert any("4,000" in text for text in _levels(checks, "fail"))


def test_an_accessory_must_be_one_mesh():
    checks = rx.check(parse_glb(_model(triangles=10, meshes=2))[0], kind="accessory")
    assert any("one mesh" in text for text in _levels(checks, "fail"))


def test_more_than_four_bones_per_point_is_warned():
    checks = rx.check(parse_glb(_model(joints_1=True))[0])
    assert any("4 bones" in text for text in _levels(checks, "warn"))


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        rx.check(parse_glb(_model())[0], kind="hat")


# ---------------------------------------------------------------- size

def test_size_is_reported_in_studs():
    result = rx.export(_model(height=1.4))
    assert result.size_studs == (round(1 / 0.28, 2), 5.0, 0.0)


def _floats(glb: bytes, accessor_index: int) -> list[float]:
    document, binary = parse_glb(glb)
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[accessor["type"]]
    return list(struct.unpack_from(f"<{accessor['count'] * width}f", binary, view["byteOffset"]))


def test_the_scale_is_baked_into_the_points_not_a_parent_node():
    """Roblox Studio ignored a scaled root node (tried 2026-11-16): the points must move."""
    glb = rx.export(_model(meshes=2)).glb
    document, _ = parse_glb(glb)
    assert len(document["nodes"]) == 2
    assert not any("scale" in node for node in document["nodes"])
    assert _floats(glb, 0)[7] == pytest.approx(1.8 / 0.28, rel=1e-5)
    assert document["accessors"][0]["max"][1] == pytest.approx(1.8 / 0.28)


def test_scaling_twice_does_not_scale_again():
    document, binary = parse_glb(rx.export(_model()).glb)
    again = rx.scale_to_studs(document, binary)
    assert again == binary


def _skinned(translation=(0.0, 1.0, 0.0), rest=(0.0, 0.0, 0.0, 1.0), turn=None,
             still_scale=False) -> bytes:
    """One triangle, one joint at `translation`, an inverse bind matrix, and a move that
    slides the joint: everything that holds a distance."""
    document, binary = parse_glb(_model())
    binary = bytearray(binary)

    def add(data: bytes, accessor: dict) -> int:
        binary.extend(b"\0" * (-len(binary) % 4))
        document["bufferViews"].append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary.extend(data)
        document["accessors"].append({"bufferView": len(document["bufferViews"]) - 1, **accessor})
        return len(document["accessors"]) - 1

    bind = np.eye(4)
    bind[:3, :3] = rx._qmat(np.array(rest))
    bind[:3, 3] = translation
    ibm = add(struct.pack("<16f", *np.linalg.inv(bind).T.reshape(-1)),
              {"count": 1, "componentType": 5126, "type": "MAT4"})
    times = add(struct.pack("<2f", 0, 1), {"count": 2, "componentType": 5126, "type": "SCALAR",
                                           "min": [0], "max": [1]})
    slide = add(struct.pack("<6f", 0, 1, 0, 0, 2, 0), {"count": 2, "componentType": 5126,
                                                      "type": "VEC3"})
    hip = {"name": "hip", "translation": list(translation), "rotation": list(rest)}
    document["nodes"].append(hip)
    document["nodes"][0]["skin"] = 0
    document["skins"] = [{"joints": [1], "inverseBindMatrices": ibm}]
    document["scenes"][0]["nodes"].append(1)
    samplers = [{"input": times, "output": slide}]
    channels = [{"sampler": 0, "target": {"node": 1, "path": "translation"}}]
    if turn is not None:
        keys = add(struct.pack("<8f", *rest, *turn), {"count": 2, "componentType": 5126, "type": "VEC4"})
        samplers.append({"input": times, "output": keys})
        channels.append({"sampler": 1, "target": {"node": 1, "path": "rotation"}})
    if still_scale:
        ones = add(struct.pack("<6f", 1, 1, 1, 1, 1, 1), {"count": 2, "componentType": 5126, "type": "VEC3"})
        samplers.append({"input": times, "output": ones})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": 1, "path": "scale"}})
    document["animations"] = [{"samplers": samplers, "channels": channels}]
    document["buffers"][0]["byteLength"] = len(binary)
    return build_glb(document, bytes(binary))


def test_joints_bind_matrices_and_moves_scale_with_the_points():
    k = 1 / 0.28
    document, binary = parse_glb(_skinned())
    glb = build_glb(document, rx.scale_to_studs(document, binary))
    assert document["nodes"][1]["translation"] == pytest.approx([0, k, 0])
    ibm = _floats(glb, document["skins"][0]["inverseBindMatrices"])
    assert ibm[13] == pytest.approx(-k, rel=1e-5)
    assert ibm[0] == 1 and ibm[5] == 1  # rotation part untouched
    sampler = document["animations"][0]["samplers"][0]
    assert _floats(glb, sampler["output"])[4] == pytest.approx(2 * k, rel=1e-5)
    assert _floats(glb, sampler["input"]) == [0, 1]  # key times are seconds, not distances


QUARTER_TURN_X = (np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4))
TILT_Z = (0.0, 0.0, np.sin(0.3), np.cos(0.3))


def _skin_matrix(glb: bytes, key: int) -> np.ndarray:
    """The hip's skinning matrix (pose @ inverse bind) at key `key` of the move."""
    document, _ = parse_glb(glb)
    hip = document["nodes"][1]
    animation = document["animations"][0]
    pose = {"translation": hip.get("translation", [0, 0, 0]), "rotation": hip.get("rotation", [0, 0, 0, 1])}
    for channel in animation["channels"]:
        values = _floats(glb, animation["samplers"][channel["sampler"]]["output"])
        width = 4 if channel["target"]["path"] == "rotation" else 3
        pose[channel["target"]["path"]] = values[key * width:(key + 1) * width]
    matrix = np.eye(4)
    matrix[:3, :3] = rx._qmat(np.array(pose["rotation"]))
    matrix[:3, 3] = pose["translation"]
    ibm = np.array(_floats(glb, document["skins"][0]["inverseBindMatrices"])).reshape(4, 4).T
    return matrix @ ibm


def test_bones_are_rebuilt_to_rest_straight_and_the_skin_moves_the_same():
    """Roblox turned a gnome's legs about the wrong axes when they rested turned (2026-11-16)."""
    source = _skinned(rest=QUARTER_TURN_X, turn=TILT_Z)
    document, binary = parse_glb(source)
    binary, note = rx.straighten_bones(document, binary)
    glb = build_glb(document, binary)
    assert document["nodes"][1]["rotation"] == pytest.approx([0, 0, 0, 1], abs=1e-6)
    for key in (0, 1):
        assert _skin_matrix(glb, key) == pytest.approx(_skin_matrix(source, key), abs=1e-5)
    assert "rest straight" in note


def test_moves_are_left_out_with_a_note():
    """Studio turned a custom skeleton's walk half round (2026-11-16): ship it still."""
    result = rx.export(_skinned(turn=TILT_Z))
    document, _ = parse_glb(result.glb)
    assert "animations" not in document
    assert document["skins"]
    assert {"level": "note", "text": rx.MOVES_LEFT_OUT} in result.checks
    assert result.passed  # standing still is not a failure


def test_a_rig_whose_moves_change_size_is_left_alone():
    document, binary = parse_glb(_skinned(still_scale=True))
    document["animations"][0]["channels"][-1]["target"]["path"] = "scale"
    binary = bytearray(binary)
    view = document["bufferViews"][document["accessors"][-1]["bufferView"]]
    binary[view["byteOffset"]:view["byteOffset"] + 4] = struct.pack("<f", 2.0)
    _, note = rx.straighten_bones(document, bytes(binary))
    assert "size" in note


# ---------------------------------------------------------------- pictures

def test_webp_pictures_become_png_and_the_extension_is_gone():
    glb = rx.export(_model(pictures=[("image/webp", _picture("WEBP"))], webp_texture=True)).glb
    document, binary = parse_glb(glb)
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["textures"][0] == {"source": 0}
    assert "extensionsUsed" not in document and "extensionsRequired" not in document
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    data = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_big_pictures_are_shrunk_to_the_limit():
    result = rx.export(_model(pictures=[("image/png", _picture("PNG", size=128))]), max_texture=64)
    document, binary = parse_glb(result.glb)
    view = document["bufferViews"][document["images"][0]["bufferView"]]
    data = binary[view["byteOffset"]:view["byteOffset"] + view["byteLength"]]
    assert Image.open(io.BytesIO(data)).size == (64, 64)
    assert any("shrunk" in note for note in result.notes)


def test_small_png_pictures_are_left_byte_for_byte():
    original = _picture("PNG")
    result = rx.export(_model(pictures=[("image/png", original)]))
    assert result.maps["color.png"] == original
    assert not result.notes


def test_ktx2_pictures_are_refused():
    ktx2 = rx.KTX2_MAGIC + b"\0" * 64
    with pytest.raises(rx.RobloxExportError, match="KTX2"):
        rx.export(_model(pictures=[("image/ktx2", ktx2)]))


def test_the_vertex_data_survives_repacking():
    source = _model(pictures=[("image/webp", _picture("WEBP"))], webp_texture=True)
    document, binary = parse_glb(rx.export(source).glb)
    view = document["bufferViews"][0]
    floats = struct.unpack_from("<9f", binary, view["byteOffset"])
    assert floats[7] == pytest.approx(1.8 / 0.28, rel=1e-5)  # survived, then scaled


def test_metal_roughness_is_split_green_to_roughness_blue_to_metal():
    pictures = [("image/png", _picture("PNG")),
                ("image/png", _picture("PNG", colour=(128, 128, 255))),
                ("image/png", _picture("PNG", colour=(0, 190, 40)))]
    maps = rx.export(_model(pictures=pictures)).maps
    assert set(maps) == {"color.png", "normal.png", "roughness.png", "metalness.png"}
    assert Image.open(io.BytesIO(maps["roughness.png"])).getpixel((0, 0)) == 190
    assert Image.open(io.BytesIO(maps["metalness.png"])).getpixel((0, 0)) == 40


def test_a_model_with_no_pictures_has_no_maps():
    assert rx.export(_model()).maps == {}


# ---------------------------------------------------------------- folder

def test_write_folder_writes_the_model_maps_record_and_how_to(tmp_path):
    result = rx.export(_model(pictures=[("image/png", _picture("PNG"))]))
    out = rx.write_folder(result, tmp_path / "raccoon_roblox")
    assert (out / "model.glb").read_bytes() == result.glb
    assert (out / "color.png").is_file()
    record = json.loads((out / "check.json").read_text())
    assert record["passed"] is True
    assert record["triangles"] == 2
    assert "studs" in (out / "HOW-TO-IMPORT.txt").read_text()


def test_a_height_in_studs_scales_the_whole_model_to_it():
    result = rx.export(_model(height=0.25), height_studs=4)
    assert result.size_studs[1] == 4.0
    assert _floats(result.glb, 0)[7] == pytest.approx(4.0, rel=1e-5)


@pytest.mark.parametrize("height", [0, -1, 5000])
def test_a_silly_height_is_refused(height):
    with pytest.raises(ValueError):
        rx.export(_model(), height_studs=height)


# ---------------------------------------------------------------- command line

def _cli():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_roblox.py"
    spec = importlib.util.spec_from_file_location("export_roblox", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_beside_the_model_and_exits_0(tmp_path, capsys):
    model = tmp_path / "chest.glb"
    model.write_bytes(_model())
    assert _cli().main([str(model), "--height-studs", "3"]) == 0
    assert (tmp_path / "chest_roblox" / "model.glb").is_file()
    assert "x 3 x" in capsys.readouterr().out


def test_cli_exits_1_when_a_check_fails(tmp_path):
    model = tmp_path / "hat.glb"
    model.write_bytes(_model(triangles=5000))
    assert _cli().main([str(model), "--kind", "accessory", "--out", str(tmp_path / "o")]) == 1


def test_cli_exits_2_on_a_packed_file(tmp_path, capsys):
    model = tmp_path / "packed.glb"
    model.write_bytes(_model(packed="EXT_meshopt_compression"))
    assert _cli().main([str(model)]) == 2
    assert "uncompressed" in capsys.readouterr().err
