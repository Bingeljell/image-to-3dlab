"""A model's stats and where it lives on disk, for the studio panel and for agents."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import model_info_api as mi

from image_to_3dlab.glb_turn import build_glb

GLB = build_glb({"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                 "accessors": [{"count": 3, "componentType": 5126, "type": "VEC3"}]}, b"")


@pytest.fixture
def output(tmp_path):
    run = tmp_path / "knight__run"
    run.mkdir()
    (run / "knight.glb").write_bytes(GLB)
    (run / "knight.png").write_bytes(b"png")
    return tmp_path


def test_info_gives_the_stats_and_the_full_path(output):
    info = mi.model_info(output, "knight__run/knight.glb")
    assert info["stats"]["triangles"] == 1
    assert info["path"] == str(output / "knight__run" / "knight.glb")
    assert info["folder"] == str(output / "knight__run")
    assert info["file"] == "knight__run/knight.glb"


@pytest.mark.parametrize("relative", ["../secret.glb", "/etc/passwd", "knight__run/../../x.glb"])
def test_a_path_outside_the_output_folder_is_refused(output, relative):
    with pytest.raises(ValueError):
        mi.resolve(output, relative)


def test_a_missing_file_is_refused(output):
    with pytest.raises(ValueError, match="no such file"):
        mi.resolve(output, "knight__run/nope.glb")


def test_stats_are_only_read_from_models(output):
    """A picture can still be shown in its folder, but it has no triangles to count."""
    info = mi.model_info(output, "knight__run/knight.png")
    assert info["stats"] is None
    assert info["path"].endswith("knight.png")


def test_the_reveal_command_selects_the_file_on_each_system():
    path = Path("/out/run/knight.glb")
    assert mi.reveal_command(path, "macos") == ["open", "-R", str(path)]
    assert mi.reveal_command(path, "windows") == ["explorer", f"/select,{path}"]
    # Linux file managers differ on selecting a file; opening its folder works everywhere
    assert mi.reveal_command(path, "linux") == ["xdg-open", str(path.parent)]


def test_reveal_runs_the_command_and_does_not_wait(output):
    ran = []
    mi.reveal(output, "knight__run/knight.glb", family="macos", launch=ran.append)
    assert ran == [["open", "-R", str(output / "knight__run" / "knight.glb")]]


@pytest.mark.parametrize("address, local", [("127.0.0.1", True), ("::1", True),
                                            ("::ffff:127.0.0.1", True), ("100.64.1.2", False)])
def test_only_this_machine_may_open_a_folder(address, local):
    """Over Tailscale or a LAN the folder would open on the server's screen, not the viewer's."""
    assert mi.is_local(address) is local


# ---------------------------------------------------------------- Export for Roblox

def _sized_glb(height: float, bones: int = 0) -> bytes:
    import struct
    positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, height, 0)
    document = {"asset": {"version": "2.0"},
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                "accessors": [{"bufferView": 0, "count": 3, "componentType": 5126, "type": "VEC3",
                               "min": [0, 0, 0], "max": [1, height, 0]}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(positions)}],
                "buffers": [{"byteLength": len(positions)}],
                "nodes": [{"mesh": 0}]}
    if bones:
        document["nodes"] += [{"name": f"b{i}"} for i in range(bones)]
        document["skins"] = [{"joints": list(range(1, bones + 1))}]
    return build_glb(document, positions)


def test_roblox_export_writes_a_folder_beside_the_model(output):
    (output / "knight__run" / "chest.glb").write_bytes(_sized_glb(0.3))
    made = mi.roblox_export(output, "knight__run/chest.glb")
    assert made["folder"] == "knight__run/chest_roblox"
    assert (output / made["file"]).is_file()
    assert made["passed"] is True


def test_an_unrigged_model_gets_the_default_prop_height(output):
    (output / "knight__run" / "chest.glb").write_bytes(_sized_glb(0.3))
    made = mi.roblox_export(output, "knight__run/chest.glb")
    assert made["size_studs"][1] == mi.PROP_HEIGHT_STUDS


def test_a_rigged_model_keeps_its_person_size(output):
    (output / "knight__run" / "hero.glb").write_bytes(_sized_glb(1.68, bones=2))
    made = mi.roblox_export(output, "knight__run/hero.glb")
    assert made["size_studs"][1] == 6.0


def test_a_chosen_height_wins(output):
    (output / "knight__run" / "chest.glb").write_bytes(_sized_glb(0.3))
    assert mi.roblox_export(output, "knight__run/chest.glb", height_studs=2)["size_studs"][1] == 2.0


def test_roblox_export_refuses_pictures_and_outside_paths(output):
    with pytest.raises(ValueError, match="only GLB"):
        mi.roblox_export(output, "knight__run/knight.png")
    with pytest.raises(ValueError):
        mi.roblox_export(output, "../x.glb")


def test_a_packed_file_is_a_plain_error(output):
    import json

    from image_to_3dlab.glb_turn import parse_glb
    document, binary = parse_glb(_sized_glb(1))
    document["extensionsUsed"] = ["EXT_meshopt_compression"]
    (output / "knight__run" / "packed.glb").write_bytes(build_glb(json.loads(json.dumps(document)), binary))
    with pytest.raises(ValueError, match="uncompressed"):
        mi.roblox_export(output, "knight__run/packed.glb")
