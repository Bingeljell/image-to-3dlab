"""Check live commands at their socket boundary without a Blender session."""

import json
import socket
import sys

import blender_attack_cycle as attack
import blender_build_rig as build_rig
import blender_fix_normals as normals
import blender_idle_cycle as idle
import blender_import_character as import_character
import blender_inspect as inspect_scene
import blender_joint_markers as markers
import blender_orbit_camera as orbit
import blender_render_asset as render
import blender_rigify_walk_cycle as rigify_walk
import blender_solidify_bake as solidify
import blender_split_regions as split_regions
import blender_stage_bake as stage
import blender_voxel_weights as voxel_weights
import blender_walk_cycle as walk
import pytest


class Connection:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.timeout = None
        self.request = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, data):
        self.request = json.loads(data)

    def recv(self, size):
        assert size == 65536
        chunk = next(self.chunks)
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


@pytest.fixture
def connect(monkeypatch):
    def with_chunks(chunks):
        connection = Connection(chunks)

        def create(address, *, timeout):
            assert address == ("localhost", 4444)
            assert timeout == 10
            return connection

        monkeypatch.setattr(socket, "create_connection", create)
        return connection

    return with_chunks


@pytest.mark.parametrize(
    ("tool", "args", "timeout"),
    [
        (normals, ["in.glb", "out/result.glb"], 1800),
        (solidify, ["in.glb", "out/result.glb", "--texture-size", "64"], 3600),
        (render, ["in.glb", "out/renders", "--culled", "--recalc-normals"], 300),
        (
            stage,
            ["walk", "--az", "30", "--label", "creature", "--output-dir", "out"],
            120,
        ),
        (
            orbit,
            [
                "--target",
                "1",
                "2",
                "3",
                "--azimuth",
                "30",
                "--elevation",
                "15",
                "--radius",
                "6",
            ],
            60,
        ),
        (
            import_character,
            ["in.blend", "--prefix", "Creature", "--position", "1", "2", "3"],
            60,
        ),
        (
            import_character,
            [
                "in.blend",
                "--prefix",
                "Creature",
                "--position",
                "1",
                "2",
                "3",
                "--facing",
                "0",
                "0",
                "0",
            ],
            60,
        ),
        (idle, ["--breath", "1.5"], 300),
        (walk, ["--gait", "walk", "--no-auto-plant"], 300),
        (voxel_weights, ["--keep-proxy"], 300),
        (attack, ["--frames", "24", "--toe", "0"], 300),
        (build_rig, ["--ear-length", "0.05"], 300),
        (rigify_walk, ["--gait", "walk"], 300),
        (split_regions, ["in.glb", "out", "--size", "64"], 300),
        (inspect_scene, [], 30),
        (inspect_scene, ["--file", "in.blend"], 30),
        (inspect_scene, ["--object", "Creature"], 30),
    ],
)
@pytest.mark.parametrize("ending", [b'"success"}', b"", TimeoutError()])
def test_live_cli_prints_reply_and_closes_connection(
    tool, args, timeout, ending, connect, monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["command", *args, "--host", "localhost", "--port", "4444"]
    )
    connection = connect([b'{"status":', ending])
    assert tool.main() == 0
    assert connection.closed and connection.timeout == timeout
    assert connection.request["type"] == "execute_code"
    assert capsys.readouterr().out.strip() == (
        '{"status":"success"}' if ending == b'"success"}' else '{"status":'
    )
    compile(connection.request["params"]["code"], "<Blender request>", "exec")
    if tool in (normals, solidify, render, split_regions):
        assert (tmp_path / "out").is_dir()


def test_file_and_live_object_inspection_cannot_be_combined(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["inspect", "--file", "in.blend", "--object", "body"]
    )
    with pytest.raises(SystemExit) as exc:
        inspect_scene.main()
    assert exc.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_joint_markers_save_round_trip_to_disk(connect, monkeypatch, tmp_path):
    destination = tmp_path / "poses" / "joints.json"
    placed = {"spine": [0, 0.5, 1], "ankle": [0, -1, 0]}
    response = {"result": {"result": json.dumps(placed)}}
    connection = connect([json.dumps(response).encode()])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "markers",
            "save",
            "--file",
            str(destination),
            "--host",
            "localhost",
            "--port",
            "4444",
        ],
    )
    assert markers.main() == 0
    assert json.loads(destination.read_text()) == placed
    assert destination.read_text().endswith("\n")
    assert connection.closed
