"""Exercise the remaining staging and turntable command boundaries."""

import json
import socket
import sys
from pathlib import Path

import blender_light_orbit as light
import blender_stage as stage
import blender_turntable as turntable
import blender_wind_demo as wind
import pytest


class Reply:
    def __init__(self, fragments):
        self.fragments = iter(fragments)
        self.request = None
        self.timeout = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, payload):
        self.request = json.loads(payload)

    def recv(self, size):
        assert size == 65536
        fragment = next(self.fragments)
        if isinstance(fragment, Exception):
            raise fragment
        return fragment


@pytest.mark.parametrize(
    ("fragments", "success", "message"),
    [
        ([b'{"status":"suc', b'cess","result":{"executed":true,"result":""}}'], True, ""),
        ([b'{"status":"error","message":"Import failed"}'], False, "Import failed"),
        ([b'{"status":', b""], False, "closed before sending a complete reply"),
        ([b'{"status":', TimeoutError()], False, "timed out before sending a complete reply"),
        ([b'{"result":200}'], False, "unexpected Blender reply"),
        ([b'{"status":"success","result":{"executed":false,"error":"Import failed"}}'],
         False, "Import failed"),
        ([b""], False, "closed before sending a complete reply"),
    ],
)
@pytest.mark.parametrize("tool", ["stage", "turntable", "wind"])
def test_render_requests_join_partial_replies_and_close_the_connection(
    fragments, success, message, tool, monkeypatch, tmp_path, capsys
):
    reply = Reply(fragments)

    def connect(address, *, timeout):
        assert address == ("workstation", 4201)
        assert timeout == 10
        return reply

    monkeypatch.setattr(socket, "create_connection", connect)
    result = '{"status":"success","result":{"executed":true,"result":""}}'
    if tool == "stage":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stage",
                "a.glb",
                "b.glb",
                "--labels",
                "The first",
                "The second",
                "--host",
                "workstation",
                "--port",
                "4201",
                "--double-sided",
                "--gap",
                "2",
            ],
        )
        assert stage.main() == (0 if success else 1)
        output = capsys.readouterr()
        if success:
            assert output.out.strip() == result
            assert not output.err
        else:
            assert not output.out
            assert message in output.err
        assert reply.timeout == 300
        expected = stage.build_code(
            ["a.glb", "b.glb"], ["A_The_first", "B_The_second"], [-1.0, 1.0], False
        )
    else:
        expected = "print('render complete')"
        launcher = turntable if tool == "turntable" else wind
        if success:
            assert launcher.send("workstation", 4201, expected, 1800) == result
        else:
            with pytest.raises(RuntimeError, match=message):
                launcher.send("workstation", 4201, expected, 1800)
        assert reply.timeout == 1800
    assert reply.closed
    assert reply.request == {"type": "execute_code", "params": {"code": expected}}


def test_light_orbit_plans_frames_and_creates_output_parent(
    monkeypatch, tmp_path, capsys
):
    source, prefix = tmp_path / "asset.glb", tmp_path / "stills" / "orbit_"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orbit",
            str(source),
            str(prefix),
            "--frames",
            "4",
            "--elevation",
            "45",
            "--res",
            "32",
            "--host",
            "workstation",
            "--port",
            "4201",
        ],
    )
    calls = []

    def send(code, host, port):
        assert (host, port) == ("workstation", 4201)
        assert prefix.parent.is_dir()
        calls.append(code)
        return "four frames saved"

    monkeypatch.setattr(light, "send", send)
    assert light.main() == 0

    from math import pi

    rotations = [[pi / 4, 0.0, angle] for angle in (0.0, pi / 2, pi, 3 * pi / 2)]
    assert calls == [
        light.blender_code(
            source.resolve(), prefix.resolve(), 4, 45.0, 32, json.dumps(rotations)
        )
    ]
    assert capsys.readouterr().out.strip() == "four frames saved"


def test_turntable_creates_frames_directory_on_first_run(tmp_path, monkeypatch):
    output = tmp_path / "renders" / "result.mp4"
    frame = None
    argv = [
        "turntable",
        str(tmp_path / "source.glb"),
        str(output),
        "--frames",
        "1",
        "--keep-frames",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert not output.parent.exists()

    def send(host, port, code, *, timeout):
        nonlocal frame
        assert (host, port, timeout) == ("127.0.0.1", 9876, 1800)
        frame_dir, = output.parent.iterdir()
        assert frame_dir.is_dir() and not list(frame_dir.iterdir())
        frame = frame_dir / "turntable_0000.png"
        frame.write_bytes(b"rendered frame")
        return "rendered"

    def encode(command, *, check):
        assert check
        assert command[0] == "ffmpeg"
        assert command[command.index("-i") + 1] == str(frame.parent / "turntable_%04d.png")
        assert Path(command[-1]).name == output.name
        Path(command[-1]).write_bytes(frame.read_bytes())

    monkeypatch.setattr(turntable, "send", send)
    monkeypatch.setattr("subprocess.run", encode)
    assert turntable.main() == 0
    assert output.read_bytes() == b"rendered frame"
    assert frame.exists()
