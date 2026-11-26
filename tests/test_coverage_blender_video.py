"""Verify frame handling before and after handing a sequence to the encoder."""

import subprocess
import sys

import blender_turntable as turntable
import blender_wind_demo as wind
import pytest


@pytest.mark.parametrize(
    ("video", "inputs", "timeout"),
    [(turntable, ["asset.glb"], 1800), (wind, ["asset.glb", "labels.glb"], 3600)],
)
@pytest.mark.parametrize("mode", ["short", "cleanup", "retain"])
def test_video_tools_only_encode_complete_frame_sequences(
    video,
    inputs,
    timeout,
    mode,
    monkeypatch,
    tmp_path,
    capsys,
):
    dest = tmp_path / "output" / "result.mp4"
    frame_dir = dest.parent / ".frames_test"
    frame_dir.mkdir(parents=True)
    stale = frame_dir / "stale.txt"
    stale.write_text("old run")
    encodes = []

    def send(host, port, code, *, timeout):
        assert (host, port) == ("localhost", 4444)
        assert timeout == expected_timeout
        assert frame_dir.is_dir() and not stale.exists()
        compile(code, "<Blender frame request>", "exec")
        for frame in range(1 if mode == "short" else 2):
            (frame_dir / f"test_{frame + 1:04d}.png").write_bytes(b"frame")
        return "sequence finished"

    def encode(argv, *, check):
        assert check
        assert argv[0] == "ffmpeg"
        assert argv[argv.index("-framerate") + 1] == "25"
        assert argv[argv.index("-i") + 1] == str(frame_dir / "test_%04d.png")
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"
        assert argv[-1] == str(dest)
        encodes.append(argv)
        dest.write_bytes(b"encoded")
        return subprocess.CompletedProcess(argv, 0)

    expected_timeout = timeout
    monkeypatch.setattr(video, "send", send)
    monkeypatch.setattr(video.subprocess, "run", encode)
    args = [
        "video",
        *inputs,
        str(dest),
        "--label",
        "test",
        "--host",
        "localhost",
        "--port",
        "4444",
        "--frames",
        "2",
        "--fps",
        "25",
    ]
    if mode == "retain":
        args += ["--keep-frames"]
    monkeypatch.setattr(sys, "argv", args)
    assert video.main() == (1 if mode == "short" else 0)
    status = capsys.readouterr()
    if mode == "short":
        assert "expected 2 frames, found 1" in status.err
        assert not dest.exists() and not encodes
    else:
        assert dest.read_bytes() == b"encoded"
        assert len(encodes) == 1
    assert frame_dir.exists() == (mode != "cleanup")
