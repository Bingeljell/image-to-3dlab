"""Verify frame handling before and after handing a sequence to the encoder."""

import subprocess
import sys
from pathlib import Path

import blender_turntable as turntable
import blender_wind_demo as wind
import pytest


@pytest.mark.parametrize(
    ("video", "inputs", "timeout"),
    [(turntable, ["asset.glb"], 1800), (wind, ["asset.glb", "labels.glb"], 3600)],
)
@pytest.mark.parametrize("mode", ["short", "gap", "offset", "extra", "cleanup", "retain", "encode_failure", "render_failure"])
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
    previous_frame_dirs = {
        dest.parent / ".frames_test", dest.parent / f"{dest.stem}_frames_earlier",
    }
    for previous_dir in previous_frame_dirs:
        previous_dir.mkdir(parents=True)
        (previous_dir / "test_0000.png").write_bytes(b"frame from an earlier run")
    encodes = []
    frame_dir = None

    def send(host, port, code, *, timeout):
        nonlocal frame_dir
        assert (host, port) == ("localhost", 4444)
        assert timeout == expected_timeout
        new_directories = {path for path in dest.parent.iterdir() if path.is_dir()}
        frame_dir, = new_directories - previous_frame_dirs
        assert not list(frame_dir.iterdir())
        compile(code, "<Blender frame request>", "exec")
        frame_numbers = {
            "short": [0], "gap": [0, 2], "offset": [1, 2], "extra": [0, 1, 2],
        }.get(mode, [0, 1])
        for frame in frame_numbers:
            (frame_dir / f"test_{frame:04d}.png").write_bytes(b"frame")
        if mode == "render_failure":
            raise RuntimeError("Blender error")
        return "sequence finished"

    def encode(argv, *, check):
        assert check
        assert argv[0] == "ffmpeg"
        assert argv[argv.index("-framerate") + 1] == "25"
        assert argv[argv.index("-i") + 1] == str(frame_dir / "test_%04d.png")
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"
        destination = Path(argv[-1])
        assert destination.parent.parent == dest.parent
        assert destination.name == dest.name
        encodes.append(argv)
        if mode == "encode_failure":
            destination.write_bytes(b"partial video")
            raise subprocess.CalledProcessError(1, argv)
        destination.write_bytes(b"encoded")
        return subprocess.CompletedProcess(argv, 0)

    expected_timeout = timeout
    monkeypatch.setattr(video, "send", send)
    monkeypatch.setattr(subprocess, "run", encode)
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
    if mode == "encode_failure":
        dest.write_bytes(b"previous video")
    successful = mode in {"cleanup", "retain"}
    assert video.main() == (0 if successful else 1)
    status = capsys.readouterr()
    assert frame_dir is not None
    for previous_dir in previous_frame_dirs:
        assert (previous_dir / "test_0000.png").read_bytes() == b"frame from an earlier run"
    if not successful:
        assert str(frame_dir) in status.err and not status.out.endswith(f"wrote {dest}\n")
        if mode == "encode_failure":
            assert dest.read_bytes() == b"previous video"
            assert set(dest.parent.iterdir()) == previous_frame_dirs | {frame_dir, dest}
        else:
            assert not dest.exists()
        assert len(encodes) == (1 if mode == "encode_failure" else 0)
        assert list(frame_dir.glob("*.png"))
    else:
        assert dest.read_bytes() == b"encoded"
        assert len(encodes) == 1
        if mode == "retain":
            assert str(frame_dir) in status.out
    assert frame_dir.exists() == (mode != "cleanup")


@pytest.mark.parametrize(
    ("video", "inputs"),
    [(turntable, ["asset.glb"]), (wind, ["asset.glb", "labels.glb"])],
)
def test_invalid_frame_count_leaves_previous_frames_in_place(video, inputs, tmp_path, monkeypatch):
    dest = tmp_path / "result.mp4"
    frame_dir = tmp_path / ".frames_test"
    frame_dir.mkdir()
    previous = frame_dir / "test_0000.png"
    previous.write_bytes(b"frame from an earlier run")
    monkeypatch.setattr(
        sys, "argv", ["video", *inputs, str(dest), "--label", "test", "--frames", "0"]
    )
    with pytest.raises(SystemExit) as error:
        video.main()
    assert error.value.code == 2
    assert previous.read_bytes() == b"frame from an earlier run"
