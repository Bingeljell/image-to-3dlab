"""Offline acceptance tests for Blender's socket and process boundaries."""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import preview_api
import pytest
import trimesh
from PIL import Image
from unit_isolation import COVERAGE_ENV, OFFLINE_ENV

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).with_name("acceptance_blender")

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="The Linux Blender fixtures use POSIX socket inheritance and shebang execution"
)


def listen_on_ephemeral_port():
    for _ in range(5):
        listener = socket.socket()
        try:
            listener.bind(("127.0.0.1", 0))
            if listener.getsockname()[1] != 9876:
                listener.listen()
                return listener
        except BaseException:
            listener.close()
            raise
        listener.close()
    raise RuntimeError("Could not reserve an ephemeral port outside the Blender port")


@pytest.fixture
def private_process_environment(tmp_path):
    # Keep the test preloads and give children no inherited credentials or user config.
    env = {key: os.environ[key] for key in COVERAGE_ENV if key in os.environ}
    env.update(OFFLINE_ENV)
    env["AF_TEST_FIXTURE_ROOT"] = os.environ["AF_TEST_FIXTURE_ROOT"]
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), os.defpath))
    env["PYTHONUTF8"] = "1"
    env["I3D_NO_USAGE"] = "1"
    for setting in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "HF_HOME", "U2NET_HOME"):
        directory = tmp_path / setting
        directory.mkdir()
        env[setting] = str(directory)
    with patch.dict(os.environ, env, clear=True):
        yield


@pytest.mark.parametrize(
    ("mode", "error"),
    [("ok", None), ("truncated", "JSONDecodeError"), ("error", "KeyError: 'result'")],
)
def test_joint_marker_save_over_loopback(tmp_path, private_process_environment, mode, error):
    marker_file = tmp_path / "placés 日本語" / "repères.json"
    receiver_path = tmp_path / "receive_scene.py"
    receiver_path.write_bytes((FIXTURES / receiver_path.name).read_bytes())

    # A listening socket is the readiness signal. The receiver inherits that open socket,
    # so we never release the port and race another process to bind it again.
    with listen_on_ephemeral_port() as listener:
        port = listener.getsockname()[1]
        receiver = subprocess.Popen(
            [sys.executable, str(receiver_path), str(listener.fileno()), mode],
            pass_fds=(listener.fileno(),),
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    try:
        sender = subprocess.run(
            [sys.executable, str(REPO / "scripts/blender_joint_markers.py"), "save",
             "--host", "127.0.0.1", "--port", str(port), "--file", str(marker_file)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
        _, receiver_error = receiver.communicate(timeout=10)
        assert receiver.returncode == 0, receiver_error
    finally:
        if receiver.poll() is None:
            receiver.kill()
        receiver.communicate()

    if error:
        assert sender.returncode != 0
        assert error in sender.stderr
        assert not marker_file.exists()
        return

    assert sender.returncode == 0, sender.stderr
    assert json.loads(marker_file.read_text(encoding="utf-8")) == {
        "head": [1.23457, -2.5, 0.125],
        "patère": [-3, 0.00001, 4],
    }


@pytest.mark.parametrize("first_mode", ["ok", "missing", "nonzero"])
def test_preview_launches_child_and_caches_complete_sheet(
    tmp_path, monkeypatch, private_process_environment, first_mode,
):
    output = tmp_path / "sorties 日本語"
    output.mkdir()
    source = output / "petit cube.glb"
    trimesh.creation.box().export(source)

    executable = tmp_path / "fixture-render"
    executable.write_bytes(b"#!/usr/bin/env python3\n" + (FIXTURES / "render_panels.py").read_bytes())
    executable.chmod(0o700)
    render_log = tmp_path / "arguments.jsonl"
    render_scratch = tmp_path / "rendus é"
    render_scratch.mkdir()
    monkeypatch.setattr(preview_api.tempfile, "tempdir", str(render_scratch))
    monkeypatch.setattr(preview_api, "RENDER_TIMEOUT", 15)
    monkeypatch.setenv("AF_TEST_RENDER_LOG", str(render_log))
    monkeypatch.setenv("AF_TEST_RENDER_MODE", first_mode)

    asset = {"model": source.name, "facing": [1, 0, 0]}
    if first_mode != "ok":
        message = "Blender could not render the preview"
        if first_mode == "nonzero":
            message += ": fixture renderer exited after writing the panels"
        with pytest.raises(RuntimeError, match=message):
            preview_api.preview(output, asset, blender=executable)
        assert not (output / ".previews").exists()
        assert not list(render_scratch.iterdir())
        monkeypatch.setenv("AF_TEST_RENDER_MODE", "ok")

    sheet, shown = preview_api.preview(output, asset, blender=executable)
    assert shown == "model"
    assert sheet.parent == output / ".previews"
    with Image.open(sheet) as png:
        assert png.format == "PNG"
        assert png.size == (306, 140)
        assert [png.getpixel((36 + 78 * index, 104)) for index in range(4)] == [
            (232, 65, 51), (59, 192, 88), (53, 115, 242), (215, 155, 45),
        ]

    # A repeated request succeeds even after the renderer has been removed.
    executable.unlink()
    assert preview_api.preview(output, asset, blender=executable) == (sheet, "model")

    invocations = [json.loads(line) for line in render_log.read_text(encoding="utf-8").splitlines()]
    assert len(invocations) == (1 if first_mode == "ok" else 2)
    for args in invocations:
        assert args[:6] == [
            "--background", "--factory-startup", "--python",
            str(REPO / "scripts/blender_preview_worker.py"), "--", str(source.resolve()),
        ]
        assert Path(args[6]).parent == render_scratch
        assert args[7:] == [
            "90.0", "512", "640", "0", "Front=0.0=;Three-quarter=45.0=;Side=90.0=;Back=180.0=",
        ]
    assert not list(render_scratch.iterdir())
