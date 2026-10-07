"""Tests for the viewer's URL construction.

The compare URL is how a browser gets pointed at two meshes, so a wrong path produces a
blank pane that reads as a broken asset rather than a broken link — the same false-negative
shape as a generation run that never started.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlparse


REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("serve", REPO / "viewer" / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load()


def test_server_script_imports_from_outside_the_repo_root(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(REPO / "viewer" / "serve.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--auto" in result.stdout


def _params(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_glb_is_served_as_a_binary_model_type():
    """A wrong MIME type makes GLTFLoader fail in a way that looks like a corrupt file."""
    assert serve.Handler.extensions_map[".glb"] == "model/gltf-binary"


# --- ./lab --auto: one command for a new user, local or remote ------------------------
def test_auto_on_your_own_machine_stays_private_and_opens_the_browser():
    plan = serve.launch_plan({}, port=8777)
    assert plan.host == "127.0.0.1" and plan.open_browser is True
    assert "http://127.0.0.1:8777/viewer/studio.html" in plan.message


def test_auto_on_a_runpod_pod_listens_for_the_proxy_and_prints_its_link():
    plan = serve.launch_plan({"RUNPOD_POD_ID": "abc123"}, port=8777)
    assert plan.host == "0.0.0.0" and plan.open_browser is False
    assert "https://abc123-8777.proxy.runpod.net/viewer/studio.html" in plan.message
    assert "8777" in plan.message and "HTTP port" in plan.message


def test_auto_over_plain_ssh_stays_private_and_explains_the_tunnel():
    # Exposing the whole repo on a network by default would be unsafe on a shared LAN.
    plan = serve.launch_plan({"SSH_CONNECTION": "1.2.3.4 5 6.7.8.9 22"}, port=8777)
    assert plan.host == "127.0.0.1" and plan.open_browser is False
    assert "ssh -L 8777:127.0.0.1:8777" in plan.message


def test_a_dropped_connection_is_quiet_and_real_errors_still_print(capsys):
    server = serve.ThreadingHTTPServer.__new__(serve.ThreadingHTTPServer)
    try:
        raise ConnectionResetError("peer closed")
    except ConnectionResetError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""
    try:
        raise RuntimeError("a real bug")
    except RuntimeError:
        server.handle_error(None, ("127.0.0.1", 1))
    assert "a real bug" in capsys.readouterr().err
