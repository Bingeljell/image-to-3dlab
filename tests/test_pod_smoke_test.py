"""scripts/pod_smoke_test.py: the pure parts. Renting a pod is exercised for real only."""

from __future__ import annotations

import http.server
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone

import pod_smoke_test as smoke
import pytest


def test_the_pod_terminates_itself_even_if_this_machine_dies():
    command = smoke.create_command("NVIDIA A40", "x", 3)
    at = command[command.index("--terminate-after") + 1]
    assert datetime.strptime(at, "%Y-%m-%dT%H:%M:%SZ")
    assert command[command.index("--cloud-type") + 1] == "SECURE"
    assert f"{smoke.PORT}/http" in command[command.index("--ports") + 1]


def test_a_community_pod_asks_for_a_public_ip():
    command = smoke.create_command("NVIDIA GeForce RTX 3090", "x", 3, "community")
    assert command[command.index("--cloud-type") + 1] == "COMMUNITY"
    assert "--public-ip" in command
    assert "--public-ip" not in smoke.create_command("NVIDIA A40", "x", 3)


def test_the_pod_is_pinned_to_a_country_outside_the_hunyuan_ban():
    command = smoke.create_command("NVIDIA GeForce RTX 3090", "x", 3, "community")
    assert command[command.index("--country-code") + 1] == "US"
    command = smoke.create_command("NVIDIA A40", "x", 3, country="CA")
    assert command[command.index("--country-code") + 1] == "CA"


@pytest.mark.parametrize("country", ["CZ", "se", "GB", "KR"])
def test_a_banned_country_is_refused_before_anything_is_rented(country, capsys):
    with pytest.raises(SystemExit):
        smoke.main(["--country", country, "--yes"])
    assert "not licensed" in capsys.readouterr().err


def test_a_banned_country_is_fine_without_hunyuan():
    # Refused for a different reason (no picture), which proves the country passed.
    with pytest.raises(SystemExit):
        smoke.main(["--country", "CZ", "--routes", "trellis", "--image", "/nope.png"])


def test_terminate_after_is_hours_from_now_in_utc():
    now = datetime(2026, 10, 2, 22, 30, tzinfo=timezone.utc)
    assert smoke.terminate_after(3, now) == "2026-10-03T01:30:00Z"


@pytest.mark.parametrize("pod,price", [({"costPerHr": 0.49}, 0.49),
                                       ({"adjustedCostPerHr": 0.5}, 0.5), ({}, None)])
def test_pod_price_reads_what_runpod_bills(pod, price):
    assert smoke.pod_price(pod) == price


def test_the_ssh_key_is_the_one_runpod_knows(tmp_path):
    # The 2026-10-02 run let ssh fall back to id_ed25519, which RunPod had never seen.
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAAdefault me@mac\n")
    (tmp_path / "runpod_key.pub").write_text("ssh-ed25519 AAAArunpod session\n")
    (tmp_path / "runpod_key").write_text("private")
    registered = {"keys": [{"key": "ssh-ed25519 AAAArunpod\n"}]}
    assert smoke.matching_private_key(registered, tmp_path) == tmp_path / "runpod_key"
    assert smoke.matching_private_key({"keys": []}, tmp_path) is None


def test_ssh_target_parses_runpods_command():
    info = {"podId": "abc", "sshCommand": "ssh root@1.2.3.4 -p 40022 -i ~/.ssh/key"}
    assert smoke.ssh_target(info) == ("root@1.2.3.4", 40022, "~/.ssh/key")
    with pytest.raises(RuntimeError):
        smoke.ssh_target({"podId": "abc"})


@pytest.mark.parametrize("output,expected", [("52428800.000", 52.4288), ("", 0.0),
                                             ("garbage", 0.0)])
def test_curl_speed_is_megabytes_per_second(output, expected):
    assert smoke.mb_per_s(output) == pytest.approx(expected)


def test_install_never_starts_the_lab_in_the_foreground():
    command = smoke.install_command("feat/x")
    assert "--ref feat/x" in command and "--yes" in command and "--no-start" in command
    assert "/feat/x/install.sh" in command  # the branch's installer, not main's


def test_the_lab_listens_for_runpods_proxy():
    command = smoke.lab_command("pod123")
    assert "RUNPOD_POD_ID=pod123" in command and "nohup ./lab" in command
    assert command.rstrip().endswith("&")


def test_starting_the_lab_returns_at_once(tmp_path):
    # 2026-10-02: `cd x && nohup ./lab ... &` backgrounded the whole list, which kept the
    # SSH channel open, and the step timed out. A pipe holds stdout the way ssh does.
    lab = tmp_path / "image-to-3dlab" / "lab"
    lab.parent.mkdir()
    lab.write_text("#!/bin/sh\nsleep 5\n")
    lab.chmod(0o755)
    started = time.monotonic()
    subprocess.run(["bash", "-c", smoke.lab_command("pod123")], capture_output=True,
                   env={"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}, timeout=10)
    assert time.monotonic() - started < 3


def test_the_viewer_calls_do_not_look_like_a_bot():
    # 2026-10-02: RunPod's proxy is behind Cloudflare, which answers Python-urllib's
    # default User-Agent with 403 (error 1010), so the lab "never answered".
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["agent"] = self.headers.get("User-Agent", "")
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.handle_request, daemon=True).start()
    status, _ = smoke.Viewer(f"http://127.0.0.1:{server.server_port}").call("GET", "/x")
    server.server_close()
    assert status == 200
    assert seen["agent"].startswith("image-to-3dlab") and "Mozilla" not in seen["agent"]


def test_a_network_blip_while_waiting_is_not_a_failed_step():
    # A 90-minute run through Cloudflare sees dropped connections; only the deadline fails.
    viewer = smoke.Viewer("http://127.0.0.1:9")  # nothing listens on the discard port
    with pytest.raises(RuntimeError, match="still running"):
        viewer.wait("/api/x/status", limit=0.3, poll=0.1)


def test_closing_the_terminal_still_deletes_the_pod():
    # SIGHUP/SIGTERM skip `finally` unless turned into KeyboardInterrupt.
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGHUP, signal.SIGTERM)}
    try:
        smoke.interrupt_on_hangup()
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(1)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_a_failed_blender_install_says_why():
    # 2026-10-02: blender.org refused the installer, and the script waited 20 minutes for
    # a Blender that was never coming, without the installer's own error.
    events = [{"phase": "setup", "message": "Downloading..."},
              {"phase": "setup", "message": "HTTP Error 403: Forbidden"},
              {"phase": "setup_done", "status": "error", "message": "exited with code 1"}]
    status, tail = smoke.setup_run_outcome(events)
    assert status == "error" and "403" in tail
    assert smoke.setup_run_outcome(events[:2])[0] == "unfinished"
    assert smoke.setup_run_outcome([{"phase": "setup_done", "status": "done"}])[0] == "done"


def test_only_a_409_that_says_wait_is_waited_out():
    # 2026-10-02: Finish was refused for a missing Blender, and the script retried that
    # for 10 minutes as if it were "busy".
    assert smoke.worth_waiting(409, {"error": "setup is running; wait for it to finish"})
    assert not smoke.worth_waiting(409, {"error": "Finish needs Blender 4.2 or newer"})
    assert not smoke.worth_waiting(500, {"error": "wait"})


def test_an_interrupt_keeps_the_steps_that_finished():
    # 2026-10-02: Ctrl-C during Finish printed an empty table after 70 minutes of steps.
    class FakePod:
        def ssh(self, command, timeout=0):
            if command.startswith("curl -sL -o /dev/null"):
                return "100000000"  # the speed check: 100 MB/s
            raise KeyboardInterrupt  # Ctrl-C during the install

    class Args:
        ref, routes = "x", []

    results = []
    with pytest.raises(KeyboardInterrupt):
        smoke.run_smoke(FakePod(), Args(), "t", None, results)
    assert [(r["step"], r["ok"]) for r in results] == [("download speed", True)]


def test_a_one_route_run_without_pixal3d_skips_blender(tmp_path, monkeypatch):
    class FakePod:
        id = "pod"

        def ssh(self, command, timeout=0):
            return "100000000"

    class FakeViewer:
        base = "https://pod"

        def __init__(self, base):
            pass

        def call(self, method, path, body=None, content_type="", timeout=0):
            if path.endswith("result.glb"):
                return 200, b"glTF" + b"0" * 2048
            return 200, b"{}"

        def json(self, method, path, payload=None, **kw):
            if path == "/api/hf/sign-in":
                return 200, {"signed_in": True, "repos": []}
            return 200, {"blender": None}

        def start(self, path, *a, **kw):
            assert "blender" not in path, "Blender installed for a run without Pixal3D"
            return {"job_id": "j"}

        def wait(self, path, limit, poll=15):
            return {"status": "done"}

    class Args:
        ref, routes = "x", ["hunyuan-cuda"]
        image = tmp_path / "in.png"

    Args.image.write_bytes(b"png")
    monkeypatch.setattr(smoke, "Viewer", FakeViewer)
    monkeypatch.setattr(smoke.time, "sleep", lambda s: None)
    results = []
    smoke.run_smoke(FakePod(), Args(), "t", tmp_path, results)
    steps = [r["step"] for r in results]
    assert "install Blender" not in steps
    assert steps[-1] == "generate hunyuan-cuda" and all(r["ok"] for r in results)


def test_the_blender_fallback_installs_where_the_lab_looks():
    command = smoke.blender_fallback_command("4.2.23")
    assert "download.blender.org/release/Blender4.2/blender-4.2.23-linux-x64.tar.xz" in command
    assert "mv blender-4.2.23-linux-x64 blender-lts" in command
    assert command.startswith("set -eo pipefail")


def test_a_failed_blender_install_still_tests_finish(tmp_path, monkeypatch):
    # 2026-10-02: the viewer's Blender install failed, the pod was killed, and Finish was
    # never tested. The run is about Finish; the install is a means.
    ran = []

    class FakePod:
        id = "pod"

        def ssh(self, command, timeout=0):
            ran.append(command)
            return "100000000"

    class FakeViewer:
        base = "https://pod"

        def __init__(self, base):
            pass

        def call(self, method, path, body=None, content_type="", timeout=0):
            if path.endswith("result.glb"):
                return 200, b"glTF" + b"0" * 2048
            if path.endswith("/events") and "finish" in path:
                return 200, b'data: {"stages": ["photo", "compress"]}\n\n'
            return 200, b"{}"

        def json(self, method, path, payload=None, **kw):
            if path == "/api/hf/sign-in":
                return 200, {"signed_in": True, "repos": []}
            return 200, {"blender": None, "blender_problem": "no Blender"}

        def start(self, path, *a, **kw):
            if "blender" in path:
                raise RuntimeError("POST /api/blender/install: HTTP 403")
            return {"job_id": "j"}

        def wait(self, path, limit, poll=15):
            return {"status": "done"}

    class Args:
        ref, routes = "x", ["pixal3d"]
        image = tmp_path / "in.png"

    Args.image.write_bytes(b"png")
    monkeypatch.setattr(smoke, "Viewer", FakeViewer)
    results = []
    smoke.run_smoke(FakePod(), Args(), "t", tmp_path, results)
    outcome = {r["step"]: r["ok"] for r in results}
    assert outcome["install Blender"] is False
    assert outcome["install Blender by hand (fallback)"] is True
    assert outcome["Finish the Pixal3D model (Pixel Match)"] is True
    assert any("blender-lts" in c for c in ran)


def test_multipart_carries_fields_and_files():
    body, ctype = smoke.multipart({"settings": json.dumps({"backend": "trellis"})},
                                  {"image": ("in.png", b"\x89PNG")})
    boundary = ctype.split("boundary=")[1]
    assert body.endswith(f"--{boundary}--\r\n".encode())
    assert b'name="image"; filename="in.png"' in body and b"\x89PNG" in body
    assert b'{"backend": "trellis"}' in body


def test_is_glb_wants_the_magic_and_some_substance():
    assert smoke.is_glb(b"glTF" + b"\0" * 2000)
    assert not smoke.is_glb(b"glTF")
    assert not smoke.is_glb(b"<html>" + b"\0" * 2000)


def test_sse_events_skip_noise():
    text = 'data: {"phase": "queued", "stages": ["photo"]}\n\n: ping\ndata: nope\n'
    assert smoke.sse_events(text) == [{"phase": "queued", "stages": ["photo"]}]


def test_announcement_names_every_route_and_its_size():
    text = smoke.announcement(["trellis", "pixal3d"], ["NVIDIA A40"], 0.8, 3)
    assert "$0.80/hr" in text and "NVIDIA A40" in text and "GB of weights" in text
    assert text.count("GB of weights") == 2


def test_every_route_is_in_the_catalogue():
    import backend_catalog
    assert all(backend_catalog.resolve(r) is not None for r in smoke.ROUTES)


def test_nothing_is_rented_without_a_yes(monkeypatch, tmp_path):
    image = tmp_path / "in.png"
    image.write_bytes(b"x")
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    monkeypatch.setattr(smoke, "rent", lambda *a: pytest.fail("rented"))
    assert smoke.main(["--image", str(image)]) == 1


def test_pods_are_rented_with_a_driver_new_enough_for_the_pixal3d_prebuilt():
    """An older host driver silently turns a one-minute install into a 15-minute compile."""
    command = smoke.create_command("NVIDIA A40", "smoke", 2.0)
    assert command[command.index("--min-cuda-version") + 1] == "12.9"
