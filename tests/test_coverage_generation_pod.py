"""Exercise pod rent and cleanup logic with local command and HTTP responses."""

import io
import json
import subprocess
import urllib.error

import pytest

from scripts import pod_smoke_test as smoke


@pytest.mark.posix
@pytest.mark.parametrize("mode", [
    "clean", "over-budget", "interrupt", "keep", "bad-result",
    "slow", "install-error", "sign-in-error", "setup-error", "generate-error",
    "pixal3d", "finish-fails", "no-photo", "classic", "never-ready", "blender-present",
    "installer-error", "installer-invisible",
])
def test_pod_cli_rents_later_gpu_and_cleans_up(
    mode, tmp_path, monkeypatch, capsys
):
    image = tmp_path / "sample.png"
    image.write_bytes(b"local image")
    monkeypatch.setattr(smoke, "REPO", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    monkeypatch.setenv("I3D_STUDIO_TOKEN", "fixture-studio-token")
    monkeypatch.setattr(smoke.signal, "signal", lambda number, handler: None)
    monkeypatch.setattr(smoke.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(smoke.time, "time", lambda: 10.0)

    ssh_key = tmp_path / "fixture-key"
    ssh_key.touch()
    rented = []
    delete_attempts = []
    process_steps = []
    pixal = mode in {"pixal3d", "finish-fails", "no-photo", "classic",
                     "blender-present", "installer-error", "installer-invisible"}
    model_route = "pixal3d" if pixal else "hunyuan-cuda"

    def run(args, **kwargs):
        if args[0] == "git":
            assert args[1:] == ["check-ref-format", "--allow-onelevel", "main"]
            return subprocess.CompletedProcess(args, 0, stdout="main\n", stderr="")
        if args[0] == "ssh":
            assert "fixture-host" in args[-2]
            assert args[args.index("-i") + 1] == str(ssh_key)
            assert kwargs["capture_output"] and kwargs["text"] and not kwargs["check"]
            if args[-1].startswith("curl -sL -o /dev/null"):
                output = "1000" if mode == "slow" else "100000000"
            elif "install" in args[-1]:
                if mode == "interrupt":
                    raise KeyboardInterrupt
                if mode == "install-error":
                    return subprocess.CompletedProcess(args, 1, stdout="", stderr="install refused")
                output = "install complete"
            elif mode in {"installer-error", "installer-invisible"} and args[-1].startswith("bash -c"):
                output = "fallback finished"
            else:
                assert "lab.log" in args[-1]
                output = "started"
            process_steps.append(output)
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")

        assert args[0] == "runpodctl"
        if args[1:3] == ["pod", "create"]:
            chosen = "GPU one" if not rented else "GPU two"
            assert chosen in args
            rented.append(chosen)
            if len(rented) == 1:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="no stock")
            value = {"id": "test-id", "costPerHr": 1.00 if mode == "over-budget" else 0.50}
        elif args[1:3] == ["pod", "delete"]:
            assert args[3] == "test-id"
            delete_attempts.append(args[3])
            if len(delete_attempts) == 1:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="transient failure")
            value = {}
        elif args[1:3] == ["ssh", "info"]:
            assert args[3] == "test-id"
            value = {"ssh_command": f"ssh -i {ssh_key} -p 22 user@fixture-host"}
        else:
            pytest.fail(f"unexpected process: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(value), stderr="")

    requests = []

    def response(request, timeout):
        route = request.selector
        requests.append((request.method, route))
        assert request.get_header("Authorization") == "Bearer fixture-studio-token"
        if route == "/api/hf/sign-in":
            assert json.loads(request.data) == {"token": "fixture-token"}
            payload = {"signed_in": True, "user": "tester", "repos": [{"repo": "gated", "access": "no"}]}
            if mode == "sign-in-error":
                payload = {"signed_in": False, "error": "sign-in refused"}
        elif route == "/api/catalog" and mode == "never-ready":
            raise OSError("lab startup timeout")
        elif route == "/api/finish/capabilities":
            calls = requests.count(("GET", route))
            payload = {"blender": "test", "blender_version": "4.2"}
            if mode == "installer-invisible" or (calls == 1 and mode != "blender-present"):
                payload = {"blender": None}
        elif route == "/api/blender/install":
            payload = {"events_url": "/api/blender/install/events"}
        elif route.endswith("/events"):
            if route.startswith("/api/blender"):
                event = {"phase": "setup_done", "status": "done"}
                if mode == "installer-error":
                    event = {"phase": "setup_error", "status": "failed", "message": "fixture"}
            else:
                event = {"stages": ["photo"] if mode != "no-photo" else ["unwrap"],
                         "message": "classic" if mode == "classic" else "Pixal3D's own clean-up"}
            stream = io.BytesIO(b"data: " + json.dumps(event).encode() + b"\n\n")
            stream.status = 200
            return stream
        elif route == "/api/generate" and requests.count(("POST", route)) == 1:
            assert b"sample.png" in request.data and b"local image" in request.data
            raise urllib.error.HTTPError(
                request.full_url, 409, "busy", hdrs=None,
                fp=io.BytesIO(b'{"error": "wait for setup"}'),
            )
        elif route in {"/api/generate", "/api/finish"}:
            payload = {"id": "job"}
        elif route.endswith("/status"):
            payload = {"status": "done", "last_event": {"phase": "done"}}
            target = {"setup-error": "/api/setup", "generate-error": "/api/generate",
                      "finish-fails": "/api/finish"}.get(mode)
            if target and route.startswith(target):
                payload = {"status": "error", "log_tail": "bad setup",
                           "last_event": {"message": "failed fixture"}}
        elif route.endswith("/result.glb"):
            stream = io.BytesIO(b"unexpected output" if mode == "bad-result" else b"glTF" + b"0" * 2048)
            stream.status = 200
            return stream
        else:
            payload = {}
        stream = io.BytesIO(json.dumps(payload).encode())
        stream.status = 200
        return stream

    monkeypatch.setattr(smoke.subprocess, "run", run)
    monkeypatch.setattr(smoke.urllib.request, "urlopen", response)
    args = [
        "--yes", "--image", str(image), "--routes", model_route,
        "--gpu", "GPU one", "--gpu", "GPU two",
    ]
    if mode == "keep":
        args.append("--keep")
    if mode == "over-budget":
        with pytest.raises(SystemExit, match="over the .* cap. Deleted"):
            smoke.main(args)
        assert not process_steps and not requests
    else:
        assert smoke.main(args) == int(mode not in {"clean", "keep", "pixal3d", "blender-present"})
        summaries = list(tmp_path.rglob("summary.json"))
        assert len(summaries) == 1
        report = json.loads(summaries[0].read_text())
        assert report["gpu"] == "GPU two" and report["price_per_hr"] == 0.50
        if mode == "interrupt":
            assert [row["step"] for row in report["steps"]] == ["download speed"]
            assert not requests
        elif mode == "bad-result":
            assert report["steps"][-1]["detail"] == "result.glb is not a GLB"
            assert not list(tmp_path.rglob("hunyuan-cuda.glb"))
        elif mode in {"slow", "install-error", "sign-in-error", "setup-error", "generate-error",
                      "finish-fails", "no-photo", "classic", "never-ready", "installer-error", "installer-invisible"}:
            failed, = [row for row in report["steps"] if not row["ok"]]
            label, fragment = {
                "slow": ("download speed", "is under"),
                "install-error": ("install", "install refused"),
                "sign-in-error": ("Hugging Face sign-in", "sign-in refused"),
                "setup-error": (f"set up {model_route}", "bad setup"),
                "generate-error": (f"generate {model_route}", "failed fixture"),
                "finish-fails": ("Finish the Pixal3D model (Pixel Match)", "failed fixture"),
                "no-photo": ("Finish the Pixal3D model (Pixel Match)", "Pixel Match did not run"),
                "classic": ("Finish the Pixal3D model (Pixel Match)", "Finish took the classic route"),
                "never-ready": ("start the lab", "viewer never answered"),
                "installer-error": ("install Blender", "installer "),
                "installer-invisible": ("install Blender", "Finish cannot see it"),
            }[mode]
            assert failed["step"] == label and fragment in failed["detail"]
            if mode.startswith("installer-"):
                assert "fallback finished" in process_steps
        else:
            assert all(row["ok"] for row in report["steps"])
            assert any("tester" in row["detail"] for row in report["steps"])
            assert requests.count(("POST", "/api/generate")) == 2
            assert summaries[0].with_name(f"{model_route}.glb").read_bytes().startswith(b"glTF")
            if pixal:
                assert summaries[0].with_name("pixal3d-finished.glb").read_bytes().startswith(b"glTF")

    assert rented == ["GPU one", "GPU two"]
    assert len(delete_attempts) == (0 if mode == "keep" else 2)
    out = capsys.readouterr().out
    assert "not available" in out
    assert ("Kept pod" if mode == "keep" else "Deleted pod test-id.") in out


def test_pod_viewer_reports_busy_deadline_and_unstructured_http_response(monkeypatch):
    monkeypatch.setenv("I3D_STUDIO_TOKEN", "fixture-token")
    client = smoke.Viewer("https://fixture.invalid")
    times = iter([0.0, 3.0, 10.0, 11.0])
    monkeypatch.setattr(smoke.time, "time", lambda: next(times))
    waited = []
    monkeypatch.setattr(smoke.time, "sleep", waited.append)

    def response(request, timeout):
        assert request.get_header("Authorization") == "Bearer fixture-token"
        if request.method == "POST":
            assert request.selector == "/api/create"
            raise urllib.error.HTTPError(request.full_url, 409, "busy", hdrs=None,
                                         fp=io.BytesIO(b'{"error": "wait for setup"}'))
        assert request.selector == "/api/job"
        stream = io.BytesIO(b"temporarily unavailable")
        stream.status = 503
        return stream

    monkeypatch.setattr(smoke.urllib.request, "urlopen", response)
    with pytest.raises(RuntimeError, match="POST /api/create: HTTP 409"):
        client.start("/api/create", patience=2)
    with pytest.raises(RuntimeError, match="/api/job still running"):
        client.wait("/api/job", limit=1, poll=0.5)
    assert not waited
    assert client.json("GET", "/api/job") == (503, {"raw": "temporarily unavailable"})


def test_pod_delete_retries_three_times_and_reports_manual_cleanup(monkeypatch, capsys):
    attempts = []
    waited = []

    def run(argv, **kwargs):
        assert argv == ["runpodctl", "pod", "delete", "fixture-id"]
        attempts.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="fixture error", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", run)
    monkeypatch.setattr(smoke.time, "sleep", waited.append)
    smoke.delete("fixture-id")
    assert len(attempts) == 3 and waited == [10, 10, 10]
    assert "Could not delete pod fixture-id" in capsys.readouterr().out


def test_pod_viewer_waits_for_completion_and_prints_event_once(monkeypatch, capsys):
    viewer = smoke.Viewer("https://fixture.invalid")
    polls = iter([
        (200, {"status": "running", "last_event": {"message": "preparing"}}),
        (200, {"status": "running", "last_event": {"message": "preparing"}}),
        (200, {"status": "done"}),
    ])
    monkeypatch.setattr(viewer, "json", lambda method, route: next(polls))
    sleeps = []
    monkeypatch.setattr(smoke.time, "sleep", sleeps.append)
    assert viewer.wait("/status", limit=10, poll=.25) == {"status": "done"}
    assert sleeps == [.25, .25]
    assert capsys.readouterr().out.count("preparing") == 1


def test_pod_rejects_unknown_route_before_any_commands(monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit) as status:
        smoke.main(["--routes", "unsupported"])
    assert status.value.code == 2


def test_empty_gpu_list_declines_rent_before_any_commands(monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit, match="No GPU from the list"):
        smoke.rent([], .5, 1)


def test_ssh_without_a_private_key_uses_agent_authentication(monkeypatch, tmp_path):
    monkeypatch.setattr(smoke.Path, "home", lambda: tmp_path)
    commands = []

    def run(args, **kwargs):
        commands.append(args)
        if args[0] == "runpodctl":
            info = {"ssh_command": "ssh -p 22 user@fixture-host"} if args[2] == "info" else {"keys": []}
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps(info), stderr="")
        assert args[:1] == ["ssh"] and "-i" not in args
        assert args[-2:] == ["user@fixture-host", "ls"]
        return subprocess.CompletedProcess(args, 0, stdout="fixture", stderr="")

    monkeypatch.setattr(smoke.subprocess, "run", run)
    assert smoke.Pod({"id": "fixture-pod"}).ssh("ls") == "fixture"
    assert [command[:3] for command in commands[:2]] == [
        ["runpodctl", "ssh", "info"], ["runpodctl", "ssh", "list-keys"],
    ]
