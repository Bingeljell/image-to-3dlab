"""Run real installer dispatch and filesystem bookkeeping with command and download substitutes."""

import json
import subprocess
import sys
import urllib.request
from types import SimpleNamespace

import bootstrap_hunyuan_cuda as cuda
import bootstrap_hunyuan_rocm as rocm
import pytest


@pytest.fixture(params=[cuda, rocm], ids=["nvidia", "amd"])
def installer(request):
    return request.param


@pytest.mark.parametrize(("target", "route", "tty", "answer", "message"), [
    ("macos", None, False, "", "Nothing downloaded"),
    ("correct", None, False, "", "Cannot install"),
    ("correct", "ok", False, "", "without --yes"),
    ("correct", "ok", True, "no", "Nothing downloaded"),
])
def test_hunyuan_never_downloads_after_preflight_or_consent_refusal(
    installer, monkeypatch, capsys, target, route, tty, answer, message,
):
    family = "linux-amd" if installer is rocm else "linux-nvidia"
    monkeypatch.setattr(installer.host, "build_target", lambda: family if target == "correct" else target)
    monkeypatch.setattr(installer, "detect", lambda: (route, "driver probe"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: tty))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    monkeypatch.setattr(installer, "install_code", lambda route: pytest.fail("started build"))
    monkeypatch.setattr(installer, "install_weights", lambda: pytest.fail("started downloads"))
    assert installer.main([]) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("mode", ["code", "weights", "both"])
def test_confirmed_hunyuan_setup_dispatches_selected_steps(installer, monkeypatch, capsys, mode):
    route = "rocm7.2" if installer is rocm else "tested"
    family = "linux-amd" if installer is rocm else "linux-nvidia"
    calls = []
    monkeypatch.setattr(installer.host, "build_target", lambda: family)
    monkeypatch.setattr(installer, "detect", lambda: (route, "driver probe"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt: " YES ")
    monkeypatch.setattr(installer, "install_code", lambda name: calls.append(("build", name)))
    monkeypatch.setattr(installer, "install_weights", lambda: calls.append(("weights", None)))
    monkeypatch.setattr(cuda, "install_background_remover", lambda: calls.append(("remover", None)))
    argv = [f"--{mode}-only"] if mode != "both" else []
    assert installer.main(argv) == 0
    assert calls == ([("build", route)] if mode == "code" else
                     [("weights", None), ("remover", None)] if mode == "weights" else
                     [("build", route), ("weights", None), ("remover", None)])
    notice = capsys.readouterr().out
    assert route in notice and "South Korea" in notice
    assert ("GB total" in notice) is (mode != "code")


@pytest.mark.parametrize("existing", [False, True])
def test_hunyuan_installer_keeps_existing_download_and_places_new_file_atomically(
    installer, monkeypatch, tmp_path, existing,
):
    plain_file = tmp_path / "ckpt" / "upscale.pth"
    local_snapshot = tmp_path / "shape"
    records = [
        ("Shape", "fake/shape", ["shape/*"], local_snapshot, .01),
        ("Paint", "fake/paint", ["paint/*"], None, .02),
        ("Upscaler", "https://example.invalid/upscale.pth", None, plain_file, .03),
    ]
    monkeypatch.setattr(installer, "WEIGHTS", records)
    downloads, plain_calls = [], []

    def download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))

    def retrieve(url, path):
        plain_calls.append((url, path))
        assert not plain_file.exists()
        path.write_bytes(b"fixture")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setattr(urllib.request, "urlretrieve", retrieve)
    if existing:
        plain_file.parent.mkdir()
        plain_file.write_bytes(b"keep")
    installer.install_weights()
    assert downloads == [
        ("fake/shape", {"local_dir": str(local_snapshot), "allow_patterns": ["shape/*"]}),
        ("fake/paint", {"allow_patterns": ["paint/*"]}),
    ]
    partial = plain_file.with_suffix(".pth.part")
    assert plain_calls == ([] if existing else [(records[-1][1], partial)])
    assert plain_file.read_bytes() == (b"keep" if existing else b"fixture")
    assert not partial.exists()


def test_install_weights_explains_missing_hub_dependency(installer, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(SystemExit, match="huggingface_hub is not installed"):
        installer.install_weights()


@pytest.mark.parametrize("already_pinned", [False, True])
def test_installer_clones_missing_checkout_and_fetches_only_specific_revision(
    installer, monkeypatch, tmp_path, already_pinned,
):
    checkout = tmp_path / "vendor" / "Hunyuan"
    monkeypatch.setattr(installer, "CHECKOUT", checkout)
    if already_pinned:
        (checkout / ".git").mkdir(parents=True)
    changes = []

    def run(argv, **kwargs):
        changes.append((argv, kwargs))
        if argv[:2] == ["git", "clone"]:
            (checkout / ".git").mkdir(parents=True)

    def git_head(argv, **kwargs):
        assert argv == ["git", "rev-parse", "HEAD"]
        assert kwargs["cwd"] == checkout
        assert (checkout / ".git").is_dir()
        return "pinned\n" if already_pinned else "upstream\n"

    monkeypatch.setattr(installer, "run", run)
    monkeypatch.setattr(subprocess, "check_output", git_head)
    installer.fetch_checkout("pinned")
    assert changes == ([] if already_pinned else [
        (["git", "clone", installer.UPSTREAM, str(checkout)], {}),
        (["git", "fetch", "origin", "pinned"], {"cwd": checkout}),
        (["git", "checkout", "--detach", "pinned"], {"cwd": checkout}),
    ])


@pytest.mark.parametrize("existing_venv", [False, True])
def test_hunyuan_code_build_marks_completion_after_all_commands(
    installer, monkeypatch, tmp_path, existing_venv,
):
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    checkout = vendor / "Hunyuan"
    checkout.mkdir()
    venv = vendor / ".venv"
    if existing_venv:
        venv.mkdir()
        (venv / "pyvenv.cfg").write_text("version_info = 3.11.14")
    python = venv / "bin/python"
    marker = vendor / ".i2l-built"
    route = "rocm7.2" if installer is rocm else "tested"
    monkeypatch.setattr(installer, "CHECKOUT", checkout)
    monkeypatch.setattr(installer, "VENDOR", vendor)
    monkeypatch.setattr(installer, "BUILT_MARKER", marker)
    monkeypatch.setattr(installer, "venv_python", lambda: python)
    monkeypatch.setattr(installer, "requirement_args", lambda: ["timm"])
    monkeypatch.setattr(installer.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(installer, "install_env", lambda: {"BUILD": "1"})
    events = []

    def run(argv, **kwargs):
        assert not marker.exists()
        events.append((argv, kwargs))

    def checkout_pinned():
        events.append("checkout")

    def compile_painter(exe):
        assert not marker.exists()
        assert exe == python
        events.append("compiled")

    monkeypatch.setattr(installer, "run", run)
    monkeypatch.setattr(installer, "fetch_checkout", checkout_pinned)
    monkeypatch.setattr(installer, "compile_mesh_painter", compile_painter)
    if installer is rocm:
        monkeypatch.setattr(rocm.hunyuan_patch, "patch", lambda path: events.append(("patched", path)))

    installer.install_code(route)

    expected = ["checkout"]
    if installer is rocm:
        expected.append(("patched", checkout))
    if not existing_venv:
        expected.append((["uv", "venv", str(venv), "--python", "3.11"], {}))
    # Assert the build phases independently of the code that assembles the command list.
    assert events[:len(expected)] == expected
    commands = events[len(expected):-1]
    assert len(commands) == 4
    assert all(args[:5] == ["uv", "pip", "install", "--python", str(python)]
               and kw == {"env": {"BUILD": "1"}} for args, kw in commands)
    assert "timm" in commands[-2][0]
    assert commands[-1][0][-1].endswith("hy3dpaint/custom_rasterizer")
    assert events[-1] == "compiled"
    assert json.loads(marker.read_text())["route"] == route
    assert installer.build_present()


def test_code_install_requires_uv_before_checkout(installer, monkeypatch):
    monkeypatch.setattr(installer.host, "find_uv", lambda: None)
    monkeypatch.setattr(installer, "fetch_checkout", lambda: pytest.fail("unexpected checkout"))
    with pytest.raises(SystemExit, match="uv is required"):
        installer.install_code("unused")
