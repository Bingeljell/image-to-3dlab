"""Test TRELLIS setup without reaching a model service or compiler."""

import importlib
import io
import sys
from types import ModuleType

import pytest

from scripts import bootstrap_trellis_cuda as cuda
from scripts import bootstrap_trellis_space_macos as mac


class GateDenied(Exception):
    pass


@pytest.fixture
def hub(monkeypatch):
    module = ModuleType("huggingface_hub")
    utils = ModuleType("huggingface_hub.utils")
    utils.GatedRepoError = GateDenied
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return module


@pytest.mark.parametrize("route", ["mac", "nvidia"])
@pytest.mark.parametrize("mode", ["not-terminal", "declined", "gated", "fetch-gated", "yes"])
def test_trellis_weights_confirm_and_check_gate_before_fetching(
    route, mode, hub, monkeypatch, tmp_path, capsys
):
    cuda_module = importlib.import_module("bootstrap_trellis_cuda") if route == "mac" else cuda
    cached = tmp_path / "cached-matte.safetensors"
    cached.touch()
    calls = []

    def auth(repo):
        assert repo == cuda.GATED
        calls.append(("auth", repo))
        if mode == "gated":
            raise GateDenied(repo)

    def fetch(repo, allow_patterns):
        calls.append(("fetch", repo, allow_patterns))
        if mode == "fetch-gated":
            raise GateDenied(repo)

    monkeypatch.setattr(hub, "auth_check", auth, raising=False)
    monkeypatch.setattr(hub, "snapshot_download", fetch, raising=False)
    monkeypatch.setattr(cuda.matte, "model_file", lambda _: cached)
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: mode != "not-terminal")

    def answer(prompt):
        assert prompt == "Continue? [y/N] "
        return "no" if mode == "declined" else "yes"

    monkeypatch.setattr("builtins.input", answer)
    apple = route == "mac"
    host = cuda.host
    monkeypatch.setattr(host, "build_target", lambda: "macos-arm64" if apple else "linux-nvidia")
    monkeypatch.setattr(host, "macos_major", lambda: mac.MIN_MACOS)
    monkeypatch.setattr(host, "driver_cuda_version", lambda: None)
    monkeypatch.setattr(host, "find_nvcc", lambda: None)
    monkeypatch.setattr(host, "nvcc_cuda_version", lambda _: None)
    monkeypatch.setattr(host, "compute_capability", lambda: None)

    if apple and mode in {"gated", "fetch-gated"}:
        with pytest.raises(SystemExit) as exc:
            mac.main(["--weights-only"])
        assert exc.value.code == 1
    else:
        assert (mac if apple else cuda).main(["--weights-only"]) == int(mode != "yes")
    out = capsys.readouterr().out
    assert "backend: TRELLIS.2" in out
    assert ("Mac clean port" if apple else "NVIDIA / CUDA") in out
    assert f"weights: {cuda.total_gb():.1f} GB" in out

    expected = [("auth", cuda.GATED)]
    if mode == "yes":
        expected.extend(("fetch", repo, patterns) for repo, patterns, _ in cuda_module.WEIGHTS)
        assert "Done." in out
    elif mode == "fetch-gated":
        repo, patterns, _ = cuda_module.WEIGHTS[0]
        expected.append(("fetch", repo, patterns))
    elif mode in {"not-terminal", "declined"}:
        expected = []
    if mode in {"gated", "fetch-gated"}:
        assert "Hugging Face refused" in out
        assert "hf auth login" in out
    assert calls == expected
    assert all("briaai/" not in call[1].lower() for call in calls)


def test_trellis_missing_hub_reports_setup_error(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    cuda.check_access()
    assert "no auth_check" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="huggingface_hub is not installed"):
        cuda.install_weights()


@pytest.mark.parametrize("build", [False, True])
def test_mac_setup_pins_all_checkouts_then_applies_patches(build, monkeypatch, tmp_path):
    import json
    import subprocess

    root = tmp_path / "trellis"
    lock = json.loads(mac.LOCK.read_text())["upstreams"]
    targets = {str(root / relative): lock[name] for name, relative in mac.TARGETS.items()}
    commits = {}
    seen = []
    monkeypatch.setattr(mac.host, "build_target", lambda: "macos-arm64")
    monkeypatch.setattr(mac.host, "macos_major", lambda: mac.MIN_MACOS)
    monkeypatch.setattr(mac.host, "find_uv", lambda: "test-uv")
    monkeypatch.delenv("MACOSX_DEPLOYMENT_TARGET", raising=False)

    def run(argv, *, cwd, env, check):
        assert check
        seen.append(argv)
        if argv[:2] == ["git", "clone"]:
            target = argv[-1]
            assert argv[2] == targets[target]["url"]
            commits[target] = "a" * 40
        elif argv[:2] == ["git", "fetch"]:
            assert argv[2:] == ["origin", targets[str(cwd)]["commit"]]
        elif argv[:2] == ["git", "checkout"]:
            assert argv[2:] == ["--detach", targets[str(cwd)]["commit"]]
            commits[str(cwd)] = argv[-1]
        elif argv[0] == sys.executable:
            assert commits == {path: info["commit"] for path, info in targets.items()}
            assert argv[2] == "--root"
        elif argv[:2] == ["test-uv", "venv"]:
            assert argv[2:] == [str(root / ".venv"), "--python", "python-test"]
        else:
            assert argv[:4] == ["test-uv", "pip", "install", "--python"]
            assert argv[4] == str(root / ".venv/bin/python")
            assert env["MACOSX_DEPLOYMENT_TARGET"] == "12.0"
        return subprocess.CompletedProcess(argv, 0)

    def output(argv, *, cwd, text):
        assert text
        if argv == ["git", "status", "--short"]:
            return ""
        assert argv == ["git", "rev-parse", "HEAD"]
        return f"{commits[str(cwd)]}\n"

    monkeypatch.setattr(mac.subprocess, "run", run)
    monkeypatch.setattr(mac.subprocess, "check_output", output)
    flags = [] if build else ["--skip-install"]
    assert mac.main(["--root", str(root), "--python", "python-test", *flags]) == 0
    patches = [argv for argv in seen if argv[0] == sys.executable]
    assert [argv[1].split("/")[-1] for argv in patches] == [
        "patch_trellis_space_core.py", "patch_trellis_metal_backends.py",
    ]
    assert [argv[-1] for argv in patches] == [str(root / "TRELLIS.2"), str(root)]
    build_commands = [argv for argv in seen if argv[0] == "test-uv"]
    if build:
        assert [argv[-1] for argv in build_commands[1:]] == [
            str(mac.REQUIREMENTS),
            *(str(root / "deps" / dependency) for dependency in
              ("utils3d", "mtlbvh", "mtldiffrast", "mtlmesh", "mtlgemm", "trellis2-apple/o-voxel")),
        ]
    else:
        assert not build_commands


@pytest.mark.parametrize("state", ["pinned", "dirty", "could-not-pin"])
def test_mac_setup_handles_existing_git_tree(state, monkeypatch, tmp_path):
    root = tmp_path / "source"
    (root / ".git").mkdir(parents=True)
    pin = "b" * 40
    existing = pin if state == "pinned" else "a" * 40
    commands = []

    def output(argv, *, cwd, text):
        assert cwd == root
        if argv == ["git", "status", "--short"]:
            return " M user-work"
        return f"{existing}\n"

    # Leave HEAD unchanged in the failure case, even when checkout reports success.
    def clean_output(argv, *, cwd, text):
        if argv == ["git", "status", "--short"]:
            return ""
        return output(argv, cwd=cwd, text=text)

    monkeypatch.setattr(mac.subprocess, "check_output", output if state == "dirty" else clean_output)
    monkeypatch.setattr(mac.subprocess, "run", lambda argv, **_: commands.append(argv))
    if state == "pinned":
        mac.clone_at("https://example.invalid/source", pin, root)
        assert not commands
    elif state == "dirty":
        with pytest.raises(RuntimeError, match="is dirty"):
            mac.clone_at("https://example.invalid/source", pin, root)
        assert not commands
    else:
        with pytest.raises(RuntimeError, match="failed to pin"):
            mac.clone_at("https://example.invalid/source", pin, root)
        assert commands == [["git", "fetch", "origin", pin], ["git", "checkout", "--detach", pin]]


def test_mac_setup_requires_uv_before_build(monkeypatch, tmp_path):
    monkeypatch.setattr(mac.host, "find_uv", lambda: None)
    with pytest.raises(RuntimeError, match="uv is required"):
        mac.install(tmp_path, "python3.11")
    assert list(tmp_path.iterdir()) == []
