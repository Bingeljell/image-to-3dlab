"""The NVIDIA TRELLIS.2 installer: picks a route from the card, announces before it fetches,
and never downloads without a yes. Nothing here touches the network."""

from __future__ import annotations

import io

import backend_catalog
import bootstrap_trellis_cuda as boot
import pytest


def test_weights_match_the_catalogue():
    """Every repo the installer fetches is one the Setup page announces, at the same size.
    TinyCLIP is the catalogue's advisory extra and is not fetched here."""
    trellis = backend_catalog.BY_ID["trellis"]
    announced = {w.source: w.bytes_expected for w in trellis.weights}
    for repo, _, size in boot.WEIGHTS:
        assert repo in announced, repo
        assert announced[repo] == pytest.approx(size * backend_catalog.GB, rel=0.05), repo


def test_the_gated_encoder_is_one_of_the_weights():
    assert boot.GATED in {repo for repo, _, _ in boot.WEIGHTS}


def test_pinned_commit_comes_from_the_audit_lock():
    assert len(boot.pinned_commit()) == 40


@pytest.mark.parametrize("driver,nvcc,expected", [
    ((13, 0), (13, 0), "cu130"),
    ((13, 0), (12, 8), "cu128"),   # newer driver, older toolkit: match the toolkit
    ((12, 8), (12, 8), "cu128"),
    ((12, 8), (13, 0), None),      # toolkit newer than the driver can run
    ((12, 4), (12, 4), None),
    (None, (12, 8), None),
    ((13, 0), None, None),
])
def test_torch_build_matches_driver_and_toolkit(driver, nvcc, expected):
    assert boot.torch_build_for(driver, nvcc) == expected


def test_torch_tag_version():
    assert boot.torch_tag_version("cu128") == (12, 8)
    assert boot.torch_tag_version("cu130") == (13, 0)


@pytest.mark.parametrize("cap,expected", [("89", "8.9"), ("120", "12.0"), ("86", "8.6"),
                                          (None, None), ("", None), ("x", None)])
def test_arch_list(cap, expected):
    assert boot.arch_list(cap) == expected


def test_blackwell_with_a_cuda13_driver_takes_the_pinned_wheels():
    route, _ = boot.choose_route((13, 0), "120", None, machine="x86_64")
    assert route == "pinned"


def test_blackwell_on_an_old_driver_compiles_instead():
    route, _ = boot.choose_route((12, 8), "120", (12, 8), machine="x86_64")
    assert route == "source"


def test_an_ada_card_with_a_toolkit_compiles():
    route, why = boot.choose_route((12, 9), "89", (12, 8), machine="x86_64")
    assert route == "source" and "cu128" in why


@pytest.mark.parametrize("driver,nvcc,needle", [
    (None, (12, 8), "nvidia-smi"),
    ((12, 4), (12, 4), "12.8"),
    ((12, 8), None, "nvcc"),
    ((12, 8), (11, 8), "toolkit"),
])
def test_no_route_says_why(driver, nvcc, needle):
    route, why = boot.choose_route(driver, "89", nvcc, machine="x86_64")
    assert route is None and needle in why


def test_pinned_plan_uses_the_runpod_pins_and_adds_our_matte():
    commands = boot.pip_commands("pinned", "uv", boot.venv_python())
    flat = [" ".join(c) for c in commands]
    assert any(str(boot.PINNED_REQUIREMENTS) in c for c in flat)
    assert any("rembg" in c for c in flat)
    assert not any("flash-attn" == c.split()[-1] for c in flat)


def test_source_plan_builds_every_extension_against_the_installed_torch():
    commands = boot.pip_commands("source", "uv", boot.venv_python(), "cu128")
    torch_at = next(i for i, c in enumerate(commands) if any(a.startswith("torch") for a in c))
    assert commands[torch_at][-1].endswith("/cu128")
    builds = [c for c in commands if "--no-build-isolation" in c]
    targets = {c[-1] for c in builds}
    for name, *_ in boot.SOURCE_EXTENSIONS:
        assert str(boot.EXTENSIONS / name) in targets
    assert str(boot.CHECKOUT / "o-voxel") in targets
    # Every build comes after torch, or it would compile against nothing.
    assert all(commands.index(c) > torch_at for c in builds)


def test_cuda_12_pins_the_torch_flash_attn_ships_a_wheel_for():
    # An unpinned torch outran flash-attn's wheels and cost a 1-2 hour compile on 8 CPUs.
    commands = boot.pip_commands("source", "uv", boot.venv_python(), "cu128")
    torch_install = next(c for c in commands if any(a.startswith("torch") for a in c))
    assert f"torch=={boot.TORCH_FOR_FLASH_WHEEL}" in torch_install
    flash = [c for c in commands if any("flash" in a for a in c)]
    assert len(flash) == 1 and flash[0][-1] == boot.flash_attn_target("cu128")
    assert flash[0][-1].endswith(".whl") and "--no-build-isolation" not in flash[0]


def test_flash_wheel_matches_the_pinned_torch_and_the_venv_python():
    url = boot.flash_attn_target("cu126")
    torch_minor = ".".join(boot.TORCH_FOR_FLASH_WHEEL.split(".")[:2])
    assert f"cu12torch{torch_minor}cxx11abiTRUE" in url
    assert "-cp311-" in url and boot.SOURCE_PYTHON == "3.11"
    assert url.startswith("https://github.com/Dao-AILab/flash-attention/releases/download/")


def test_cuda_13_has_no_flash_wheel_and_compiles_it():
    commands = boot.pip_commands("source", "uv", boot.venv_python(), "cu130")
    torch_install = next(c for c in commands if any(a.startswith("torch") for a in c))
    assert "torch" in torch_install and not any("==" in a for a in torch_install)
    assert any(c[-1] == boot.FLASH_ATTN and "--no-build-isolation" in c for c in commands)


def test_o_voxel_does_not_rebuild_the_extensions_just_built():
    # o-voxel names CuMesh and FlexGEMM as git dependencies, so a plain install fetched and
    # compiled both a second time over the ones built just before it; on a test pod that
    # hidden rebuild is where setup failed. Its small dependencies come from BASIC instead.
    commands = boot.pip_commands("source", "uv", boot.venv_python(), "cu128")
    o_voxel = next(c for c in commands if c[-1] == str(boot.CHECKOUT / "o-voxel"))
    assert "--no-deps" in o_voxel
    for package in ("numpy", "plyfile", "trimesh", "tqdm", "zstandard", "easydict"):
        assert package in boot.BASIC


def test_source_plan_needs_a_torch_tag():
    with pytest.raises(ValueError):
        boot.pip_commands("source", "uv", boot.venv_python(), None)


def test_build_env_sets_arch_jobs_and_toolkit(monkeypatch):
    monkeypatch.setattr(boot.host, "build_jobs", lambda: 6)
    env = boot.build_env("89", "/usr/local/cuda/bin/nvcc", {"PATH": "/usr/bin"})
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert env["MAX_JOBS"] == "6"
    assert env["PATH"].startswith("/usr/local/cuda/bin")
    assert env["CUDA_HOME"] == "/usr/local/cuda"


def test_announcement_names_backend_route_and_size():
    text = boot.announcement("source", "PyTorch cu128, extensions compiled for this card")
    assert "TRELLIS.2" in text and "NVIDIA" in text
    assert f"{boot.total_gb():.1f} GB" in text
    assert boot.GATED in text
    assert "30-60 minutes" in text
    assert "BRIA" in text


@pytest.fixture
def linux_nvidia(monkeypatch):
    monkeypatch.setattr(boot.host, "build_target", lambda *a, **k: "linux-nvidia")
    monkeypatch.setattr(boot, "detect", lambda: ("source", "test route", "cu128"))
    calls = []
    monkeypatch.setattr(boot, "check_access", lambda *a: calls.append("access"))
    monkeypatch.setattr(boot, "install_code", lambda *a: calls.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: calls.append("weights"))
    # the remover download has its own test; here it must never reach the network
    monkeypatch.setattr(boot, "install_background_remover", lambda *a, **k: None)
    return calls


def test_refuses_without_yes_when_nobody_can_answer(monkeypatch, linux_nvidia, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert boot.main([]) == 1
    assert linux_nvidia == []
    assert "Refusing" in capsys.readouterr().out


def test_a_no_downloads_nothing(monkeypatch, linux_nvidia):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert boot.main([]) == 1
    assert linux_nvidia == []


def test_yes_checks_access_before_building(linux_nvidia):
    assert boot.main(["--yes"]) == 0
    assert linux_nvidia == ["access", "code", "weights"]


def test_code_only_and_weights_only(linux_nvidia):
    boot.main(["--yes", "--code-only"])
    boot.main(["--yes", "--weights-only"])
    assert linux_nvidia == ["access", "code", "access", "weights"]


def test_gated_refusal_stops_before_anything_is_built(monkeypatch, linux_nvidia, capsys):
    def refuse(*_):
        raise boot.GatedAccess(boot.GATED)

    monkeypatch.setattr(boot, "check_access", refuse)
    assert boot.main(["--yes"]) == 1
    assert linux_nvidia == []
    assert "hf auth login" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["macos-arm64", "windows-nvidia", None])
def test_other_machines_are_refused_before_any_download(monkeypatch, target, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda *a, **k: target)
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Windows is not supported" in capsys.readouterr().out


def test_no_route_refuses_before_asking(monkeypatch, linux_nvidia, capsys):
    monkeypatch.setattr(boot, "detect", lambda: (None, "no nvcc", None))
    assert boot.main(["--yes"]) == 1
    assert linux_nvidia == []
    assert "no nvcc" in capsys.readouterr().out


def test_build_is_present_only_after_the_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "BUILT_MARKER", tmp_path / ".done")
    assert boot.build_present() is False
    (tmp_path / ".done").write_text("{}")
    assert boot.build_present() is True


def test_every_route_installs_with_patient_downloads(monkeypatch):
    monkeypatch.setattr(boot.host, "compute_capability", lambda: "86")
    monkeypatch.setattr(boot.host, "find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    for route in ("pinned", "source"):
        assert "UV_HTTP_TIMEOUT" in boot.install_env(route, {"PATH": "/usr/bin"})


def test_setup_fetches_every_weight_the_setup_card_counts():
    # On a real NVIDIA pod the card counted TinyCLIP but setup never fetched it, so a
    # finished setup read "partial" and Generate said TRELLIS was not installed.
    fetched = {repo for repo, _, _ in boot.WEIGHTS}
    listed = {w.source for w in backend_catalog.BY_ID["trellis"].weights}
    assert fetched | {boot.matte.LITE_URL} == listed  # the remover comes from GitHub


def test_nvidia_setup_installs_the_background_remover(tmp_path):
    # Without it, the first generation fetched u2net (176 MB) unannounced.
    fetched = []
    boot.install_background_remover(tmp_path / "lite.onnx", download=fetched.append)
    assert fetched == [tmp_path / "lite.onnx"]
    assert "BiRefNet-lite" in boot.announcement("source", "test")


def test_installs_the_basics_from_the_lock():
    assert boot.basic_args() == ["-r", str(boot.PACKAGE_LOCK)] and boot.PACKAGE_LOCK.is_file()
    commands = boot.pip_commands("source", "uv", boot.venv_python(), "cu128")
    assert any(c[-2:] == ["-r", str(boot.PACKAGE_LOCK)] for c in commands)
    lock = boot.PACKAGE_LOCK.read_text()
    assert "\ntransformers==4.57.3\n" in lock
    assert not any(line.startswith(("torch==", "torchvision==", "nvidia-"))
                   for line in lock.splitlines())


def test_source_extensions_are_pinned_to_commits():
    # CuMesh and FlexGEMM were cloned at whatever HEAD was that day.
    for name, _url, ref, _recursive in boot.SOURCE_EXTENSIONS:
        assert ref, name
        if name != "nvdiffrast":  # nvdiffrast is pinned by its release tag
            assert len(ref) == 40 and all(ch in "0123456789abcdef" for ch in ref), name


def test_a_commit_ref_is_checked_out_after_cloning(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(boot, "run", lambda command, **kwargs: calls.append(command))
    sha = "a" * 40
    boot.clone("https://example/x.git", tmp_path / "x", sha, recursive=True)
    assert "--branch" not in calls[0]
    assert ["git", "checkout", "--detach", sha] in [c[:4] for c in calls]
    boot.clone("https://example/y.git", tmp_path / "y", "v0.4.0")
    assert ["--branch", "v0.4.0"] == calls[-1][2:4]
