"""The Pixal3D installer: one script for a Mac build and an NVIDIA prebuilt, never a
download without an explicit yes."""

from __future__ import annotations

import io
import tarfile
import types
import zipfile
from pathlib import Path

import backend_catalog
import bootstrap_pixal3d as boot
import pytest


def test_weight_size_matches_the_catalogue():
    """The Setup page and this script must not quote different numbers for one download."""
    pixal = next(b for b in backend_catalog.CATALOG if b.id == "pixal3d")
    assert sum(w.bytes_expected for w in pixal.weights) == pytest.approx(
        boot.WEIGHTS_GB * backend_catalog.GB + boot.matte.LITE_BYTES, rel=0.01)


@pytest.fixture
def new_driver(monkeypatch):
    """A driver new enough for upstream's CUDA 12.9 prebuilt, and no compiler."""
    monkeypatch.setattr(boot, "driver_cuda", lambda: (12, 9))
    monkeypatch.setattr(boot, "find_nvcc", lambda: None)


@pytest.mark.parametrize("key,route,size", [
    ("macos-arm64", "built from source with Metal", "Xcode"),
    ("linux-nvidia", "CUDA 12 prebuilt", "~640 MB"),
    ("windows-nvidia", "CUDA 12 prebuilt", "~610 MB"),
])
def test_announcement_names_backend_route_size_and_licence(monkeypatch, new_driver,
                                                           key, route, size):
    monkeypatch.setattr(boot, "target", lambda: key)
    text = boot.announcement()
    for needle in ("Pixal3D", route, size, "8.4 GB", "MIT", "DINOv3"):
        assert needle in text


def test_unsupported_machine_is_refused_before_anything_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(boot, "target", lambda: None)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Apple Silicon Mac, or Linux/Windows with an NVIDIA card" in capsys.readouterr().out


def test_no_yes_and_no_terminal_means_no_download(monkeypatch, new_driver):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main([]) == 1


def test_rerun_on_an_installed_tree_does_not_rebuild(monkeypatch, capsys, new_driver):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "build_present", lambda: True)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("rebuilt"))
    fetched = []
    monkeypatch.setattr(boot, "install_weights", lambda *a: fetched.append(1))
    assert boot.main(["--yes"]) == 0
    assert "already installed" in capsys.readouterr().out
    assert fetched == [1]  # the weight fetch resumes, it does not restart


def test_build_only_and_weights_only_do_one_half_each(monkeypatch, new_driver):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "build_present", lambda: False)
    called = []
    monkeypatch.setattr(boot, "install_build", lambda *a: called.append("build"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: called.append("weights"))
    assert boot.main(["--yes", "--build-only"]) == 0
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert called == ["build", "weights"]


# The CUDA asset names on pixal3d.cpp's v0.10.1-desktop-alpha release, 2026-09-23.
RELEASE = [{"name": n, "browser_download_url": f"https://x/{n}"} for n in (
    "trellis-cuda-linux-x64.tar.gz", "trellis-cuda-windows-x64.zip",
    "trellis-cuda12-linux-x64.tar.gz", "trellis-cuda12-windows-x64.zip",
    "trellis-vulkan-linux-x64.tar.gz", "trellis-metal-macos-arm64.tar.gz",
)]


@pytest.mark.parametrize("key,expected", [
    ("linux-nvidia", "trellis-cuda12-linux-x64.tar.gz"),
    ("windows-nvidia", "trellis-cuda12-windows-x64.zip"),
])
def test_prebuilt_is_the_cuda12_build_for_this_os(key, expected):
    """CUDA 12, not the unversioned (newer) CUDA build: it runs on older drivers."""
    assert boot.pick_prebuilt(RELEASE, key)["name"] == expected


def test_missing_prebuilt_is_none():
    assert boot.pick_prebuilt(RELEASE[4:], "linux-nvidia") is None


def _tarball(path: Path, members: dict[str, bytes], links: dict[str, str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)


def test_unpacking_a_linux_prebuilt_keeps_library_symlinks_and_runs(tmp_path, monkeypatch):
    """The CUDA tarball carries `libcudart.so.12 -> libcudart.so.12.9.79` style links; a
    flattened copy that loses them fails to load at run time, not at install time."""
    monkeypatch.setattr(boot.host, "os_family", lambda *a: "linux")
    archive = tmp_path / "t.tar.gz"
    _tarball(archive, {"./trellis-cli": b"x", "./libcudart.so.12.9.79": b"x"},
             {"libcudart.so.12": "libcudart.so.12.9.79"})
    build = tmp_path / "build"
    cli = boot.unpack_prebuilt(archive, build)
    assert cli == build / "trellis-cli"
    assert cli.stat().st_mode & 0o111
    assert (build / "libcudart.so.12").is_symlink()


def test_unpacking_a_windows_prebuilt_finds_the_exe(tmp_path, monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda *a: "windows")
    archive = tmp_path / "t.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("trellis-cli.exe", "x")
        z.writestr("cudart64_12.dll", "x")
    assert boot.unpack_prebuilt(archive, tmp_path / "build").name == "trellis-cli.exe"


def test_an_archive_without_the_cli_is_an_error_not_a_silent_success(tmp_path, monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda *a: "linux")
    archive = tmp_path / "t.tar.gz"
    _tarball(archive, {"./README": b"x"}, {})
    with pytest.raises(SystemExit, match="no trellis-cli"):
        boot.unpack_prebuilt(archive, tmp_path / "build")


def test_matting_model_is_moved_flat_where_trellis_cli_looks(tmp_path):
    (tmp_path / "q8").mkdir()
    (tmp_path / "q8" / "birefnet.gguf").write_text("x")
    boot.flatten_matte(tmp_path)
    assert (tmp_path / "birefnet.gguf").exists()
    assert not (tmp_path / "q8").exists()
    boot.flatten_matte(tmp_path)  # idempotent


def test_the_viewer_runs_this_script_with_yes():
    """The browser dialog is the confirmation; asking again on a detached stdin hangs."""
    import download_api

    command = download_api.COMMANDS["pixal3d"]
    assert command[1].endswith("bootstrap_pixal3d.py") and "--yes" in command


# Pixal3D's CUDA 12 prebuilt is compiled with CUDA 12.9. On the RunPod 4090 (driver 570,
# CUDA 12.8) it died at its first kernel: "the provided PTX was compiled with an
# unsupported toolchain". On a newer driver it runs, at about half the speed of a local
# compile (390 s against ~190 s, RunPod 2026-09-23), but the compile took 15 minutes on an
# A40 pod (2026-10-02). So the prebuilt wins whenever the driver can run it; a compiler is
# the fallback for an old driver.
NVCC = "/usr/local/cuda/bin/nvcc"


@pytest.mark.parametrize("key,cuda,nvcc,nvcc_cuda,expected", [
    ("linux-nvidia", (12, 9), None, None, "prebuilt"),
    ("linux-nvidia", (13, 0), NVCC, (12, 8), "prebuilt"),
    # Driver too old for the prebuilt: compile instead.
    ("linux-nvidia", (12, 8), NVCC, (12, 8), "cuda-source"),
    # A version we could not read is tried rather than refused.
    ("linux-nvidia", (12, 8), NVCC, None, "cuda-source"),
    # A toolkit newer than the driver builds kernels the driver cannot run.
    ("linux-nvidia", (13, 0), NVCC, (13, 1), "prebuilt"),
    ("linux-nvidia", (12, 8), NVCC, (13, 0), None),
    ("linux-nvidia", (12, 8), None, None, None),
    ("linux-nvidia", None, None, None, None),
    ("windows-nvidia", (12, 9), None, None, "prebuilt"),
    # A Windows source build is a Visual Studio project of its own; not offered.
    ("windows-nvidia", (12, 8), "C:/cuda/nvcc.exe", (12, 8), None),
    ("windows-nvidia", (13, 0), "C:/cuda/nvcc.exe", (12, 8), "prebuilt"),
    ("macos-arm64", None, None, None, "metal-source"),
])
def test_the_driver_and_compiler_pick_the_route(key, cuda, nvcc, nvcc_cuda, expected):
    assert boot.build_kind(key, cuda, nvcc, nvcc_cuda) == expected


def test_a_compile_can_be_asked_for_when_it_would_run():
    """For an agent following the README's "compile it locally" note."""
    assert boot.build_kind("linux-nvidia", (13, 0), NVCC, (12, 8),
                           prefer_compile=True) == "cuda-source"
    # Asking for it with no compiler still gets the prebuilt.
    assert boot.build_kind("linux-nvidia", (13, 0), None, None,
                           prefer_compile=True) == "prebuilt"


def test_the_compile_flag_reaches_the_route(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "driver_cuda", lambda: (13, 0))
    monkeypatch.setattr(boot, "find_nvcc", lambda: NVCC)
    monkeypatch.setattr(boot, "nvcc_cuda", lambda _: (12, 8))
    monkeypatch.setattr(boot, "build_present", lambda: False)
    chosen = []
    monkeypatch.setattr(boot, "install_build", lambda key, kind: chosen.append(kind))
    monkeypatch.setattr(boot, "install_weights", lambda *a: None)
    assert boot.main(["--yes", "--compile"]) == 0
    assert boot.main(["--yes"]) == 0
    assert chosen == ["cuda-source", "prebuilt"]


def test_an_old_driver_without_a_compiler_is_told_what_to_update(monkeypatch, capsys):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "driver_cuda", lambda: (12, 8))
    monkeypatch.setattr(boot, "find_nvcc", lambda: None)
    monkeypatch.setattr(boot, "build_present", lambda: False)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "575" in out and "12.8" in out


def test_an_old_driver_with_a_compiler_announces_a_local_cuda_build(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "driver_cuda", lambda: (12, 8))
    monkeypatch.setattr(boot, "find_nvcc", lambda: NVCC)
    monkeypatch.setattr(boot, "nvcc_cuda", lambda _: (12, 8))
    text = boot.announcement()
    assert "compiled locally with CUDA" in text and "prebuilt" not in text
    # Says why it is compiling, and that a driver update skips the wait.
    assert "12.8" in text and "575" in text and "about a minute" in text


def test_choosing_to_compile_on_a_new_driver_gets_no_driver_advice(monkeypatch, new_driver):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(boot, "find_nvcc", lambda: NVCC)
    monkeypatch.setattr(boot, "nvcc_cuda", lambda _: (12, 8))
    assert "575" not in boot.announcement(prefer_compile=True)


def test_cuda_source_build_targets_this_card(monkeypatch):
    flags = boot.cmake_flags("cuda-source", nvcc="/usr/local/cuda/bin/nvcc", arch="89")
    assert "-DGGML_CUDA=ON" in flags
    assert "-DCMAKE_CUDA_ARCHITECTURES=89" in flags
    assert "-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" in flags


def test_cuda_source_build_without_a_known_card_lets_cmake_ask():
    flags = boot.cmake_flags("cuda-source", nvcc="nvcc", arch=None)
    assert "-DCMAKE_CUDA_ARCHITECTURES=native" in flags


def test_metal_build_passes_no_cuda_flags():
    assert not any("CUDA" in f for f in boot.cmake_flags("metal-source"))


# --- AMD: HIP, compiled here, because upstream ships no AMD prebuilt ----------------------
# GGML_HIP is ggml's own switch; HIPBLAS is the math library underneath it. Neither is CUDA
# on an AMD card, so a build that passes either would compile nothing that can run.
HIPCC = "/opt/rocm/bin/hipcc"


@pytest.mark.parametrize("hipcc,gfx,expected", [
    (HIPCC, ["gfx1100"], "hip-source"),
    # The compiler without the card would build for an ISA nobody has.
    (HIPCC, [], None),
    (HIPCC, None, None),
    # A card with no ROCm toolchain: the machine can be named AMD and still have no route.
    (None, ["gfx1100"], None),
])
def test_amd_compiles_with_hip_when_both_halves_are_there(hipcc, gfx, expected):
    assert boot.build_kind("linux-amd", None, None, None, False, hipcc, gfx) == expected


def test_amd_ignores_the_nvidia_answer():
    """An AMD box with an old NVIDIA driver left in the machine must not be sent down the
    prebuilt route: it has no AMD prebuilt."""
    assert boot.build_kind("linux-amd", (12, 9), None, None, False, HIPCC,
                           ["gfx1100"]) == "hip-source"


def test_the_hip_build_is_given_rocms_clang_not_the_hipcc_wrapper(monkeypatch, tmp_path):
    """CMake refuses hipcc outright ("CMAKE_HIP_COMPILER is set to the hipcc wrapper ... This
    is not supported"), so naming it makes configure fail on every AMD machine. It is the
    Clang beside it that goes in the flag."""
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    monkeypatch.setattr(boot.Path, "exists",
                        lambda self: str(self).endswith(("llvm/bin/clang++",)))
    assert boot.hip_compiler(HIPCC) == "/opt/rocm/llvm/bin/clang++"
    flags = boot.cmake_flags("hip-source", hipcc=HIPCC, gfx=["gfx1100"])
    compiler = next(f for f in flags if f.startswith("-DCMAKE_HIP_COMPILER="))
    assert compiler == "-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++"


def test_no_rocm_clang_means_no_compiler_flag_at_all(monkeypatch, tmp_path):
    """Better an absent flag CMake can resolve itself than a wrong one. This is the shape
    of the failure: the HIP SDK without its compiler, which is what a partial ROCm install
    gives you."""
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    monkeypatch.setattr(boot.Path, "exists", lambda self: False)
    assert boot.hip_compiler(HIPCC) is None
    assert not any("HIP_COMPILER" in f
                   for f in boot.cmake_flags("hip-source", hipcc=HIPCC, gfx=["gfx1100"]))


def test_no_rocm_at_all_means_no_compiler(monkeypatch):
    monkeypatch.setattr(boot, "find_hipcc", lambda: None)
    assert boot.hip_compiler() is None


def test_hip_build_names_the_card_in_both_spellings():
    """ggml reads AMDGPU_TARGETS and CMake's HIP language reads CMAKE_HIP_ARCHITECTURES.
    ggml only falls back to the first when the second is unset, so both get the list."""
    flags = boot.cmake_flags("hip-source", hipcc=HIPCC, gfx=["gfx1100"])
    assert "-DGGML_HIP=ON" in flags
    assert "-DCMAKE_HIP_ARCHITECTURES=gfx1100" in flags
    assert "-DAMDGPU_TARGETS=gfx1100" in flags
    assert not any("CUDA" in f for f in flags)


def test_hip_build_points_cmake_at_rocm():
    """Without ROCm's own directory, CMake's HIP support cannot find the runtime."""
    flags = boot.cmake_flags("hip-source", hipcc=HIPCC, gfx=["gfx1100"])
    assert any("CMAKE_PREFIX_PATH" in f for f in flags)
    assert any(f.startswith("-DROCM_PATH=") for f in flags)


def test_two_cards_compile_for_both():
    flags = boot.cmake_flags("hip-source", hipcc=HIPCC, gfx=["gfx1100", "gfx1200"])
    assert "-DAMDGPU_TARGETS=gfx1100;gfx1200" in flags


def test_extra_cmake_flags_come_from_the_environment():
    """A card ROCm needs help with (a HIPBLASLt build, say) must not need a fork of this
    script to pass one flag."""
    assert boot.extra_cmake_flags({}) == []
    env = {boot.EXTRA_CMAKE_FLAGS: "-DGGML_HIPBLASLT=ON -DGGML_VMM=OFF"}
    assert boot.extra_cmake_flags(env) == ["-DGGML_HIPBLASLT=ON", "-DGGML_VMM=OFF"]


def test_amd_announces_a_hip_compile_and_its_size(monkeypatch):
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    monkeypatch.setattr(boot, "gfx_targets", lambda: ["gfx1100"])
    text = boot.announcement()
    assert "HIP" in text and "8.4 GB" in text and "MIT" in text
    # The 10-20 minutes is the thing a user needs to know before agreeing to it.
    assert "10-20 minutes" in text


def test_amd_without_the_hip_sdk_is_told_which_package_installs_it(monkeypatch, capsys):
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "find_hipcc", lambda: None)
    monkeypatch.setattr(boot, "build_present", lambda: False)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "rocm-hip-sdk" in out and "Nothing downloaded" in out


def test_amd_with_a_compiler_but_no_readable_card_can_be_told_the_target(monkeypatch, capsys):
    """rocminfo can be missing from a container that has the SDK. Refusing outright would
    strand it, so the answer is how to name the card by hand."""
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    monkeypatch.setattr(boot, "gfx_targets", list)
    monkeypatch.setattr(boot, "build_present", lambda: False)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "PIXAL3D_CMAKE_FLAGS" in out and "gfx1100" in out


def test_an_unreadable_amd_card_does_not_reach_the_weights(monkeypatch):
    """The one case that must never fall through: hipcc present, no gfx target, so CMake
    would pick an ISA of its own choosing and the binary would miss every kernel."""
    monkeypatch.setattr(boot, "target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    monkeypatch.setattr(boot, "gfx_targets", list)
    monkeypatch.setattr(boot, "build_present", lambda: False)
    monkeypatch.setattr(boot, "install_build", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1


def test_the_hip_build_says_hip_not_cuda_on_its_progress_line(monkeypatch, capsys):
    """A 20-minute compile that announces CUDA on an AMD machine is a user who stops and
    assumes the tool picked the wrong card."""
    ran = []
    monkeypatch.setattr(boot.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(boot, "fetch_source", lambda ref: None)
    monkeypatch.setattr(boot, "apply_steps_patch", lambda: True)
    monkeypatch.setattr(boot, "gfx_targets", lambda: ["gfx1100"])
    monkeypatch.setattr(boot, "find_hipcc", lambda: HIPCC)
    # ROCm's Clang is looked up on disk; this test is about the label, not the lookup,
    # and must pass on a machine with no ROCm installed.
    monkeypatch.setattr(boot, "hip_compiler", lambda hipcc=None: "/opt/rocm/llvm/bin/clang++")
    monkeypatch.setattr(boot, "build_present", lambda: True)
    monkeypatch.setattr(boot.subprocess, "run",
                        lambda command, **kw: ran.append(command))
    boot.build_from_source("hip-source")
    assert "Building (HIP)" in capsys.readouterr().out
    configure = next(c for c in ran if c[0] == "cmake")
    assert "-DGGML_HIP=ON" in configure
    assert not any("CUDA" in flag for flag in configure)


def test_the_hip_build_fetches_the_tested_commit():
    """The same commit the CUDA build pins, so AMD and NVIDIA run the same code."""
    assert boot.source_ref("hip-source") == boot.PREBUILT_COMMIT


def test_a_hip_build_rebuilds_in_place_like_any_other_source_build():
    assert boot.build_decision(True, True, "hip-source") == "rebuild"


def test_the_build_command_names_a_job_count():
    """A bare `-j` is unbounded: dozens of nvcc jobs on a 96-CPU, 31 GB pod got OOM-killed."""
    command = boot.build_command(jobs=10)
    assert command[-2:] == ["-j", "10"]
    assert "trellis-cli" in command


def test_a_failed_steps_patch_does_not_fail_the_install(capsys):
    """Upstream moving the anchor must cost the speed-up, not the whole install."""
    def runner(command, **kwargs):
        assert command[-1].endswith("patch_pixal3d_steps.py")
        return types.SimpleNamespace(returncode=1, stdout="", stderr="anchor not found")
    assert boot.apply_steps_patch(runner) is False
    assert "12 steps" in capsys.readouterr().out


def test_a_successful_steps_patch_is_reported(capsys):
    def runner(command, **kwargs):
        return types.SimpleNamespace(returncode=0, stdout="patched flow_runner.cpp", stderr="")
    assert boot.apply_steps_patch(runner) is True


# --- Rebuilding an existing install so it picks up our patches ------------------------------


@pytest.mark.parametrize("present, rebuild, kind, expected", [
    (False, False, "metal-source", "install"),
    (False, True, "metal-source", "install"),
    (True, False, "metal-source", "keep"),
    (True, True, "metal-source", "rebuild"),
    (True, True, "cuda-source", "rebuild"),
    (True, True, "prebuilt", "cannot-rebuild"),
])
def test_an_existing_build_is_only_rebuilt_when_asked(present, rebuild, kind, expected):
    """0.3.4's 8-step default needs the steps patch compiled in; before --rebuild, an
    existing install printed "leaving it alone" and the release notes' command did nothing."""
    assert boot.build_decision(present, rebuild, kind) == expected


def test_rebuild_reaches_the_build_decision(monkeypatch, capsys):
    """--rebuild on an existing Mac build recompiles instead of leaving it alone."""
    calls = []
    monkeypatch.setattr(boot, "target", lambda: "macos-arm64")
    monkeypatch.setattr(boot, "current_kind", lambda key, prefer=False: "metal-source")
    monkeypatch.setattr(boot, "build_present", lambda: True)
    monkeypatch.setattr(boot, "rebuild_existing", lambda: calls.append("rebuild"))
    monkeypatch.setattr(boot, "build_from_source", lambda kind: calls.append("fresh"))
    monkeypatch.setattr(boot, "install_weights", lambda: calls.append("weights"))
    assert boot.main(["--build-only", "--rebuild", "--yes"]) == 0
    assert calls == ["rebuild"]
    assert "Rebuilding" in capsys.readouterr().out


def test_a_rebuild_patches_then_compiles_without_the_fresh_install_checks(monkeypatch):
    """No Metal-compiler probe: a fresh Terminal usually cannot see it, and C++ is all
    that changes."""
    ran = []

    def runner(command, **kwargs):
        ran.append(command)
        return types.SimpleNamespace(returncode=0, stdout="patched", stderr="")

    monkeypatch.setattr(boot, "build_present", lambda: True)
    boot.rebuild_existing(runner)
    assert ran[0][-1].endswith("patch_pixal3d_steps.py")
    assert ran[1][:2] == ["cmake", "--build"]
    assert not any("xcrun" in part for command in ran for part in command)


def test_announcement_names_the_background_remover_it_brings(monkeypatch, new_driver):
    monkeypatch.setattr(boot, "target", lambda: "linux-nvidia")
    assert "BiRefNet-lite" in boot.announcement()


def test_background_remover_is_fetched_once_and_only_when_missing(tmp_path):
    """A fresh install fell back to u2net and lost a robot's arms (2026-09-30)."""
    target = tmp_path / "birefnet-general-lite.onnx"
    fetched = []
    boot.install_background_remover(target, download=lambda t: fetched.append(t))
    assert fetched == [target]
    target.write_bytes(b"x")
    boot.install_background_remover(target, download=lambda t: pytest.fail("refetched"))


def test_the_cuda_source_build_fetches_the_tested_commit():
    assert boot.source_ref("cuda-source") == boot.PREBUILT_COMMIT
    assert len(boot.PREBUILT_COMMIT) == 40
    assert boot.source_ref("metal-source") == "HEAD"


def test_weights_come_from_the_tested_revisions(monkeypatch, tmp_path):
    """A fresh install must get the files that were tested, not whatever was pushed since."""
    seen = {}
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda repo, **kw: seen.setdefault(repo, kw["revision"])
    hub.hf_hub_download = lambda repo, name, **kw: seen.setdefault(repo, kw["revision"])
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", hub)
    monkeypatch.setattr(boot, "install_background_remover", lambda: None)
    boot.install_weights(tmp_path)
    assert seen == {boot.WEIGHTS_REPO: boot.WEIGHTS_REVISION,
                    boot.MATTE_REPO: boot.MATTE_REVISION}
