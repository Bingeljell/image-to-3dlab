"""What machine is this: the one question every bootstrap and the viewer ask first."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from image_to_3dlab import host


def _smi(returncode: int, stdout: str):
    def run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")
    return run


def test_os_family_names_the_three_we_support():
    assert host.os_family("darwin") == "macos"
    assert host.os_family("linux") == "linux"
    assert host.os_family("win32") == "windows"
    assert host.os_family("freebsd13") == "other"


def test_apple_silicon_is_decided_without_asking_for_a_gpu():
    def never(*a, **k):
        raise AssertionError("a Mac must not probe for nvidia-smi")
    assert host.host_platform("darwin", "arm64", nvidia=never) == host.APPLE


def test_intel_mac_is_not_apple_silicon():
    assert host.host_platform("darwin", "x86_64", nvidia=lambda: False) == "other"


def test_linux_and_windows_with_an_nvidia_card_are_nvidia():
    assert host.host_platform("linux", "x86_64", nvidia=lambda: True, amd=lambda: False) == host.NVIDIA
    assert host.host_platform("win32", "AMD64", nvidia=lambda: True, amd=lambda: False) == host.NVIDIA


def test_linux_with_an_amd_card_is_amd():
    assert host.host_platform("linux", "x86_64", nvidia=lambda: False, amd=lambda: True) == host.AMD


def test_nvidia_is_preferred_when_a_box_has_both():
    """A board with both is rare, but "which driver does the user have" is not a coin flip
    to be flipped by probe order."""
    assert host.host_platform("linux", "x86_64", nvidia=lambda: True,
                              amd=lambda: True) == host.NVIDIA


def test_amd_is_never_reported_on_a_mac():
    assert host.host_platform("darwin", "x86_64", nvidia=lambda: False,
                              amd=lambda: True) == host.OTHER


def test_linux_without_a_card_is_other():
    assert host.host_platform("linux", "x86_64", nvidia=lambda: False, amd=lambda: False) == "other"


def test_nvidia_gpu_needs_nvidia_smi_on_path():
    assert host.has_nvidia_gpu(which=lambda _: None) is False


# Real `rocminfo` output from the RX 7900 XTX + Ryzen 9800X3D box this work was done on,
# cut down to the agent blocks. ROCr lists the APU as a GPU agent too, with the
# placeholder id `GPU-XX` where a real card gets a unique one.
ROCMINFO_DISCRETE = """
==========
HSA Agents
==========
Agent 1
*******
  Name:                    AMD Ryzen 7 9800X3D 8-Core Processor
  Uuid:                    CPU-XX
  Device Type:             CPU

Agent 2
*******
  Name:                    gfx1100
  Uuid:                    GPU-3a90a60dbaf2e93f
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1100
      Marketing Name:          AMD Radeon Graphics
    ISA 2
      Name:                    amdgcn-amd-amdhsa--gfx11-generic

Agent 3
*******
  Name:                    gfx1036
  Uuid:                    GPU-XX
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1036
      Name:                    amdgcn-amd-amdhsa--gfx10-3-generic
"""

# The same box with the card removed, i.e. an APU-only machine.
ROCMINFO_APU_ONLY = """
  Name:                    gfx1036
  Uuid:                    GPU-XX
  Device Type:             GPU
"""

# An AMD CPU with no graphics hardware at all.
ROCMINFO_CPU_ONLY = """
  Name:                    AMD Ryzen 7 9800X3D 8-Core Processor
  Uuid:                    CPU-XX
  Device Type:             CPU
"""


def test_nvidia_gpu_needs_a_listed_gpu():
    def which(_):
        return "/usr/bin/nvidia-smi"
    listed = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-abc)\n"
    assert host.has_nvidia_gpu(which=which, run=_smi(0, listed)) is True
    # A driver installed with no card, or a container without the GPU mounted.
    assert host.has_nvidia_gpu(which=which, run=_smi(0, "")) is False
    assert host.has_nvidia_gpu(which=which, run=_smi(9, "NVIDIA-SMI has failed")) is False


def test_nvidia_smi_that_hangs_or_vanishes_means_no_gpu():
    def which(_):
        return "/usr/bin/nvidia-smi"

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    def vanished(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    assert host.has_nvidia_gpu(which=which, run=hangs) is False
    assert host.has_nvidia_gpu(which=which, run=vanished) is False


# --- AMD: the same question, answered by the AMD kernel driver and ROCm's own tools -------


def test_a_bound_amd_card_is_found_without_running_anything(tmp_path: Path):
    """A topology node per bound card is what the amdgpu driver leaves behind, and it is
    what HIP needs too. Reading it costs nothing, unlike rocminfo."""
    (tmp_path / "1").mkdir()

    def never(*a, **k):
        raise AssertionError("a bound card must not need rocminfo")
    assert host.has_amd_gpu(which=never, run=never, nodes=tmp_path) is True


def test_no_node_and_no_rocm_tools_is_not_an_amd_box(tmp_path: Path):
    assert host.has_amd_gpu(which=lambda _: None, run=_smi(0, ""), nodes=tmp_path) is False


def test_an_amd_cpu_on_its_own_is_not_an_amd_gpu(tmp_path: Path):
    """The Ryzen 9800X3D in the test box reports itself as a CPU agent; calling that a GPU
    would offer a download for a machine with no card."""
    def which(_):
        return "/opt/rocm/bin/rocminfo"
    assert host.has_amd_gpu(which=which, run=_smi(0, ROCMINFO_CPU_ONLY),
                            nodes=tmp_path) is False


def test_rocminfo_finds_the_card_when_sysfs_is_hidden(tmp_path: Path):
    """A container can expose ROCm's tools without the driver's topology nodes."""
    def which(_):
        return "/opt/rocm/bin/rocminfo"
    assert host.has_amd_gpu(which=which, run=_smi(0, ROCMINFO_DISCRETE),
                            nodes=tmp_path) is True


def test_rocminfo_that_hangs_or_fails_is_not_a_gpu(tmp_path: Path):
    def which(_):
        return "/opt/rocm/bin/rocminfo"

    def hangs(*a, **k):
        raise subprocess.TimeoutExpired("rocminfo", 20)

    def missing(*a, **k):
        raise FileNotFoundError("rocminfo")

    assert host.has_amd_gpu(which=which, run=hangs, nodes=tmp_path) is False
    assert host.has_amd_gpu(which=which, run=missing, nodes=tmp_path) is False
    assert host.has_amd_gpu(which=which, run=_smi(1, ROCMINFO_DISCRETE),
                            nodes=tmp_path) is False


def test_the_gfx_target_comes_from_the_card_not_the_generic_isa():
    """`gfx11-generic` is what a card's KFD node reports on some ROCm versions; a compiler
    is given the real target, or every second kernel it emits is wrong for the card."""
    def which(_):
        return "/opt/rocm/bin/rocminfo"
    assert host.rocm_gfx_targets(which=which, run=_smi(0, ROCMINFO_DISCRETE)) == ["gfx1100"]


def test_an_integrated_gpu_stands_in_when_it_is_the_only_card():
    def which(_):
        return "/opt/rocm/bin/rocminfo"
    assert host.rocm_gfx_targets(which=which, run=_smi(0, ROCMINFO_APU_ONLY)) == ["gfx1036"]


def test_no_rocm_tools_means_no_target_to_compile_for():
    assert host.rocm_gfx_targets(which=lambda _: None) == []
    assert host.rocm_gfx_targets(which=lambda _: "/opt/rocm/bin/rocminfo",
                                 run=_smi(1, ROCMINFO_DISCRETE)) == []


def test_the_rocm_version_comes_from_the_install_it_looks_in(tmp_path: Path):
    # /opt/rocm is a symlink to the installed runtime, so this follows an upgrade.
    info = tmp_path / ".info"
    info.mkdir()
    (info / "version").write_text("7.2.4\n")
    assert host.rocm_version(info / "version") == (7, 2)


def test_an_unreadable_or_silly_rocm_version_is_not_a_guess(tmp_path: Path):
    missing = tmp_path / "version"
    assert host.rocm_version(missing, which=lambda _: None) is None
    missing.write_text("not a version")
    # A file we cannot read is a file we cannot trust: fall back rather than half-believe.
    assert host.rocm_version(missing, which=lambda _: None) is None


def test_the_rocm_version_file_is_the_install_root_and_not_bin():
    # ROCm keeps .info beside bin/, not inside it. Reading /opt/rocm/bin/.info/version
    # finds nothing on every real install, and a silent None there means an AMD machine
    # falls through to PyPI's CUDA torch and never sees its card.
    assert host.ROCM_VERSION_FILE == host.ROCM_BIN.parent / ".info" / "version"


@pytest.mark.skipif(not host.ROCM_VERSION_FILE.is_file(), reason="no ROCm on this machine")
def test_this_machines_rocm_is_read_and_gets_a_published_torch_index():
    # The real thing on the box, not a copy of it: this is the check that catches a path
    # that is wrong on every machine at once.
    version = host.rocm_version()
    assert version is not None
    assert host.torch_rocm_index(version) is not None


def test_an_unreadable_rocm_version_falls_back_to_hipconfig(tmp_path: Path):
    # An install that is not at /opt/rocm -- a versioned directory with no symlink -- has
    # no version file. Refusing setup there would turn away a machine with a good ROCm.
    missing = tmp_path / "version"
    def which(name):
        return "/opt/rocm-6.4.2/bin/hipconfig" if name == "hipconfig" else None
    def run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "7.2.53211-3d9ef42\n", "")
    assert host.rocm_version(missing, which=which, run=run) == (7, 2)


def test_a_rocm_that_will_not_answer_is_not_a_guess(tmp_path: Path):
    missing = tmp_path / "version"

    def refuses(cmd, **kw):
        raise OSError("no such file")

    assert host.rocm_version(missing, which=lambda _: None) is None
    assert host.rocm_version(missing, which=lambda _: "/bin/hipconfig", run=refuses) is None


def test_the_rocm_torch_index_is_the_one_the_runtime_can_run():
    # Each ROCm wheel is built against one ROCm and fails at its first kernel, not at load
    # time, if the runtime is older -- so a 7.2 machine must not be handed a 7.1 build.
    assert host.torch_rocm_index((7, 2)).endswith("/rocm7.2")
    assert host.torch_rocm_index((7, 0)).endswith("/rocm7.0")
    assert host.torch_rocm_index((6, 4)).endswith("/rocm6.4")
    assert host.torch_rocm_index(None) is None
    assert host.torch_rocm_index((6, 1)) is None  # older than any index PyTorch publishes


def test_find_hipcc_prefers_path_then_the_default_install(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(host.shutil, "which", lambda _: "/opt/rocm/bin/hipcc")
    assert host.find_hipcc() == "/opt/rocm/bin/hipcc"
    # ROCm is often not on PATH outside a ROCm shell, which is why the default is checked.
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    monkeypatch.setattr(host, "ROCM_BIN", tmp_path)
    assert host.find_hipcc() is None  # nothing at the default location either
    (tmp_path / "hipcc").write_text("#!/bin/sh\n")
    assert host.find_hipcc() == str(tmp_path / "hipcc")


def test_rocm_root_is_the_parent_of_the_compiler(monkeypatch, tmp_path: Path):
    """CMake's HIP support and the runtime loader both want ROCm's own directory, and
    hipcc is the only reliable way to find it on a box with several ROCm installs."""
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    monkeypatch.setattr(host, "ROCM_BIN", tmp_path / "rocm" / "bin")
    (tmp_path / "rocm" / "bin").mkdir(parents=True)
    (tmp_path / "rocm" / "bin" / "hipcc").write_text("#!/bin/sh\n")
    assert host.rocm_root() == tmp_path / "rocm"


def test_no_rocm_means_no_root():
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(host.shutil, "which", lambda _: None)
    monkeypatch.setattr(host, "ROCM_BIN", Path("/nowhere"))
    assert host.rocm_root() is None
    monkeypatch.undo()


def test_executable_gets_exe_only_on_windows(tmp_path: Path):
    assert host.executable(tmp_path, "sd-cli", "windows") == tmp_path / "sd-cli.exe"
    assert host.executable(tmp_path, "sd-cli", "linux") == tmp_path / "sd-cli"
    assert host.executable(tmp_path, "sd-cli", "macos") == tmp_path / "sd-cli"


@pytest.mark.parametrize("platform_id,family,expected", [
    ("apple-silicon", "macos", "macos-arm64"),
    ("nvidia", "linux", "linux-nvidia"),
    ("nvidia", "windows", "windows-nvidia"),
    ("amd", "linux", "linux-amd"),
    # ROCm on Windows is not a thing anyone has shipped yet.
    ("amd", "windows", None),
    ("other", "linux", None),
    ("other", "macos", None),
])
def test_build_target_maps_the_machine_to_a_prebuilt(platform_id, family, expected):
    assert host.build_target(platform_id, family) == expected


# The header `nvidia-smi` printed on the RunPod 4090 used for the 2026-09-23 test.
SMI_HEADER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
"""


def test_driver_cuda_version_is_read_from_the_smi_header():
    def which(_):
        return "/usr/bin/nvidia-smi"
    assert host.driver_cuda_version(which=which, run=_smi(0, SMI_HEADER)) == (12, 8)


# Driver 610 renamed the field to "CUDA UMD Version". Real header from an RTX 5090 on
# Windows 11, reported by paisanllc in #50.
SMI_HEADER_610 = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.88                 KMD Version: 610.88            CUDA UMD Version: 13.3  |
|-----------------------------------------+------------------------+----------------------+
"""


def test_driver_cuda_version_reads_the_driver_610_spelling():
    def which(_):
        return "/usr/bin/nvidia-smi"
    assert host.driver_cuda_version(which=which, run=_smi(0, SMI_HEADER_610)) == (13, 3)


def test_driver_cuda_version_is_none_without_a_driver_or_a_header():
    assert host.driver_cuda_version(which=lambda _: None) is None
    def which(_):
        return "/usr/bin/nvidia-smi"
    assert host.driver_cuda_version(which=which, run=_smi(0, "no header here")) is None
    assert host.driver_cuda_version(which=which, run=_smi(9, SMI_HEADER)) is None


def test_compute_capability_drops_the_dot():
    """CMake wants `89`, nvidia-smi says `8.9`."""
    def which(_):
        return "/usr/bin/nvidia-smi"
    assert host.compute_capability(which=which, run=_smi(0, "8.9\n")) == "89"
    assert host.compute_capability(which=which, run=_smi(0, "garbage")) is None
    assert host.compute_capability(which=lambda _: None) is None


def test_find_nvcc_prefers_path(monkeypatch):
    monkeypatch.setattr(host.shutil, "which", lambda _: "/opt/cuda/bin/nvcc")
    assert host.find_nvcc() == "/opt/cuda/bin/nvcc"


def test_nvcc_cuda_version_is_read_from_its_release_line():
    out = ("nvcc: NVIDIA (R) Cuda compiler driver\n"
           "Cuda compilation tools, release 12.8, V12.8.93\n")
    assert host.nvcc_cuda_version("nvcc", run=_smi(0, out)) == (12, 8)
    assert host.nvcc_cuda_version("nvcc", run=_smi(0, "garbage")) is None
    assert host.nvcc_cuda_version("nvcc", run=_smi(1, out)) is None
    assert host.nvcc_cuda_version(None) is None


def test_nvcc_that_cannot_start_has_no_version():
    def missing(*a, **k):
        raise FileNotFoundError("nvcc")
    assert host.nvcc_cuda_version("/nowhere/nvcc", run=missing) is None


# A RunPod 4090 container reported 96 CPUs and had 31 GB; an unbounded `cmake --build -j`
# ran dozens of nvcc jobs at once and the kernel killed them (2026-09-23, pod run #3).
GB = 1024 ** 3


def test_build_jobs_are_capped_by_memory_not_just_cpus():
    assert host.build_jobs(cpus=96, memory_bytes=31 * GB, per_job_bytes=3 * GB) == 10


def test_build_jobs_are_capped_by_cpus_when_memory_is_plentiful():
    assert host.build_jobs(cpus=8, memory_bytes=256 * GB, per_job_bytes=3 * GB) == 8


def test_build_jobs_never_drop_below_one():
    assert host.build_jobs(cpus=4, memory_bytes=1 * GB, per_job_bytes=3 * GB) == 1


def test_with_no_arguments_it_measures_this_machine():
    assert host.build_jobs() >= 1


def test_container_cpu_quota_beats_the_host_count(tmp_path):
    (tmp_path / "cpu.max").write_text("1020000 100000\n")
    assert host.cgroup_cpus(tmp_path) == 10
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert host.cgroup_cpus(tmp_path) is None
    assert host.cgroup_cpus(tmp_path / "absent") is None


def test_container_memory_limit_is_read_from_cgroup_v2_or_v1(tmp_path):
    (tmp_path / "memory.max").write_text("30999998464\n")
    assert host.cgroup_memory(tmp_path) == 30999998464
    (tmp_path / "memory.max").write_text("max\n")
    assert host.cgroup_memory(tmp_path) is None
    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.limit_in_bytes").write_text("8589934592\n")
    assert host.cgroup_memory(v1) == 8589934592
    # cgroup v1 spells "no limit" as a huge number, which must not read as a limit.
    (v1 / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert host.cgroup_memory(v1) is None


# --- PyTorch CUDA build for Windows (PyPI only has CPU-only torch there) -------------

def test_a_cuda_13_driver_gets_the_cu130_build():
    assert host.torch_cuda_index((13, 3)) == "https://download.pytorch.org/whl/cu130"
    assert host.torch_cuda_index((13, 0)) == "https://download.pytorch.org/whl/cu130"


def test_a_cuda_12_8_driver_gets_the_cu128_build():
    assert host.torch_cuda_index((12, 8)) == "https://download.pytorch.org/whl/cu128"
    assert host.torch_cuda_index((12, 9)) == "https://download.pytorch.org/whl/cu128"


def test_an_older_driver_or_no_driver_gets_no_cuda_build():
    """12.8 is also the floor for RTX 50-series cards; below it, updating the driver is
    the fix, not an older wheel."""
    assert host.torch_cuda_index((12, 6)) is None
    assert host.torch_cuda_index(None) is None


def test_the_installer_command_prints_the_index_for_this_driver(capsys):
    def which(_):
        return "/usr/bin/nvidia-smi"
    code = host.main(["torch-index"], which=which, run=_smi(0, SMI_HEADER_610))
    assert code == 0
    assert capsys.readouterr().out.strip() == "https://download.pytorch.org/whl/cu130"


def test_the_installer_command_prints_nothing_without_a_usable_driver(capsys):
    assert host.main(["torch-index"], which=lambda _: None) == 0
    assert capsys.readouterr().out.strip() == ""


def test_torch_has_cuda_is_false_when_torch_is_missing_or_cpu_only(capsys):
    assert host.main(["torch-has-cuda"], torch_cuda=lambda: None) == 1
    assert host.main(["torch-has-cuda"], torch_cuda=lambda: "13.0") == 0
    assert capsys.readouterr().err == ""


def test_an_unknown_command_is_an_error():
    assert host.main(["nonsense"]) == 2


def test_patient_downloads_lengthen_uv_timeouts_without_overriding_the_user():
    # A 700 MB cuDNN wheel timed out at uv's 30 s default on a slow pod and killed setup.
    env = host.patient_downloads({"PATH": "/usr/bin"})
    assert int(env["UV_HTTP_TIMEOUT"]) >= 300 and int(env["UV_HTTP_RETRIES"]) >= 5
    mine = host.patient_downloads({"UV_HTTP_TIMEOUT": "30"})
    assert mine["UV_HTTP_TIMEOUT"] == "30"
