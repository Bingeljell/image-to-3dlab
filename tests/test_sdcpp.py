"""Reading stable-diffusion.cpp's start-up chatter to tell a GPU run from a CPU one."""

from __future__ import annotations

from image_to_3dlab import sdcpp

# Lines as the Linux Vulkan build printed them on the RunPod 4090, 2026-09-23.
VULKAN_OK = [
    "ggml_vulkan: Found 1 Vulkan devices:",
    "ggml_vulkan: 0 = NVIDIA GeForce RTX 4090 (NVIDIA) | uma: 0 | fp16: 1 | bf16: 0",
    "load_backend: loaded Vulkan backend from /workspace/lab/vendor/sdcpp/libggml-vulkan.so",
    "load_backend: loaded CPU backend from /workspace/lab/vendor/sdcpp/libggml-cpu-haswell.so",
]
# The same build before `apt install libegl1 libgl1`: the driver's Vulkan layer could not
# load, so only the CPU backend came up, and the run sat at 1,700% CPU for minutes.
CPU_ONLY = [
    "[VERBOSE] main.cpp:696  - version: stable-diffusion.cpp version unknown, commit 28b454b",
    "load_backend: loaded CPU backend from /workspace/lab/vendor/sdcpp/libggml-cpu-haswell.so",
]


def test_backend_lines_are_classified():
    assert sdcpp.backend_of(VULKAN_OK[0]) == "gpu"
    assert sdcpp.backend_of(VULKAN_OK[2]) == "gpu"
    assert sdcpp.backend_of("load_backend: loaded CUDA backend from C:/sd/ggml-cuda.dll") == "gpu"
    assert sdcpp.backend_of("ggml_cuda_init: found 1 CUDA devices:") == "gpu"
    # The ROCm build reuses the CUDA log prefix for its device enumeration.
    assert sdcpp.backend_of(
        "ggml_cuda_init: found 2 ROCm devices (Total VRAM: 55486 MiB):") == "gpu"
    assert sdcpp.backend_of("load_backend: loaded ROCm backend from /x/libggml-hip.so") == "gpu"
    assert sdcpp.backend_of(CPU_ONLY[1]) == "cpu"
    assert sdcpp.backend_of("[DEBUG] loading model") is None


def test_zero_devices_found_is_not_a_gpu():
    assert sdcpp.backend_of("ggml_vulkan: Found 0 Vulkan devices:") is None


def test_gpu_found_in_a_whole_probe_output():
    assert sdcpp.gpu_found("\n".join(VULKAN_OK)) is True
    assert sdcpp.gpu_found("\n".join(CPU_ONLY)) is False


def test_watch_flags_a_cpu_fallback_only_when_the_gpu_never_showed_up():
    watch = sdcpp.BackendWatch()
    assert [watch.feed(line) for line in VULKAN_OK] == [False] * 4

    watch = sdcpp.BackendWatch()
    assert [watch.feed(line) for line in CPU_ONLY] == [False, True]


def test_the_help_names_the_fix_that_worked():
    assert "sudo apt install libegl1 libgl1" in sdcpp.NO_GPU_HELP
    assert "nvidia-smi" in sdcpp.NO_GPU_HELP


def test_the_amd_help_points_at_rocm_and_the_vulkan_fallback():
    assert "rocminfo" in sdcpp.NO_AMD_GPU_HELP
    assert "vulkan" in sdcpp.NO_AMD_GPU_HELP.lower()
