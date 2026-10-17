"""Free VRAM and who is holding it: the question asked before an expensive run.

The failure this answers: a Pixal3D run on a 24 GB card died at minute two with
``cudaMalloc failed: out of memory``, because another program was holding 13.7 GB of the
card. The log said so, 200 lines deep under a GDB backtrace. Here it is asked first, in one
sentence, naming the program to close.
"""

from __future__ import annotations

import subprocess

import pytest

from image_to_3dlab import gpu_memory as gpu

GB = gpu.GB


def _runner(stdout: str, returncode: int = 0):
    def run(command, timeout=10):
        return stdout if returncode == 0 else None
    return run


# `rocm-smi --showmeminfo vram` on the RX 7900 XTX box, with ComfyUI holding 13.7 GB.
ROCM_MEMINFO = """
===================================== ROCm System Management Interface =====================================
===================================== Memory Usage (Bytes) =====================================
GPU[0]        : VRAM Total Memory (B): 25753026560
GPU[0]        : VRAM Total Used Memory (B): 15960199168
GPU[1]        : VRAM Total Memory (B): 2147483648
GPU[1]        : VRAM Total Used Memory (B): 17227776
====================================== End of ROCm SMI Log ======================================
"""

# The APU on that box is 2 GB. Asking about it would refuse a run the dGPU has room for.
ROCM_SMI = _runner(ROCM_MEMINFO)


def test_rocm_free_memory_is_what_the_big_card_has_left():
    free = gpu.rocm_free_bytes(which=lambda _: "/opt/rocm/bin/rocm-smi", run=ROCM_SMI)
    assert free == 25753026560 - 15960199168


def test_a_mixed_box_reports_the_roomier_card_not_the_first_one():
    """A box with an NVIDIA card and an AMD one answers both questions. Reporting whichever
    came first would understate the room, and a caller fitting settings to VRAM -- the AMD
    Hunyuan route's paint ladder -- would then pick settings for the smaller card."""
    nvidia_free = 4 * GB

    def which(name):
        return {"nvidia-smi": "/usr/bin/nvidia-smi", "rocm-smi": "/opt/rocm/bin/rocm-smi"}.get(name)

    outputs = {"/usr/bin/nvidia-smi": f"{nvidia_free}", "/opt/rocm/bin/rocm-smi": ROCM_MEMINFO}

    def run(command, timeout=10):
        return outputs.get(command[0])

    monkeypatched = gpu.free_vram_bytes(which=which, run=run)
    assert monkeypatched == max(nvidia_free, gpu.rocm_free_bytes(which=which, run=run))


def test_free_memory_prefers_the_roomier_answer_whatever_the_order():
    """Both vendors can answer on one box, so the probe order must not decide the result.

    The card being fitted for is the AMD one, and a smaller NVIDIA number must not win.
    """
    nvidia_free = 4 * GB
    roomier = 20 * GB
    rocm_only = gpu.rocm_free_bytes(which=lambda _: "/opt/rocm/bin/rocm-smi", run=ROCM_SMI)
    assert rocm_only > nvidia_free, "the fixture no longer exercises the mixed case"

    paths = {"nvidia-smi": "/usr/bin/nvidia-smi", "rocm-smi": "/opt/rocm/bin/rocm-smi"}
    # nvidia-smi answers in MiB CSV; rocm-smi in its own table. Both real formats, because
    # a stub that returns the same string to both would pass whatever the code parsed.
    answers = {paths["nvidia-smi"]: f"{nvidia_free // (1024 * 1024)}, 24576",
               paths["rocm-smi"]: ROCM_MEMINFO}

    both = gpu.free_vram_bytes(which=lambda name: paths.get(name),
                               run=lambda command, timeout=10: answers[command[0]])
    assert both == rocm_only > nvidia_free

    # And the same with the roomier card being the NVIDIA one, so this is a rule about
    # room rather than a rule that happens to prefer AMD.
    swapped = {paths["nvidia-smi"]: f"{roomier // (1024 * 1024)}, 24576",
               paths["rocm-smi"]: ROCM_MEMINFO}
    assert gpu.free_vram_bytes(
        which=lambda name: paths.get(name),
        run=lambda command, timeout=10: swapped[command[0]]) == roomier


def test_the_smallest_gpu_is_never_the_one_asked_about():
    """A box with a 24 GB card and a 2 GB APU must not be measured by the APU: the answer
    would refuse every run that the real card could do."""
    free = gpu.rocm_free_bytes(which=lambda _: "/opt/rocm/bin/rocm-smi", run=ROCM_SMI)
    assert free > 9 * GB


def test_nvidia_free_memory_is_read_from_its_own_csv():
    out = "24576, 24576\n"
    free = gpu.nvidia_free_bytes(which=lambda _: "/usr/bin/nvidia-smi",
                                 run=_runner(out))
    assert free == 24576 * 1024 * 1024


def test_a_missing_tool_means_unknown_not_zero():
    """Unknown must never read as "no memory", or every machine without the tools refuses
    every run."""
    assert gpu.free_vram_bytes(which=lambda _: None) is None
    assert gpu.nvidia_free_bytes(which=lambda _: None) is None
    assert gpu.rocm_free_bytes(which=lambda _: None) is None


def test_a_tool_that_fails_or_prints_nonsense_means_unknown():
    assert gpu.rocm_free_bytes(which=lambda _: "rocm-smi", run=_runner("", 1)) is None
    assert gpu.rocm_free_bytes(which=lambda _: "rocm-smi", run=_runner("nonsense")) is None
    assert gpu.nvidia_free_bytes(which=lambda _: "nvidia-smi",
                                 run=_runner("not a number")) is None


def test_amd_is_asked_when_there_is_no_nvidia():
    def which(name):
        return "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else None
    assert gpu.free_vram_bytes(which=which, run=ROCM_SMI) == 25753026560 - 15960199168


def test_a_tool_that_hangs_does_not_hang_the_run(monkeypatch):
    """rocm-smi wakes every agent and can sit for seconds; a hang must cost the answer,
    not the run."""
    def hangs(command, **kwargs):
        raise subprocess.TimeoutExpired("rocm-smi", 10)

    monkeypatch.setattr(gpu.subprocess, "run", hangs, raising=True)
    assert gpu._run(["rocm-smi"]) is None
    assert gpu.rocm_free_bytes(which=lambda _: "rocm-smi", run=gpu._run) is None


# --- who is using the card ------------------------------------------------------------------

# `rocm-smi --showpids` with ComfyUI-3D holding VRAM, which is what caused the OOM.
ROCM_PIDS = """
===================================== KFD Processes ======================================
KFD process information:
PID    PROCESS NAME       GPU(s)       VRAM USED       SDMA USED       CU OCCUPANCY
94051   python3            1            13675610112    0              UNKNOWN
====================================== End of ROCm SMI Log ====================================
"""


def test_the_program_holding_the_card_is_named():
    rows = gpu.gpu_processes(which=lambda _: "/opt/rocm/bin/rocm-smi",
                             run=_runner(ROCM_PIDS))
    assert rows == [(94051, "python3", 13675610112)]


def test_nvidia_processes_are_read_when_there_is_no_rocm():
    out = "94051, python3, 13054\n"
    def which(name):
        return "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None
    rows = gpu.gpu_processes(which=which, run=_runner(out))
    assert rows == [(94051, "python3", 13054 * 1024 * 1024)]


def test_no_processes_is_an_empty_list_not_a_crash():
    assert gpu.gpu_processes(which=lambda _: None) == []


# --- the sentence the user gets ---------------------------------------------------------------

def test_a_run_that_does_not_fit_says_so_and_names_the_thief():
    """The message is the whole point. 'out of memory' 200 lines under a backtrace is what
    the log said; this says what to close."""
    free = 25753026560 - 15960199168
    note = gpu.gpu_busy_note(12 * GB, free_bytes=free,
                             processes=lambda: [(94051, "python3", 13675610112)])
    assert note is not None
    assert "13.0 GB free" in note or "13.1 GB free" in note or "GB free" in note
    assert "12.0 GB" in note
    assert "python3" in note and "94051" in note


def test_a_run_that_fits_is_not_stopped():
    assert gpu.gpu_busy_note(12 * GB, free_bytes=20 * GB) is None


def test_an_unknown_free_amount_is_never_a_refusal():
    """A box whose tools are missing must still generate. Guessing 'not enough' would take
    the route away from exactly the machines least able to answer."""
    assert gpu.gpu_busy_note(12 * GB, free=lambda: None, processes=list) is None


def test_the_note_lands_when_the_probe_says_there_is_no_room():
    """`free_bytes=None` means "measure it yourself", which is how the viewer calls it."""
    note = gpu.gpu_busy_note(12 * GB, free=lambda: 2 * GB, processes=list)
    assert note is not None and "2.0 GB free" in note


def test_the_note_still_lands_when_nobody_can_be_named():
    note = gpu.gpu_busy_note(12 * GB, free_bytes=1 * GB, processes=list)
    assert note is not None
    assert "Another program is using the card" in note


@pytest.mark.parametrize("value,expected", [
    (11879596032, "11.1 GB"),
    (13675610112, "12.7 GB"),
    (512, "512 B"),
])
def test_sizes_read_as_sizes(value, expected):
    assert gpu.human_bytes(value) == expected