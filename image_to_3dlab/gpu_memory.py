"""How much GPU memory is free, and who is using it.

A run that dies at minute two with "out of memory" cost two minutes to learn nothing.
This module answers the question *before* the run, and in the words a user needs:

    >>> free_vram_bytes() < 12 * GB
    'the GPU has 2.1 GB free of 25.8 GB, and this backend wants about 12 GB. Something
     else is using the card: ...'

Both vendors are asked the same question in their own dialect: NVIDIA through `nvidia-smi`,
AMD through `rocm-smi`. Neither tool being present is not an error -- it means the answer is
unknown, and every caller treats unknown as "say nothing and run".
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

GB = 1024 ** 3

# rocm-smi --showmeminfo vram prints one line per GPU:
#   GPU[0]        : VRAM Total Memory (B): 25753026560
#   GPU[0]        : VRAM Total Used Memory (B): 15960199168
_NVIDIA_FREE = re.compile(r"^(?:MiB\s+)?(\d+)\s*(?:,|/)\s*(\d+)")
_ROCM_TOTAL = re.compile(r"^GPU\[\d+\]\s*:\s*VRAM Total Memory \(B\):\s*(\d+)")
_ROCM_USED = re.compile(r"^GPU\[\d+\]\s*:\s*VRAM Total Used Memory \(B\):\s*(\d+)")


def _run(command: list[str], timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def nvidia_free_bytes(which: Callable = shutil.which,
                      run: Callable = _run) -> int | None:
    """Free VRAM on the first NVIDIA card, in bytes."""
    tool = which("nvidia-smi")
    if not tool:
        return None
    out = run([tool, "--query-gpu=memory.free,memory.total",
               "--format=csv,noheader,nounits"])
    match = _NVIDIA_FREE.match((out or "").strip().splitlines()[0] if out and out.strip() else "")
    if not match:
        return None
    free_mib, _total_mib = int(match[1]), int(match[2])
    return free_mib * 1024 * 1024


def rocm_free_bytes(which: Callable = shutil.which,
                    run: Callable = _run) -> int | None:
    """Free VRAM on the largest AMD card, in bytes.

    `rocm-smi` numbers its GPUs, and the smallest of a multi-GPU box can be an APU with a
    2 GB ceiling. Asking about the largest one is the question a generator actually cares
    about, because that is the card it will be placed on.
    """
    tool = which("rocm-smi")
    if not tool:
        return None
    out = run([tool, "--showmeminfo", "vram"])
    if not out:
        return None
    total = [int(m[1]) for m in (_ROCM_TOTAL.match(line.strip()) for line in out.splitlines()) if m]
    used = [int(m[1]) for m in (_ROCM_USED.match(line.strip()) for line in out.splitlines()) if m]
    if not total or len(total) != len(used):
        return None
    free = max(t - u for t, u in zip(total, used))
    return max(0, free)


def free_vram_bytes(which: Callable = shutil.which,
                    run: Callable = _run) -> int | None:
    """Free VRAM on this machine's GPU, or None when it cannot be read.

    None means *unknown*, not *zero*: a box whose tools are missing or that is mid-install
    still generates, so callers report nothing rather than refusing.
    """
    for probe in (nvidia_free_bytes, rocm_free_bytes):
        found = probe(which=which, run=run)
        if found is not None:
            return found
    return None


def gpu_processes(which: Callable = shutil.which,
                  run: Callable = _run) -> list[tuple[int, str, int]]:
    """(pid, name, vram bytes) for the programs using the card right now.

    Both vendors list this. It is what turns "out of memory" into an answer: the user is
    told which program to close, not merely that there is not enough room.
    """
    out = run([which("rocm-smi"), "--showpids"]) if which("rocm-smi") else None
    if out:
        rows = []
        for line in out.splitlines():
            match = re.match(r"^(\d+)\s+(\S+)\s+\d+\s+(\d+)\s", line.strip())
            if match:
                rows.append((int(match[1]), match[2], int(match[3])))
        if rows:
            return rows
    tool = which("nvidia-smi")
    if not tool:
        return []
    out = run([tool, "--query-compute-apps=pid,process_name,used_gpu_memory",
               "--format=csv,noheader,nounits"])
    rows = []
    for line in (out or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit() and parts[2].isdigit():
            rows.append((int(parts[0]), parts[1], int(parts[2]) * 1024 * 1024))
    return rows


def human_bytes(value: int) -> str:
    """`11879596032` -> `'11.1 GB'`. Two decimals below 10 GB, one above: 11.1 and 12.4 both
    read better than 11.08 and 12.35 in a sentence a user has to act on."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def gpu_busy_note(wanted_bytes: int, free_bytes: int | None = None,
                  free: Callable[[], int | None] = free_vram_bytes,
                  processes: Callable[..., list[tuple[int, str, int]]] = gpu_processes) -> str | None:
    """Why a run will not fit, or None when it will (or when we cannot tell).

    `wanted_bytes` is what the backend needs up front, which is a floor: a generator's real
    peak comes and goes. It is only ever used to refuse a run that cannot possibly succeed,
    so the figure passed in is deliberately an under-estimate.

    `free_bytes` short-circuits the probe; `free` is the probe itself, injectable so a test
    can answer "this box cannot tell us" without one being installed.
    """
    if free_bytes is None:
        free_bytes = free()
    if free_bytes is None or free_bytes >= wanted_bytes:
        return None
    note = (f"the GPU has {human_bytes(free_bytes)} free and this backend wants about "
            f"{human_bytes(wanted_bytes)}")
    rows = processes()
    if rows:
        listed = ", ".join(f"{name} (pid {pid}, {human_bytes(vram)})"
                           for pid, name, vram in sorted(rows, key=lambda row: -row[2]))
        note += f". Close what is using the card first: {listed}"
    else:
        note += ". Another program is using the card; close it and try again"
    return note + "."