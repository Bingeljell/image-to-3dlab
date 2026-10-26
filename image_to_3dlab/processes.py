"""Start, check and stop background jobs the same way on every platform.

Each viewer job runs in its own process group so stopping it also stops its children.
POSIX does that with sessions and `os.killpg`; Windows has neither, so it gets a new
process group at start and `taskkill /T` (kill the whole tree) at stop. Found on the first
real Windows run (issue #36): `os.killpg` does not exist there.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
from pathlib import PureWindowsPath

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
_JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_IS_WINDOWS = os.name == "nt"
# Jobs the viewer starts: Python workers, and Blender for rig rebinds.
_OUR_IMAGES = ("python", "blender")

# Python children print UTF-8 whatever the locale (PEP 686 makes it the default only from
# 3.15; on Windows the pipe is cp1252 until then, and a Chinese prompt or a name outside it
# raised UnicodeEncodeError in the child), and the studio reads them as UTF-8.
UTF8_ENV = {"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
TEXT_PIPE = {"text": True, "encoding": "utf-8", "errors": "replace"}


def child_env(**extra: str) -> dict:
    """This process's environment for a child job: UTF-8 and unbuffered, plus `extra`."""
    return {**os.environ, **UTF8_ENV, **extra}


def group_popen_kwargs(windows: bool = _IS_WINDOWS) -> dict:
    """Popen keyword arguments that put the child in its own process group.

    A group job is tracked by its pid file and outlives the server on purpose (an orphan
    is reconciled at the next start), so on Windows it also leaves the server's job object
    when it is in one (see kill_children_with_this_process)."""
    if windows:
        return {"creationflags": CREATE_NEW_PROCESS_GROUP | outlive_kwargs(windows).get("creationflags", 0)}
    return {"start_new_session": True}


# The job object this process put itself in, kept open for its whole life: the kernel
# closes the last handle when the process ends, however it ends.
_JOB = None
# Whether a child may leave that job, tried once: if the server could not get a job of its
# own, a breakaway flag on a job that forbids it makes CreateProcess fail.
_BREAKAWAY = False
_CREATE_NO_WINDOW = 0x08000000


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


def kill_children_with_this_process(windows: bool = _IS_WINDOWS) -> bool:
    """Windows: every process this one starts from now on dies when it dies.

    A server stopped from Task Manager or with Stop-Process gets TerminateProcess, which
    runs no handler, so sd-cli and the other workers kept running (and holding GPU memory)
    after it. A job object with KILL_ON_JOB_CLOSE ends them all when the server's last
    handle to it closes, which the kernel does whatever killed it. Children are put in the
    job by Windows itself; one that must outlive the server starts with `outlive_kwargs()`.
    False elsewhere, or when Windows refuses (a job that forbids nesting), and nothing
    changes then."""
    global _JOB, _BREAKAWAY
    if not windows:
        return False
    if _JOB is not None:
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
    kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = (_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                               | _JOB_OBJECT_LIMIT_BREAKAWAY_OK)
    if not (kernel32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                             ctypes.byref(limits), ctypes.sizeof(limits))
            and kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())):
        kernel32.CloseHandle(job)
        return False
    _JOB = job
    try:
        # the full path: a bare "cmd.exe" is looked up in the working folder first on Windows
        cmd = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        subprocess.run([cmd, "/c", "exit 0"], check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=CREATE_BREAKAWAY_FROM_JOB | _CREATE_NO_WINDOW)
        _BREAKAWAY = True
    except (OSError, subprocess.CalledProcessError):
        _BREAKAWAY = False
    return True


def outlive_kwargs(windows: bool = _IS_WINDOWS) -> dict:
    """Popen keyword arguments for a program that must outlive the server: it leaves the
    server's job object (see kill_children_with_this_process), when Windows lets it."""
    return {"creationflags": CREATE_BREAKAWAY_FROM_JOB} if windows and _BREAKAWAY else {}


def windows_group_alive(pid: int, kernel32) -> bool:
    """True if `pid` is a running Python or Blender process. Windows recycles pids quickly, so a
    stale pid that now belongs to some other program must read as dead, or cleanup would
    kill that program's tree."""
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        if code.value != _STILL_ACTIVE:
            return False
        buffer = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_ulong(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return False
        return PureWindowsPath(buffer.value).name.lower().startswith(_OUR_IMAGES)
    finally:
        kernel32.CloseHandle(handle)


def group_alive(pid: int, windows: bool = _IS_WINDOWS) -> bool:
    """True if the process group led by `pid` still exists."""
    if windows:
        return windows_group_alive(pid, ctypes.windll.kernel32)
    try:
        os.killpg(pid, 0)  # signal 0: existence check only, sends nothing
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal


def process_alive(pid: int, windows: bool = _IS_WINDOWS) -> bool:
    """True if the single process `pid` exists (a server, not a job's group)."""
    if windows:
        return windows_group_alive(pid, ctypes.windll.kernel32)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pid_record(pid: int, owner: int | None = None) -> str:
    """What a job's `pid` file holds: the job's pid, then the pid of the process tracking it.

    The owner is what lets a starting server tell a dead server's orphan from a job that
    something else alive is still running (a second viewer, a script driving the Finish
    code). Without it, starting the viewer killed a live Finish run (2026-09-29)."""
    return f"{pid} {os.getpid() if owner is None else owner}"


def parse_pid_record(text: str) -> tuple[int, int | None]:
    """(job pid, owner pid). Files from before the owner was recorded hold one number.

    Raises ValueError for anything that is not one or two whole numbers."""
    fields = text.split()
    if len(fields) not in (1, 2):
        raise ValueError(f"not a pid record: {text!r}")
    numbers = [int(field) for field in fields]
    return numbers[0], numbers[1] if len(numbers) == 2 else None


def terminate_group(pid: int, windows: bool = _IS_WINDOWS) -> None:
    """Best-effort stop of the group led by `pid`; one already gone is not an error."""
    if windows:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
