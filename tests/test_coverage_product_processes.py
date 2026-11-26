"""Run process bookkeeping against controlled operating-system replies."""

import ctypes
import os
import struct
from types import SimpleNamespace

import pytest

from image_to_3dlab import processes as p


class Kernel:
    def __init__(self, failure=None):
        self.failure = failure
        self.closed = []
        self.CreateJobObjectW = lambda *_: 0 if failure == "create" else 8
        self.GetCurrentProcess = lambda: 42
        self.SetInformationJobObject = lambda *args: self.set_limits(*args)
        self.AssignProcessToJobObject = lambda *args: self.assign(*args)
        self.CloseHandle = lambda handle: self.closed.append(handle)

    def set_limits(self, handle, kind, ref, size):
        assert handle == 8 and kind == 9
        limits = ref._obj
        assert size == ctypes.sizeof(limits)
        assert limits.BasicLimitInformation.LimitFlags == 0x2800
        return self.failure != "limits"

    def assign(self, job, process):
        assert (job, process) == (8, 42)
        return self.failure != "assign"

    def OpenProcess(self, access, inherit, pid):
        assert (access, inherit, pid) == (0x1000, False, 123)
        return 7

    def GetExitCodeProcess(self, handle, code):
        assert handle == 7
        code._obj.value = 259
        return self.failure != "exit"

    def QueryFullProcessImageNameW(self, handle, flags, buffer, size):
        assert (handle, flags, size._obj.value) == (7, 0, len(buffer))
        buffer.value = r"C:\Python\Python311.exe"
        return self.failure != "image"

    def GetProcessTimes(self, handle, created, ended, in_kernel, in_user):
        assert handle == 7
        created._obj.value = 133000000000000000
        return self.failure != "times"


@pytest.mark.parametrize("failure", [None, "create", "limits", "assign", "probe"])
def test_windows_job_lifetime_and_breakaway_probe(monkeypatch, failure):
    kernel = Kernel(failure)
    probed = []

    def probe(argv, **kwargs):
        probed.append(argv)
        assert kwargs["check"]
        assert kwargs["creationflags"] == p.CREATE_BREAKAWAY_FROM_JOB | 0x08000000
        if failure == "probe":
            raise OSError("breakaway forbidden")

    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setattr(p.ctypes, "WinDLL", lambda name, use_last_error: kernel, raising=False)
    monkeypatch.setattr(p.subprocess, "run", probe)
    monkeypatch.setattr(p, "_JOB", None)
    monkeypatch.setattr(p, "_BREAKAWAY", False)

    installed = failure in (None, "probe")
    assert p.kill_children_with_this_process(windows=True) is installed
    assert kernel.closed == ([8] if failure in ("limits", "assign") else [])
    expected = p.CREATE_BREAKAWAY_FROM_JOB if failure is None else 0
    assert p.group_popen_kwargs(windows=True)["creationflags"] == p.CREATE_NEW_PROCESS_GROUP | expected
    assert p.outlive_kwargs(windows=False) == {}
    if installed:
        assert p.kill_children_with_this_process(windows=True)
        assert probed == [[os.path.join(r"C:\Windows", "System32", "cmd.exe"), "/c", "exit 0"]]
    else:
        assert not probed


@pytest.mark.parametrize("failure", [None, "exit", "image", "times"])
def test_windows_process_queries_close_all_handles(monkeypatch, failure):
    kernel = Kernel(failure)
    monkeypatch.setattr(p.ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    assert p.process_alive(123, windows=True) is (failure not in ("exit", "image"))
    assert p.group_alive(123, windows=True) is (failure not in ("exit", "image"))
    assert p.process_started(123, windows=True) == (
        None if failure in ("exit", "times") else "133000000000000000"
    )
    assert kernel.closed == [7, 7, 7]


@pytest.mark.parametrize("platform", ["linux", "freebsd"])
def test_start_time_missing_or_unavailable(monkeypatch, platform):
    def proc_file(path):
        assert path == "/proc/123/stat"
        return SimpleNamespace(read_text=lambda **kwargs: "123 (short) S 1")

    monkeypatch.setattr(p, "Path", proc_file)
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform=platform))
    assert p.process_started(123, windows=False) is None


@pytest.mark.parametrize(("return_code", "size", "expected"), [
    (0, 648, "123.000045"), (0, 0, None), (1, 648, None),
])
def test_mac_start_time_reads_kernel_timeval(monkeypatch, return_code, size, expected):
    def sysctl(asked, count, buffer, written, new, newlen):
        assert list(asked) == [1, 14, 1, 123]
        assert count == 4 and new is None and newlen == 0
        written._obj.value = size
        buffer.raw = struct.pack("=qi", 123, 45)
        return return_code
    monkeypatch.setattr(p, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(p.ctypes, "CDLL", lambda *args, **kw: SimpleNamespace(sysctl=sysctl))
    assert p.process_started(123, windows=False) == expected


@pytest.mark.posix
def test_protected_process_is_alive_but_cannot_be_killed(monkeypatch):
    calls = []

    def protected(pid, sig):
        calls.append((pid, sig))
        raise PermissionError("owned by another user")

    monkeypatch.setattr(p.os, "kill", protected)
    monkeypatch.setattr(p.os, "killpg", protected)
    assert p.process_alive(123, windows=False)
    assert p.group_alive(123, windows=False)
    p.kill_group(123, windows=False)
    assert calls == [(123, 0), (123, 0), (123, p.signal.SIGKILL)]
