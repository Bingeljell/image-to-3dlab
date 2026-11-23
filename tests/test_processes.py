"""Background jobs start, get checked and get stopped the same way on every platform.

Found on the first real Windows run (issue #36, 2026-09-24): the viewer called `os.killpg`,
which Windows Python does not have, so a leftover pid file crashed the server on every
start. Windows paths are tested with fakes here; the POSIX paths run for real.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from image_to_3dlab import processes

REPO = Path(__file__).resolve().parents[1]
POSIX = sys.platform != "win32"


def test_popen_kwargs_start_a_new_group_on_each_platform():
    assert processes.group_popen_kwargs(windows=False) == {"start_new_session": True}
    assert processes.group_popen_kwargs(windows=True) == {
        "creationflags": processes.CREATE_NEW_PROCESS_GROUP
    }


def test_a_stoppable_child_leads_a_group_but_does_not_leave_the_servers_job():
    assert processes.stoppable_popen_kwargs(windows=False) == {"start_new_session": True}
    # Windows stops a tree by pid. A child that left the server's job object would outlive a
    # server killed from Task Manager, and nothing tracks an add-on run to clean it up later.
    flags = processes.stoppable_popen_kwargs(windows=True).get("creationflags", 0)
    assert not flags & processes.CREATE_BREAKAWAY_FROM_JOB


@pytest.mark.skipif(not POSIX, reason="real process groups on POSIX")
def test_a_live_group_is_alive_and_terminating_it_ends_it():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        **processes.group_popen_kwargs(),
    )
    try:
        assert processes.group_alive(child.pid) is True
        processes.terminate_group(child.pid)
        child.wait(timeout=10)
        time.sleep(0.1)
        assert processes.group_alive(child.pid) is False
    finally:
        if child.poll() is None:
            child.kill()


@pytest.mark.posix
def test_a_group_that_ignores_being_asked_to_stop_can_still_be_stopped():
    deaf = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(30)")
    child = subprocess.Popen([sys.executable, "-c", deaf], stdout=subprocess.PIPE,
                             **processes.group_popen_kwargs())
    try:
        assert child.stdout.readline().strip() == b"ready"      # the signal is being ignored by now
        processes.terminate_group(child.pid)
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(timeout=0.5)                             # asked, and it carried on

        processes.kill_group(child.pid)

        assert child.wait(timeout=10) != 0
    finally:
        if child.poll() is None:
            child.kill()
        child.stdout.close()


def test_stopping_a_group_for_certain_is_the_same_forced_stop_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(processes.subprocess, "run", lambda argv, **_more: calls.append(argv))

    processes.kill_group(1234, windows=True)

    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"]]


def test_terminating_a_group_that_is_gone_is_not_an_error():
    child = subprocess.Popen([sys.executable, "-c", "pass"], **processes.group_popen_kwargs())
    child.wait(timeout=10)
    processes.terminate_group(child.pid)  # must not raise
    assert processes.group_alive(child.pid) is False


def test_windows_terminate_kills_the_whole_tree(monkeypatch):
    calls = []
    monkeypatch.setattr(processes.subprocess, "run", lambda args, **kw: calls.append(args))
    processes.terminate_group(1234, windows=True)
    assert calls == [["taskkill", "/PID", "1234", "/T", "/F"]]


class _FakeKernel32:
    """Just enough of kernel32 to exercise the Windows liveness check."""

    def __init__(self, exists=True, exit_code=259, image="C:\\py\\python.exe"):
        self.exists, self.exit_code, self.image = exists, exit_code, image
        self.closed = []

    def OpenProcess(self, access, inherit, pid):
        return 42 if self.exists else 0

    def GetExitCodeProcess(self, handle, code_ref):
        code_ref._obj.value = self.exit_code
        return 1

    def QueryFullProcessImageNameW(self, handle, flags, buffer, size_ref):
        buffer.value = self.image
        return 1

    def GetProcessTimes(self, handle, created_ref, ended_ref, kernel_ref, user_ref):
        created_ref._obj.value = 133_000_000_000_000_000      # a FILETIME: 100 ns steps since 1601
        return 1

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return 1


def test_windows_alive_needs_a_running_python_process():
    assert processes.windows_group_alive(7, _FakeKernel32()) is True
    assert processes.windows_group_alive(7, _FakeKernel32(exists=False)) is False
    assert processes.windows_group_alive(7, _FakeKernel32(exit_code=0)) is False


def test_windows_alive_counts_blender_since_rig_jobs_run_it_directly():
    kernel = _FakeKernel32(image="C:\\Program Files\\Blender\\blender.exe")
    assert processes.windows_group_alive(7, kernel) is True


def test_windows_never_calls_a_reused_pid_ours():
    """Windows recycles pids fast. A stale pid that now belongs to, say, a browser must
    read as dead, or the orphan cleanup would taskkill someone's browser tree."""
    kernel = _FakeKernel32(image="C:\\Program Files\\Browser\\browser.exe")
    assert processes.windows_group_alive(7, kernel) is False
    assert kernel.closed == [42]


def test_no_viewer_module_uses_posix_only_process_calls():
    """Every job start/stop in the viewer goes through image_to_3dlab.processes."""
    offenders = []
    for path in sorted((REPO / "viewer").glob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"os\.killpg\(|start_new_session\s*=\s*True", code):
                offenders.append(f"{path.name}:{number}")
    assert offenders == []


def test_a_pid_record_names_the_job_and_its_owner_and_when_each_started():
    assert processes.pid_record(4242, owner=5151, started="then", owner_started="before") == "4242 5151 then before"
    assert processes.pid_record(4242, started="then", owner_started="before") == f"4242 {os.getpid()} then before"


def test_a_pid_record_marks_a_start_it_cannot_read():
    job, owner = _a_process_that_has_ended(), _a_process_that_has_ended()
    assert processes.pid_record(job, owner=owner) == f"{job} {owner} - -"


def test_pid_records_old_and_new_all_parse():
    assert processes.parse_pid_record("4242 5151 then before\n") == (4242, 5151, "then", "before")
    assert processes.parse_pid_record("4242 5151 - before") == (4242, 5151, None, "before")
    assert processes.parse_pid_record("4242 5151 then") == (4242, 5151, "then", None)
    assert processes.parse_pid_record("4242 5151\n") == (4242, 5151, None, None)
    assert processes.parse_pid_record("4242") == (4242, None, None, None)


@pytest.mark.parametrize("text", ["", "abc", "1 2 3 4 5", "12 x", "x 2 3"])
def test_a_malformed_pid_record_is_rejected(text):
    with pytest.raises(ValueError):
        processes.parse_pid_record(text)


# --- a job's number can be given to another process once the job has ended ----------------
#
# A pid note named the job by its number alone. When the job had ended and the number had
# gone to another process since (a terminal's shell leads a group of its own), the next
# studio start stopped that process and everything in its group. A note now keeps when
# the job started, which a later process given the same number cannot share.

def _a_process_that_has_ended():
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait()
    return ended.pid


def test_a_pid_record_keeps_when_its_job_started():
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as job:
        try:
            note = processes.parse_pid_record(processes.pid_record(job.pid))
            assert (note.pid, note.owner) == (job.pid, os.getpid())
            assert note.started is not None and " " not in note.started
            assert note.started == processes.process_started(job.pid), "the start reads differently a moment later"
            assert note.owner_started == processes.process_started(os.getpid()) is not None
        finally:
            job.kill()


@pytest.mark.posix
def test_two_processes_do_not_share_a_start():
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"]) as job:
        try:
            assert processes.process_started(job.pid) != processes.process_started(os.getppid())
        finally:
            job.kill()


def test_a_process_that_has_ended_has_no_start():
    assert processes.process_started(_a_process_that_has_ended()) is None


def test_linux_reads_the_start_from_field_22_of_the_process_stat_whatever_its_name(monkeypatch):
    # The name sits in brackets and may hold spaces and brackets of its own; the fields after
    # it are counted from its last closing bracket. Field 22 is the start, in ticks since boot.
    stat = ("4242 (a (strange) name) S 1 4242 4242 0 -1 4194560 1203 0 0 0 12 3 0 0 20 0 1 0 "
            "987654 12345678 1000 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 3 0 0 0 0 0\n")
    asked = []

    class Proc:
        def __init__(self, path):
            asked.append(str(path))

        def read_text(self, **_how):
            return stat
    monkeypatch.setattr(processes.sys, "platform", "linux")
    monkeypatch.setattr(processes, "Path", Proc)

    assert processes.process_started(4242, windows=False) == "987654"
    assert asked == ["/proc/4242/stat"]


def test_windows_reads_when_a_process_started_from_its_times():
    kernel = _FakeKernel32()
    assert processes.windows_process_started(7, kernel) == "133000000000000000"
    assert kernel.closed == [42]
    assert processes.windows_process_started(7, _FakeKernel32(exists=False)) is None


@pytest.mark.parametrize("now, recorded, same", [
    ("Mon-Nov-23-10:00:00-2026", "Mon-Nov-23-10:00:00-2026", True),
    ("Tue-Nov-24-09:00:00-2026", "Mon-Nov-23-10:00:00-2026", False),     # the number went to another
    (None, "Mon-Nov-23-10:00:00-2026", True),       # the leader is gone; its group, if any, is the job's
    ("Tue-Nov-24-09:00:00-2026", None, True),       # an older note: nothing to tell them apart by
], ids=["still the job", "another process", "the leader has ended", "an older note"])
def test_a_job_is_only_taken_for_the_process_now_at_its_number_when_their_starts_agree(
        monkeypatch, now, recorded, same):
    monkeypatch.setattr(processes, "process_started", lambda _pid, windows=False: now)
    assert processes.still_the_process(4242, recorded) is same


@pytest.mark.posix
def test_this_process_is_alive_and_a_free_pid_is_not():
    assert processes.process_alive(os.getpid(), windows=False)
    assert not processes.process_alive(2**22 + 12345, windows=False)


# ---- Windows: the server's children die with it ----------------------------------------

REPO = Path(__file__).resolve().parents[1]
CHILD = """
import subprocess, sys, time
sys.path.insert(0, {repo!r})
from image_to_3dlab import processes
before = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])   # as the browser
print(processes.kill_children_with_this_process(), flush=True)
held = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
kept = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **processes.outlive_kwargs())
job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **processes.group_popen_kwargs())
print(held.pid, kept.pid, job.pid, before.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(os.name != "nt", reason="Windows job objects")
def test_a_killed_server_takes_its_children_with_it_but_not_what_must_outlive_it():
    """TerminateProcess (Task Manager, Stop-Process) runs no handler, so sd-cli kept running
    after the server died; a job object that kills on close ends it. A program that breaks
    away stays, and so does a group job, which outlives the server by design and is
    reconciled at the next start. So does what the server started before joining the job,
    such as the browser it opens."""
    server = subprocess.Popen([sys.executable, "-c", CHILD.format(repo=str(REPO))],
                              stdout=subprocess.PIPE, text=True)
    held = kept = job = before = None
    try:
        assert server.stdout.readline().strip() == "True"
        held, kept, job, before = (int(x) for x in server.stdout.readline().split())
        assert processes.process_alive(held) and processes.process_alive(kept)
        server.kill()                              # TerminateProcess, as Task Manager does
        server.wait(10)
        deadline = time.monotonic() + 10
        while processes.process_alive(held) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not processes.process_alive(held)
        assert processes.process_alive(kept) and processes.process_alive(job)
        assert processes.process_alive(before)
    finally:
        server.kill()
        for pid in (held, kept, job, before):
            if pid:
                processes.terminate_group(pid)


def test_elsewhere_nothing_is_set_up():
    assert processes.kill_children_with_this_process(windows=False) is False
    assert processes.outlive_kwargs(windows=False) == {}
    assert processes.group_popen_kwargs(windows=False) == {"start_new_session": True}


def test_outside_the_servers_job_a_group_job_starts_as_before():
    # the suite, the command line and the MCP server never make the job: no breakaway flag
    assert processes.group_popen_kwargs(windows=True) == {"creationflags": processes.CREATE_NEW_PROCESS_GROUP}
