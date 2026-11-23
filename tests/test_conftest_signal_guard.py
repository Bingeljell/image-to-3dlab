"""The session-wide signal guard in conftest.py: a test never signals a process it did not start.

The POSIX tests are marked `posix`; the Windows half runs everywhere with a faked `subprocess.run`.
The target is a real process that is not our child (a grandchild in its own session). The
test never aims a real signal at os.getppid() or pid 1.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

posix = pytest.mark.posix

GRANDCHILD_PARENT = r"""
import subprocess, sys
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
print(g.pid, flush=True)
sys.stdin.readline()
g.kill()
g.wait()
"""


def _alive(pid: int) -> bool:
    os.kill(pid, 0)  # signal 0 passes the guard; raises if gone
    return True


@pytest.fixture
def grandchild():
    """(child, grandchild_pid): the grandchild is in its own session and is not our child."""
    child = subprocess.Popen(
        [sys.executable, "-c", GRANDCHILD_PARENT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        pid = int(child.stdout.readline())
        yield child, pid
    finally:
        # Ask the child to kill and reap the grandchild, then wait for it, with a timeout.
        try:
            child.stdin.write("\n")
            child.stdin.close()
        except OSError:
            pass
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)
        child.stdout.close()


@posix
def test_guard_is_installed():
    assert getattr(os.kill, "_i3d_guard", False)
    assert getattr(os.killpg, "_i3d_guard", False)


@posix
def test_kill_of_a_non_child_is_refused_as_gone(grandchild):
    _, pid = grandchild
    with pytest.raises(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    assert _alive(pid)


@posix
def test_killpg_of_a_non_child_group_is_refused_as_gone(grandchild):
    _, pid = grandchild
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, signal.SIGTERM)
    assert _alive(pid)


@posix
def test_signal_zero_still_works_on_a_non_child(grandchild):
    _, pid = grandchild
    os.kill(pid, 0)
    os.killpg(pid, 0)


@posix
def test_made_up_pid_is_refused():
    with pytest.raises(ProcessLookupError):
        os.killpg(4242, signal.SIGKILL)
    with pytest.raises(ProcessLookupError):
        os.kill(4242, signal.SIGKILL)


@posix
def test_kill_stops_a_real_direct_child():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        os.kill(child.pid, signal.SIGTERM)
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@posix
def test_killpg_stops_a_real_direct_child_group():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True
    )
    try:
        os.killpg(child.pid, signal.SIGTERM)
        assert child.wait(timeout=10) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


@posix
def test_an_exited_unreaped_child_keeps_its_exit_code():
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    for _ in range(20):  # the guard peeks with WNOWAIT, so it must not reap the exit code
        os.kill(child.pid, signal.SIGCONT)
        time.sleep(0.02)
    assert child.wait(timeout=10) == 7
