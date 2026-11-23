"""Shared pytest setup: make the repo's own modules importable the ordinary way.

`viewer/` and `scripts/` are not packages and are not on Python's search path, so test
files had to hand-load the modules they exercise with `importlib`. That works, but it
quietly breaks Python's one-module-one-object rule: **every hand-load creates a new copy**.
When production code imports the same module normally, the test and the code under test
end up holding two different objects with the same name, and a monkeypatch applied to one
is invisible to the other.

That is not hypothetical. `tests/test_download_api.py` patched `backend_catalog` while
`download_api` held a different copy, so the "an unsupported machine is refused before
anything downloads" guard passed without refusing anything. It failed only in the file
order the alphabetical suite never takes, so it sat green for weeks.

Putting the two directories on the path here lets a test write `import download_api` and
get the same object everyone else has. Neither directory shadows a standard-library name
(checked: 7 and 116 modules, no clashes), so this is additive.

Existing files that hand-load are not wrong to do so where nothing else imports the same
module -- a private copy of a module no one shares cannot diverge from anyone. Migrate
them when convenient; migrate immediately if production code imports the same name.
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for directory in ("viewer", "scripts"):
    path = str(REPO / directory)
    if path not in sys.path:
        sys.path.insert(0, path)

# The suite never counts or sends usage from a real install (tests of it pass their own env).
os.environ["I3D_NO_USAGE"] = "1"


import pytest


def _waitid_peek(pid: int) -> None:
    """`waitid(P_PID, pid, WEXITED | WNOHANG | WNOWAIT)`; raises ChildProcessError if not ours.

    Python's `os.waitid` is missing on macOS before 3.13 (the project runs 3.11), so call
    libc directly there. The constants differ per platform, so they are spelled out.
    """
    if hasattr(os, "waitid"):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    wnowait = 0x20 if sys.platform == "darwin" else 0x01000000
    info = ctypes.create_string_buffer(256)  # siginfo_t, larger than any platform's
    ctypes.set_errno(0)
    if libc.waitid(1, ctypes.c_uint(pid), info, 4 | 1 | wnowait) != 0:  # P_PID, WEXITED|WNOHANG
        err = ctypes.get_errno()
        raise (ChildProcessError if err == errno.ECHILD else OSError)(err, os.strerror(err))


def _is_our_child(pid: int) -> bool:
    """Whether `pid` is a child of this process (live, or exited but not yet reaped).

    The WNOWAIT peek leaves an exited child unreaped, so `Popen` still gets its exit code.
    A process that is not our child raises ChildProcessError.
    """
    if pid <= 0:
        return False
    try:
        _waitid_peek(pid)
    except OSError:  # ChildProcessError (not ours) or anything else: do not trust it
        return False
    return True


def _install_signal_guard() -> None:
    """Make `os.kill` / `os.killpg` unable to signal a process this test run did not start.

    Many tests use a fake worker with a made-up pid (often 4242) whose `poll()` never
    finishes. The runners' cleanup now really stops a still-running worker through
    `os.killpg(pid, SIGTERM/SIGKILL)`, and skin-build Cancel does so from a timer about
    10 s later. Left alone, the suite could signal whatever process group 4242 is on the
    developer's or CI machine. A test must never signal a process it did not start.

    A non-zero signal goes through only when the target is a child of this process (for
    `killpg`, when the group leader, whose pid is the group id, is our child). Anything
    else raises ProcessLookupError, as if the target were gone; production code already
    treats that as "nothing to stop". A process signalling itself is allowed. Signal 0 (the existence check behind
    `group_alive` / `process_alive`) passes through unchanged.

    This is installed once at conftest import time, not as a per-test monkeypatch, because
    a timer can fire after the test that started it has finished and its patch was undone.
    Tests that monkeypatch `os.killpg` themselves still work: their patch sits on top.
    """
    if sys.platform == "win32":
        return
    if getattr(os.kill, "_i3d_guard", False):
        return
    real_kill, real_killpg = os.kill, os.killpg

    def guarded_kill(pid, sig):
        # A negative pid addresses a group; pid 0 / -1 address our own group / everyone.
        # A test signalling itself (a signal-handler test) is its own business and allowed.
        if sig != 0 and pid != os.getpid() and not _is_our_child(abs(pid) if pid < -1 else pid):
            raise ProcessLookupError(3, "test guard: not a child of the test process", pid)
        return real_kill(pid, sig)

    def guarded_killpg(pgid, sig):
        if sig != 0 and not _is_our_child(pgid):
            raise ProcessLookupError(3, "test guard: group leader is not a child of the test process", pgid)
        return real_killpg(pgid, sig)

    guarded_kill._i3d_guard = True  # type: ignore[attr-defined]
    guarded_killpg._i3d_guard = True  # type: ignore[attr-defined]
    os.kill = guarded_kill
    os.killpg = guarded_killpg


_install_signal_guard()

_SYMLINKS: list[bool] = []


def symlinks_allowed() -> bool:
    """Whether this machine lets the user running the tests make a symlink.

    Windows refuses without admin rights or Developer Mode (WinError 1314). The product does
    not need them there (it falls back to hard links or copies), but some fixtures are built
    from symlinks. Asked once, by trying, so an admin Windows runner still runs those tests.
    """
    if not _SYMLINKS:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "target"
            target.write_text("x", encoding="utf-8")
            try:
                (Path(folder) / "link").symlink_to(target)
                _SYMLINKS.append(True)
            except OSError:
                _SYMLINKS.append(False)
    return _SYMLINKS[0]


NO_SYMLINKS = "this machine refuses symlinks (Windows without admin rights or Developer Mode)"


def pytest_collection_modifyitems(config, items):
    """Skip `@pytest.mark.symlinks` where links are refused and `@pytest.mark.posix` on Windows."""
    for item in items:
        if "symlinks" in item.keywords and not symlinks_allowed():
            item.add_marker(pytest.mark.skip(reason=NO_SYMLINKS))
        if "posix" in item.keywords and sys.platform == "win32":
            item.add_marker(pytest.mark.skip(reason="POSIX-only behaviour, not run on Windows"))


@pytest.fixture
def needs_symlinks():
    """For a fixture built from symlinks: skips the test where they are refused."""
    if not symlinks_allowed():
        pytest.skip(NO_SYMLINKS)


@pytest.fixture(autouse=True)
def _private_model_folder(tmp_path_factory, monkeypatch):
    """Every test sees an empty background-remover folder, never the real ~/.u2net.

    Two installer tests used to pass only because this machine already had the remover in
    ~/.u2net, so their "install it if missing" step quietly skipped a real download. On a
    clean machine, or after the remover file changed, they reached the internet instead.
    """
    monkeypatch.setenv("U2NET_HOME", str(tmp_path_factory.mktemp("u2net")))
