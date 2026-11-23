"""A real worker that ignores SIGTERM, so a runner's cleanup can be tested against it.

A fake process cannot show that a runner stops a worker that shrugs off the polite stop, so
these helpers launch a genuine child that ignores SIGTERM and swap it in for a module's
`subprocess.Popen`. Shared by the tests of the runners that own a worker process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types
from pathlib import Path

_DEAF_WORKER = (
    "import pathlib, signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "pathlib.Path(sys.argv[1]).touch()\n"
    "time.sleep(60)\n"
)


def _deaf_popen(tmp_path):
    """A Popen that really launches a child which ignores SIGTERM (a Blender or a model mid-way
    through a long call does the same), and waits until the child has said so, so the test does
    not race the signal. Its waits are capped at a second so a runner that waits 30 does not
    make the test slow; that cap changes how long a wait lasts, not what the runner does."""
    ready = tmp_path / "deaf.ready"

    class DeafPopen(subprocess.Popen):
        def __init__(self, command, *args, **kwargs):
            super().__init__([sys.executable, "-c", _DEAF_WORKER, str(ready)], *args, **kwargs)
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(0.01)

        def wait(self, timeout=None):
            return super().wait(timeout=None if timeout is None else min(timeout, 1))

    return DeafPopen


def use_deaf_popen(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "subprocess", types.SimpleNamespace(
        **{**vars(subprocess), "Popen": _deaf_popen(tmp_path)}))


def kill_and_reap(process):
    if process is None:
        return
    if process.poll() is None:
        try:
            os.killpg(process.pid, 9)
        except OSError:
            process.kill()
    process.wait(timeout=10)


def failing_pid_note(monkeypatch):
    original = Path.write_text

    def write_text(self, *args, **kwargs):
        if self.name == "pid":
            raise OSError("No space left on device")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)


def watch_slot(manager, job):
    """Record whether the worker was still running at the moment the slot was freed."""
    seen = {}
    freed = manager.finish

    def finish(finishing):
        seen["alive_when_freed"] = job.process is not None and job.process.poll() is None
        freed(finishing)

    manager.finish = finish
    return seen


def unremovable_pid_note(monkeypatch):
    """Make removing a job's pid note fail with something other than "not found"."""
    original = Path.unlink

    def unlink(self, *args, **kwargs):
        if self.name == "pid":
            raise PermissionError(13, "Operation not permitted", str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
