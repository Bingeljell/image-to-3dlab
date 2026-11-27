"""The stage bake waits long enough for a real action and fails when Blender did not finish.

The Blender socket is replaced by a fake that records the read timeout it was given.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_stage_bake.py"


def _load():
    spec = importlib.util.spec_from_file_location("stage_bake_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Socket:
    def __init__(self, reply: str, timeouts: list):
        self._data = reply.encode()
        self._timeouts = timeouts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def settimeout(self, value):
        self._timeouts.append(value)

    def sendall(self, data):
        pass

    def recv(self, size):
        data, self._data = self._data, b""
        return data


def _run(monkeypatch, reply: dict, *extra: str):
    stage = _load()
    timeouts: list = []
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: _Socket(json.dumps(reply), timeouts),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stage",
            "walk",
            "--label",
            "x",
            "--az",
            "35",
            "--output-dir",
            "/tmp/o",
            *extra,
        ],
    )
    return stage.main(), timeouts


GOOD = {"status": "success", "result": "STAGE_BAKE {}"}
FAILED = {"status": "error", "message": "no action named 'walk' in this file"}


def test_a_long_frame_range_gets_more_than_two_minutes(monkeypatch):
    _code, timeouts = _run(
        monkeypatch, GOOD, "--frame-start", "1", "--frame-end", "100"
    )
    assert timeouts and max(timeouts) > 120, f"read timeouts were {timeouts}"


def test_a_whole_action_gets_more_than_two_minutes(monkeypatch):
    _code, timeouts = _run(monkeypatch, GOOD)
    assert timeouts and max(timeouts) > 120, f"read timeouts were {timeouts}"


def test_a_finished_bake_exits_zero(monkeypatch):
    code, _ = _run(monkeypatch, GOOD)
    assert code == 0


def test_a_failed_bake_exits_nonzero(monkeypatch):
    code, _ = _run(monkeypatch, FAILED)
    assert code != 0, "Blender reported an error but the script exited 0"


def test_an_empty_reply_exits_nonzero(monkeypatch):
    stage = _load()
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Socket("", []))
    monkeypatch.setattr(
        sys,
        "argv",
        ["stage", "walk", "--label", "x", "--az", "35", "--output-dir", "/tmp/o"],
    )
    assert stage.main() != 0, "no reply at all (timed out) exited 0"
