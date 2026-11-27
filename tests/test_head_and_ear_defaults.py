"""A script default must not suit one creature: head yaw and ear turn start at 0.

The options stay available. Each test runs the script's real argument parser
(through `main`) with the network send replaced, and reads what it would send.
"""

from __future__ import annotations

import blender_idle_cycle as idle
import blender_walk_cycle as walk
import pytest


def _sent(module, monkeypatch, argv=()):
    sent = []
    monkeypatch.setattr(
        module, "send", lambda code, host, port: sent.append(code) or ""
    )
    monkeypatch.setattr("sys.argv", [module.__name__, *argv])
    assert module.main() == 0
    return sent[0]


def test_walk_cycle_head_yaw_defaults_to_zero(monkeypatch):
    assert "HEAD_YAW = math.radians(0.0)" in _sent(walk, monkeypatch)


def test_idle_cycle_head_yaw_defaults_to_zero(monkeypatch):
    assert "HEAD_YAW = math.radians(0.0)" in _sent(idle, monkeypatch)


def test_idle_cycle_ear_turn_defaults_to_zero(monkeypatch):
    assert "EAR_TURN = math.radians(0.0)" in _sent(idle, monkeypatch)


@pytest.mark.parametrize("module", [walk, idle])
def test_an_explicit_head_yaw_still_reaches_the_program(module, monkeypatch):
    assert "HEAD_YAW = math.radians(-32.0)" in _sent(
        module, monkeypatch, ["--head-yaw", "-32"]
    )


def test_an_explicit_ear_turn_still_reaches_the_program(monkeypatch):
    assert "EAR_TURN = math.radians(-62.0)" in _sent(
        idle, monkeypatch, ["--ear-turn", "-62"]
    )
