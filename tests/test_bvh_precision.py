"""Tests for the BVH precision report.

The number that matters is `sign_flip_risk`: the share of samples whose distance error
exceeds eps. Narrow-band dual contouring decides "does the surface cross here?" by comparing
against eps, so an error that size is not a rounding detail - it is a wrong answer, and 22%
of them produce the lattice we ship.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_bvh_precision.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_bvh_precision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mbp = _load()


def test_sphere_ground_truth_is_exact():
    """A point at 2r from the centre is exactly r from the surface."""
    pts = np.array([[0.8, 0.0, 0.0], [0.0, 0.2, 0.0]])
    assert mbp.sphere_ground_truth(pts, 0.4) == pytest.approx([0.4, 0.2])


def test_perfect_measurement_has_no_flip_risk():
    exact = np.linspace(0, 0.01, 100)
    stats = mbp.error_report(exact.copy(), exact, eps=0.001)
    assert stats["max"] == pytest.approx(0.0)
    assert stats["sign_flip_risk"] == 0.0


def test_error_larger_than_eps_is_counted_as_flip_risk():
    exact = np.zeros(100)
    measured = np.full(100, 0.002)          # every sample off by 2x eps
    stats = mbp.error_report(measured, exact, eps=0.001)
    assert stats["sign_flip_risk"] == 1.0
    assert stats["error_over_eps_p99"] == pytest.approx(2.0, rel=1e-6)


def test_flip_risk_is_a_fraction_not_a_count():
    exact = np.zeros(100)
    measured = np.concatenate([np.full(25, 0.002), np.zeros(75)])
    stats = mbp.error_report(measured, exact, eps=0.001)
    assert stats["sign_flip_risk"] == pytest.approx(0.25)
    assert stats["samples"] == 100


def test_report_carries_the_eps_it_was_judged_against():
    """Without eps in the output the error is uninterpretable later."""
    stats = mbp.error_report(np.zeros(10), np.zeros(10), eps=0.00097942)
    assert stats["eps"] == pytest.approx(0.00097942)
