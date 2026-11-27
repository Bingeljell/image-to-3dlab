"""Tests for splitting a mesh into per-region material slots.

The point of the split is that 2048 is the cap *per material*: three regions means three
atlases, so the budget triples without changing any generation setting. These cover the
classification and the density arithmetic that justifies it.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_split_regions.py"


def _load():
    spec = importlib.util.spec_from_file_location("split_regions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load()


def test_faces_land_in_the_expected_region(mod):
    """Cuts are fractions of the mesh's own extent: 0 is the tail end, 1 the head end."""
    got = mod.classify_faces([0.0, 50.0, 100.0], head_cut=2 / 3, tail_cut=1 / 3)
    assert got == ["tail", "body", "head"]


def test_boundaries_are_exclusive_and_fall_to_body(mod):
    """A face exactly on a cut must not be double-claimed; body is the safe default."""
    got = mod.classify_faces([0.0, 25.0, 75.0, 100.0], head_cut=0.75, tail_cut=0.25)
    assert got == ["tail", "body", "body", "head"]


def test_a_mesh_with_no_extent_is_refused(mod):
    with pytest.raises(ValueError):
        mod.classify_faces([4.0, 4.0, 4.0], head_cut=2 / 3, tail_cut=1 / 3)


def test_every_face_gets_exactly_one_region(mod):
    positions = [i / 100 - 0.5 for i in range(101)]
    got = mod.classify_faces(positions, head_cut=0.25, tail_cut=-0.10)
    assert len(got) == len(positions)
    assert set(got) <= set(mod.REGION_NAMES)
    assert sum(mod.region_counts(got).values()) == len(positions)


def test_inverted_cuts_are_rejected(mod):
    """head_cut below tail_cut would silently produce an empty body region."""
    with pytest.raises(ValueError):
        mod.classify_faces([0.0], head_cut=-0.2, tail_cut=0.2)


def test_density_triples_when_one_atlas_becomes_three(mod):
    """The arithmetic the whole exercise rests on."""
    total = 101_298
    before = mod.texel_density(total, 2048)
    thirds = [mod.texel_density(total // 3, 2048) for _ in range(3)]
    assert before == pytest.approx(41.4, abs=0.5)
    for d in thirds:
        assert d / before == pytest.approx(3.0, abs=0.05)


def test_density_matches_the_measured_split(mod):
    """Pinned to the real face counts, so a change in the cuts is visible."""
    assert mod.texel_density(35_824, 2048) == pytest.approx(117.0, abs=1.0)
    assert mod.texel_density(28_647, 2048) == pytest.approx(146.4, abs=1.0)


def test_empty_region_does_not_divide_by_zero(mod):
    assert mod.texel_density(0, 2048) == 0.0


def test_cuts_are_fractions_of_the_meshs_own_extent(mod):
    """Old bug: cuts were absolute positions off one creature. A mesh that sits at 100..200
    then lands wholly in "head". Cuts of 2/3 and 1/3 must split any mesh into thirds."""
    got = mod.classify_faces([100.0, 150.0, 200.0], head_cut=2 / 3, tail_cut=1 / 3)
    assert got == ["tail", "body", "head"]


def test_default_cuts_are_fractions_not_one_creatures_coordinates(
    mod, monkeypatch, tmp_path
):
    sent: list[str] = []
    monkeypatch.setattr(mod, "send", lambda code, *a, **k: sent.append(code) or "{}")
    monkeypatch.setattr(
        sys,
        "argv",
        ["blender_split_regions.py", str(tmp_path / "a.glb"), str(tmp_path)],
    )
    mod.main()
    head = float(re.search(r"^HEAD_CUT = (\S+)", sent[0], re.MULTILINE).group(1))
    tail = float(re.search(r"^TAIL_CUT = (\S+)", sent[0], re.MULTILINE).group(1))
    assert 0.0 <= tail < head <= 1.0, (head, tail)
