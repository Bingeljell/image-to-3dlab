"""Tests for decode winding repair.

Two properties matter, and both were learned by getting them wrong:

* **Merge by position first.** Repairing the exported mesh reversed 376 of 282,610 faces and
  still failed to converge, because attribute seams split the surface into pieces that
  cannot see their neighbours - orientation has no path to propagate along.
* **Per-component orientation.** The decode arrives in 130,373 connected components, so a
  single global flip is meaningless; each component must be judged by its own volume.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "repair_decode.py"


def _load():
    spec = importlib.util.spec_from_file_location("repair_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rd = _load()


def test_flips_an_inside_out_mesh():
    box = trimesh.creation.box()
    flipped = box.copy()
    flipped.faces = np.fliplr(flipped.faces)
    assert flipped.volume < 0

    _mesh, before, after = rd.repair(flipped.vertices, flipped.faces)
    assert before["inverted_components"] == 1
    assert after["inverted_components"] == 0
    assert after["total_volume"] == pytest.approx(1.0, rel=1e-6)


def test_orients_each_component_independently():
    """Two separate boxes, one inside-out. A global flip cannot fix that; per-component can."""
    good = trimesh.creation.box()
    bad = trimesh.creation.box()
    bad.apply_translation([5, 0, 0])
    bad.faces = np.fliplr(bad.faces)
    combined = trimesh.util.concatenate([good, bad])

    _mesh, before, after = rd.repair(combined.vertices, combined.faces)
    # The failure trimesh's global fix_inversion cannot see: the two volumes cancel to 0,
    # so it concludes there is nothing to do and leaves one box inside-out.
    assert before["components"] == 2
    assert before["inverted_components"] == 1
    assert before["total_volume"] == pytest.approx(0.0, abs=1e-9)

    assert after["inverted_components"] == 0
    assert after["total_volume"] == pytest.approx(2.0, rel=1e-6)


def test_merges_seam_split_vertices_so_orientation_can_propagate():
    """An unmerged mesh has no adjacency; repair must merge before it can do anything."""
    box = trimesh.creation.box()
    split = box.copy()
    split.unmerge_vertices()
    assert len(split.vertices) == 36

    mesh, _before, after = rd.repair(split.vertices, split.faces)
    assert len(mesh.vertices) == 8          # merged back down to real positions
    assert after["components"] == 1         # and adjacency exists again
    assert after["inverted_components"] == 0


def test_a_clean_mesh_is_left_alone():
    box = trimesh.creation.box()
    _mesh, before, after = rd.repair(box.vertices, box.faces)
    assert before["inverted_components"] == 0
    assert after["total_volume"] == pytest.approx(before["total_volume"], rel=1e-9)


def test_summarise_reports_the_deciding_fields():
    stats = rd.summarise(trimesh.creation.box())
    assert set(stats) == {
        "faces", "vertices", "components", "inverted_components",
        "total_volume", "outward_volume",
    }
    assert stats["faces"] == 12


def test_outward_volume_exposes_cancellation():
    """total_volume cancels on a half-inverted mesh; outward_volume does not."""
    good = trimesh.creation.box()
    bad = trimesh.creation.box()
    bad.apply_translation([5, 0, 0])
    bad.faces = np.fliplr(bad.faces)
    stats = rd.summarise(trimesh.util.concatenate([good, bad]))
    assert stats["total_volume"] == pytest.approx(0.0, abs=1e-9)
    assert stats["outward_volume"] == pytest.approx(2.0, rel=1e-6)
