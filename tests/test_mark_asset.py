"""Tests for the asset verdict register.

The register's whole value is that a verdict survives being wrong about everything else,
so the tests lean on the failure modes rather than the happy path:

* a broken/unparseable asset must still record the human judgement (`measure` degrades)
* a verdict with no reason is refused - "good" alone is not a data point
* `judged_backface_culled` must distinguish False from None, because "judged double-sided"
  and "nobody said" are very different strengths of evidence and Python conflates them
  under a plain truthiness check
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mark_asset.py"


def _load():
    spec = importlib.util.spec_from_file_location("mark_asset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ma = _load()


@pytest.fixture
def asset(tmp_path) -> Path:
    path = tmp_path / "box.glb"
    path.write_bytes(trimesh.creation.box().export(file_type="glb"))
    return path


# --- recording ---------------------------------------------------------------------

def test_records_verdict_with_measurements(asset):
    entry = ma.build_entry(asset, "good", "silhouette holds", tags=["ears"], culled=True)

    assert entry["verdict"] == "good"
    assert entry["note"] == "silhouette holds"
    assert entry["tags"] == ["ears"]
    assert entry["judged_backface_culled"] is True
    assert len(entry["asset_sha256"]) == 64
    # The point of the register: the numbers are frozen next to the judgement.
    assert entry["measured"]["faces"] == 12
    assert entry["measured"]["winding_consistent"] is True


def test_requires_a_reason(asset):
    with pytest.raises(ValueError, match="not a data point"):
        ma.build_entry(asset, "good", "   ")


def test_rejects_unknown_verdict(asset):
    with pytest.raises(ValueError, match="verdict must be one of"):
        ma.build_entry(asset, "amazing", "looks great")


def test_double_sided_is_distinct_from_unknown(asset):
    """False and None must not collapse - they are different strengths of evidence."""
    stated = ma.build_entry(asset, "good", "n", culled=False)
    unstated = ma.build_entry(asset, "good", "n")
    assert stated["judged_backface_culled"] is False
    assert unstated["judged_backface_culled"] is None


def test_verdict_survives_an_unmeasurable_asset(tmp_path):
    """A tooling failure must never cost us the human judgement."""
    broken = tmp_path / "broken.glb"
    broken.write_bytes(b"not a glb at all")

    entry = ma.build_entry(broken, "bad", "shotgun holes everywhere", culled=True)
    assert entry["verdict"] == "bad"
    assert "error" in entry["measured"]


def test_links_provenance_sidecar_when_present(asset):
    assert ma.build_entry(asset, "good", "n")["provenance"] is None
    asset.with_suffix(".provenance.json").write_text("{}")
    assert ma.build_entry(asset, "good", "n")["provenance"] is not None


def test_hash_identifies_content_not_path(asset, tmp_path):
    """Renaming an asset must not orphan its verdict."""
    renamed = tmp_path / "renamed.glb"
    renamed.write_bytes(asset.read_bytes())
    assert (
        ma.build_entry(asset, "good", "n")["asset_sha256"]
        == ma.build_entry(renamed, "good", "n")["asset_sha256"]
    )


# --- the register file -------------------------------------------------------------

def test_append_only_roundtrip(asset, tmp_path):
    register = tmp_path / "verdicts.jsonl"
    when = dt.datetime(2026, 8, 13, 9, 0, tzinfo=dt.timezone.utc)

    ma.append_entry(register, ma.build_entry(asset, "bad", "hollow", culled=True, now=when))
    ma.append_entry(register, ma.build_entry(asset, "good", "fixed", culled=True, now=when))

    entries = ma.load_register(register)
    assert [e["verdict"] for e in entries] == ["bad", "good"]
    # Disagreement is a second row, never an overwrite.
    assert len(register.read_text().strip().splitlines()) == 2


def test_load_register_missing_file_is_empty(tmp_path):
    assert ma.load_register(tmp_path / "nope.jsonl") == []


def test_load_register_tolerates_blank_lines(tmp_path):
    register = tmp_path / "verdicts.jsonl"
    register.write_text(json.dumps({"verdict": "good"}) + "\n\n")
    assert len(ma.load_register(register)) == 1


def test_select_filters_by_verdict_and_source():
    entries = [
        {"verdict": "good", "source_image": "fox.png"},
        {"verdict": "bad", "source_image": "fox.png"},
        {"verdict": "good", "source_image": "snag.png"},
    ]
    assert len(ma.select(entries, verdict="good")) == 2
    assert len(ma.select(entries, source="fox.png")) == 2
    assert len(ma.select(entries, verdict="good", source="fox.png")) == 1
    assert len(ma.select(entries)) == 3
