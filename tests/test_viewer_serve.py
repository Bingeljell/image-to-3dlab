"""Tests for the viewer's URL construction.

The compare URL is how a browser gets pointed at two meshes, so a wrong path produces a
blank pane that reads as a broken asset rather than a broken link — the same false-negative
shape as a generation run that never started.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("serve", REPO / "viewer" / "serve.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load()


def _params(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}


def test_two_assets_become_a_and_b():
    p = _params(serve.compare_url(["output/one.glb", "assets_to_test/two.glb"], 8777))
    assert p["a"] == "output/one.glb"
    assert p["b"] == "assets_to_test/two.glb"


def test_absolute_paths_inside_the_repo_are_made_relative():
    p = _params(serve.compare_url([str(REPO / "output" / "x.glb")], 8777))
    assert p["a"] == "output/x.glb"


def test_paths_outside_the_repo_are_refused():
    """Better a clear error than a pane that silently 404s and reads as a bad mesh."""
    with pytest.raises(ValueError, match="outside the repo"):
        serve.compare_url(["/etc/hosts"], 8777)


def test_labels_are_attached_positionally():
    p = _params(serve.compare_url(["a.glb", "b.glb"], 8777, labels=["control", "ours"]))
    assert p["la"] == "control"
    assert p["lb"] == "ours"


def test_at_most_three_assets():
    p = _params(serve.compare_url(["1.glb", "2.glb", "3.glb", "4.glb"], 8777))
    assert {k for k in p if len(k) == 1} == set("abc")


def test_port_is_honoured():
    assert ":9001/" in serve.compare_url(["a.glb"], 9001)


def test_glb_is_served_as_a_binary_model_type():
    """A wrong MIME type makes GLTFLoader fail in a way that looks like a corrupt file."""
    assert serve.Handler.extensions_map[".glb"] == "model/gltf-binary"
