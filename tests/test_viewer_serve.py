"""Tests for the viewer's URL construction.

The compare URL is how a browser gets pointed at two meshes, so a wrong path produces a
blank pane that reads as a broken asset rather than a broken link — the same false-negative
shape as a generation run that never started.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
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


def test_viewer_styles_are_split_by_responsibility():
    """The app shell must not grow back into a single inline implementation file."""
    html = (REPO / "viewer" / "index.html").read_text()
    expected = ("base.css", "compare.css", "generate.css", "animate.css", "rig.css")

    assert "<style>" not in html
    for name in expected:
        assert f'href="./styles/{name}"' in html
        assert (REPO / "viewer" / "styles" / name).is_file()


def test_viewer_javascript_has_an_external_entry_point():
    """Keep the HTML as an app shell so modes can be composed from ES modules."""
    html = (REPO / "viewer" / "index.html").read_text()

    assert '<script type="module" src="./app.js"></script>' in html
    assert '<script type="module">' not in html
    assert (REPO / "viewer" / "app.js").is_file()


def test_compare_mode_is_an_es_module():
    app = (REPO / "viewer" / "app.js").read_text()
    compare = REPO / "viewer" / "modes" / "compare.js"

    assert "import './modes/compare.js';" in app
    assert compare.is_file()
    assert "function mountModel" in compare.read_text()


def test_generate_mode_is_an_es_module():
    app = (REPO / "viewer" / "app.js").read_text()
    generate = REPO / "viewer" / "modes" / "generate.js"

    assert "import './modes/generate.js';" in app
    assert generate.is_file()
    assert "function startGenerateStream" in generate.read_text()


def test_animate_mode_is_a_workshop_room():
    html = (REPO / "viewer" / "index.html").read_text()
    app = (REPO / "viewer" / "app.js").read_text()
    animate = REPO / "viewer" / "modes" / "animate.js"

    assert 'id="mode-animate"' in html
    assert 'id="animate-view"' in html
    assert "import './modes/animate.js';" in app
    assert "animate: byId('animate-view')" in app
    assert animate.is_file()
    source = animate.read_text()
    assert "new AnimationPlayer" in source
    assert "new THREE.SkeletonHelper" in source
    assert "new BonePicker" in source
    assert "describeBone" in source


def test_rig_review_is_a_sidecar_aware_workshop_room():
    html = (REPO / "viewer" / "index.html").read_text()
    app = (REPO / "viewer" / "app.js").read_text()
    rig_review = REPO / "viewer" / "modes" / "rig-review.js"

    assert 'id="mode-rig"' in html
    assert 'id="rig-view"' in html
    assert "import './modes/rig-review.js';" in app
    assert "rig: byId('rig-view')" in app
    source = rig_review.read_text()
    assert "new BonePicker" in source
    assert "new FitSkeletonOverlay" in source
    assert "new RigCorrectionSession" in source
    assert "fingerprintAsset" in source
    assert "rig-export" in html


def test_rig_review_only_references_existing_controls():
    html = (REPO / "viewer" / "index.html").read_text()
    source = (REPO / "viewer" / "modes" / "rig-review.js").read_text()
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    referenced_ids = set(re.findall(r"element\('([^']+)'\)", source))

    assert referenced_ids
    assert referenced_ids <= html_ids


def test_animate_mode_only_references_existing_controls():
    html = (REPO / "viewer" / "index.html").read_text()
    source = (REPO / "viewer" / "modes" / "animate.js").read_text()
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    referenced_ids = set(re.findall(r"element\('([^']+)'\)", source))

    assert referenced_ids
    assert referenced_ids <= html_ids


def test_compare_uses_the_shared_model_viewport():
    compare = (REPO / "viewer" / "modes" / "compare.js").read_text()
    viewport = REPO / "viewer" / "components" / "model-viewport.js"

    assert "createModelViewport" in compare
    assert viewport.is_file()
    source = viewport.read_text()
    assert "export function createModelViewport" in source
    assert "export function disposeModelViewport" in source


def test_compare_uses_shared_asset_file_specs():
    compare = (REPO / "viewer" / "modes" / "compare.js").read_text()
    asset_files = REPO / "viewer" / "core" / "asset-files.js"

    assert "from '../core/asset-files.js'" in compare
    assert asset_files.is_file()
    assert "export function specsFromFiles" in asset_files.read_text()


def test_app_shell_owns_mode_navigation():
    app = (REPO / "viewer" / "app.js").read_text()
    generate = (REPO / "viewer" / "modes" / "generate.js").read_text()

    assert "function setMode" in app
    assert "mode-${mode}" in app
    assert "setGenerateMode" not in generate


def test_generate_uses_shared_job_progress():
    generate = (REPO / "viewer" / "modes" / "generate.js").read_text()
    progress = REPO / "viewer" / "components" / "job-progress.js"

    assert "new JobProgressPanel" in generate
    assert progress.is_file()
