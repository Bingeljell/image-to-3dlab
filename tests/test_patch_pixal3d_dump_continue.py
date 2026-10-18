"""Tests for letting Pixal3D save its clean-up input and still write the model.

What matters: only the multi-view dump branch changes (its single-image twin stays as
upstream wrote it), the run keeps upstream's stop-after-dump behaviour unless asked, the
dump's weld carries the colours, the patch is idempotent, and it refuses rather than
guesses when upstream moves an anchor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_pixal3d_dump_continue.py"
VENDOR_SOURCE = REPO / "vendor" / "pixal3d-cpp" / "src" / "trellis_cli.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_dump_continue", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The shape of upstream's two dump branches (raven38/pixal3d.cpp at 1f432fd and c89126d):
# the multi-view one this patches, and the single-image TRELLIS twin it must leave alone.
ORIGINAL = '''#include "image_preprocess.h"
#include <filesystem>

int trellis_run_mv(const trellis::TrellisParams& cfg) {
    if (!cfg.dump_post.empty()) {
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr, 1.0f / ((float)so.res * 8.0f));
        trellis::fill_small_holes(mesh.faces);
        fclose(dfp);
        printf("[6/6] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\\n", cfg.dump_post.c_str(), dV, dFc,
               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);
        printf("done in %.1fs -> %s\\n", now() - t0, cfg.dump_post.c_str());
        return 0;
    }
    trellis::weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors, 1.0f / ((float)so.res * 8.0f));
}

int trellis_run(const trellis::TrellisParams& cfg) {
    if (!cfg.dump_post.empty()) {
        trellis::weld_vertices(mesh.verts, mesh.faces, nullptr,
                               1.0f / ((float)so.res * 8.0f));
        printf("[7/7] cleaned dump -> %s (V=%d F=%d%s, PBR=%d @res%d)\\n",
               cfg.dump_post.c_str(), dV, dFc,
               rm.F() > 0 ? " remeshed" : " raw (remesh empty)", Mv, res);
        printf("done in %.1fs -> %s\\n", now() - t0, cfg.dump_post.c_str());
        return 0;
    }
}
'''


@pytest.mark.parametrize("raw, expected", [
    ("1", True), ("1yes", True), (None, False), ("", False), ("0", False), ("yes", False),
])
def test_only_a_one_carries_on(raw, expected):
    """Unset (or anything else) keeps upstream's stop-after-dump behaviour."""
    assert patch.continues(raw) is expected


def test_the_multiview_branch_carries_on_when_asked():
    patched = patch.apply(ORIGINAL)
    mv = patched.split("int trellis_run(")[0]
    assert 'std::getenv("PIXAL3D_DUMP_CONTINUE")' in mv
    assert "[0] != '1') return 0;" in mv
    assert "carrying on to write the model" in mv


def test_the_dump_weld_takes_the_colours_like_the_normal_path():
    mv = patch.apply(ORIGINAL).split("int trellis_run(")[0]
    assert "weld_vertices(mesh.verts, mesh.faces, nullptr" not in mv
    assert mv.count("weld_vertices(mesh.verts, mesh.faces, colors.empty() ? nullptr : &colors") == 2


def test_the_single_image_twin_is_left_alone():
    patched = patch.apply(ORIGINAL)
    assert patched.split("int trellis_run(")[1] == ORIGINAL.split("int trellis_run(")[1]


def test_the_patch_brings_its_own_include():
    """getenv reaches trellis_cli.cpp only through other headers today."""
    assert "#include <cstdlib>" in patch.apply(ORIGINAL)


def test_rerunning_changes_nothing():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once


@pytest.mark.parametrize("broken", [
    ORIGINAL.replace("#include <filesystem>\n", ""),
    ORIGINAL.replace('printf("[6/6] cleaned dump', 'printf("[6/7] cleaned dump'),
    ORIGINAL.replace("nullptr, 1.0f / ((float)so.res * 8.0f));\n        trellis::fill",
                     "&colors, 1.0f / ((float)so.res * 8.0f));\n        trellis::fill"),
])
def test_a_moved_anchor_is_refused(broken):
    with pytest.raises(SystemExit):
        patch.apply(broken)


def test_a_doubled_anchor_is_refused():
    mv = ORIGINAL.split("int trellis_run(")[0]
    with pytest.raises(SystemExit):
        patch.apply(ORIGINAL + mv)


@pytest.mark.skipif(not VENDOR_SOURCE.is_file(), reason="pixal3d.cpp is not checked out here")
def test_the_vendored_source_takes_the_patch():
    source = VENDOR_SOURCE.read_text()
    patched = patch.apply(source)
    assert patch.is_patched(patched)
