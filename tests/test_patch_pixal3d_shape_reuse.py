"""Tests for saving a Pixal3D shape and repainting it later from new pictures.

What matters: with the load variable set, the structure and shape stages are skipped and
the saved coordinates and latent take their place; the save writes after the shape latent
exists and never on a repaint; the single-image route stays as upstream wrote it; the
patch brings its own include, is idempotent, and refuses rather than guesses when
upstream moves an anchor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_pixal3d_shape_reuse.py"
VENDOR_SOURCE = REPO / "vendor" / "pixal3d-cpp" / "src" / "trellis_cli.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_shape_reuse", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The shape of upstream's multi-view route (raven38/pixal3d.cpp at c89126d), cut down to the
# lines the patch anchors on, plus the single-image route's twin of the voxel check.
ORIGINAL = '''#include "image_preprocess.h"
#include <filesystem>

int trellis_run_mv(const trellis::TrellisParams& cfg) {
    printf("[2/6] SS proj conditioning + flow\\n");
    vector<std::array<int,3>> coords;
    {
        trellis::Pixal3dCond c;
    }
    cond_stage_end("SS");
    printf("      active voxels @res32 = %d  (%.1fs)\\n", (int)coords.size(), now() - t_stage);
    if (coords.empty()) { fprintf(stderr, "no voxels produced\\n"); return 1; }
    cond_stage_begin();

    printf("[3/6] shape SLAT flow (LR 512 -> upsample -> HR %d cascade, max_tok=%d)\\n", cfg.hr_res, cfg.max_tokens);
    vector<float> lr_norm, lr_dn;
    {
        trellis::Pixal3dCond c;
    }
    vector<std::array<int,3>> hr_coords;
    { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf", gpu);
      hr_coords = trellis::shape_upsample(m, lr_dn, coords); m.free(); }
    for (;;) {
        hr_res -= 128;
    }
    const int grid = hr_res / 16;
    const int RES = hr_res;

    vector<float> slat_norm;
    {
        trellis::Pixal3dCond c;
    }
    const int N = (int)shc.size();
    cond_lap("denorm + stats (+ dump)");
    cond_stage_end("HR shape SLAT");
}

int trellis_run(const trellis::TrellisParams& cfg) {
    printf("      active voxels @res32 = %d\\n", (int)coords.size());
    if (coords.empty()) { fprintf(stderr, "no voxels produced\\n"); return 1; }
}
'''


def _mv(source: str) -> str:
    return source.split("int trellis_run(")[0]


def test_the_load_variable_skips_both_shape_stages():
    mv = _mv(patch.apply(ORIGINAL))
    assert 'std::getenv("PIXAL3D_LOAD_SHAPE")' in mv
    assert mv.count("if (!i2l_reuse) {\n") == 2               # SS stage, LR shape stage
    assert 'if (!i2l_reuse) { trellis::Model m = trellis::Model::load(M + "/shape_dec.gguf"' in mv
    assert "if (!i2l_reuse && coords.empty())" in mv


def test_the_saved_shape_replaces_the_hr_stage():
    mv = _mv(patch.apply(ORIGINAL))
    assert "if (i2l_reuse) { hr_res = i2l_res; shc = i2l_shc; }" in mv
    assert mv.index("shc = i2l_shc") < mv.index("const int grid = hr_res / 16;")
    assert "if (i2l_reuse) slat_norm = i2l_norm;\n    else {" in mv


def test_the_load_happens_before_stage_two():
    mv = _mv(patch.apply(ORIGINAL))
    assert mv.index('getenv("PIXAL3D_LOAD_SHAPE")') < mv.index("SS proj conditioning")


def test_the_save_writes_after_the_shape_exists_and_never_on_a_repaint():
    mv = _mv(patch.apply(ORIGINAL))
    save = mv.index('getenv("PIXAL3D_SAVE_SHAPE")')
    assert mv.index("const int N = (int)shc.size();") < save < mv.index('cond_stage_end("HR shape SLAT")')
    assert '"PIXAL3D_SAVE_SHAPE"); sp && *sp && !i2l_reuse)' in mv


def test_the_file_layout_matches_on_both_sides():
    """Written as N, res, coords, latent; read back in the same order and sizes."""
    mv = _mv(patch.apply(ORIGINAL))
    assert "fwrite(&n,4,1,f); fwrite(&res,4,1,f);" in mv
    assert "fread(&n,4,1,f) != 1 || fread(&i2l_res,4,1,f) != 1" in mv
    assert "fwrite(xyz,4,3,f)" in mv and "fread(xyz,4,3,f) != 3" in mv
    assert "fwrite(slat_norm.data(),4,slat_norm.size(),f)" in mv
    assert "i2l_norm.resize((size_t)32 * n)" in mv


def test_the_single_image_route_is_left_alone():
    patched = patch.apply(ORIGINAL)
    assert patched.split("int trellis_run(")[1] == ORIGINAL.split("int trellis_run(")[1]


def test_the_patch_brings_its_own_include():
    """getenv reaches trellis_cli.cpp only through other headers today."""
    assert "#include <cstdlib>   // i2l_shape_reuse" in patch.apply(ORIGINAL)


def test_rerunning_changes_nothing():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once


@pytest.mark.parametrize("broken", [
    ORIGINAL.replace('printf("[2/6] SS proj', 'printf("[2/7] SS proj'),
    ORIGINAL.replace("    vector<float> slat_norm;\n    {\n", "    vector<float> slat_norm;\n"),
    ORIGINAL.replace('cond_lap("denorm + stats (+ dump)");', 'cond_lap("denorm");'),
    ORIGINAL.replace("    const int grid = hr_res / 16;\n", ""),
])
def test_a_moved_anchor_is_refused(broken):
    with pytest.raises(SystemExit):
        patch.apply(broken)


def test_a_doubled_anchor_is_refused():
    with pytest.raises(SystemExit):
        patch.apply(ORIGINAL + _mv(ORIGINAL))


@pytest.mark.skipif(not VENDOR_SOURCE.is_file(), reason="pixal3d.cpp is not checked out here")
def test_the_vendored_source_takes_the_patch():
    patched = patch.apply(VENDOR_SOURCE.read_text(encoding="utf-8"))
    assert patch.is_patched(patched)
