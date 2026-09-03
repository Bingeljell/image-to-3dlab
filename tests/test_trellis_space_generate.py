"""Pure contract tests for the clean-port full-generation wrapper.

These import the real module (no re-derived copies) and exercise only the torch-free helpers,
so they run in the dev venv as the cheap gate before any 25-minute run. The heavy ``generate``
path is exercised manually via ``--check`` then a real run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_space_generate.py"
SPEC = importlib.util.spec_from_file_location("trellis_space_generate", SCRIPT)
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)


# --- resolution -> pipeline_type ---
def test_pipeline_type_mapping():
    assert gen.pipeline_type_for_resolution("512") == "512"
    assert gen.pipeline_type_for_resolution("1024") == "1024_cascade"
    assert gen.pipeline_type_for_resolution("1536") == "1536_cascade"


def test_pipeline_type_rejects_unknown():
    with pytest.raises(ValueError):
        gen.pipeline_type_for_resolution("2048")


def test_demo_resolution_is_1024_cascade():
    assert gen.pipeline_type_for_resolution(gen.DEMO_PARAMS["resolution"]) == "1024_cascade"


# --- sampler params come straight from the demo defaults, as fresh copies ---
@pytest.mark.parametrize(
    "stage,expected",
    [
        ("sparse_structure", {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}),
        ("shape_slat", {"steps": 12, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}),
        ("tex_slat", {"steps": 12, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}),
    ],
)
def test_sampler_params_match_demo(stage, expected):
    assert gen.sampler_params(stage) == expected


def test_sampler_params_returns_a_copy():
    p = gen.sampler_params("shape_slat")
    p["steps"] = 999
    assert gen.DEMO_PARAMS["shape_slat"]["steps"] == 12  # source of truth untouched


def test_sampler_params_rejects_unknown_stage():
    with pytest.raises(ValueError):
        gen.sampler_params("nonsense")


# --- alpha guardrail logic ---
@pytest.mark.parametrize(
    "mode,alpha_min,expected",
    [
        ("RGBA", 0, True),
        ("RGBA", 254, True),
        ("RGBA", 255, False),   # fully opaque = no real foreground mask
        ("RGB", None, False),
        ("LA", 0, False),       # not RGBA
    ],
)
def test_alpha_is_transparent(mode, alpha_min, expected):
    assert gen.alpha_is_transparent(mode, alpha_min) is expected


# --- "has alpha" is not "is cut out" ---
# 2026-09-03: 2dDog.png passed alpha_is_transparent on its letterbox bars alone while its
# subject sat on an opaque white backdrop. TRELLIS duly rebuilt the backdrop as geometry --
# 45 minutes for a slab. The border ring is what separates the two cases.
def _letterboxed(size=64, bar=8):
    """Opaque artwork with only transparent bars top and bottom -- the 2dDog.png shape."""
    import numpy as np

    a = np.full((size, size), 255, dtype=np.uint8)
    a[:bar] = 0
    a[-bar:] = 0
    return a


def _real_cutout(size=64, margin=8):
    """An opaque subject with a transparent frame all the way round."""
    import numpy as np

    a = np.zeros((size, size), dtype=np.uint8)
    a[margin:-margin, margin:-margin] = 255
    return a


def test_border_opaque_fraction_flags_letterboxed_image():
    # left/right edges of the un-barred rows are still opaque, so the ring is far from clear
    assert gen.border_opaque_fraction(_letterboxed()) > gen.BORDER_OPAQUE_LIMIT


def test_border_opaque_fraction_passes_real_cutout():
    assert gen.border_opaque_fraction(_real_cutout()) == 0.0


def test_border_opaque_fraction_ignores_interior_opacity():
    """A subject filling most of the frame is fine so long as the frame itself is clear."""
    import numpy as np

    a = np.full((64, 64), 255, dtype=np.uint8)
    a[:3] = a[-3:] = 0
    a[:, :3] = a[:, -3:] = 0
    assert a.mean() > 200  # overwhelmingly opaque overall...
    assert gen.border_opaque_fraction(a) == 0.0  # ...but properly framed


def test_border_opaque_fraction_survives_degenerate_input():
    import numpy as np

    assert gen.border_opaque_fraction(np.zeros((3, 3), dtype=np.uint8)) == 0.0
    assert gen.border_opaque_fraction(np.zeros((8, 8, 3), dtype=np.uint8)) == 0.0


def test_uncut_foreground_message_is_actionable():
    msg = gen.uncut_foreground_message("dog.png", 0.39)
    assert "dog.png" in msg
    assert "39%" in msg           # the measurement, so the user can judge it
    assert "--allow-uncut" in msg  # the override, so it is not a dead end


# --- degenerate-face filter (the MPS decode -1 index crash) ---
def test_valid_face_mask_drops_out_of_range():
    faces = [[0, 1, 2], [59990, 59991, -1], [3, 4, 5], [1, 2, 10]]
    # V=10 -> index 10 is out of range, and -1 is the degenerate marker
    mask = gen.valid_face_mask(faces, num_vertices=10)
    assert list(mask) == [True, False, True, False]


def test_valid_face_mask_boundary_index_is_valid():
    # index num_vertices-1 is the last valid vertex
    assert list(gen.valid_face_mask([[0, 9, 9]], num_vertices=10)) == [True]
    # all-good faces stay
    assert gen.valid_face_mask([[0, 1, 2], [2, 3, 4]], num_vertices=5).all()


# --- environment configuration ---
def test_configure_environment_sets_sdpa(monkeypatch, tmp_path):
    for key in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "PYTORCH_ENABLE_MPS_FALLBACK",
                "FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
        monkeypatch.delenv(key, raising=False)
    gen.configure_environment(tmp_path, "sdpa")
    import os
    assert os.environ["ATTN_BACKEND"] == "sdpa"
    assert os.environ["SPARSE_ATTN_BACKEND"] == "sdpa"
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_configure_environment_overrides_stale_attention_env(monkeypatch, tmp_path):
    """A stale inherited ATTN_BACKEND must not silently win (the conv_none crash class)."""
    import os

    monkeypatch.setenv("ATTN_BACKEND", "flash_attn")
    gen.configure_environment(tmp_path, "sdpa")
    assert os.environ["ATTN_BACKEND"] == "sdpa"


def test_require_flex_gemm_pins_flex_gemm(monkeypatch):
    import os
    import sys
    import types

    monkeypatch.setitem(sys.modules, "flex_gemm", types.ModuleType("flex_gemm"))
    monkeypatch.delenv("SPARSE_CONV_BACKEND", raising=False)
    gen.require_flex_gemm()
    assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"


def test_require_flex_gemm_raises_instead_of_invalid_none(monkeypatch):
    """flex_gemm is mandatory on MPS; 'none' is not a real backend, so fail loudly."""
    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "flex_gemm":
            raise ImportError("dlopen: library not loaded, stale rpath")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(RuntimeError, match="flex_gemm"):
        gen.require_flex_gemm()


# --- filesystem verification ---
def test_verify_paths_flags_missing_build(tmp_path):
    problems = gen.verify_paths(tmp_path / "does-not-exist")
    assert problems  # every expected artifact is missing
    assert any(".venv python" in p for p in problems)


# --- manifest builder ---
def test_build_manifest_shape():
    m = gen.build_manifest(
        image="in.png", output="out.glb",
        params={"decimation_target": 300000}, pipeline_type="1024_cascade", seed=0,
        timings={"total": 1.0}, artifacts={"glb": {"bytes": 10}},
        load_rembg=False, sparse_attn_backend="sdpa",
    )
    assert m["schema_version"] == 1
    assert m["attn_backend"] == "sdpa"
    assert m["device"] == "mps"
    assert m["load_rembg"] is False
    assert m["pipeline_type"] == "1024_cascade"
    assert m["params"]["decimation_target"] == 300000
    assert m["timings_seconds"]["total"] == 1.0


# --- CPU pre-cap ratio (the MPS decode->GLB path: fast_simplification before Metal to_glb) ---
def test_precap_ratio_math():
    # 20M faces -> 4M cap means removing 80% of faces
    assert gen.precap_ratio(20_000_000, 4_000_000) == 0.8
    assert gen.precap_ratio(8_000_000, 4_000_000) == 0.5
    # already at or under the cap -> no-op (0.0 = fast_simplification keeps everything)
    assert gen.precap_ratio(4_000_000, 4_000_000) == 0.0
    assert gen.precap_ratio(1_000_000, 4_000_000) == 0.0


def test_precap_ratio_rejects_bad_cap():
    with pytest.raises(ValueError):
        gen.precap_ratio(100, 0)
    with pytest.raises(ValueError):
        gen.precap_ratio(100, -5)


# --- guard the demo constants against accidental drift ---
def test_demo_params_integrity():
    assert gen.DEMO_PARAMS["seed"] == 0
    assert gen.DEMO_PARAMS["texture_size"] == 2048
    # decimation_target is under active review (300k app.py vs 3M live demo); assert it is a
    # positive int rather than pinning a value we may deliberately change.
    assert isinstance(gen.DEMO_PARAMS["decimation_target"], int)
    assert gen.DEMO_PARAMS["decimation_target"] > 0
    assert gen.DEMO_PARAMS["remesh"] == {"remesh": True, "remesh_band": 1, "remesh_project": 0}
    for stage in ("sparse_structure", "shape_slat", "tex_slat"):
        assert gen.DEMO_PARAMS[stage]["steps"] == 12


# --- post-pre-cap corruption filter (fast_simplification emits stray out-of-range indices) ---
def test_filter_out_of_range_faces_drops_bad():
    import torch

    faces = torch.tensor([[0, 1, 2], [3, 4, 5], [1, 2, 99], [4, 5, 6]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 1
    assert kept.tolist() == [[0, 1, 2], [3, 4, 5], [4, 5, 6]]


def test_filter_out_of_range_faces_clean_input_untouched():
    import torch

    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 0
    assert kept.shape[0] == 2
    assert kept.dtype == faces.dtype


def test_filter_out_of_range_faces_negative_index():
    import torch

    faces = torch.tensor([[0, 1, 2], [-1, 4, 5]])
    kept, removed = gen.filter_out_of_range_faces(faces, num_vertices=10)
    assert removed == 1
    assert kept.tolist() == [[0, 1, 2]]
