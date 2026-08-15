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
