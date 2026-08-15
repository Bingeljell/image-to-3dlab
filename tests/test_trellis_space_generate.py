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
