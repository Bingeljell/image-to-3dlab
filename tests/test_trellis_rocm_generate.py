"""scripts/trellis_rocm_generate.py: the AMD twin, and the things it must not get wrong.

The twin exists so there is one pipeline and two cards. These tests check both halves of
that: that the shared surface really is shared (same objects, not copies that can drift), and
that the few things which must differ actually do — most importantly the attention backend,
which fails silently if it is left as the NVIDIA value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import trellis_cuda_generate as cuda
import trellis_rocm_generate as rocm
import trellis_space_generate as space

#: Everything the AMD route delegates to the shared implementation. If one of these is copied
#: instead of re-exported, that is the drift this file exists to prevent.
SHARED = (
    "generate", "generate_from_latents", "check_environment", "load_pipeline",
    "build_manifest", "build_provenance", "verify_paths", "backend_revision",
    "prepare_image", "configure_environment", "checkout", "matted_path",
    "provenance_path", "forbid_bria", "bria_refusal",
)


# --- it really is the same pipeline ---------------------------------------------------------

@pytest.mark.parametrize("name", SHARED)
def test_the_pipeline_surface_is_the_cuda_one_not_a_copy(name):
    """A copy is a second thing to fix when upstream moves."""
    here, shared = getattr(rocm, name), getattr(cuda, name)
    # Functions taking a route are bound with functools.partial; the rest are passed through.
    original = getattr(here, "func", here)
    assert original is shared, f"{name} was copied rather than re-exported"


def test_the_demo_settings_are_the_same_object():
    assert rocm.DEMO_PARAMS is space.DEMO_PARAMS
    assert rocm.DEMO_PARAMS == cuda.DEMO_PARAMS


def test_the_sampler_ladder_is_shared():
    """Sampler params come from the Mac route, as they do for the CUDA one."""
    assert cuda.space.sampler_params("sparse_structure") == \
        space.sampler_params("sparse_structure")
    assert rocm.DEMO_PARAMS["resolution"] == space.DEMO_PARAMS["resolution"]


# --- what must differ, and why ---------------------------------------------------------------

def test_it_is_the_rocm_route():
    assert rocm.ROUTE.provenance_route == "amd-rocm"
    assert rocm.DEFAULT_VENDOR == REPO / "vendor" / "trellis-rocm"


def test_attention_is_aule_and_not_flash_attn():
    env = rocm.configure_environment({})
    assert env["ATTN_BACKEND"] == "aule"
    assert env["SPARSE_ATTN_BACKEND"] == "aule"
    assert "flash_attn" not in "".join(env.values())


def test_no_allocator_configuration_on_rocm():
    """expandable_segments gave silent NaNs here, not a crash."""
    assert "PYTORCH_CUDA_ALLOC_CONF" not in rocm.configure_environment({})


def test_the_re_exports_are_pinned_to_this_route():
    """Without the binding they keep the NVIDIA defaults and quietly select flash-attn."""
    for name in SHARED:
        function = getattr(rocm, name)
        if hasattr(function, "keywords") and "route" in function.keywords:
            assert function.keywords["route"] is rocm.ROUTE, name
    # And a function with no route parameter is passed through unchanged.
    assert not hasattr(rocm.matted_path, "keywords")
    assert rocm.checkout is cuda.checkout


def test_the_cuda_route_still_says_flash_attn():
    """The binding must not have leaked the other way."""
    assert cuda.configure_environment({})["ATTN_BACKEND"] == "flash_attn"


def test_the_default_vendor_root_differs_per_route():
    assert cuda.parse_args([]).vendor_root.name == "trellis-cuda"
    assert rocm.parse_args([]).vendor_root.name == "trellis-rocm"


# --- the records it writes --------------------------------------------------------------------

def test_the_manifest_names_the_amd_run():
    manifest = rocm.build_manifest(image="in.png", output="out.glb",
                                   params=rocm.DEMO_PARAMS, pipeline_type="1024_cascade",
                                   seed=1, timings={}, artifacts={}, matte_model=None,
                                   route=rocm.ROUTE)
    assert manifest["generator"] == "trellis_rocm_generate.py"
    assert manifest["attn_backend"] == "aule"
    assert manifest["sparse_attn_backend"] == "aule"
    assert "ROCm" in manifest["port"]


def test_the_provenance_route_sorts_amd_output_apart(tmp_path):
    image, output = tmp_path / "in.png", tmp_path / "out.glb"
    image.write_bytes(b"not really a png")
    output.write_bytes(b"not really a glb")
    record = rocm.build_provenance(image=image, output=output, parameters={},
                                   matte_model=None, backend_revision=None,
                                   route=rocm.ROUTE)
    assert record["model"]["route"] == "amd-rocm"
    # The licence record is identical in substance on both cards.
    assert record["schema_version"] == 1
    assert any(c.get("commercial_status") == "blocked" for c in record["components"])


def test_bria_is_still_refused():
    """The licence guardrail is not a CUDA-route detail; it applies to every card."""
    message = rocm.bria_refusal(REPO / "vendor" / "does-not-exist")
    assert message is not None
    assert "BRIA" in message


def test_main_refuses_an_empty_invocation():
    """Nothing runs without an image and an output, so a mistyped command costs seconds."""
    with pytest.raises(SystemExit):
        rocm.main([])


def test_check_reports_rather_than_raising_on_an_absent_checkout(tmp_path, capsys):
    assert rocm.check_environment(tmp_path) == 1
    assert "FAIL" in capsys.readouterr().out