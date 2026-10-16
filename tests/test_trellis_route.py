"""scripts/trellis_route.py: the per-card differences, in one place.

The AMD route is only trustworthy if it is the *same* pipeline as the NVIDIA one. That is
easy to say and easy to lose: a second copy of the decode-and-bake is a second thing to fix
when upstream moves. So these tests pin the shared surface by identity, and pin the handful
of things that must differ.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import trellis_route as routes


def test_both_routes_exist_and_are_distinct():
    assert routes.CUDA is not routes.ROCM
    assert routes.CUDA.name != routes.ROCM.name
    assert routes.CUDA.vendor != routes.ROCM.vendor


def test_the_vendor_roots_are_the_ones_the_bootstraps_build():
    """A route pointing at the wrong checkout fails in a way that reads like a model problem."""
    assert routes.CUDA.vendor == REPO / "vendor" / "trellis-cuda"
    assert routes.ROCM.vendor == REPO / "vendor" / "trellis-rocm"
    assert routes.CUDA.bootstrap == "scripts/bootstrap_trellis_cuda.py"
    assert routes.ROCM.bootstrap == "scripts/bootstrap_trellis_rocm.py"


def test_each_route_names_its_own_generator():
    """The manifest says what produced the file; a stale name here is a false record."""
    assert routes.CUDA.generator == "trellis_cuda_generate.py"
    assert routes.ROCM.generator == "trellis_rocm_generate.py"


def test_the_attention_backend_is_pinned_not_inherited():
    """A stale ATTN_BACKEND in the viewer's environment once selected a backend that did
    not exist. `environ` must overwrite, never default."""
    env = routes.ROCM.environ({"ATTN_BACKEND": "flash_attn"})
    assert env["ATTN_BACKEND"] == "aule"
    assert env["SPARSE_ATTN_BACKEND"] == "aule"


def test_the_amd_route_selects_a_backend_that_can_be_built_here():
    """flash-attn has no ROCm build and no wheel that fits; aule is pure Python + Triton."""
    assert routes.ROCM.backend_env["ATTN_BACKEND"] == "aule"
    assert routes.ROCM.backend_env["SPARSE_ATTN_BACKEND"] == "aule"
    assert routes.CUDA.backend_env["ATTN_BACKEND"] == "flash_attn"


def test_both_routes_use_the_sparse_conv_backend_that_does_build():
    """FlexGEMM dispatches on IS_HIP_EXTENSION, so one name covers both."""
    for route in (routes.CUDA, routes.ROCM):
        assert route.backend_env["SPARSE_CONV_BACKEND"] == "flex_gemm"


def test_the_amd_route_sets_no_allocator_configuration():
    """On ROCm, expandable_segments produced silent NaNs rather than an error, which is worse
    than not setting it. Guarded so it cannot be "tidied up" back in by symmetry."""
    assert "PYTORCH_CUDA_ALLOC_CONF" not in routes.ROCM.extra_env
    assert routes.CUDA.extra_env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_environ_does_not_mutate_the_caller_and_copies_by_default():
    base = {"KEEP": "me"}
    env = routes.ROCM.environ(base)
    assert env["KEEP"] == "me"
    assert "ATTN_BACKEND" not in base, "the caller's environment must not be written to"
    env["KEEP"] = "changed"
    assert base["KEEP"] == "me"


def test_environ_with_no_base_writes_through_to_the_process(monkeypatch):
    """TRELLIS reads these at import time, in a module loaded later. A copy nothing points
    at changes nothing -- and the symptom is a run that selected flash_attn on an AMD card
    without ever saying so."""
    monkeypatch.delenv("ATTN_BACKEND", raising=False)
    routes.ROCM.environ()
    assert os.environ["ATTN_BACKEND"] == "aule"
    assert os.environ["SPARSE_ATTN_BACKEND"] == "aule"


def test_environ_sets_upstreams_openexr_flag():
    """Upstream's example.py and app.py both set it; upstream does not default it."""
    assert routes.CUDA.environ({})["OPENCV_IO_ENABLE_OPENEXR"] == "1"
    assert routes.ROCM.environ({})["OPENCV_IO_ENABLE_OPENEXR"] == "1"


def test_the_provenance_routes_sort_output_apart():
    assert routes.CUDA.provenance_route == "nvidia-cuda"
    assert routes.ROCM.provenance_route == "amd-rocm"


def test_the_build_marker_is_the_same_name_on_both_routes():
    """Both bootstraps write `.i2l-build-complete`; readiness reads what the route asks for."""
    assert routes.CUDA.marker_name == routes.ROCM.marker_name == ".i2l-build-complete"