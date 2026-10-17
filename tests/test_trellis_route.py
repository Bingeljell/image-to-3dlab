"""scripts/trellis_route.py: the per-card differences, in one place.

The AMD route is only trustworthy if it is the *same* pipeline as the NVIDIA one. That is
easy to say and easy to lose: a second copy of the decode-and-bake is a second thing to fix
when upstream moves. So these tests pin the shared surface by identity, and pin the handful
of things that must differ.
"""

from __future__ import annotations

import dataclasses
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


def test_the_weights_come_from_the_repo_when_there_is_no_local_tree(monkeypatch, tmp_path):
    monkeypatch.delenv("I2L_TRELLIS_MODEL_PATH", raising=False)
    route = dataclasses.replace(routes.ROCM, vendor=tmp_path / "not-built")
    assert route.model_path() == "microsoft/TRELLIS.2-4B"


def test_a_conventional_local_tree_is_used_without_being_asked(monkeypatch, tmp_path):
    """vendor/<route>/models/TRELLIS.2-4B holding a pipeline.json is a weights tree. Reading
    the environment alone would leave a machine that assembled one by hand still being told
    it is missing 15 GB."""
    monkeypatch.delenv("I2L_TRELLIS_MODEL_PATH", raising=False)
    route = dataclasses.replace(routes.ROCM, vendor=tmp_path)
    assert route.model_path() == "microsoft/TRELLIS.2-4B"
    (route.vendor / "models" / "TRELLIS.2-4B").mkdir(parents=True)
    assert route.model_path() == "microsoft/TRELLIS.2-4B", "an empty directory is not a tree"
    (route.vendor / "models" / "TRELLIS.2-4B" / "pipeline.json").write_text("{}")
    assert route.model_path() == str(route.vendor / "models" / "TRELLIS.2-4B")


def test_the_weights_can_be_pointed_at_a_local_directory(monkeypatch, tmp_path):
    """For machines that already have the weights -- a shared store, an offline box -- where
    re-fetching 16 GB is not on. Both routes honour it, since they load the same repo."""
    monkeypatch.setenv("I2L_TRELLIS_MODEL_PATH", str(tmp_path))
    for route in (routes.CUDA, routes.ROCM):
        assert route.model_path() == str(tmp_path)


def test_the_build_marker_is_the_same_name_on_both_routes():
    """Both bootstraps write `.i2l-build-complete`; readiness reads what the route asks for."""
    assert routes.CUDA.marker_name == routes.ROCM.marker_name == ".i2l-build-complete"

def test_the_rocm_route_drops_an_inherited_allocator_setting():
    """expandable_segments gives silent NaNs on ROCm. Not setting it is not enough when the
    viewer was started from a shell that exports it."""
    env = routes.ROCM.environ({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                               "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"})
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env and "PYTORCH_HIP_ALLOC_CONF" not in env
    assert routes.CUDA.environ({})["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_the_cuda_route_keeps_a_users_own_allocator_setting():
    """Upstream's default, not a rule: a user who tuned it for their card keeps their value.
    The attention backends are still pinned, since a stale one selects a backend that is
    not there."""
    env = routes.CUDA.environ({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                               "ATTN_BACKEND": "stale"})
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"
    assert env["ATTN_BACKEND"] == "flash_attn"


def test_a_tree_under_the_other_route_is_used_too(monkeypatch, tmp_path):
    """The Setup card counts weights under either route; a run that only looked under its
    own would read "present" and then download 16 GB."""
    monkeypatch.delenv("I2L_TRELLIS_MODEL_PATH", raising=False)
    route = dataclasses.replace(routes.ROCM, vendor=tmp_path / "trellis-rocm")
    other = tmp_path / "trellis-cuda" / "models" / "TRELLIS.2-4B"
    other.mkdir(parents=True)
    (other / "pipeline.json").write_text("{}")
    assert route.model_path() == str(other)
    own = route.vendor / "models" / "TRELLIS.2-4B"
    own.mkdir(parents=True)
    (own / "pipeline.json").write_text("{}")
    assert route.model_path() == str(own), "its own tree still comes first"


def test_the_run_and_the_setup_card_look_in_the_same_places():
    sys.path.insert(0, str(REPO))
    from viewer import backend_catalog

    assert routes.ROUTE_DIRS == backend_catalog.TRELLIS_ROUTE_DIRS
    assert routes.LOCAL_MODEL_DIR == backend_catalog.TRELLIS_LOCAL_MODEL_DIR
    for route in (routes.CUDA, routes.ROCM):
        assert set(route.local_trees()) == set(backend_catalog._local_trellis()) - {
            backend_catalog.trellis_local_models()}
