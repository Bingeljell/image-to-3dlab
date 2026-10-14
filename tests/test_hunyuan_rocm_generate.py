"""scripts/hunyuan_rocm_generate.py: the parts that can be checked without a GPU.

The interesting logic is `fit_paint_settings`: upstream's paint defaults want about 21 GB
and a 20 GB card does not have that, so the run has to walk down a ladder of views,
resolution and attention chunk. Getting that wrong is an out-of-memory crash twenty
minutes in, or a silently worse result, so it is checked here rather than found there.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import hunyuan_rocm_generate as gen

GB = 1024 ** 3


def test_a_big_card_is_left_on_upstreams_defaults():
    """Chunking costs a little speed, so a card that does not need it should not pay it."""
    fitted = gen.fit_paint_settings(48 * GB)
    assert fitted == {"max_num_view": 6, "paint_resolution": 512, "sdpa_query_chunk": 0}


def test_a_twenty_gigabyte_card_gets_chunked_but_keeps_the_full_settings():
    """20 GB is the common consumer card and it is one rung below the unchunked defaults:
    the same picture, done more slowly, rather than a smaller picture."""
    fitted = gen.fit_paint_settings(20 * GB)
    assert fitted["max_num_view"] == 6 and fitted["paint_resolution"] == 512
    assert fitted["sdpa_query_chunk"] > 0


@pytest.mark.parametrize("gigabytes", [48, 24, 20, 16, 12, 8, 4, 1])
def test_the_ladder_only_ever_gives_back_what_it_has_to(gigabytes):
    """Every rung must be at least as tight as the one above it, or a bigger card would
    get a worse result than a smaller one.

    Compared between neighbouring rungs of the table rather than between the sampled card
    sizes: several sampled sizes share a rung, and 24 GB and 20 GB differ in chunk only,
    so "at least as tight" has to mean per-column across the table, not per-sample.
    """
    rungs = gen.FIT_LADDER
    for (floor_a, views_a, res_a, chunk_a), (floor_b, views_b, res_b, chunk_b) in itertools.pairwise(
            rungs):
        assert floor_b < floor_a, (floor_a, floor_b)
        assert views_b <= views_a, (views_a, views_b)
        assert res_b <= res_a, (res_a, res_b)
        # 0 means "do not chunk at all", which is the loosest setting there is and belongs
        # at the top. Below that, fewer tokens at a time means less memory at a time, so
        # the chunk must not grow as the card gets smaller.
        if chunk_a:
            assert chunk_b <= chunk_a, (chunk_a, chunk_b)
        else:
            assert chunk_b > 0, "a card below the top rung must still chunk"


def test_bigger_cards_are_never_given_a_smaller_picture():
    for bigger, smaller in ((48, 24), (20, 16), (16, 12), (12, 8), (8, 4), (4, 1)):
        above = gen.fit_paint_settings(bigger * GB)
        below = gen.fit_paint_settings(smaller * GB)
        assert below["max_num_view"] <= above["max_num_view"], bigger
        assert below["paint_resolution"] <= above["paint_resolution"], bigger


def test_unknown_free_memory_still_runs_rather_than_refusing():
    """A box whose rocm-smi is missing is not a box that cannot generate. It starts one
    rung down from the best, which is the safe guess."""
    fitted = gen.fit_paint_settings(None)
    assert fitted["sdpa_query_chunk"] > 0
    assert fitted["max_num_view"] >= 1


def test_every_rung_produces_settings_the_validator_accepts():
    """A ladder rung the validator rejects would fail only after a model was loaded."""
    for gigabytes in (48, 24, 20, 16, 12, 8, 1):
        fitted = gen.fit_paint_settings(gigabytes * GB)
        merged = gen.validate({**fitted, "octree_resolution": 384})
        assert merged["max_num_view"] in gen.VALID_VIEWS
        assert merged["paint_resolution"] in gen.VALID_PAINT_RESOLUTION


def test_an_explicit_request_beats_the_ladder():
    """Someone who passed --max-num-view 9 on a 12 GB card meant it, even if it will not
    fit; silently giving them 4 views and a smaller texture is not what they asked for."""
    fitted = gen.fit_paint_settings(12 * GB, wanted_views=9, wanted_resolution=768)
    assert fitted["max_num_view"] == 9 and fitted["paint_resolution"] == 768
    # The chunk still comes from the ladder: it is a safety measure, not a setting.
    assert fitted["sdpa_query_chunk"] == 1024


def test_a_chunk_larger_than_the_sequence_is_left_alone_by_the_patch():
    """The patch skips chunking when there are fewer tokens than the chunk asks for, so a
    generous chunk on a big card costs one comparison rather than any memory."""
    assert gen.fit_paint_settings(48 * GB)["sdpa_query_chunk"] == 0


def test_the_ladder_goes_all_the_way_down_for_a_tiny_card():
    fitted = gen.fit_paint_settings(1 * GB)
    assert fitted["max_num_view"] == 1 and fitted["paint_resolution"] == 128


def test_the_chunk_is_read_by_the_patch_under_the_name_the_patch_expects():
    assert gen.CHUNK_ENV == "HY3D_SDPA_QUERY_CHUNK"


def test_the_chunk_reaches_the_environment_the_patch_reads(tmp_path):
    fresh: dict = {}
    gen.configure_environment(tmp_path, chunk=1024, environ=fresh)
    assert fresh[gen.CHUNK_ENV] == "1024"


def test_no_chunk_means_the_variable_is_not_set_at_all(tmp_path):
    fresh: dict = {}
    gen.configure_environment(tmp_path, chunk=0, environ=fresh)
    assert gen.CHUNK_ENV not in fresh, "an empty value would be read as a chunk of zero rows"


def test_the_expandable_segments_trap_the_nvidia_route_sets_is_not_set_here(tmp_path):
    """It has produced silent NaNs on ROCm, and a run that writes nothing is worse than a
    slow one. Its absence here is deliberate, so it is pinned."""
    fresh: dict = {}
    gen.configure_environment(tmp_path, chunk=512, environ=fresh)
    assert not any(key.startswith("PYTORCH_CUDA_ALLOC_CONF") for key in fresh), fresh


def test_the_shape_weights_are_where_the_bootstrap_put_them(tmp_path):
    fresh: dict = {}
    gen.configure_environment(tmp_path, chunk=0, environ=fresh)
    assert fresh["HY3DGEN_MODELS"] == str(tmp_path / "models")


def test_the_log_says_what_was_chosen_and_why():
    line = gen.describe_fit(gen.fit_paint_settings(20 * GB), 20 * GB)
    assert "6 views" in line and "512px" in line
    assert "20.0 GB" in line and "chunk 4096" in line


def test_the_log_says_so_when_it_cannot_read_the_card():
    assert "unknown" in gen.describe_fit(gen.fit_paint_settings(None), None)


def test_verify_paths_names_the_bootstrap_that_builds_it(tmp_path):
    """Wrong script name in a "run this to fix it" message is the whole message."""
    problems = gen.verify_paths(tmp_path)
    joined = " ".join(problems)
    assert "bootstrap_hunyuan_rocm.py" in joined
    assert "bootstrap_hunyuan_cuda.py" not in joined
    for problem in problems:
        assert "missing" in problem


def test_a_missing_checkout_is_reported_before_any_model_loads(tmp_path):
    problems = gen.verify_paths(tmp_path)
    assert any(".venv python" in p for p in problems)


def test_the_manifest_says_this_is_the_rocm_route(tmp_path):
    manifest = gen.build_manifest(image="a.png", output="a.glb", params={}, shape_only=False,
                                  timings={}, artifacts={}, matte_model=None,
                                  fitted={"sdpa_query_chunk": 1024}, free_bytes=12 * GB)
    assert manifest["device"] == "hip"
    assert "ROCm" in manifest["port"]
    assert manifest["generator"] == "hunyuan_rocm_generate.py"
    # What the run actually did, not what was asked for: the ladder decides the views.
    assert manifest["fitted"]["sdpa_query_chunk"] == 1024
    assert manifest["free_vram_bytes"] == 12 * GB


def test_the_provenance_records_the_rocm_route_not_the_cuda_one(tmp_path):

    image = tmp_path / "in.png"
    output = tmp_path / "out.glb"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    output.write_bytes(b"glTF")
    record = gen.build_provenance(image=image, output=output, parameters={},
                                  matte_model=None, shape_only=False, backend_revision="abc")
    assert record["model"]["route"] == "amd-rocm"
    assert "Hunyuan" in record["license"]["name"]
    assert record["software"]["backend_revision"] == "abc"


def test_shape_only_does_not_claim_the_paint_models_it_never_loaded(tmp_path):
    """The provenance travels with the file; claiming a licence for a model that did not
    run is a false record."""
    image = tmp_path / "in.png"
    output = tmp_path / "out.glb"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    output.write_bytes(b"glTF")
    names = [c["component"] for c in gen.build_provenance(
        image=image, output=output, parameters={}, matte_model=None, shape_only=True,
        backend_revision=None)["components"]]
    assert "facebook/dinov2-giant" not in names
    assert "RealESRGAN_x4plus" not in names


def test_validation_rejects_what_the_pipeline_cannot_do():
    for bad in ({"octree_resolution": 999}, {"paint_resolution": 999},
                {"max_num_view": 0}, {"steps": 0}):
        with pytest.raises(ValueError):
            gen.validate(bad)


def test_validation_fills_in_the_defaults():
    merged = gen.validate({})
    assert merged["seed"] == gen.DEFAULTS["seed"]
    assert merged["octree_resolution"] == 384


def test_the_intermediate_files_are_named_after_the_output():
    assert gen.work_dir(Path("/o/fox.glb")).name == "fox__hunyuan"
    assert gen.matted_path(Path("/o/fox.glb")).name == "fox__matted.png"