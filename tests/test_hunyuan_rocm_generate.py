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


def test_a_big_card_is_left_on_upstreams_settings():
    """Chunking costs a little speed, so a card that does not need it should not pay it."""
    fitted = gen.fit_paint_settings(48 * GB)
    assert fitted == {"max_num_view": 6, "paint_resolution": 256, "sdpa_query_chunk": 0}


def test_a_twenty_gigabyte_card_gets_chunked_but_keeps_every_view():
    """20 GB is the common consumer card and it is one rung below the unchunked settings:
    the same picture, done more slowly, rather than fewer views."""
    fitted = gen.fit_paint_settings(20 * GB)
    assert fitted["max_num_view"] == 6 and fitted["paint_resolution"] == 256
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
    assert "6 views" in line and "256px" in line
    assert "20.0 GB" in line and "chunk 4096" in line


def test_the_log_says_so_when_it_cannot_read_the_card():
    assert "unknown" in gen.describe_fit(gen.fit_paint_settings(None), None)


# --- the 512px black texture, found by running this on a real card -----------------------

def test_this_route_never_asks_for_the_resolution_that_comes_back_black():
    """Upstream offers 512 and 768. On this card the paint UNet's multiview output has a
    maximum of 0.02 across the whole image at 512 -- uniformly black -- where at 256 it is
    an ordinary picture with no non-finite values. Every rung, and the CLI, agree.

    The cap is not because 512 runs out of memory: it finishes with 20 GB free. It is
    because one run in a row is worse than one that is never wrong."""
    assert gen.MAX_PAINT_RESOLUTION == 256
    assert gen.VALID_PAINT_RESOLUTION == (128, 256)
    for gigabytes in (48, 24, 20, 16, 12, 8, 1):
        assert gen.fit_paint_settings(gigabytes * GB)["paint_resolution"] <= 256, gigabytes


def test_a_black_texture_is_told_apart_from_a_dark_one(tmp_path):
    """The check is on the spread, not the brightness, so a genuinely black model is not
    mistaken for the failure and told to re-run."""
    pytest.importorskip("numpy")
    numpy = pytest.importorskip("numpy")
    from PIL import Image

    black = tmp_path / "black.jpg"
    Image.fromarray(numpy.zeros((64, 64, 3), dtype="uint8")).save(black)
    assert gen.texture_is_blank(black) is True

    dark = tmp_path / "dark.jpg"
    varying = numpy.zeros((64, 64, 3), dtype="uint8")
    varying[:, :, 0] = numpy.linspace(0, 90, 64)[None, :]  # a dark but varied surface
    Image.fromarray(varying).save(dark)
    assert gen.texture_is_blank(dark) is False


def test_a_run_that_baked_a_black_texture_stops_rather_than_shipping_it(tmp_path):
    """Without this the viewer shows a finished model with no texture and says nothing.
    Found on a real card at 512px, where the paint stage returned cleanly and wrote a
    valid GLB whose base colour was entirely black."""
    pytest.importorskip("numpy")
    numpy = pytest.importorskip("numpy")
    from PIL import Image

    Image.fromarray(numpy.zeros((64, 64, 3), dtype="uint8")).save(
        tmp_path / "textured_mesh.jpg")
    with pytest.raises(SystemExit) as raised:
        gen._refuse_a_blank_texture(tmp_path, {"paint_resolution": 512})
    message = str(raised.value)
    assert "all-black texture" in message
    assert "--paint-resolution 256" in message, "and say how to get a real texture"
    assert "not a problem with the picture" in message


def test_a_good_run_is_not_interrupted(tmp_path):
    pytest.importorskip("numpy")
    numpy = pytest.importorskip("numpy")
    from PIL import Image

    varying = numpy.zeros((64, 64, 3), dtype="uint8")
    varying[:, :, 0] = numpy.linspace(0, 200, 64)[None, :]
    Image.fromarray(varying).save(tmp_path / "textured_mesh.jpg")
    assert gen._refuse_a_blank_texture(tmp_path, {"paint_resolution": 256}) is None


def test_no_texture_at_all_is_not_this_failure(tmp_path):
    """An older layout, or a run that wrote the GLB some other way, must not be reported
    as a black texture: there is nothing to judge."""
    assert gen._refuse_a_blank_texture(tmp_path, {"paint_resolution": 512}) is None


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
    # The default must itself be a value this route can actually produce.
    assert merged["paint_resolution"] <= gen.MAX_PAINT_RESOLUTION


def test_the_intermediate_files_are_named_after_the_output():
    assert gen.work_dir(Path("/o/fox.glb")).name == "fox__hunyuan"
    assert gen.matted_path(Path("/o/fox.glb")).name == "fox__matted.png"

def test_a_small_card_gets_fewer_views_when_nobody_asked_for_more():
    """The defaults used to be filled in before the fit, so they read as a request and the
    ladder only ever chose the chunk: 6 views at 256px on a card with 4 GB free."""
    settings, fitted = gen.plan_paint({"seed": 1}, 4 * GB)
    assert (settings["max_num_view"], settings["paint_resolution"]) == (1, 128)
    assert fitted["sdpa_query_chunk"] == 512


def test_an_explicit_request_still_beats_the_ladder():
    settings, _ = gen.plan_paint({"max_num_view": 4, "paint_resolution": 256}, 4 * GB)
    assert (settings["max_num_view"], settings["paint_resolution"]) == (4, 256)


def test_free_memory_is_asked_of_torch_before_the_vendor_tool():
    """On an APU the tool's "VRAM" is a 1 GB slice of a 124 GB shared pool, and ROCm 10 has
    no rocm-smi at all. Torch answers for what a run can really allocate."""
    assert gen.measured_free_bytes(lambda: 100 * GB, lambda: 1 * GB) == 100 * GB
    assert gen.measured_free_bytes(lambda: None, lambda: 8 * GB) == 8 * GB
    assert gen.measured_free_bytes(lambda: None, lambda: None) is None


def test_a_black_texture_at_the_top_size_suggests_a_smaller_one(tmp_path):
    """Already at 256, "re-run with 256" is no advice at all."""
    numpy = pytest.importorskip("numpy")
    from PIL import Image

    Image.fromarray(numpy.zeros((64, 64, 3), dtype="uint8")).save(
        tmp_path / "textured_mesh.jpg")
    with pytest.raises(SystemExit) as raised:
        gen._refuse_a_blank_texture(tmp_path, {"paint_resolution": 256})
    assert "--paint-resolution 128" in str(raised.value)


def _all_weights(vendor: Path, hub: Path) -> None:
    shape = vendor / "models" / "tencent" / "Hunyuan3D-2.1"
    (shape / "hunyuan3d-dit-v2-1").mkdir(parents=True)
    (shape / "hunyuan3d-vae-v2-1").mkdir(parents=True)
    paint = hub / "models--tencent--Hunyuan3D-2.1" / "snapshots" / "abc"
    (paint / "hunyuan3d-paintpbr-v2-1").mkdir(parents=True)
    dino = hub / "models--facebook--dinov2-giant" / "snapshots" / "abc"
    dino.mkdir(parents=True)
    (dino / "model.safetensors").write_bytes(b"x")
    gen.realesrgan_path(vendor).parent.mkdir(parents=True)
    gen.realesrgan_path(vendor).write_bytes(b"x")


def test_nothing_missing_means_nothing_to_refuse(tmp_path):
    _all_weights(tmp_path / "vendor", tmp_path / "hub")
    assert gen.missing_weights(tmp_path / "vendor", tmp_path / "hub") == []


def test_every_missing_weight_set_is_named(tmp_path):
    missing = gen.missing_weights(tmp_path / "vendor", tmp_path / "hub")
    text = "\n".join(missing)
    for name in ("shape model", "shape VAE", "PBR paint", "DINOv2", "RealESRGAN"):
        assert name in text


def test_shape_only_does_not_ask_for_the_paint_weights(tmp_path):
    missing = gen.missing_weights(tmp_path / "vendor", tmp_path / "hub", shape_only=True)
    assert len(missing) == 2 and all("shape" in line for line in missing)


def test_the_cache_follows_the_usual_variables(tmp_path):
    assert gen.hf_hub_dir({"HF_HUB_CACHE": "/x"}) == Path("/x")
    assert gen.hf_hub_dir({"HF_HOME": "/y"}) == Path("/y/hub")


def test_a_run_with_no_weights_refuses_instead_of_downloading(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    for problem in gen.verify_paths(vendor):  # a built checkout, minus the weights
        path = Path(problem.rsplit(": ", 1)[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(gen, "prepare_image", lambda *a, **k: pytest.fail("got past the check"))
    with pytest.raises(SystemExit, match="unannounced") as refused:
        gen.generate(tmp_path / "in.png", tmp_path / "out.glb", vendor, {})
    assert "bootstrap_hunyuan_rocm.py --weights-only" in str(refused.value)


def test_check_reports_missing_weights_without_loading_anything(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(gen, "_enter_upstream", lambda *a, **k: pytest.fail("loaded upstream"))
    assert gen.check_environment(tmp_path / "vendor") == 1
    assert "--weights-only" in capsys.readouterr().out
