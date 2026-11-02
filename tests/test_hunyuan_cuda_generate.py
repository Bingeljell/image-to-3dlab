"""The NVIDIA Hunyuan3D-2.1 generator's torch-free contract: upstream's settings, weights
found where the installer put them, our own matte, and a complete licence record."""

from __future__ import annotations

import json

import bootstrap_hunyuan_cuda as boot
import hunyuan_cuda_generate as gen
import pytest


def _vendor(tmp_path, built=True):
    vendor = tmp_path / "hunyuan-cuda"
    root = gen.checkout(vendor)
    for relative in ("hy3dshape/hy3dshape/pipelines.py", "hy3dpaint/textureGenPipeline.py",
                     "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml", ".venv/bin/python"):
        base = vendor if relative.startswith(".venv") else root
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / relative).write_text("")
    if built:
        (vendor / gen.BUILT_MARKER_NAME).write_text(json.dumps({"commit": "abc123"}))
    return vendor


def test_defaults_are_upstreams():
    """demo.py: 6 views at 512; the shape pipeline's own steps, octree and guidance."""
    assert gen.DEFAULTS == {"seed": 1234, "steps": 50, "octree_resolution": 384,
                            "guidance_scale": 5.0, "max_num_view": 6, "paint_resolution": 512}


@pytest.mark.parametrize("bad", [{"octree_resolution": 1024}, {"max_num_view": 5},
                                 {"max_num_view": 10}, {"paint_resolution": 1024},
                                 {"steps": 0}])
def test_bad_settings_are_refused_before_a_model_loads(bad):
    with pytest.raises(ValueError):
        gen.validate(bad)


def test_validate_fills_defaults():
    assert gen.validate({"seed": 7})["octree_resolution"] == 384


def test_shape_weights_are_read_from_where_the_installer_put_them(tmp_path):
    env = gen.configure_environment(tmp_path / "hunyuan-cuda", {})
    assert env["HY3DGEN_MODELS"] == str(tmp_path / "hunyuan-cuda" / boot.MODELS.name)


def test_the_real_installer_and_generator_agree_on_the_models_folder():
    env = gen.configure_environment(gen.DEFAULT_VENDOR, {})
    shape = next(local for label, _, _, local, _ in boot.WEIGHTS if "shape" in label)
    assert shape == gen.DEFAULT_VENDOR / env["HY3DGEN_MODELS"].rsplit("/", 1)[-1] / boot.HUNYUAN_REPO


def test_import_paths_match_upstreams_demo(tmp_path):
    root = gen.checkout(tmp_path)
    assert gen.import_paths(tmp_path) == [root, root / "hy3dshape", root / "hy3dpaint"]


def test_paint_config_paths_are_absolute(tmp_path):
    """Upstream's defaults are relative to its own folder; the viewer runs from elsewhere."""
    overrides = gen.paint_config_overrides(tmp_path)
    assert all(value.startswith(str(tmp_path)) for value in overrides.values())
    assert overrides["realesrgan_ckpt_path"] == str(gen.realesrgan_path(tmp_path))


def test_realesrgan_is_found_where_the_installer_puts_it():
    assert gen.realesrgan_path(boot.VENDOR) == boot.REALESRGAN


def test_check_needs_a_finished_build(tmp_path, capsys):
    vendor = _vendor(tmp_path, built=False)
    assert gen.check_environment(vendor) == 1
    assert "completed build" in capsys.readouterr().out


def test_check_needs_realesrgan(tmp_path, capsys):
    vendor = _vendor(tmp_path)
    assert gen.check_environment(vendor) == 1
    assert "RealESRGAN" in capsys.readouterr().out


def test_generate_refuses_missing_weights_before_the_hub(tmp_path, monkeypatch):
    vendor = _vendor(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setattr(gen, "prepare_image", lambda *a, **k: pytest.fail("matted"))
    with pytest.raises(SystemExit, match="about 19.5 GB") as raised:
        gen.generate(tmp_path / "in.png", tmp_path / "out.glb", vendor, {})
    message = str(raised.value)
    assert "Hunyuan3D-2.1" in message and "NVIDIA / CUDA" in message
    assert "missing weights" in message


def test_generate_refuses_a_missing_install_before_touching_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "prepare_image", lambda *a, **k: pytest.fail("matted"))
    with pytest.raises(SystemExit, match="bootstrap_hunyuan_cuda"):
        gen.generate(tmp_path / "in.png", tmp_path / "out.glb", tmp_path / "nothing", {})


def test_intermediates_stay_beside_the_output(tmp_path):
    out = tmp_path / "run" / "model.glb"
    assert gen.work_dir(out).parent == out.parent
    assert gen.matted_path(out).parent == out.parent


def test_provenance_records_territory_licence_and_paint_parts(tmp_path):
    image, output = tmp_path / "in.png", tmp_path / "out.glb"
    image.write_bytes(b"png")
    output.write_bytes(b"glb")
    record = gen.build_provenance(image=image, output=output, parameters={"seed": 1},
                                  matte_model="birefnet-general-lite", shape_only=False,
                                  backend_revision="abc123")
    assert record["model"] == {"backend": "hunyuan3d-2.1", "route": "nvidia-cuda",
                               "parameters": {"seed": 1}}
    assert record["output"]["classification"] == "territory-restricted"
    assert any("EU" in condition for condition in record["license"]["conditions"])
    names = [c["component"] for c in record["components"]]
    assert names == ["rembg/birefnet-general-lite", "facebook/dinov2-giant", "RealESRGAN_x4plus"]
    assert record["software"]["backend_revision"] == "abc123"


def test_shape_only_provenance_lists_no_paint_parts(tmp_path):
    image, output = tmp_path / "in.png", tmp_path / "out.glb"
    image.write_bytes(b"png")
    output.write_bytes(b"glb")
    record = gen.build_provenance(image=image, output=output, parameters={}, matte_model=None,
                                  shape_only=True, backend_revision=None)
    assert record["components"] == []


def test_manifest_says_cuda_and_the_matte():
    manifest = gen.build_manifest(image="in.png", output="out.glb", params={}, shape_only=False,
                                  timings={}, artifacts={}, matte_model="u2net")
    assert manifest["device"] == "cuda" and manifest["matte_model"] == "u2net"
    assert manifest["backend"] == "hunyuan3d-2.1"


def test_an_unmatted_image_is_cut_out_by_our_remover(tmp_path, monkeypatch):
    """upstream's demo checks for RGB after converting to RGBA, so it never cuts anything."""
    from PIL import Image

    from image_to_3dlab import matte

    source = tmp_path / "in.png"
    Image.new("RGB", (32, 32), "white").save(source)
    cut = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    monkeypatch.setattr(matte, "cut_out", lambda image: (cut, "birefnet-general-lite"))
    path, model = gen.prepare_image(source, tmp_path / "out.glb", force_matte=False)
    assert model == "birefnet-general-lite"
    assert path == gen.matted_path(tmp_path / "out.glb") and path.is_file()


def test_a_real_cutout_goes_straight_in(tmp_path, monkeypatch):
    from PIL import Image

    from image_to_3dlab import matte

    source = tmp_path / "in.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    image.paste((255, 0, 0, 255), (8, 8, 24, 24))
    image.save(source)
    monkeypatch.setattr(matte, "cut_out", lambda *a: pytest.fail("re-matted"))
    assert gen.prepare_image(source, tmp_path / "out.glb", force_matte=False) == (source, None)


def test_main_requires_image_and_output():
    with pytest.raises(SystemExit):
        gen.main([])


def test_main_requires_a_glb(tmp_path):
    image = tmp_path / "in.png"
    image.write_bytes(b"png")
    with pytest.raises(SystemExit, match=".glb"):
        gen.main([str(image), str(tmp_path / "out.obj")])



def test_a_finished_run_exits_without_interpreter_teardown(monkeypatch):
    # Seen on a real pod: the GLB, manifest and provenance were written, then the process
    # segfaulted during Python's shutdown (bpy/open3d teardown), so the viewer read a
    # finished run as failed and showed no model. Exit straight after flushing.
    exits = []
    monkeypatch.setattr(gen.os, "_exit", exits.append)
    gen.finish(0)
    assert exits == [0]


def test_the_model_load_is_announced_before_it_starts(monkeypatch, capsys, tmp_path):
    # Seen on a real pod: loading the shape model took ~2.5 min and the page showed nothing,
    # so the run looked stuck. A line goes out before each slow load.
    import sys
    import types

    class Loaded(Exception):
        pass

    def from_pretrained(*_args, **_kwargs):
        raise Loaded

    pipelines = types.SimpleNamespace(
        Hunyuan3DDiTFlowMatchingPipeline=types.SimpleNamespace(from_pretrained=from_pretrained))
    monkeypatch.setitem(sys.modules, "hy3dshape", types.ModuleType("hy3dshape"))
    monkeypatch.setitem(sys.modules, "hy3dshape.pipelines", pipelines)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    with pytest.raises(Loaded):
        gen.generate_shape(tmp_path / "in.png", tmp_path / "m.glb", {})
    assert gen.LOADING_SHAPE in capsys.readouterr().out
    assert "minute" in gen.LOADING_SHAPE and "minute" in gen.LOADING_PAINT
