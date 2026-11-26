"""Check Hunyuan command wiring and local model guards without model weights."""

import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import trimesh
from PIL import Image

from scripts import hunyuan_cuda_generate as cuda
from scripts import hunyuan_rocm_generate as rocm


def test_cuda_weight_check_uses_same_models_folder_as_loader(tmp_path, monkeypatch):
    default_models = tmp_path / "default-models"
    monkeypatch.setattr(cuda.bootstrap, "MODELS", default_models)
    vendor = tmp_path / "custom-vendor"
    for model in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        (default_models / cuda.bootstrap.HUNYUAN_REPO / model).mkdir(parents=True)
    absent = cuda.missing_weights(vendor, tmp_path / "hub", shape_only=True)
    assert len(absent) == 2
    expected = Path(cuda.configure_environment(vendor, {})["HY3DGEN_MODELS"])
    for model in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        (expected / cuda.bootstrap.HUNYUAN_REPO / model).mkdir(parents=True)
    assert cuda.missing_weights(vendor, tmp_path / "hub", shape_only=True) == []


@pytest.fixture
def local_hunyuan(monkeypatch, tmp_path):
    root = tmp_path / "vendor"
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for key in ("HY3DGEN_MODELS", "PYTORCH_CUDA_ALLOC_CONF", rocm.CHUNK_ENV):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        rocm.measured_free_bytes, "__defaults__", (rocm._torch_free_bytes, lambda: None)
    )
    for module in (cuda, rocm):
        vendor = root / module.bootstrap.VENDOR.name
        for relative in (
            ".venv/bin/python", module.BUILT_MARKER_NAME,
            f"{module.checkout(vendor).name}/hy3dshape/hy3dshape/pipelines.py",
            f"{module.checkout(vendor).name}/hy3dpaint/textureGenPipeline.py",
            f"{module.checkout(vendor).name}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml",
        ):
            path = vendor / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"commit": "local-test-revision"}))
    return root, hub


@pytest.mark.parametrize("module", [cuda, rocm], ids=["cuda", "rocm"])
def test_hunyuan_cli_rejects_invalid_paths_settings_and_absent_weights(
    module, local_hunyuan, tmp_path
):
    root, _ = local_hunyuan
    vendor = root / module.bootstrap.VENDOR.name
    image = tmp_path / "input.png"
    output = tmp_path / "out.glb"
    vendor_args = ["--vendor-root", str(vendor)]
    with pytest.raises(SystemExit, match="image and output are required"):
        module.main(vendor_args)
    with pytest.raises(SystemExit, match="missing input image"):
        module.main([*vendor_args, str(image), str(output)])
    Image.new("RGBA", (8, 8)).save(image)
    with pytest.raises(SystemExit, match="must be a .glb path"):
        module.main([*vendor_args, str(image), str(output.with_suffix(".obj"))])
    with pytest.raises(ValueError, match="steps must be at least 1"):
        module.main([*vendor_args, str(image), str(output), "--steps", "0"])
    with pytest.raises(SystemExit, match="about 19.5 GB") as refused:
        module.main([*vendor_args, str(image), str(output), "--shape-only"])
    assert f"bootstrap_hunyuan_{'cuda' if module is cuda else 'rocm'}.py --weights-only" in str(refused.value)
    assert "shape model" in str(refused.value)
    assert "PBR paint" not in str(refused.value)
    assert not output.exists()


@pytest.mark.parametrize(
    ("module", "mode"),
    [(cuda, "shape"), (cuda, "paint"), (cuda, "paint-missing"), (cuda, "no-gpu"),
     (rocm, "shape"), (rocm, "paint"), (rocm, "paint-missing"), (rocm, "paint-black"),
     (rocm, "flash-failure"), (rocm, "no-gpu")],
    ids=lambda x: x.__name__.split("_")[1] if isinstance(x, ModuleType) else x,
)
def test_hunyuan_run_records_real_artifacts_or_refuses_failed_backend(
    module, mode, local_hunyuan, tmp_path, monkeypatch, capsys
):
    root, hub = local_hunyuan
    vendor = root / module.bootstrap.VENDOR.name
    setup = module.bootstrap
    for name in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        path = vendor / "models" / setup.HUNYUAN_REPO / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")

    for repo, name in (
        (setup.HUNYUAN_REPO, "hunyuan3d-paintpbr-v2-1/config.json"),
        (setup.DINO_REPO, "model.safetensors"),
    ):
        model = hub / f"models--{repo.replace('/', '--')}" / "snapshots/test" / name
        model.parent.mkdir(parents=True)
        model.write_bytes(b"test file")
    upscaler = module.realesrgan_path(vendor)
    upscaler.parent.mkdir(parents=True, exist_ok=True)
    upscaler.touch()

    source = tmp_path / "source.png"
    picture = Image.new("RGBA", (8, 8))
    picture.paste((100, 120, 140, 255), (2, 2, 6, 6))
    picture.save(source)
    destination = tmp_path / "delivery" / "model.GLB"
    calls = []
    shape_only = mode in {"shape", "flash-failure", "no-gpu"}
    seed = 17
    resolution = 512 if module is cuda else 256

    class Shape:
        @staticmethod
        def from_pretrained(repo):
            assert repo == setup.HUNYUAN_REPO
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert Path(os.environ["HY3DGEN_MODELS"]) == vendor / "models"
            calls.append("load shape")
            return Shape()

        def enable_flashvdm(self):
            calls.append("flash")
            if mode == "flash-failure":
                raise RuntimeError("fixture exception")

        def __call__(self, **kwargs):
            assert kwargs["image"].mode == "RGBA"
            assert kwargs["image"].getpixel((0, 0))[3] == 0
            assert kwargs["num_inference_steps"] == 7
            assert kwargs["guidance_scale"] == 3.5
            assert kwargs["generator"] == ("seed", seed)
            calls.append("shape")
            return [trimesh.creation.box()]

    class Paint:
        def __init__(self, config):
            expected = module.paint_config_overrides(vendor)
            assert all(getattr(config, field) == value for field, value in expected.items())
            assert config.views == 6
            assert config.resolution == resolution
            calls.append("load paint")

        def __call__(self, *, mesh_path, image_path, output_mesh_path):
            assert Path(mesh_path) == module.work_dir(destination) / "shape.glb"
            assert Path(mesh_path).is_file()
            assert Path(image_path) == source
            obj = Path(output_mesh_path)
            assert obj.name == "textured_mesh.obj"
            calls.append("paint")
            if mode != "paint-missing":
                trimesh.creation.icosphere(subdivisions=1).export(obj.with_suffix(".glb"))
            if mode == "paint-black":
                Image.new("RGB", (16, 16)).save(obj.with_suffix(".jpg"))

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: mode != "no-gpu",
        mem_get_info=lambda: (16 * 1024**3, 24 * 1024**3),
        empty_cache=lambda: calls.append("free"),
    )
    torch.manual_seed = lambda value: ("seed", value)
    shape = ModuleType("hy3dshape.pipelines")
    shape.Hunyuan3DDiTFlowMatchingPipeline = Shape
    paint = ModuleType("textureGenPipeline")
    paint.Hunyuan3DPaintConfig = lambda views, resolution: SimpleNamespace(
        views=views, resolution=resolution
    )
    paint.Hunyuan3DPaintPipeline = Paint
    for dependency in (torch, shape, paint):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    monkeypatch.setitem(sys.modules, "torchvision_fix", None)
    argv = [
        str(source), str(destination), "--vendor-root", str(vendor),
        "--seed", str(seed), "--steps", "7", "--guidance-scale", "3.5",
        "--max-num-view", "6", "--paint-resolution", str(resolution),
    ]
    if shape_only:
        argv.append("--shape-only")
    if module is rocm and mode not in {"no-gpu", "shape"}:
        argv.append("--flash-vdm")
    error = {
        "paint-black": "all-black texture",
        "paint-missing": "wrote no GLB",
        "no-gpu": "torch cannot see",
    }.get(mode)
    if error:
        with pytest.raises(SystemExit, match=error):
            module.main(argv)
        assert not destination.exists()
        assert not module.provenance_path(destination).exists()
        if mode == "no-gpu":
            assert not calls
        return

    assert module.main(argv) == 0
    assert destination.is_file()
    manifest = json.loads(destination.with_suffix(".json").read_text())
    assert manifest["params"]["seed"] == seed
    assert manifest["shape_only"] is shape_only
    assert manifest["matte_model"] is None
    assert ("paint" in manifest["timings"]) is not shape_only
    provenance = json.loads(module.provenance_path(destination).read_text())
    assert provenance["software"]["backend_revision"] == "local-test-revision"
    assert provenance["output"]["classification"] == "territory-restricted"
    assert provenance["model"]["route"] == ("nvidia-cuda" if module is cuda else "amd-rocm")
    if mode == "flash-failure":
        assert "fixture exception" in capsys.readouterr().out
    assert calls.count("free") == (1 if shape_only else 2)


@pytest.mark.parametrize("hardware", ["no-gpu", "wrong-build", "working", "mem-info-error"])
def test_rocm_check_reports_device_and_import_failures(
    hardware, local_hunyuan, tmp_path, monkeypatch, capsys
):
    root, hub = local_hunyuan
    vendor = root / rocm.bootstrap.VENDOR.name
    setup = rocm.bootstrap
    for name in ("hunyuan3d-dit-v2-1", "hunyuan3d-vae-v2-1"):
        (vendor / "models" / setup.HUNYUAN_REPO / name).mkdir(parents=True)
    for repo, name in ((setup.HUNYUAN_REPO, "hunyuan3d-paintpbr-v2-1/config.json"),
                       (setup.DINO_REPO, "model.safetensors")):
        file = hub / f"models--{repo.replace('/', '--')}" / "snapshots/local" / name
        file.parent.mkdir(parents=True)
        file.touch()
    upscaler = rocm.realesrgan_path(vendor)
    upscaler.parent.mkdir(parents=True, exist_ok=True)
    upscaler.touch()
    patched = []

    def apply_fix():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["HY3DGEN_MODELS"] == str(vendor / "models")
        patched.append("shim")

    def mem_get_info():
        if hardware == "mem-info-error":
            raise RuntimeError("test unavailable")
        return 12 * 1024**3, 16 * 1024**3

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: hardware != "no-gpu",
        get_device_name=lambda ordinal: "Test device" if ordinal == 0 else pytest.fail("ordinal"),
        mem_get_info=mem_get_info,
    )
    torch.version = SimpleNamespace(hip=None if hardware == "wrong-build" else "7.2")
    shim = ModuleType("torchvision_fix")
    shim.apply_fix = apply_fix
    renderer = ModuleType("DifferentiableRenderer")
    renderer.mesh_inpaint_processor = object()
    paint = ModuleType("textureGenPipeline")
    paint.Hunyuan3DPaintConfig = paint.Hunyuan3DPaintPipeline = object()
    shape = ModuleType("hy3dshape.pipelines")
    shape.Hunyuan3DDiTFlowMatchingPipeline = object()
    for dependency in (torch, shim, renderer, paint, shape, ModuleType("custom_rasterizer")):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)

    works = hardware in {"working", "mem-info-error"}
    assert rocm.main(["--check", "--vendor-root", str(vendor)]) == int(not works)
    text = capsys.readouterr().out
    expected = {
        "no-gpu": "torch cannot see a GPU",
        "wrong-build": "Re-run scripts/bootstrap_hunyuan_rocm.py --code-only",
        "working": "Test device (ROCm 7.2)",
        "mem-info-error": "VRAM unknown",
    }[hardware]
    assert expected in text
    assert ("fit:" in text) is works
    assert patched == ["shim"]


def test_rocm_prepare_forced_matte_saves_image_and_reports_fallback(tmp_path, monkeypatch, capsys):
    from image_to_3dlab import matte

    source = tmp_path / "source.png"
    Image.new("RGBA", (12, 12)).save(source)

    def cut_out(image):
        assert image.mode == "RGB" and image.size == (12, 12)
        return Image.new("RGBA", image.size, (10, 20, 30, 40)), "test-remover"

    monkeypatch.setattr(matte, "cut_out", cut_out)
    monkeypatch.setattr(matte, "fallback_note", lambda model: f"fallback for {model}")
    result, method = rocm.prepare_image(source, tmp_path / "delivery/model.glb", force_matte=True)
    assert result.name == "model__matted.png" and method == "test-remover"
    with Image.open(result) as image:
        assert image.getpixel((0, 0)) == (10, 20, 30, 40)
    assert "fallback for test-remover" in capsys.readouterr().out
    output = tmp_path / "delivery/model.glb"
    trimesh.creation.box().export(output)
    record = rocm.build_provenance(image=source, output=output, parameters={},
                                   matte_model=method, shape_only=True, backend_revision=None)
    remover, = record["components"]
    assert remover["component"] == "rembg/test-remover"
    assert remover["purpose"] == "background removal"


def test_rocm_command_exit_preserves_preflight_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setattr(sys, "argv", [rocm.__file__, "--check", "--vendor-root", str(tmp_path)])
    monkeypatch.setattr(os, "_exit", lambda code: sys.exit(code))
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(rocm.__file__, run_name="__main__")
    assert exc.value.code == 1
    assert "missing shape pipeline" in capsys.readouterr().out


def test_rocm_generation_reports_missing_checkout_before_opening_image(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    output = tmp_path / "out.glb"
    with pytest.raises(SystemExit, match="bootstrap_hunyuan_rocm.py"):
        rocm.generate(tmp_path / "missing-input.png", output, tmp_path / "missing-vendor", {})
    assert not output.exists()
