"""Drive the paint launchers with controlled model outputs and real texture math."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

mx = pytest.importorskip("mlx.core")

from hunyuan_mlx.paint import hy3dpaint_mlx
from hunyuan_mlx.paint.hy3dpaint_mlx import mesh_render, raster, scheduler

pytestmark = pytest.mark.mlx_unit


@pytest.fixture
def paint_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(mesh_render, "_RASTER_BACKEND", "numpy")
    monkeypatch.setenv("PAINT_RES", "8")
    monkeypatch.setenv("PAINT_STEPS", "1")
    monkeypatch.setenv("PAINT_TEX", "9")
    monkeypatch.delenv("PAINT_OUT", raising=False)
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    source = tmp_path / "plane.glb"
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(source)
    image = tmp_path / "front.png"
    Image.new("RGB", (4, 4), (255, 0, 128)).save(image)
    monkeypatch.setenv("PAINT_MESH", str(source))
    monkeypatch.setenv("PAINT_IMG", str(image))
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx", hy3dpaint_mlx)
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.mesh_render", mesh_render)
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.raster", raster)
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.scheduler", scheduler)

    def uv_unwrap(points, indices):
        np.testing.assert_array_equal(points, vertices)
        np.testing.assert_array_equal(indices, faces)
        return np.arange(len(points)), indices, (points[:, :2] + 1) / 2

    monkeypatch.setitem(sys.modules, "xatlas", SimpleNamespace(parametrize=uv_unwrap))
    return vertices, faces, image


@pytest.mark.parametrize(("pbr", "use_sr"), [(False, False), (False, True), (True, False), (True, True)])
def test_paint_command_loads_local_configs_guides_once_and_writes_matching_materials(
    tmp_path, monkeypatch, paint_inputs, capsys, pbr, use_sr,
):
    vertices, faces, image = paint_inputs
    route = "hunyuan3d-paintpbr-v2-1" if pbr else "hunyuan3d-paint-v2-0"
    relative_weights = Path("weights") / route
    module_key = "run_paint_pbr" if pbr else "run_paint"
    model_key = "UNet2p5DPBRConditionModel" if pbr else "UNet2p5DConditionModel"
    unet_module = "unet2p5d_pbr" if pbr else "unet2p5d"
    expected_groups = ["vae", "unet", "dinov2"] if pbr else ["vae", "unet"]
    expected_files = {
        group: relative_weights / group / ("model.safetensors" if group == "dinov2"
                                          else "diffusion_pytorch_model.safetensors")
        for group in expected_groups
    }
    for group in ("vae", "unet"):
        config = relative_weights / group / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text(json.dumps({"fixture_route": route, "fixture_stage": group}))
    # Real schedulers exercise the terminal step in the two supported code paths.
    schedule = "ddim" if use_sr else "unipc"
    monkeypatch.setenv("PAINT_SCHED", schedule)
    monkeypatch.setenv("PAINT_SR", "1" if use_sr else "0")
    loaded = []
    calls = []
    decoded = []
    calls_sr = []
    albedo_rgb = [0.2, 0.6, 1.0]
    metallic_rgb = [1.0, 0.0, 0.3]

    def load(path):
        path = Path(path)
        stage = path.parent.name
        assert path == expected_files[stage]
        assert not path.exists()
        return stage

    def assign(model, stage, renames=()):
        assert (type(model).__name__, stage) in {("Vae", "vae"), ("Unet", "unet"), ("Dino", "dinov2")}
        expected_renames = {
            "vae": [(".to_out.0.", ".to_out.")],
            "unet": () if pbr else [("transformer_blocks.0.transformer.", "transformer_blocks.0.")],
            "dinov2": (),
        }
        assert renames == expected_renames[stage]
        loaded.append(stage)

    class Vae:
        @classmethod
        def from_config(cls, config):
            assert config == {"fixture_route": route, "fixture_stage": "vae"}
            return cls()

        def encode_mean(self, pixels):
            assert pixels.shape[1:] == (8, 8, 3)
            if pixels.shape[0] == 1:
                np.testing.assert_allclose(np.asarray(pixels[0, 0, 0]), [1, -1, 1 / 255], atol=1e-7)
            return mx.zeros((pixels.shape[0], 1, 1, 4))

        def decode(self, slab):
            assert len(calls) == 3
            initial = calls[-2][1][0, len(decoded)] if pbr else calls[-2][1][0]
            guided_velocity = 1.4 if pbr else 1.0
            alpha = 1e-8 if schedule == "ddim" else 2 ** -24
            expected = (np.sqrt(alpha) * initial - np.sqrt(1 - alpha) * guided_velocity) / 0.18215
            np.testing.assert_allclose(np.asarray(slab), expected, atol=2e-6)
            decoded.append(slab.shape)
            color = metallic_rgb if len(decoded) == 2 else albedo_rgb
            return mx.broadcast_to(mx.array(color) * 2 - 1, (slab.shape[0], 8, 8, 3))

    class Unet:
        def __init__(self, config):
            assert config == {"fixture_route": route, "fixture_stage": "unet"}
            self.unet = SimpleNamespace(learned_text_clip_gen=mx.ones((1, 2, 4)))

        def compute_condition_embed(self, reference):
            assert not pbr and reference.shape == (1, 1, 1, 1, 4)
            calls.append("prepare")
            return {"cached": True}

        def prepare(self, reference, dino, posmap, h, n):
            assert pbr and reference.shape == (1, 1, 1, 1, 4)
            assert dino.shape == (1, 1, 2)
            assert posmap.shape == (1, 6, 8, 8, 3)
            assert (h, n) == (1, 6)
            calls.append("prepare")
            return {"condition_embed_dict": {"cached": True}, "dino": dino, "rope": {"cached": True}}

        def __call__(self, latents, timestep, *args, **options):
            conditional = options["ref_scale"] == 1
            assert timestep == 999 and options["mva_scale"] == 1.0
            if pbr:
                assert latents.shape == (1, 2, 6, 1, 1, 4)
                cond = options["cond"]
                assert cond["condition_embed_dict"] == ({"cached": True} if conditional else None)
                assert cond["rope"] == {"cached": True}
                assert float(cond["dino"].sum()) == (2 if conditional else 0)
            else:
                assert latents.shape == (1, 6, 1, 1, 4)
                assert options["condition_embed_dict"] == ({"cached": True} if conditional else None)
                text, _, _, _, cameras, _ = args
                assert float(text.sum()) == (8 if conditional else 0)
                np.testing.assert_array_equal(np.asarray(cameras), [[0, 1, 2, 3, 4, 5]])
            calls.append(("conditional" if conditional else "unconditional", np.asarray(latents)))
            return (0.6 if conditional else 0.2) * mx.ones_like(latents)

    class Dino:
        def __call__(self, rgb):
            np.testing.assert_allclose(np.asarray(rgb[0, 0, 0]), [
                (1 - 0.485) / 0.229, -0.456 / 0.224, ((128 / 255) - 0.406) / 0.225,
            ], atol=1e-6)
            return mx.ones((1, 1, 2))

    def sr_load():
        assert use_sr
        calls_sr.append("load")
        return "fixture"

    def sr_sample(model, pixels, *, tile):
        assert model == "fixture" and tile == 256 and pixels.shape == (8, 8, 3)
        calls_sr.append("sample")
        return np.repeat(np.repeat(pixels, 2, axis=0), 2, axis=1)

    monkeypatch.setattr(mx, "load", load)
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.convert", SimpleNamespace(load_torch_weights=assign))
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.vae", SimpleNamespace(AutoencoderKL=Vae))
    monkeypatch.setitem(sys.modules, f"hy3dpaint_mlx.{unet_module}", SimpleNamespace(**{model_key: Unet}))
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.dinov2", SimpleNamespace(Dinov2Model=Dino))
    monkeypatch.setitem(sys.modules, "hy3dpaint_mlx.realesrgan",
                        SimpleNamespace(load_rrdbnet=sr_load, upscale=sr_sample))
    runpy.run_module(f"hunyuan_mlx.paint.scripts.{module_key}", run_name="__main__")

    assert loaded == expected_groups
    assert calls[0] == "prepare" and [c[0] for c in calls[1:]] == ["conditional", "unconditional"]
    np.testing.assert_array_equal(calls[1][1], calls[2][1])
    assert len(decoded) == (2 if pbr else 1)
    assert calls_sr == (["load"] + ["sample"] * (12 if pbr else 6) if use_sr else [])
    output = next((tmp_path / "outputs").iterdir())
    glb = output / ("textured_mesh_pbr.glb" if pbr else "textured_mesh.glb")
    assert str(glb) in capsys.readouterr().out
    record = json.loads(glb.with_suffix(".provenance.json").read_text())
    assert record["input"]["path"] == str(image)
    assert record["components"][0]["weights"] == str(relative_weights)
    result = trimesh.load(glb, force="mesh", process=False)
    np.testing.assert_array_equal(result.vertices, vertices)
    np.testing.assert_array_equal(result.faces, faces)
    material = result.visual.material
    assert material.baseColorTexture.getpixel((4, 4))[:3] == pytest.approx([51, 153, 255], abs=1)
    if pbr:
        assert material.metallicRoughnessTexture.getpixel((4, 4))[:3] == pytest.approx([255, 0, 76], abs=1)
        assert (output / "pbr_albedo_views.png").exists()
    else:
        with Image.open(output / "textured_preview.png") as preview:
            assert preview.size == (1536, 512)
        assert (output / "paint_views.png").exists()
