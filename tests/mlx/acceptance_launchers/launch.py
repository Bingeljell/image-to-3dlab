"""Install tiny model fixtures, then execute the shipped MLX entry points."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import yaml
from mlx import nn

REPO = Path(__file__).resolve().parents[3]
SHAPE_ROOT = REPO / "hunyuan_mlx" / "shape"
PAINT_ROOT = REPO / "hunyuan_mlx" / "paint"
PAINT_SCRIPT = PAINT_ROOT / "scripts" / "run_paint_pbr.py"
PAINT_WEIGHTS = Path("weights/hunyuan3d-paintpbr-v2-1")


def shape_models(root, *, octree, empty=False):
    from hy3dmlx import convert
    from hy3dmlx.models import dinov2, dit, shape_vae

    class Dino(nn.Module):
        def __init__(self, *, hidden, ffn_dim, **kwargs):
            super().__init__()
            assert (hidden, ffn_dim) == (2, 3)

        def __call__(self, pixels):
            print(f"shape model inference on {mx.default_device()}")
            return pixels[:, :1, 0, :2]

        def unconditional_embedding(self, batch, dtype):
            return mx.zeros((batch, 1, 2), dtype=dtype)

    class Velocity(nn.Module):
        def __init__(self, *, fixture):
            super().__init__()
            assert fixture == "velocity"

        def __call__(self, x, t, cond):
            return mx.ones_like(x) * cond[:, :, :1]

    class Vae(nn.Module):
        latent_shape = (1, 2)
        scale_factor = 2

        def __init__(self, *, fixture):
            super().__init__()
            assert fixture == "field"

        def decode(self, state):
            assert state.shape == (1, 1, 2)
            assert bool(mx.all(mx.isfinite(state)))
            return state

        def query_grid(self, context, *, bounds, octree_resolution, num_chunks):
            assert octree_resolution == octree
            values = mx.broadcast_to(mx.array([-1, 0, 1])[None, None, :], (3, 3, 3))
            return np.asarray(values), np.full(3, -bounds), np.full(3, bounds), (2, 2, 2)

    dinov2.DinoImageEncoder = Dino
    dit.Hunyuan3DDiT = Velocity
    shape_vae.ShapeVAE = Vae

    root.mkdir()
    if empty:
        return
    cfg = {
        "model": {"target": "Hunyuan3DDiT", "params": {"fixture": "velocity"}},
        "vae": {"params": {"fixture": "field"}},
        "conditioner": {"params": {"main_image_encoder": {"kwargs": {"config": {
            "hidden_size": 2, "num_attention_heads": 1, "num_hidden_layers": 1,
            "patch_size": 2, "use_swiglu_ffn": False, "layer_norm_eps": 1e-5,
        }}}}},
    }
    (root / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    mx.save_safetensors(str(root / "model.fp16.safetensors"), {
        convert.COND_PREFIX + "model.encoder.layer.0.mlp.fc2.weight": mx.zeros((2, 3)),
    })


def paint_files(root, *, bad_config):
    for stage in ("vae", "unet", "dinov2"):
        directory = root / PAINT_WEIGHTS / stage
        directory.mkdir(parents=True)
        if stage != "dinov2":
            config = "{" if stage == "unet" and bad_config else json.dumps({"fixture": stage})
            (directory / "config.json").write_text(config, encoding="utf-8")
        checkpoint = "model.safetensors" if stage == "dinov2" else "diffusion_pytorch_model.safetensors"
        mx.save_safetensors(str(directory / checkpoint), {"amplitude": mx.array(1.0)})


def paint_models():
    from hy3dpaint_mlx import dinov2, mesh_render, unet2p5d_pbr, vae

    class Vae(nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = mx.array(0.0)
            self.decoded = 0

        @classmethod
        def from_config(cls, config):
            assert config == {"fixture": "vae"}
            return cls()

        def encode_mean(self, pixels):
            print(f"paint model inference on {mx.default_device()}")
            assert pixels.shape[1:] == (8, 8, 3)
            return mx.zeros((pixels.shape[0], 1, 1, 4))

        def decode(self, slab):
            assert slab.shape == (6, 1, 1, 4)
            assert bool(mx.all(mx.isfinite(slab)))
            self.decoded += 1
            rgb = [0.2, 0.6, 1.0] if self.decoded == 1 else [1.0, 0.0, 0.3]
            return mx.broadcast_to((mx.array(rgb) * 2 - 1) * self.amplitude, (6, 8, 8, 3))

    class Unet(nn.Module):
        def __init__(self, config):
            super().__init__()
            assert config == {"fixture": "unet"}
            self.amplitude = mx.array(0.0)

        def prepare(self, reference, dino, posmap, h, n):
            assert (h, n) == (1, 6)
            return {"condition_embed_dict": None, "dino": dino, "rope": None}

        def __call__(self, latents, timestep, *args, ref_scale, **options):
            assert timestep == 999
            return mx.ones_like(latents) * self.amplitude * (0.2 + 0.4 * ref_scale)

    class Dino(nn.Module):
        def __init__(self):
            super().__init__()
            self.amplitude = mx.array(0.0)

        def __call__(self, pixels):
            assert pixels.shape == (1, 518, 518, 3)
            return mx.ones((1, 1, 2)) * self.amplitude

    def unwrap(vertices, faces):
        xy = vertices[:, :2]
        uv = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0)
        return np.arange(len(vertices)), faces, uv

    vae.AutoencoderKL = Vae
    unet2p5d_pbr.UNet2p5DPBRConditionModel = Unet
    dinov2.Dinov2Model = Dino
    mesh_render._RASTER_BACKEND = "numpy"
    sys.modules["xatlas"] = SimpleNamespace(parametrize=unwrap)


def run_shape(image, output, scenario):
    assert Path.cwd() == SHAPE_ROOT
    sys.path.insert(0, str(SHAPE_ROOT))
    weights = image.parent / "shape-weights"
    shape_models(weights, octree=2, empty=scenario == "empty")
    sys.argv = ["hy3dmlx.pipeline", str(image), "--weights", str(weights), "--out", str(output),
                "--steps", "3", "--guidance", "2.5", "--octree", "2", "--seed", "11"]
    runpy.run_module("hy3dmlx.pipeline", run_name="__main__", alter_sys=True)


def run_wrapper(image, output, scenario):
    assert Path.cwd() == REPO
    sys.path.insert(0, str(REPO))
    from scripts import hunyuan_mlx_xiong_generate as wrapper

    weights = image.parent / "shape-weights"
    shape_models(weights, octree=256)
    paint_root = image.parent / "paint"
    paint_files(paint_root, bad_config=scenario == "bad-config")
    wrapper.SHAPE_MODELS = {"2.0": weights}
    wrapper.PAINT_ROOT = paint_root
    wrapper.PAINT_PYTHON = Path(sys.executable)

    def run_paint_child(command, *, cwd, env, check):
        # The offline guard blocks model script names in subprocess arguments.
        # The runner below invokes the verified target after installing the fixtures.
        python, unbuffered, script = command
        assert Path(script) == PAINT_SCRIPT and unbuffered == "-u"
        assert cwd == str(paint_root)
        return subprocess.run(
            [python, unbuffered, __file__, "paint"],
            cwd=cwd, env=env, check=check, timeout=30,
        )

    wrapper.subprocess = SimpleNamespace(run=run_paint_child)
    os.environ["PAINT_SR"] = "0"
    sys.argv = [wrapper.__file__, str(image), str(output), "--octree-resolution", "256",
                "--no-compile-dit", "--no-octree-decode", "--quantize", "0",
                "--paint-seed", "7", "--paint-res", "8", "--paint-steps", "1", "--paint-tex", "9"]
    wrapper.main()


if __name__ == "__main__":
    assert sys.version_info[:2] == (3, 12)
    assert os.environ["AF_OFFLINE_TESTS"] == "1" and "unit_isolation" in sys.modules
    mx.set_default_device(mx.cpu)
    stage, *args = sys.argv[1:]
    if stage == "paint":
        assert not args
        sys.path.insert(0, str(PAINT_ROOT))
        paint_models()
        sys.argv = [str(PAINT_SCRIPT)]
        runpy.run_path(str(PAINT_SCRIPT), run_name="__main__")
    else:
        image, output, scenario = args
        launchers = {"shape": run_shape, "wrapper": run_wrapper}
        launchers[stage](Path(image), Path(output), scenario)
