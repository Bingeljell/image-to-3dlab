"""Exercise diagnostic control flow with CPU tensors and synthetic reference models."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from hunyuan_mlx.shape.scripts import measure_mps, oracle_compare, oracle_compare_v21


@pytest.fixture
def reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []

    class Model:
        def __init__(self, tag):
            self.tag = tag

        def eval(self):
            return self

        def float(self):
            return self

        def to(self, device, dtype):
            assert device.type == "cpu"
            assert dtype == torch.float16
            calls.append(("to", self.tag))

        def load_state_dict(self, state, strict):
            assert not strict
            assert set(state) == {"value"}
            expected = torch.float16 if self.tag.startswith("mps") else torch.float32
            assert state["value"].dtype == expected
            calls.append(("load", self.tag))
            return SimpleNamespace(missing_keys=["bias"])

    class Denoiser(Model):
        def __call__(self, latents, steps, context, **kwargs):
            torch.testing.assert_close(latents[0], latents[1])
            torch.testing.assert_close(steps, torch.full_like(steps, 0.5))
            if self.tag.startswith("mps"):
                assert latents.shape == (2, 3, 64)
                assert context["main"].shape == (2, 1370, 4)
                assert latents.dtype == torch.float16
            else:
                torch.testing.assert_close(context["main"], torch.tensor([[2.0], [1.0]]))
            calls.append(("denoise", self.tag, kwargs))
            if self.tag.startswith("mps"):
                calls.append(("mps-input", latents[0].clone()))
            return torch.stack([torch.full_like(latents[0], 2), torch.ones_like(latents[1])])

    class Vae(Model):
        scale_factor = 2.0
        geo_decoder = object()

        def __init__(self, tag, num_latents):
            super().__init__(tag)
            self.num_latents = num_latents

        def __call__(self, latents):
            calls.append(("decode", latents.clone()))
            return latents

        def volume_decoder(self, decoded, decoder, **kwargs):
            torch.testing.assert_close(decoded, calls[-1][1])
            assert decoder is self.geo_decoder
            assert kwargs == dict(bounds=1.01, num_chunks=8000,
                                  octree_resolution=96, enable_pbar=False)
            return torch.tensor([[-1.0, 0.0, 1.0, 2.0]])

    class Condition(Model):
        def __init__(self, tag, main_image_encoder):
            super().__init__(tag)
            self.main_image_encoder = SimpleNamespace(
                transform=lambda pixels: 2 * pixels,
                model=lambda pixels: SimpleNamespace(last_hidden_state=pixels + 1),
            )

        def __call__(self, image):
            torch.testing.assert_close(image, torch.tensor([[-1.0, 1.0]]))
            return {"main": torch.tensor([[2.0]])}

        def unconditional_embedding(self, count):
            assert count == 1
            return {"main": torch.tensor([[1.0]])}

    class Schedule:
        config = SimpleNamespace(num_train_timesteps=10)

        def __init__(self, tag):
            assert tag == "schedule"

        def set_timesteps(self, sigmas, device):
            assert device == "cpu"
            np.testing.assert_allclose(sigmas, np.linspace(0, 1, 30))
            self.timesteps = torch.tensor([5, 5])

        def step(self, values, timestep, latents):
            assert timestep == 5
            torch.testing.assert_close(values, torch.full_like(latents, 6))
            return SimpleNamespace(prev_sample=latents + values)

    def processor(tag):
        assert tag == "processor"

        def load_image(path):
            assert path == oracle_compare.IMG
            return {"image": torch.tensor([[-1.0, 1.0]])}

        return load_image

    def checkpoint(path, device):
        assert device == "cpu"
        assert path.endswith("/model.fp16.safetensors")
        return {f"{prefix}.value": torch.ones(1, dtype=torch.float16)
                for prefix in ("model", "vae", "conditioner", "ignored")}

    def config(folder, tag, target="upstream.Hunyuan3DDiT"):
        folder = Path(folder)
        folder.mkdir(parents=True)
        data = {
            "model": {"params": {"tag": f"{tag}-dit"}, "target": target},
            "vae": {"params": {"tag": f"{tag}-vae", "num_latents": 3}},
            "conditioner": {"params": {
                "tag": f"{tag}-cond",
                "main_image_encoder": {"kwargs": {"config": {"hidden_size": 4}}},
            }},
            "scheduler": {"params": {"tag": "schedule"}},
            "image_processor": {"params": {"tag": "processor"}},
        }
        (folder / "config.yaml").write_text(yaml.safe_dump(data))

    def install(*, runtime=torch, mini=Denoiser, plain=Denoiser):
        modules = {
            "safetensors.torch": {"load_file": checkpoint},
            "hy3dshape.models.autoencoders.model": {"ShapeVAE": Vae},
            "hy3dshape.models.conditioner": {"SingleImageEncoder": Condition},
            "hy3dshape.models.denoisers.hunyuan3ddit": {"Hunyuan3DDiT": mini},
            "hy3dshape.models.denoisers.hunyuandit": {"HunYuanDiTPlain": plain},
            "hy3dshape.schedulers": {"FlowMatchEulerDiscreteScheduler": Schedule},
            "hy3dshape.preprocessors": {"ImageProcessorV2": processor},
        }
        for name, members in modules.items():
            module = ModuleType(name)
            vars(module).update(members)
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setitem(sys.modules, "torch", runtime)
        monkeypatch.setattr(sys, "path", sys.path.copy())

    return SimpleNamespace(calls=calls, dit=Denoiser, config=config, install=install)


@pytest.fixture
def redirect_dumps(tmp_path, monkeypatch):
    save = np.save

    def save_in_temp(path, array):
        path = Path(path)
        assert path.parent == Path("/tmp")
        save(tmp_path / path.name, array)

    monkeypatch.setattr(np, "save", save_in_temp)


def test_mini_reference_exports_tensors(reference, redirect_dumps, capsys):
    reference.config(oracle_compare.WD, "mini")
    reference.install()
    runpy.run_path(oracle_compare.__file__, run_name="__main__")
    np.testing.assert_array_equal(np.load("dino_pixels.npy"), [[0.0, 2.0]])
    np.testing.assert_array_equal(np.load("dino_out_torch.npy"), [[1.0, 3.0]])
    np.testing.assert_array_equal(np.load("cond_torch.npy"), [[2.0], [1.0]])
    noise = np.random.RandomState(0).randn(1, 512, 64).astype(np.float32)
    np.testing.assert_array_equal(np.load("noise.npy"), noise)
    latents = np.load("latents_torch.npy")
    np.testing.assert_allclose(latents, noise + 12, atol=2e-6)
    torch.testing.assert_close(reference.calls[-1][1], torch.from_numpy(latents / 2))
    assert reference.calls[:3] == [
        ("load", "mini-dit"), ("load", "mini-vae"), ("load", "mini-cond"),
    ]
    assert reference.calls[3:5] == [("denoise", "mini-dit", {"guidance": None})] * 2
    np.testing.assert_array_equal(np.load("grid_torch.npy"), [-1.0, 0.0, 1.0, 2.0])
    console = capsys.readouterr().out
    assert "mean 2.0000 std 1.4142" in console
    assert "range -1.0000..2.0000 frac>0 0.5000" in console


def test_v21_reference_exports_single_forward(reference, redirect_dumps, capsys):
    reference.config(oracle_compare_v21.WD, "v21")
    reference.install()
    runpy.run_path(oracle_compare_v21.__file__, run_name="__main__")
    np.testing.assert_array_equal(np.load("v21_pixels.npy"), [[0.0, 2.0]])
    np.testing.assert_array_equal(np.load("v21_dino.npy"), [[1.0, 3.0]])
    np.testing.assert_array_equal(np.load("v21_cond.npy"), [[2.0], [1.0]])
    np.testing.assert_array_equal(np.load("v21_noise.npy"),
                                  np.random.RandomState(0).randn(1, 3, 64).astype(np.float32))
    np.testing.assert_array_equal(np.load("v21_dit_v.npy"),
                                  np.stack([np.full((3, 64), 2), np.ones((3, 64))]))
    assert reference.calls == [
        ("load", "v21-dit"), ("load", "v21-cond"), ("denoise", "v21-dit", {}),
    ]
    assert "DiT single-forward v (2, 3, 64) std 0.5007 mean 1.5000" in capsys.readouterr().out


@pytest.mark.parametrize("plain", [False, True])
def test_mps_report_uses_configured_model_and_memory_samples(reference, monkeypatch, capsys, plain):
    reference.config("model", "mps",
                     "upstream.HunYuanDiTPlain" if plain else "upstream.Hunyuan3DDiT")

    def dit(variant):
        def build(**params):
            reference.calls.append(("variant", variant))
            return reference.dit(**params)
        return build

    samples = iter([2_500_000_000, 3_750_000_000])

    def device(name):
        assert name == "mps"
        return torch.device("cpu")

    torch_cpu = SimpleNamespace(
        device=device, float16=torch.float16, no_grad=torch.no_grad,
        randn=torch.randn, zeros=torch.zeros, tensor=torch.tensor, cat=torch.cat,
        mps=SimpleNamespace(
            synchronize=lambda: reference.calls.append(("sync",)),
            current_allocated_memory=lambda: next(samples),
            driver_allocated_memory=lambda: 4_500_000_000,
        ),
    )
    reference.install(runtime=torch_cpu, mini=dit("mini"), plain=dit("plain"))
    monkeypatch.setattr(measure_mps.resource, "getrusage", lambda who: (
        SimpleNamespace(ru_maxrss=5_250_000_000)
    ))
    monkeypatch.setattr(sys, "argv", ["measure_mps.py", "model"])
    runpy.run_path(measure_mps.__file__, run_name="__main__")
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert reference.calls[0] == ("variant", "plain" if plain else "mini")
    assert reference.calls[1:9] == [
        ("load", "mps-dit"), ("load", "mps-vae"), ("load", "mps-cond"),
        ("to", "mps-dit"), ("to", "mps-vae"), ("to", "mps-cond"),
        ("sync",), ("denoise", "mps-dit", {}),
    ]
    assert reference.calls[-2][0] == "decode"
    assert reference.calls[9][0] == "mps-input"
    assert reference.calls[10] == ("sync",)
    assert reference.calls[-2][1].shape == (1, 3, 64)
    torch.testing.assert_close(reference.calls[-2][1][0], reference.calls[9][1] / 2)
    assert reference.calls[-1] == ("sync",)
    console = capsys.readouterr().out
    assert console.startswith("MPS  model")
    assert "weights  2.50 GB | mps-current  3.75 GB | mps-driver  4.50 GB | process-RSS  5.25 GB" in console
