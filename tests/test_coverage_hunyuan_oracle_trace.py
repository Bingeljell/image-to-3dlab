"""Exercise the Torch diagnostic commands without the upstream checkout or checkpoints."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "hunyuan_mlx/shape/scripts"


@pytest.fixture
def oracle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    original_load, original_save, original_savez = np.load, np.save, np.savez

    def move(path):
        assert Path(path).parent == Path("/tmp")
        return tmp_path / Path(path).name

    monkeypatch.setattr(np, "load", lambda path: original_load(move(path)))
    monkeypatch.setattr(np, "save", lambda path, a: original_save(move(path), a))
    monkeypatch.setattr(np, "savez", lambda path, **kw: original_savez(move(path), **kw))

    class DiT:
        time_factor = 100
        guidance_embed = True

        def __init__(self, hidden):
            assert hidden == 3

        def eval(self):
            return self

        def float(self):
            return self

        def load_state_dict(self, state, strict):
            assert not strict
            assert set(state) == {"gain"}
            torch.testing.assert_close(state["gain"], torch.tensor(2.0))
            return SimpleNamespace(missing_keys=["bias"])

        def latent_in(self, x):
            return x * 2

        def time_in(self, value):
            return value + 1

        def cond_in(self, ctx):
            return ctx + 3

        @property
        def double_blocks(self):
            return [lambda img, txt, vec, pe: (img + vec, txt - vec),
                    lambda img, txt, vec, pe: (img * 2, txt + 1)]

        @property
        def single_blocks(self):
            return [lambda cat, vec, pe: cat + 6, lambda cat, vec, pe: cat * 2]

        def final_layer(self, value, vec):
            return value + vec

        def __call__(self, x, t, cond, guidance):
            assert x.shape == (1, 512, 64)
            assert cond["main"].shape == (1, 1370, 1536)
            torch.testing.assert_close(t, torch.tensor([0.3]))
            torch.testing.assert_close(guidance, torch.tensor([5.0]))
            return x * guidance - t

    def timestep_embedding(time, width, factor):
        torch.testing.assert_close(time, torch.tensor([0.5]))
        assert (width, factor) == (256, 100)
        return torch.tensor([2.0])

    def load_file(path, device="cpu"):
        assert device == "cpu"
        assert not Path(path).exists()
        return {"model.gain": torch.tensor(2, dtype=torch.float16), "vae.ignored": None}

    modules = {
        "hy3dshape.models.denoisers.hunyuan3ddit": {
            "Hunyuan3DDiT": DiT, "timestep_embedding": timestep_embedding,
        },
        "safetensors.torch": {"load_file": load_file},
        "safetensors": {},
    }
    for name, members in modules.items():
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["safetensors"].torch = sys.modules["safetensors.torch"]
    for suffix in ("mini", "mini-turbo"):
        path = tmp_path / f"weights/Hunyuan3D-2mini/hunyuan3d-dit-v2-{suffix}"
        path.mkdir(parents=True)
        (path / "config.yaml").write_text("model:\n  params:\n    hidden: 3\n")
    return tmp_path, original_load, original_save


def test_oracle_trace_keeps_first_and_last_block_outputs(oracle, capsys):
    folder, load, save = oracle
    save(folder / "noise.npy", [[[1., 2.], [3., 4.]], [[-1., -1.], [-1., -1.]]])
    save(folder / "cond_torch.npy", [[[4., 6.]], [[-1., -1.]]])
    runpy.run_path(str(SCRIPTS / "oracle_dit_trace.py"), run_name="__main__")
    with load(folder / "dit_trace.npz") as trace:
        np.testing.assert_array_equal(trace["latent_in"], [[[2, 4], [6, 8]]])
        np.testing.assert_array_equal(trace["cond_in"], [[[7, 9]]])
        np.testing.assert_array_equal(trace["vec"], [3])
        np.testing.assert_array_equal(trace["db0_img"], [[[5, 7], [9, 11]]])
        np.testing.assert_array_equal(trace["db0_txt"], [[[4, 6]]])
        np.testing.assert_array_equal(trace["db_img"], [[[10, 14], [18, 22]]])
        np.testing.assert_array_equal(trace["db_txt"], [[[5, 7]]])
        np.testing.assert_array_equal(trace["sb0"], [[[11, 13], [16, 20], [24, 28]]])
        np.testing.assert_array_equal(trace["sb"], [[[32, 40], [48, 56]]])
        np.testing.assert_array_equal(trace["out"], [[[35, 43], [51, 59]]])
    assert "std 8.9443 mean 47.0000" in capsys.readouterr().out


@pytest.mark.parametrize("args", [[], ["torch"]])
def test_turbo_reference_exports_guidance_inputs(oracle, capsys, monkeypatch, args):
    folder, load, _ = oracle
    script = str(SCRIPTS / "oracle_turbo_check.py")
    monkeypatch.setattr(sys, "argv", [script, *args])
    runpy.run_path(script, run_name="__main__")
    np.testing.assert_allclose(load(folder / "turbo_v.npy"), load(folder / "turbo_x.npy") * 5 - 0.3)
    np.testing.assert_array_equal(load(folder / "turbo_t.npy"), np.array([0.3], dtype=np.float32))
    np.testing.assert_array_equal(load(folder / "turbo_g.npy"), [5.0])
    assert "guidance_embed: True | missing 1" in capsys.readouterr().out
