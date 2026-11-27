"""Run the MLX diagnostics with tiny CPU tensors and fixed external observations."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import trimesh

from hunyuan_mlx.shape.hy3dmlx import layers, sampler

pytestmark = pytest.mark.mlx_unit
SCRIPTS = Path(__file__).resolve().parents[2] / "hunyuan_mlx/shape/scripts"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    original_np_load = np.load
    original_mx_load = mx.load
    original_mx_save = mx.save_safetensors
    calls = []

    def local(path):
        assert Path(path).parent == Path("/tmp")
        return tmp_path / Path(path).name

    monkeypatch.setattr(np, "load", lambda path: original_np_load(local(path)))
    monkeypatch.setattr(mx, "load", lambda path: original_mx_load(str(local(path))))
    monkeypatch.setattr(mx, "save_safetensors", lambda path, a: original_mx_save(str(local(path)), a))

    class Dino:
        hidden = 2

        def __call__(self, pixels):
            assert pixels.shape == (1, 1, 1, 3)
            return pixels[..., :2].reshape(1, 1, 2)

        def parameters(self):
            return {"weight": mx.array(1)}

    class DiT:
        time_factor = 100

        def __call__(self, x, t, cond, guidance=None):
            calls.append(("velocity", np.array(x), np.array(t), np.array(cond)))
            if guidance is not None:
                return x * guidance - t
            return mx.ones_like(x) * cond[:, :1, :1]

        def parameters(self):
            return {"weight": mx.array(1)}

        def latent_in(self, x):
            return x * 2

        def time_in(self, x):
            assert x.shape == (1, 256)
            return x[:, :1] * 0 + 3

        def cond_in(self, ctx):
            return ctx + 3

        @property
        def double_blocks(self):
            return [lambda img, txt, vec: (img + vec, txt - vec),
                    lambda img, txt, vec: (img * 2, txt + 1)]

        @property
        def single_blocks(self):
            return [lambda cat, vec: cat + 6, lambda cat, vec: cat * 2]

        def final_layer(self, latent, vec):
            return latent + vec

    class Vae:
        scale_factor = 2
        latent_shape = (1, 64)

        def decode(self, latents):
            calls.append(("decode", np.array(latents)))
            return latents

        def parameters(self):
            return {"weight": mx.array(1)}

        def geo_decoder(self, points, context):
            assert context.shape == (1, 512, 64)
            return points[..., :1]

        def query_grid(self, context, *, octree_resolution, num_chunks, bounds=None):
            calls.append(("dense", octree_resolution, num_chunks, bounds, np.array(context)))
            v = 1 if octree_resolution != 192 or num_chunks == 8000 else 2
            return np.array([[[-1., v], [0., 0.]], [[1., 0.], [0., 1.]]]), -1, 1, 2

        def query_grid_octree(self, context, *, octree_resolution):
            calls.append(("octree", octree_resolution, np.array(context)))
            return np.array([[[float("nan"), 1], [0, 0]], [[1, 0], [0, 1]]]), -1, 1, 2

    class Pipeline:
        vae = Vae()

        def __init__(self, dino, dit, vae, cfg, dtype):
            assert dtype == mx.float32
            self.vae = vae

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls(*load_models(path, **kwargs), dtype=kwargs["dtype"])

        def encode_image(self, path):
            assert path == "reference/Hunyuan3D-2.1/assets/demo.png"
            return mx.array([[[2., 3.]], [[1., 1.]]])

        def _grid_to_mesh(self, grid, bmin, bmax, gs, level):
            calls.append(("mesh", grid.shape, bmin, bmax, gs, level))
            return trimesh.creation.box()

    def load_models(path, *, dtype, verbose, quantize=None):
        calls.append(("load", path, dtype, verbose, quantize))
        return Dino(), DiT(), Vae(), {"kind": "synthetic"}

    for name, members in {
        "hy3dmlx.convert": {"load_models": load_models},
        "hy3dmlx.pipeline": {"Hunyuan3DShapePipeline": Pipeline},
    }.items():
        module = ModuleType(name)
        vars(module).update(members)
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setitem(sys.modules, "hy3dmlx.sampler", sampler)
    monkeypatch.setitem(sys.modules, "hy3dmlx.layers", layers)

    def run(name, *args):
        path = str(SCRIPTS / f"{name}.py")
        monkeypatch.setattr(sys, "argv", [path, *args])
        runpy.run_path(path, run_name="__main__")

    monkeypatch.chdir(tmp_path)
    return SimpleNamespace(
        run=run, calls=calls, path=tmp_path,
        save=original_mx_save, load_tensor=original_mx_load,
    )


def test_mini_parity_schedules_guidance_and_isolates_each_stage(studio, capsys):
    p = studio.path
    np.save(p / "dino_pixels.npy", [[[[2]], [[3]], [[4]]]])
    np.save(p / "dino_out_torch.npy", [[[2, 3]]])
    np.save(p / "noise.npy", [[[1., 2.]]])
    np.save(p / "cond_torch.npy", [[[2., 3.]], [[1., 1.]]])
    np.save(p / "latents_torch.npy", [[[4., 8.]]])
    np.save(p / "grid_torch.npy", np.zeros((2, 2, 2)))
    studio.run("mlx_compare")
    vel = [call for call in studio.calls if call[0] == "velocity"]
    assert len(vel) == 29
    np.testing.assert_allclose(vel[0][1], [[[1, 2]], [[1, 2]]])
    np.testing.assert_allclose(vel[-1][2], [28 / 29, 28 / 29])
    decode = next(call for call in studio.calls if call[0] == "decode")
    np.testing.assert_array_equal(decode[1], [[[2, 4]]])
    assert studio.calls[-1][:4] == ("dense", 96, 8000, 1.01)
    output = capsys.readouterr().out
    assert "[DINO]  cosine 1.000000  maxabs 0.0000" in output
    assert "[DiT]" in output and "maxabs 3.0000" in output
    assert "mlx range -1.0000..1.0000 (frac>0 0.3750)" in output


def test_v21_parity_uses_duplicate_noise_and_half_step(studio, capsys):
    p = studio.path
    np.save(p / "v21_pixels.npy", [[[[2]], [[3]], [[4]]]])
    np.save(p / "v21_dino.npy", [[[2, 3]]])
    np.save(p / "v21_noise.npy", [[[1., 2.]]])
    np.save(p / "v21_cond.npy", [[[2., 3.]], [[1., 1.]]])
    np.save(p / "v21_dit_v.npy", [[[2, 2]], [[1, 1]]])
    studio.run("mlx_compare_v21")
    assert len(studio.calls) == 2
    _, x, t, cond = studio.calls[1]
    np.testing.assert_array_equal(x, [[[1, 2]], [[1, 2]]])
    np.testing.assert_array_equal(t, [.5, .5])
    np.testing.assert_array_equal(cond, [[[2, 3]], [[1, 1]]])
    assert "[HunYuanDiTPlain] cosine 1.000000  maxabs 0.0000" in capsys.readouterr().out


def test_mlx_trace_compares_intermediate_tensors(studio, capsys):
    p = studio.path
    np.save(p / "noise.npy", [[[1., 2.], [3., 4.]], [[-1., -1.], [-1., -1.]]])
    np.save(p / "cond_torch.npy", [[[4., 6.]]])
    np.savez(p / "dit_trace.npz",
             latent_in=[[[2, 4], [6, 8]]], cond_in=[[[7, 9]]], vec=[3],
             db0_img=[[[5, 7], [9, 11]]], db0_txt=[[[4, 6]]],
             db_img=[[[10, 14], [18, 22]]], db_txt=[[[5, 7]]],
             sb0=[[[11, 13], [16, 20], [24, 28]]],
             sb=[[[32, 40], [48, 56]]], out=[[[35, 43], [51, 59]]])
    studio.run("mlx_dit_trace")
    output = capsys.readouterr().out
    assert output.count("cos 1.000000  maxabs 0.0000") == 10
    assert "mlx std 8.9443 ref std 8.9443" in output


def test_turbo_mlx_parity_uses_guidance_fixture(studio, capsys):
    p = studio.path
    np.save(p / "turbo_x.npy", [[1., 2.]])
    np.save(p / "turbo_cond.npy", [[[2., 3.]]])
    np.save(p / "turbo_t.npy", [0.5])
    np.save(p / "turbo_g.npy", [5.0])
    np.save(p / "turbo_v.npy", [[4.5, 9.5]])
    studio.run("oracle_turbo_check", "mlx")
    assert "[turbo DiT] cosine 1.0000000  maxabs 0.00000" in capsys.readouterr().out


@pytest.mark.parametrize("octree", [None, "192"])
def test_grid_benchmark_measures_requested_chunks_and_drift(studio, capsys, octree):
    studio.run("bench_grid", "weights/local", *([octree] if octree else []))
    queries = [call for call in studio.calls if call[0] == "dense"]
    counts = [8000, 32000, 100000, 300000, 600000]
    assert [call[1:3] for call in queries] == [(int(octree or 256), n) for n in counts]
    assert queries[0][-1].dtype == np.float16
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(counts)
    assert "max-drift" not in lines[0]
    assert lines[-1].endswith("1.00e+00" if octree else "0.00e+00")


def save_run_fixture(studio, *, dino=False):
    studio.save(str(studio.path / "run_fixture.safetensors"), {
        "cond": mx.array([[[2., 3.]], [[1., 1.]]]),
        "noise": mx.array([[[1., 2.]]]),
        "sigmas": mx.array([0., 0.5, 1.]),
    })
    if dino:
        studio.save(str(studio.path / "dino_fixture.safetensors"),
                    {"pixels": mx.array([[[[2., 3., 4.]]]])})


@pytest.mark.parametrize("resolution", [None, "96"])
def test_stage_benchmark_runs_each_stage_for_cold_and_warm(studio, capsys, resolution):
    save_run_fixture(studio, dino=True)
    studio.run("bench_stages", *([resolution] if resolution else []))
    queries = [call for call in studio.calls if call[0] == "dense"]
    assert [call[1:4] for call in queries] == [(int(resolution or 256), 50000, None)] * 2
    for query in queries:
        np.testing.assert_array_equal(query[-1], [[[5.5, 6.]]])
    assert len([call for call in studio.calls if call[0] == "velocity"]) == 4
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    for phase, line in zip(("cold", "warm"), lines, strict=True):
        assert line.startswith(f"[{phase}] R={resolution or 256}")
        assert "denoise " in line and " | total " in line and line.endswith("(8 verts)")


@pytest.mark.parametrize("octree", [False, True])
def test_fixture_runner_selects_grid_and_exports_mesh(studio, capsys, octree):
    save_run_fixture(studio)
    if octree:
        out = studio.path / "fast.glb"
        studio.run("python_from_fixture", "64", str(out), "octree")
    else:
        (studio.path / "outputs").mkdir()
        out = studio.path / "outputs/python_from_fixture.glb"
        studio.run("python_from_fixture")
    assert out.is_file()
    assert trimesh.load(out).to_geometry().vertices.shape == (8, 3)
    assert len([call for call in studio.calls if call[0] == "velocity"]) == 2
    query = studio.calls[-2]
    expected = ("octree", 64) if octree else ("dense", 128, 50000, None)
    assert query[:-1] == expected
    np.testing.assert_array_equal(query[-1], [[[3.5, 4.]]])
    assert f"active {7 if octree else 8} -> 8 verts" in capsys.readouterr().out


@pytest.mark.parametrize("quant", [None, "4"])
def test_memory_benchmark_clears_cache_and_resets_peak(studio, capsys, monkeypatch, quant):
    events = studio.calls

    def clear():
        events.append(("clear",))
        if quant:
            raise RuntimeError("allocator already shut down")

    monkeypatch.setattr(mx, "clear_cache", clear)
    monkeypatch.setattr(mx, "metal", SimpleNamespace(
        clear_cache=(lambda: events.append(("metal-clear",))) if quant else None))
    monkeypatch.setattr(mx, "get_active_memory", lambda: 2_000_000_000)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: events.append(("reset",)))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 3_125_000_000)
    studio.run("measure_mlx", "weights/local", "fp16", *([quant] if quant else []))
    assert events[0] == ("load", "weights/local", mx.float16, False, 4 if quant else None)
    expected = ["load", "clear"] + (["metal-clear"] if quant else [])
    assert [entry[0] for entry in events] == expected + ["reset", "velocity", "decode", "dense"]
    assert events[-1][:4] == ("dense", 128, 8000, None)
    assert "resident-weights  2.00 GB | inference-peak  3.12 GB" in capsys.readouterr().out


def test_dit_fixture_writes_seeded_inputs_and_outputs(studio, capsys):
    studio.run("dump_dit_fixture")
    dumped = studio.load_tensor(str(studio.path / "dit_fixture.safetensors"))
    assert dumped["x"].shape == (1, 512, 64)
    assert dumped["cond"].shape == (1, 1370, 1536)
    np.testing.assert_allclose(np.array(dumped["t"]), [0.3])
    np.testing.assert_allclose(np.array(dumped["x"][0, 0, :2]), [1.7640524, 0.40015721])
    np.testing.assert_array_equal(np.array(dumped["v"]), np.broadcast_to(
        np.array(dumped["cond"][..., :1])[:, :1, :1], (1, 512, 64)))
    assert "| v std" in capsys.readouterr().out


def test_swift_fixtures_write_grid_samples_and_schedule(studio, capsys):
    studio.run("dump_swift_fixtures")
    vae = studio.load_tensor(str(studio.path / "vae_fixture.safetensors"))
    assert vae["lat"].shape == (1, 512, 64)
    assert vae["q"].shape == (1, 256, 3)
    assert np.abs(np.array(vae["q"])).max() <= 1
    np.testing.assert_array_equal(np.array(vae["sdf"]), np.array(vae["q"][..., :1]))
    run = studio.load_tensor(str(studio.path / "run_fixture.safetensors"))
    assert run["noise"].shape == (1, 512, 64)
    np.testing.assert_array_equal(np.array(run["cond"]), [[[2, 3]], [[1, 1]]])
    np.testing.assert_allclose(np.array(run["sigmas"]), np.linspace(0, 1, 30))
    assert "run fixture:" in capsys.readouterr().out
