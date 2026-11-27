"""Run shape launch and scheduling against a synthetic field and model boundaries."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")

from hunyuan_mlx.shape.hy3dmlx import convert, pipeline

pytestmark = pytest.mark.mlx_unit


@pytest.mark.parametrize("distilled", [False, True])
def test_shape_pipeline_routes_guidance_and_mesh_queries_without_a_model(tmp_path, capsys, distilled):
    image = tmp_path / "red.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image)
    latent_shape, seed, scale = (1, 1, 2), 11, 2.5
    expected_velocity = 6 if distilled else (1 - 0.485) / 0.229 * scale
    stochastic = np.asarray(mx.random.normal(latent_shape, key=mx.random.key(seed), dtype=mx.float32))
    events = []

    class Dino:
        def __call__(self, pixels):
            assert pixels.shape == (1, 518, 518, 3)
            np.testing.assert_allclose(np.asarray(pixels[0, 0, 0]), [
                (1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225,
            ], atol=1e-6)
            events.append("encode")
            return pixels[:, :1, 0, :2]

        def unconditional_embedding(self, batch, dtype):
            assert batch == 1
            return mx.zeros((batch, 1, 2), dtype=dtype)

    class Velocity:
        guidance_embed = distilled

        def __call__(self, x, t, cond, guidance=None):
            assert t.shape == (x.shape[0],)
            if distilled:
                assert guidance is not None and cond.shape == (1, 1, 2)
                return mx.ones_like(x) * (guidance[:, None, None] + 3.5)
            assert guidance is None and cond.shape == (2, 1, 2)
            return mx.ones_like(x) * cond[:, :, :1]

    class Vae:
        latent_shape = (1, 2)
        scale_factor = 2

        def decode(self, state):
            np.testing.assert_allclose(np.asarray(state), (stochastic + expected_velocity) / 2, atol=2e-6)
            return state

        def query_grid(self, context, *, bounds, octree_resolution, num_chunks):
            assert not distilled
            return self.field(context, bounds, octree_resolution, num_chunks)

        def query_grid_octree(self, context, *, bounds, octree_resolution, num_chunks, mc_level):
            assert distilled and mc_level == 0
            return self.field(context, bounds, octree_resolution, num_chunks)

        def field(self, context, bounds, octree_resolution, num_chunks):
            assert context.shape == latent_shape
            assert (bounds, octree_resolution, num_chunks) == (2, 2, 3)
            events.append("octree" if distilled else "dense")
            values = np.broadcast_to(np.array([-1, 0, 1])[:, None, None], (3, 3, 3))
            return values, np.array([-2, -2, -2]), np.array([2, 2, 2]), (2, 2, 2)

    cfg = {"scheduler": {"target": "ConsistencyScheduler", "params": {"pcm_timesteps": 4}}} if distilled else {}
    model = pipeline.Hunyuan3DShapePipeline(Dino(), Velocity(), Vae(), cfg)
    mesh = model.generate(
        str(image), num_inference_steps=3, guidance_scale=scale, octree_resolution=2,
        num_chunks=3, box_v=2, seed=seed, border_ratio=0, octree_decode=distilled,
        compile_dit=distilled, verbose=distilled,
    )
    assert events == ["encode", "octree" if distilled else "dense"]
    np.testing.assert_allclose(mesh.bounds, [[0, -2, -2], [0, 2, 2]])
    assert mesh.area == pytest.approx(16)
    assert ("guidance-embed" in capsys.readouterr().out) == distilled


@pytest.mark.parametrize("explicit", [False, True])
def test_shape_command_passes_cli_options_and_writes_output_licence(tmp_path, monkeypatch, capsys, explicit):
    image = tmp_path / "front.png"
    image.write_bytes(b"fixture")
    local = tmp_path / "shape"
    local.mkdir()
    output = tmp_path / "shape.glb"
    recorded = []

    def load(path, **kwargs):
        assert path == str(local)
        assert kwargs == {"dtype": mx.float16 if explicit else mx.float32,
                          "quantize": 8 if explicit else None, "verbose": True}
        recorded.append("load")
        return "encoder", "velocity", "vae", {}

    def export(path):
        Path(path).write_bytes(b"fixture mesh")
        recorded.append("export")

    def generate(self, path, **options):
        assert self.dino == "encoder" and self.dit == "velocity" and self.vae == "vae"
        assert path == str(image)
        assert options == {
            "num_inference_steps": 4 if explicit else 30, "guidance_scale": 3 if explicit else 5,
            "octree_resolution": 8 if explicit else 256, "seed": 2 if explicit else 0,
            "octree_decode": explicit, "compile_dit": explicit,
        }
        recorded.append("generate")
        return SimpleNamespace(export=export)

    monkeypatch.setattr(convert, "load_models", load)
    monkeypatch.setattr(pipeline.Hunyuan3DShapePipeline, "generate", generate)
    args = ["shape", str(image), "--weights", str(local), "--out", str(output)]
    if explicit:
        args += ["--steps", "4", "--guidance", "3", "--octree", "8", "--seed", "2",
                 "--octree-decode", "--compile-dit", "--quantize", "8", "--dtype", "float16"]
    monkeypatch.setattr(sys, "argv", args)
    repo = str(Path(pipeline.__file__).resolve().parents[3])
    path = [item for item in sys.path if item != repo]
    monkeypatch.setattr(sys, "path", path if explicit else [repo, *path])
    pipeline.main()
    assert recorded == ["load", "generate", "export"]
    sidecar = output.with_suffix(".provenance.json")
    assert str(sidecar) in capsys.readouterr().out
    data = json.loads(sidecar.read_text())
    assert data["backend"] == "hunyuan-mlx-shape"
    assert data["input"]["path"] == str(image)
    assert data["model"]["parameters"]["weights"] == str(local)
    assert len(data["components"]) == 1
    assert data["components"][0]["weights"] == str(local)



def test_shape_script_help_describes_optional_model_and_mesh_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["shape", "--help"])
    monkeypatch.delitem(sys.modules, pipeline.__name__)
    with pytest.raises(SystemExit) as status:
        runpy.run_module(pipeline.__name__, run_name="__main__")
    assert status.value.code == 0
    out = capsys.readouterr().out
    assert "--weights" in out and "--octree-decode" in out and "--compile-dit" in out
