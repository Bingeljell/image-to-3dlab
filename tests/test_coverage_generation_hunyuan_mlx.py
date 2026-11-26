"""Exercise both local Apple Hunyuan shape-to-paint adapters."""

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import trimesh
from PIL import Image

from scripts import hunyuan_mlx_generate as dgrauet
from scripts import hunyuan_mlx_xiong_generate as xiong


@pytest.mark.parametrize("module", [dgrauet, xiong], ids=["dgrauet", "xiong"])
@pytest.mark.parametrize("mode", ["success", "paint-failed", "paint-missing", "weights-missing"])
def test_hunyuan_mlx_passes_shape_to_paint_and_records_run(
    module, mode, tmp_path, monkeypatch
):
    image = tmp_path / "picture.png"
    Image.new("RGBA", (8, 8)).save(image)
    output = tmp_path / "delivery/model.glb"
    paint_root = tmp_path / "paint"
    monkeypatch.setattr(module, "PAINT_ROOT", paint_root)
    monkeypatch.setattr(module, "PAINT_PYTHON", paint_root / "python")
    monkeypatch.setattr(module, "PAINT_SCRIPT", paint_root / "run.py")
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")

    model = next(iter(xiong.SHAPE_MODELS))
    if module is xiong:
        weights = tmp_path / "xiong"
        monkeypatch.setitem(xiong.SHAPE_MODELS, model, weights)
    else:
        weights = hub / "models--dgrauet--hunyuan3d-2.1-mlx/snapshots/test"
    if mode != "weights-missing":
        weights.mkdir(parents=True)
        (weights / "config.json").write_text("{}")

    shape = trimesh.creation.box()
    events = []

    class Pipeline:
        @classmethod
        def from_pretrained(cls, local, **kwargs):
            assert Path(local) == weights
            if module is xiong:
                assert kwargs == {"dtype": "f16", "quantize": None}
            else:
                assert not kwargs
            events.append("load")
            return cls()

        def __call__(self, path, **kwargs):
            assert path == str(image)
            assert kwargs["octree_resolution"] == 384
            assert kwargs["seed"] == 7
            assert kwargs["num_inference_steps"] == (30 if module is xiong else 50)
            if module is xiong:
                assert kwargs["octree_decode"] is False and kwargs["compile_dit"] is False
            events.append("shape")
            return shape

        generate = __call__

    mm = ModuleType("mlx")
    mx = ModuleType("mlx.core")
    mx.float16 = "f16"
    mx.clear_cache = lambda: events.append("clear cache")
    mm.core = mx
    xiong_pipeline = ModuleType("hy3dmlx.pipeline")
    xiong_pipeline.Hunyuan3DShapePipeline = Pipeline
    dgrauet_pipeline = ModuleType("hy3dshape.pipeline_mlx")
    dgrauet_pipeline.ShapePipeline = Pipeline
    fs = ModuleType("fast_simplification")

    def simplify(vertices, faces, *, target_count):
        assert vertices is shape.vertices and faces is shape.faces and target_count == 8
        events.append("simplify")
        return vertices, faces[:target_count]

    fs.simplify = simplify
    for dependency in (mm, mx, xiong_pipeline, dgrauet_pipeline, fs):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)

    paint_dirs = []

    def run(argv, *, cwd, env, check):
        assert argv == [str(module.PAINT_PYTHON), "-u", str(module.PAINT_SCRIPT)]
        assert cwd == str(paint_root) and not check
        assert env["PAINT_IMG"] == str(image)
        assert (env["PAINT_SEED"], env["PAINT_RES"], env["PAINT_STEPS"], env["PAINT_TEX"]) == (
            "9", "256", "10", "512"
        )
        obj = trimesh.load(env["PAINT_MESH"], force="mesh", process=False)
        expected_faces = 8 if mode == "success" else 12
        assert len(obj.faces) == expected_faces
        dest = Path(env["PAINT_OUT"])
        assert dest.is_relative_to(paint_root / "outputs")
        assert dest.is_dir()
        paint_dirs.append(dest)
        events.append("paint")
        if mode == "paint-failed":
            return subprocess.CompletedProcess(argv, 3)
        if mode != "paint-missing":
            shape.export(dest / "textured_mesh_pbr.glb")
            (dest / "textured_mesh_pbr.provenance.json").write_text(
                json.dumps({"output": {"path": "temporary"}})
            )
            Image.new("RGB", (8, 8)).save(dest / "pbr_albedo_texture.png")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(module.subprocess, "run", run)
    argv = [
        module.__file__, str(image), str(output), "--seed", "7", "--octree-resolution", "384",
        "--decimation-target", str(8 if mode == "success" else 16),
        "--paint-seed", "9", "--paint-res", "256", "--paint-steps", "10", "--paint-tex", "512",
    ]
    if module is xiong:
        argv += ["--model", model, "--quantize", "0", "--no-octree-decode", "--no-compile-dit"]
    monkeypatch.setattr(sys, "argv", argv)

    if mode == "weights-missing":
        with pytest.raises(SystemExit, match="not downloaded|needs its weights"):
            module.main()
        assert not events and not output.exists()
        return
    if mode != "success":
        error = "exited with code 3" if mode == "paint-failed" else "did not produce"
        with pytest.raises(RuntimeError, match=error):
            module.main()
        assert not output.exists()
        assert not output.with_suffix(".provenance.json").exists()
        return

    assert module.main() is None
    assert events == ["load", "shape", "simplify", *(["clear cache"] if module is xiong else []), "paint"]
    assert output.is_file()
    assert output.with_name("model_pbr_albedo_texture.png").is_file()
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["faces"] == 8 and report["parameters"]["remeshed"] is True
    record = json.loads(output.with_suffix(".provenance.json").read_text())
    assert record["output"]["path"] == str(output)
    assert record["output"]["classification"] == "territory-restricted"
    assert (paint_dirs[0] / "textured_mesh_pbr.provenance.json").is_file() is (module is xiong)
