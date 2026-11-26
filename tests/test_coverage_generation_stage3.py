"""Exercise Stage 3's cache and metadata wiring against small local bundles."""

import json
import os
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import trellis_stage3 as stage


@pytest.mark.parametrize("mode", ["missing-flex"])
def test_stage3_reports_progress_and_writes_compatible_cache(mode, tmp_path, monkeypatch, capsys):
    vendor = tmp_path / "vendor"
    loader = stage.space_patch.pipeline_path(vendor / "TRELLIS.2")
    loader.parent.mkdir(parents=True)
    loader.write_text(
        "class Fixture:\n    def setup(self, new_pipeline):\n        new_pipeline.rembg_model = None\n"
    )
    (vendor / "stubs").mkdir()

    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    for name in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND",
                 "FLEX_GEMM_AUTOTUNE_CACHE_PATH", "PYTORCH_ENABLE_MPS_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    weights = hub / "models--microsoft--TRELLIS.2-4B/snapshots/local"
    weights.mkdir(parents=True)
    (weights / "pipeline.json").write_text("{}")

    latents = tmp_path / "first_latents.pt"
    latents.write_bytes(b"tiny latents")
    geometry = tmp_path / "first_decode.pt"
    if mode != "missing-geometry":
        geometry.touch()
    source = tmp_path / "source.png"
    if mode != "missing-image":
        image = Image.new("RGBA", (8, 8))
        image.paste((10, 20, 30, 255), (2, 2, 6, 6))
        image.save(source)
    destination = tmp_path / "delivery/candidate_material.pt"
    tex = destination.with_name("candidate_tex_latent.pt")
    phases = []
    saved = {}

    class Tensor(np.ndarray):
        def to(self, device):
            assert device == "mps"
            return self

        def cpu(self):
            return self

        def detach(self):
            return self

        def median(self, axis=None):
            value = np.median(np.asarray(self), axis=axis)
            return SimpleNamespace(values=value) if axis is not None else value

    attrs = np.asarray([[0.125, 0.25, 0.5, 0.25, 0.5, 1.0]]).view(Tensor)
    coords = np.asarray([[0, 2, 3, 4]]).view(Tensor)
    layout = {"base_color": [0, 1, 2], "metallic": 3, "roughness": 4, "alpha": 5}

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and not weights_only
        if path == latents:
            return {
                "shape_slat_feats": attrs, "coords": coords, "res": 512,
                "images": [str(source)], "pipeline_type": "512", "seed": 4,
            }
        assert path == geometry
        return {"attr_layout": layout, "res": 512, "original_image": str(source)}

    def save(value, path):
        saved[path] = value
        path.write_bytes(b"synthetic data")

    torch = ModuleType("torch")
    torch.device = lambda device: device
    torch.manual_seed = lambda seed: phases.append(("torch seed", seed))
    torch.mps = SimpleNamespace(manual_seed=lambda seed: phases.append(("mps seed", seed)))
    torch.load = load
    torch.save = save

    class Pipeline:
        models = {"tex_slat_flow_model_512": "material-flow-512"}

        @staticmethod
        def from_pretrained(path, *, load_rembg):
            assert Path(path) == weights and not load_rembg
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"
            phases.append("load")
            return Pipeline()

        def to(self, device):
            assert device == "mps"

        def preprocess_image(self, image):
            assert image.mode == "RGBA"
            phases.append("preprocess")
            return image.convert("RGB")

        def get_cond(self, images, resolution):
            assert len(images) == 1 and images[0].mode == "RGB"
            assert resolution == 512
            phases.append("condition")
            return "conditioning"

        def sample_tex_slat(self, conditioning, model, shape, params):
            assert conditioning == "conditioning" and model == "material-flow-512"
            assert shape.feats is attrs and shape.coords is coords
            assert params["steps"] == 9 and params["guidance_interval"] == (0.25, 0.5)
            phases.append("sample")
            return SimpleNamespace(feats=attrs, coords=coords)

        def decode_shape_slat(self, shape, resolution):
            assert resolution == 512 and shape.feats is attrs
            phases.append("decode-shape")
            return [], "subs"

        def decode_tex_slat(self, texture, subs):
            assert texture.feats is attrs and subs == "subs"
            phases.append("decode-material")
            return [texture]

    pipeline = ModuleType("trellis2.pipelines.trellis2_image_to_3d")
    pipeline.Trellis2ImageTo3DPipeline = Pipeline
    sparse = ModuleType("trellis2.modules.sparse")
    sparse.SparseTensor = SimpleNamespace
    for module in (torch, sparse, pipeline):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "flex_gemm", None if mode == "missing-flex" else ModuleType("flex_gemm"))

    argv = [
        stage.__file__, str(latents), str(destination), "--vendor-root", str(vendor),
        "--texture-seed", "7", "--steps", "9", "--guidance-interval", "0.25", "0.5",
    ]
    if mode == "dry":
        argv += ["--dry-run"]
    if mode == "sample":
        argv += ["--sample-only", "--preprocessed-image"]
    monkeypatch.setattr(sys, "argv", argv)

    if mode.startswith("missing-"):
        error = {"missing-flex": "flex_gemm", "missing-image": "conditioning image",
                 "missing-geometry": "geometry decode"}[mode]
        expected = RuntimeError if mode == "missing-flex" else SystemExit
        with pytest.raises(expected, match=error):
            stage.main()
        assert not phases and not saved
        return

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(stage.__file__, run_name="__main__")
    assert exc.value.code == 0
    if mode == "dry":
        assert not phases and not saved
        assert json.loads(capsys.readouterr().out)["shape_seed"] == 4
        return

    assert saved[tex]["texture_seed"] == 7
    assert saved[tex]["shape_latents_sha256"] == stage.sha256(latents)
    metadata = json.loads(destination.with_suffix(".json").read_text())
    assert metadata["experiment"]["texture_seed"] == 7
    assert metadata["artifacts"]["texture_latent"]["sha256"] == stage.sha256(tex)
    expected_phases = ["load", *(["preprocess"] if mode == "full" else []), "condition",
                       ("torch seed", 7), ("mps seed", 7), "sample"]
    if mode == "sample":
        assert phases == expected_phases
        assert "voxel_attributes" not in metadata and destination not in saved
    else:
        assert phases == [*expected_phases, "decode-shape", "decode-material"]
        assert "material_cache" in metadata["artifacts"]
        assert metadata["voxel_attributes"]["metallic_median"] == 0.25
        assert metadata["voxel_attributes"]["base_color_mean"] == [0.125, 0.25, 0.5]
        np.testing.assert_array_equal(saved[destination]["coords"], [[2, 3, 4]])
        assert saved[destination]["res"] == 512
        assert saved[destination]["original_image"] == str(source)
