"""Verify the TRELLIS probe's local inputs, runtime boundary and report."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import runpod_trellis2_cuda_probe as probe


@pytest.fixture
def local_probe(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    snapshot = hub / "models--microsoft--TRELLIS.2-4B" / "snapshots" / "test"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    for name in (
        "HF_HUB_OFFLINE", "OPENCV_IO_ENABLE_OPENEXR", "PYTORCH_CUDA_ALLOC_CONF",
        "ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND",
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    source, latent, out = tmp_path / "input.png", tmp_path / "latent.pt", tmp_path / "report"
    Image.new("RGBA", (2, 2), (255, 0, 0, 128)).save(source)
    monkeypatch.setattr(sys, "argv", ["probe", "--image", str(source), "--latents", str(latent),
                                     "--output-dir", str(out), "--texture-seed", "17"])
    return source, latent, out, snapshot


def test_cuda_probe_refuses_missing_local_weights_before_import(monkeypatch, tmp_path, local_probe):
    _, _, out, _ = local_probe
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", None)
    with pytest.raises(SystemExit, match=r"TRELLIS.2 \(NVIDIA probe\).*16.5 GB"):
        runpy.run_path(probe.__file__, run_name="__main__")
    assert not out.exists()


@pytest.fixture
def fake_vendor(monkeypatch):
    import torch

    operations = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "test device")
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: operations.append(("gpu-seed", seed)))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: operations.append(("cpu-seed", seed)))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)

    class SparseTensor:
        def __init__(self, *, feats, coords):
            self.feats = feats
            self.coords = coords

    class ForbiddenBackgroundRemover:
        def __init__(self):
            pytest.fail("gated background-removal model was constructed")

    rembg = SimpleNamespace(BiRefNet=ForbiddenBackgroundRemover)
    layout = {"base_color": [0, 1, 2], "metallic": 3, "roughness": 4, "alpha": 5}
    atlas = torch.tensor([
        [0.0, 0.25, 0.5, 0.25, 0.5, 1.0],
        [0.25, 0.5, 0.75, 0.5, 0.75, 1.0],
        [0.5, 0.75, 1.0, 0.75, 1.0, 1.0],
    ])

    class Pipeline:
        models = {"tex_slat_flow_model_512": "frozen-flow"}
        pbr_attr_layout = layout

        @classmethod
        def from_pretrained(cls, path):
            operations.append(("pretrained-local", Path(path)))
            assert rembg.BiRefNet() is None
            return cls()

        def cuda(self):
            operations.append(("activate-device",))

        def preprocess_image(self, image):
            assert image.mode == "RGBA"
            return Image.new("RGB", image.size, (127, 0, 0))

        def get_cond(self, images, size):
            assert size == 512 and len(images) == 1
            return "image-condition"

        def sample_tex_slat(self, condition, model, shape, sampler):
            assert condition == "image-condition"
            assert model == "frozen-flow"
            assert shape.feats.tolist() == [[8.0], [9.0], [10.0]]
            assert sampler["guidance_interval"] == (0.6, 0.9)
            operations.append(("sample", sampler["steps"]))
            return SparseTensor(feats=torch.tensor([[2.0], [4.0], [6.0]]), coords=shape.coords)

        def decode_shape_slat(self, shape, size):
            assert size == 512
            return None, shape.coords

        def decode_tex_slat(self, texture, subs):
            assert torch.equal(texture.coords, subs)
            return [SparseTensor(feats=atlas, coords=subs)]

    monkeypatch.setitem(sys.modules, "trellis2.modules.sparse", SimpleNamespace(SparseTensor=SparseTensor))
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", SimpleNamespace(
        Trellis2ImageTo3DPipeline=Pipeline, rembg=rembg,
    ))
    return torch, operations, rembg, ForbiddenBackgroundRemover


def test_cuda_probe_emits_auditable_local_report_without_gated_background_removal(
    local_probe, fake_vendor, capsys,
):
    source, latent, out, snapshot = local_probe
    torch, operations, remover, original = fake_vendor
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]])
    torch.save({
        "res": 512, "coords": coords, "shape_slat_feats": torch.tensor([[8.], [9.], [10.]]),
    }, latent)
    assert probe.main() == 0
    assert operations == [
        ("pretrained-local", snapshot), ("activate-device",), ("cpu-seed", 17), ("gpu-seed", 17),
        ("sample", 12),
    ]
    assert remover.BiRefNet is original
    report = json.loads((out / "cuda-probe.json").read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["device"] == "test device"
    assert report["resolution"] == 512 and report["texture_seed"] == 17
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["latents_sha256"] == hashlib.sha256(latent.read_bytes()).hexdigest()
    assert report["processed_sha256"] == hashlib.sha256((out / "preprocessed-black-matte.png").read_bytes()).hexdigest()
    assert report["voxel_attributes"] == {
        "count": 3, "base_color_mean": [0.25, 0.5, 0.75], "base_color_median": [0.25, 0.5, 0.75],
        "metallic_median": 0.5, "roughness_median": 0.75, "alpha_median": 1.0,
    }
    assert report["texture_latent"]["mean"] == 4.0
    assert report["texture_latent"]["std"] == 2.0
    material = torch.load(out / "cuda-material-field.pt", weights_only=True)
    assert torch.equal(material["coords"], coords[:, 1:])
    assert material["attrs"].shape == (3, 6)


def test_cuda_probe_rejects_unsupported_shape_resolution_before_constructing_model(
    local_probe, fake_vendor,
):
    _, latent, out, _ = local_probe
    torch, operations, _, _ = fake_vendor
    torch.save({"res": 1024}, latent)
    with pytest.raises(ValueError, match="expects a frozen 512 shape, got 1024"):
        probe.main()
    assert operations == []
    assert not list(out.iterdir())


def test_cuda_probe_refuses_cpu_before_loading_latents(local_probe, fake_vendor, monkeypatch):
    _, _, out, _ = local_probe
    torch, operations, _, _ = fake_vendor
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        probe.main()
    assert not operations and not out.exists()
