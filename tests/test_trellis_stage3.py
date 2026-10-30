"""Pure contract tests for the Stage-3-only TRELLIS runner."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trellis_stage3.py"
SPEC = importlib.util.spec_from_file_location("trellis_stage3", SCRIPT)
stage3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage3)


def _opaque_rgba_png(path: Path, size: int = 4) -> None:
    from PIL import Image

    Image.new("RGBA", (size, size), (128, 64, 32, 255)).save(path)


def _write_latents(path: Path) -> None:
    torch.save(
        {
            "shape_slat_feats": torch.zeros((7, 32)),
            "coords": torch.zeros((7, 4), dtype=torch.int32),
            "res": 1024,
            "pipeline_type": "1024_cascade",
            "images": ["source.png"],
        },
        path,
    )


class _FakeSparseTensor:
    def __init__(self, feats, coords):
        self.feats = feats
        self.coords = coords


class _FakeSlat:
    feats = torch.zeros((1, 1))
    coords = torch.zeros((1, 4), dtype=torch.int32)


class _FakeVoxel:
    feats = torch.zeros((1, 6))
    coords = torch.tensor([[0, 0, 0, 0]])


class _FakePipeline:
    recorded: dict = {}

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.recorded = {"model_id": model_id, "kwargs": dict(kwargs)}
        return cls()

    def to(self, device):
        return self

    def preprocess_image(self, image):
        return image.convert("RGB") if image.mode == "RGBA" else image

    def get_cond(self, images, resolution):
        return "cond"

    @property
    def models(self):
        return {"tex_slat_flow_model_1024": "tex_model"}

    def sample_tex_slat(self, cond, model, shape_slat, params):
        return _FakeSlat()

    def decode_shape_slat(self, slat, res):
        return ([], "subs")

    def decode_tex_slat(self, slat, subs):
        return [_FakeVoxel()]


class _FakeMatte:
    def __init__(self, is_matted: bool = False):
        self.is_matted_return = is_matted
        self.cut_calls: list = []

    def is_matted(self, image):
        return self.is_matted_return

    def cut_out(self, image, session=None):
        from PIL import Image

        self.cut_calls.append(image)
        return (Image.new("RGBA", image.size, (128, 64, 32, 255)), "u2net")


def _install_fake_trellis(monkeypatch):
    fake_trellis2 = types.ModuleType("trellis2")
    fake_trellis2.modules = types.ModuleType("trellis2.modules")
    fake_trellis2.modules.sparse = types.ModuleType("trellis2.modules.sparse")
    fake_trellis2.modules.sparse.SparseTensor = _FakeSparseTensor
    fake_trellis2.pipelines = types.ModuleType("trellis2.pipelines")
    fake_trellis2.pipelines.trellis2_image_to_3d = types.ModuleType(
        "trellis2.pipelines.trellis2_image_to_3d"
    )
    fake_trellis2.pipelines.trellis2_image_to_3d.Trellis2ImageTo3DPipeline = _FakePipeline

    monkeypatch.setitem(sys.modules, "trellis2", fake_trellis2)
    monkeypatch.setitem(sys.modules, "trellis2.modules", fake_trellis2.modules)
    monkeypatch.setitem(sys.modules, "trellis2.modules.sparse", fake_trellis2.modules.sparse)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", fake_trellis2.pipelines)
    monkeypatch.setitem(
        sys.modules,
        "trellis2.pipelines.trellis2_image_to_3d",
        fake_trellis2.pipelines.trellis2_image_to_3d,
    )


def _run_main(monkeypatch, tmp_path: Path, *, preprocessed: bool = False, transparent: bool = False):
    _FakePipeline.recorded = {}
    _install_fake_trellis(monkeypatch)
    fake_matte = _FakeMatte(is_matted=transparent)
    monkeypatch.setattr(stage3, "matte", fake_matte, raising=False)

    latent_path = tmp_path / "latents.pt"
    _write_latents(latent_path)
    geom_path = tmp_path / "decode.pt"
    geom_path.write_bytes(b"dummy")
    image_path = tmp_path / "source.png"
    _opaque_rgba_png(image_path)
    output_path = tmp_path / "material.pt"
    vendor_root = tmp_path / "vendor"
    vendor_root.mkdir()
    pipeline = vendor_root / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text("        new_pipeline.rembg_model = None\n")
    snap = tmp_path / "trellis-snapshot"
    snap.mkdir()
    (snap / "pipeline.json").write_text("{}\n")
    from image_to_3dlab import weight_gate
    monkeypatch.setattr(weight_gate, "snapshot_dir", lambda *a, **k: snap)

    argv = [
        "trellis_stage3.py",
        str(latent_path),
        str(output_path),
        "--image",
        str(image_path),
        "--texture-seed",
        "123",
        "--geometry-decode",
        str(geom_path),
        "--sample-only",
        "--vendor-root",
        str(vendor_root),
    ]
    if preprocessed:
        argv.append("--preprocessed-image")
    monkeypatch.setattr(sys, "argv", argv)

    stage3.main()
    return _FakePipeline.recorded, fake_matte


def _bundle():
    return {
        "shape_slat_feats": torch.zeros((7, 32)),
        "coords": torch.zeros((7, 4), dtype=torch.int32),
        "res": 1024,
        "pipeline_type": "1024_cascade",
        "images": ["source.png"],
    }


def test_stage3_refuses_an_unpatched_tree_before_the_pipeline(tmp_path, monkeypatch):
    latents = tmp_path / "latents.pt"
    latents.write_bytes(b"not read")
    image = tmp_path / "source.png"
    image.write_bytes(b"not used")
    vendor = tmp_path / "vendor"
    pipeline = vendor / "TRELLIS.2" / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_text(
        "        new_pipeline.rembg_model = getattr(rembg, "
        "args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
    )
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "pipeline.json").write_text("{}\n")
    from image_to_3dlab import weight_gate

    monkeypatch.setattr(weight_gate, "snapshot_dir", lambda *a, **k: snap)
    monkeypatch.setitem(sys.modules, "torch", None)
    argv = [
        "trellis_stage3.py",
        str(latents),
        str(tmp_path / "out.pt"),
        "--image",
        str(image),
        "--texture-seed",
        "1",
        "--vendor-root",
        str(vendor),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="BRIA RMBG-2.0"):
        stage3.main()


def test_stage3_refuses_when_the_snapshot_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    latents = tmp_path / "latents.pt"
    latents.write_bytes(b"not read")
    image = tmp_path / "source.png"
    image.write_bytes(b"not used")
    argv = ["trellis_stage3.py", str(latents), str(tmp_path / "out.pt"),
            "--image", str(image), "--texture-seed", "1", "--vendor-root", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="about 16.5 GB"):
        stage3.main()


def test_valid_bundle_is_accepted():
    stage3.validate_bundle(_bundle())


@pytest.mark.parametrize(
    "resolution,expected",
    [
        (512, (512, "tex_slat_flow_model_512")),
        (1024, (1024, "tex_slat_flow_model_1024")),
        (1536, (1024, "tex_slat_flow_model_1024")),
    ],
)
def test_material_stage_contract_selects_matching_model(resolution, expected):
    payload = _bundle()
    payload["res"] = resolution
    assert stage3.material_stage_contract(payload) == expected


class _Voxel:
    feats = torch.ones((3, 6))
    coords = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]])


def test_material_payload_supports_clean_space_decode_schema():
    geometry = {
        "vertices": "V",
        "faces": "F",
        "attrs": "old attrs",
        "coords": "old coords",
        "attr_layout": {"base_color": slice(0, 3)},
        "res": 512,
    }
    bundle = _bundle()
    bundle["res"] = 512
    payload = stage3.build_material_payload(geometry, _Voxel(), bundle, 42, {"steps": 12})
    assert payload["vertices"] == "V"
    assert payload["faces"] == "F"
    assert payload["attrs"].shape == (3, 6)
    assert payload["coords"].shape == (3, 3)
    assert payload["texture_seed"] == 42
    assert payload["res"] == 512


def test_material_payload_preserves_legacy_split_cache_schema():
    geometry = {
        "geometry_ref": "geometry.pt",
        "layout": {"base_color": slice(0, 3)},
        "voxel_size": 0.5,
    }
    payload = stage3.build_material_payload(geometry, _Voxel(), _bundle(), 7, {"steps": 12})
    assert payload["geometry_ref"] == "geometry.pt"
    assert payload["voxel_size"] == 0.5
    assert payload["texture_seed"] == 7


def test_missing_shape_latent_is_rejected():
    payload = _bundle()
    del payload["shape_slat_feats"]
    with pytest.raises(ValueError, match="shape_slat_feats"):
        stage3.validate_bundle(payload)


def test_row_mismatch_is_rejected():
    payload = _bundle()
    payload["coords"] = torch.zeros((6, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="row mismatch"):
        stage3.validate_bundle(payload)


def test_default_geometry_path_matches_generate_outputs(tmp_path):
    path = tmp_path / "snag_latents.pt"
    assert stage3.default_geometry_decode(path) == tmp_path / "snag_decode.pt"


def test_official_texture_defaults_are_explicit():
    args = argparse.Namespace(
        steps=12,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        rescale_t=3.0,
    )
    assert stage3.sampler_params(args) == {
        "steps": 12,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "guidance_interval": (0.6, 0.9),
        "rescale_t": 3.0,
    }


def test_runner_can_target_the_clean_space_port_and_stop_after_sampling():
    source = SCRIPT.read_text()
    assert '"--vendor-root"' in source
    assert 'default="sdpa"' in source
    assert '"metal_flash"' in source
    assert '"--sample-only"' in source
    assert '"--preprocessed-image"' in source
    assert "material decode intentionally skipped" in source
    assert "load_rembg=False" in source


def test_opaque_image_loads_pipeline_without_rembg_and_cuts_out(tmp_path, monkeypatch):
    recorded, fake_matte = _run_main(monkeypatch, tmp_path)
    assert recorded["kwargs"]["load_rembg"] is False
    assert len(fake_matte.cut_calls) == 1
    assert fake_matte.cut_calls[0].mode == "RGB"
    assert fake_matte.cut_calls[0].size == (4, 4)
    metadata = json.loads((tmp_path / "material.json").read_text())
    assert metadata["experiment"]["matted_with"] == "u2net"


def test_preprocessed_image_skips_matting_and_loads_pipeline_without_rembg(tmp_path, monkeypatch):
    recorded, fake_matte = _run_main(monkeypatch, tmp_path, preprocessed=True)
    assert recorded["kwargs"]["load_rembg"] is False
    assert len(fake_matte.cut_calls) == 0
    metadata = json.loads((tmp_path / "material.json").read_text())
    assert metadata["experiment"]["matted_with"] is None


def test_transparent_image_skips_matting_and_loads_pipeline_without_rembg(tmp_path, monkeypatch):
    recorded, fake_matte = _run_main(monkeypatch, tmp_path, transparent=True)
    assert recorded["kwargs"]["load_rembg"] is False
    assert len(fake_matte.cut_calls) == 0
    metadata = json.loads((tmp_path / "material.json").read_text())
    assert metadata["experiment"]["matted_with"] is None
