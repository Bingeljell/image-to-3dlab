"""Exercise the SF3D adapter against local files and a vendor-runtime substitute."""

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import trimesh
from PIL import Image

from image_to_3dlab import host
from image_to_3dlab import sf3d_backend as s
from image_to_3dlab.glb_turn import glb_faces_front


def test_sf3d_reports_missing_checkout_then_missing_extension(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor"
    monkeypatch.setattr(host, "host_platform", lambda: host.AMD)
    with pytest.raises(RuntimeError, match="checkout not found.*bootstrap_sf3d_rocm"):
        s._load_sf3d(vendor)
    package = vendor / "sf3d"
    package.mkdir(parents=True)
    (package / "system.py").touch()
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "sf3d", None)
    with pytest.raises(RuntimeError, match="compiled extension could not be imported"):
        s._load_sf3d(vendor)


@pytest.mark.parametrize("platform", [host.AMD, "macos-arm64"])
def test_sf3d_refuses_missing_remote_weights(monkeypatch, tmp_path, platform):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(host, "host_platform", lambda: platform)
    route = "AMD" if platform == host.AMD else "Mac"
    with pytest.raises(SystemExit, match=rf"Stable Fast 3D \({route}\).*5\.5 GB"):
        s.local_model("vendor/missing")


@pytest.mark.parametrize(("gpu", "failure", "retry", "expected_devices", "expected_caches"), [
    ("cuda", None, True, ["cuda"], ["cuda"]),
    ("cpu", None, True, ["cpu"], []),
    ("mps", None, True, ["mps"], ["mps"]),
    ("mps", "MPS out of memory", True, ["mps", "cpu"], ["mps", "mps"]),
    ("mps", "MPS out of memory", False, ["mps"], ["mps"]),
    ("mps", "invalid texture", True, ["mps"], ["mps"]),
    ("cuda", "CUDA out of memory", True, ["cuda"], ["cuda"]),
])
def test_sf3d_writes_prepared_input_rotates_output_and_clears_cache(
    gpu, failure, retry, expected_devices, expected_caches, tmp_path, monkeypatch, capsys,
):
    image = tmp_path / "source.png"
    source = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    source.putpixel((0, 0), (0, 0, 0, 0))
    source.save(image)
    devices, caches, resize_ratios = [], [], []
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    repo = tmp_path / "checkout"
    (repo / "sf3d").mkdir(parents=True)
    (repo / "sf3d/system.py").touch()
    cfg = s.SF3DOptions(repo, str(model_dir), texture_resolution=32, foreground_ratio=.75,
                        remesh="triangle", target_vertices=12, cpu_fallback=retry)
    cuda = SimpleNamespace(is_available=lambda: gpu == "cuda", empty_cache=lambda: caches.append("cuda"))
    mps = SimpleNamespace(is_available=lambda: gpu == "mps", empty_cache=lambda: caches.append("mps"))
    monkeypatch.setitem(sys.modules, "torch",
                        SimpleNamespace(cuda=cuda, mps=mps, backends=SimpleNamespace(mps=mps),
                                        inference_mode=contextlib.nullcontext))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("SF3D_USE_CPU", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    class VendorModel:
        @classmethod
        def from_pretrained(cls, identity, **kwargs):
            assert identity == str(model_dir)
            assert kwargs == {"config_name": "config.yaml", "weight_name": "model.safetensors"}
            return cls()

        def to(self, device):
            devices.append(device)
            return self

        def eval(self):
            self.ready = True

        def run_image(self, images, **kwargs):
            assert self.ready
            assert images[0].getpixel((0, 0))[3] == 0
            assert kwargs == {"bake_resolution": 32, "remesh": "triangle", "vertex_count": 12}
            if failure and len(devices) == 1:
                raise RuntimeError(failure)
            mesh = trimesh.creation.box()
            mesh.apply_translation((1, 2, 3))
            return mesh, None

    def resize_foreground(picture, ratio):
        resize_ratios.append(ratio)
        return picture

    monkeypatch.setitem(sys.modules, "sf3d", ModuleType("sf3d"))
    monkeypatch.setitem(sys.modules, "sf3d.system", SimpleNamespace(SF3D=VendorModel))
    monkeypatch.setitem(sys.modules, "sf3d.utils", SimpleNamespace(resize_foreground=resize_foreground))
    monkeypatch.setattr(s.matte, "cut_out", lambda *a, **kw: pytest.fail("input already has a matte"))

    succeeds = failure is None or len(expected_devices) == 2
    if succeeds:
        result = s.generate_sf3d(image, tmp_path, cfg)
        assert result == tmp_path / "source_sf3d.glb"
        assert glb_faces_front(result.read_bytes())
        mesh = trimesh.load(str(result), force="mesh")
        assert mesh.centroid == pytest.approx((-1, 2, -3))
    else:
        with pytest.raises(RuntimeError, match=failure):
            s.generate_sf3d(image, tmp_path, cfg)
        assert not (tmp_path / "source_sf3d.glb").exists()
    assert devices == expected_devices
    assert caches == expected_caches
    assert resize_ratios == [.75] * len(devices)
    assert (tmp_path / "source_sf3d_input.png").is_file()
    assert ("retrying SF3D on CPU" in capsys.readouterr().err) is (len(devices) == 2)
