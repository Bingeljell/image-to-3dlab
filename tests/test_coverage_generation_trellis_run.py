"""Exercise TRELLIS image and cached-decode entry points on local files."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from scripts import trellis_cuda_generate as cuda
from scripts import trellis_space_generate as mac


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.device = "cpu"

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __getitem__(self, index):
        return Tensor(self.values[index])


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in (
        "ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND",
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH", "PYTORCH_ENABLE_MPS_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.mark.parametrize("module", [mac, cuda], ids=["mac", "cuda"])
def test_trellis_cli_checks_paths_before_backend_import(module, offline, tmp_path, capsys):
    vendor = tmp_path / "vendor"
    output = tmp_path / "out.glb"
    source = tmp_path / "picture.png"
    args = ["--vendor-root", str(vendor)]
    assert module.main([*args, "--check"]) == 1
    assert "FAIL:" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="image and output are required"):
        module.main(args)
    with pytest.raises(SystemExit, match="missing input image"):
        module.main([*args, str(source), str(output)])
    with pytest.raises(SystemExit, match="output GLB path"):
        module.main([*args, "--from-latents", str(tmp_path / "missing.pt")])
    expected = "missing latents bundle" if module is mac else "existing --from-latents"
    with pytest.raises(SystemExit, match=expected):
        module.main([*args, "--from-latents", str(tmp_path / "missing.pt"), str(output)])
    assert not output.exists()


@pytest.mark.parametrize("mode", ["rgb", "uncut", "no-cache", "bria"])
def test_mac_trellis_guards_alpha_and_weights_before_import(mode, offline, tmp_path, monkeypatch):
    source = tmp_path / "image.png"
    vendor = tmp_path / "empty-vendor"
    args = [str(source), str(tmp_path / "out.glb"), "--vendor-root", str(vendor)]
    if mode == "rgb":
        Image.new("RGB", (16, 16)).save(source)
        error = "no transparent alpha foreground"
    elif mode == "uncut":
        img = Image.new("RGBA", (16, 16), (10, 10, 10, 255))
        img.putpixel((8, 8), (0, 0, 0, 0))
        img.save(source)
        error = "background"
    else:
        img = Image.new("RGBA", (16, 16), (10, 10, 10, 255))
        img.paste((0, 0, 0, 0), (0, 0, 16, 4))
        img.paste((0, 0, 0, 0), (0, 12, 16, 16))
        img.paste((0, 0, 0, 0), (0, 4, 4, 12))
        img.paste((0, 0, 0, 0), (12, 4, 16, 12))
        img.save(source)
        error = "about 16.5 GB"
        if mode == "bria":
            snapshot = tmp_path / "empty-hub/models--microsoft--TRELLIS.2-4B/snapshots/test"
            snapshot.mkdir(parents=True)
            (snapshot / "pipeline.json").write_text("{}")
            error = "BRIA"
    with pytest.raises(SystemExit, match=error):
        mac.main(args)


@pytest.mark.parametrize("resume", [False, True])
def test_cuda_unpatched_checkout_never_reaches_pipeline_load(resume, offline, tmp_path):
    source = tmp_path / "in.png"
    source.touch()
    vendor = tmp_path / "vendor"
    output = tmp_path / "out.glb"
    args = ["--vendor-root", str(vendor)]
    if resume:
        args.extend(["--from-latents", str(source), str(output)])
    else:
        args.extend([str(source), str(output)])
    with pytest.raises(SystemExit, match="still loads BRIA"):
        cuda.main(args)
    assert not output.exists()


@pytest.mark.parametrize("picture_matches", [False, True])
def test_mac_resume_bakes_real_glb_and_preserves_provenance(
    picture_matches, offline, tmp_path, monkeypatch
):
    vendor = tmp_path / "vendor"
    decoded = tmp_path / "first_decode.pt"
    output = tmp_path / "out/model.glb"
    picture = tmp_path / "source.png"
    Image.new("RGBA", (8, 8)).save(picture)
    decoded.write_bytes(b"local cached mesh")
    mesh = trimesh.creation.box()
    bundle = {
        **mac.bundle_origin(picture, "birefnet-general-lite"),
        "vertices": Tensor(mesh.vertices), "faces": Tensor(mesh.faces),
        "attrs": Tensor([[1, 1, 1]]), "coords": Tensor([[0, 1, 2]]),
        "attr_layout": {"base_color": [0, 1, 2]}, "res": 512,
        "pipeline_type": "512", "seed": 99,
    }
    if not picture_matches:
        Image.new("RGBA", (4, 4)).save(picture)
    seen = []

    def load(path, *, map_location, weights_only):
        assert (path, map_location, weights_only) == (decoded, "cpu", False)
        return bundle

    def bake(**kwargs):
        assert kwargs["vertices"] is bundle["vertices"]
        assert kwargs["faces"] is bundle["faces"]
        assert kwargs["attr_volume"] is bundle["attrs"]
        assert kwargs["coords"] is bundle["coords"]
        assert kwargs["grid_size"] == 512
        assert kwargs["decimation_target"] == 33
        assert kwargs["texture_size"] == 64
        seen.append(kwargs)

        def export(path, *, extension_webp):
            assert path == str(output) and extension_webp
            mesh.export(path)

        return SimpleNamespace(export=export)

    torch = ModuleType("torch")
    torch.load = load
    voxel = ModuleType("o_voxel")
    voxel.postprocess = SimpleNamespace(to_glb=bake)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "o_voxel", voxel)

    with pytest.raises(SystemExit, match="output GLB path"):
        mac.main(["--from-decode", str(decoded)])
    with pytest.raises(SystemExit, match="missing decode bundle"):
        mac.main(["--from-decode", str(tmp_path / "missing.pt"), str(output)])
    assert mac.main([
        "--vendor-root", str(vendor), "--from-decode", str(decoded),
        "--pre-cap", "25", "--decimation-target", "33", "--texture-size", "64", str(output),
    ]) == 0

    assert len(seen) == 1
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["resumed_from_decode"] == str(decoded)
    assert manifest["seed"] == 99
    assert manifest["artifacts"]["glb"]["sha256"] == mac.sha256(output)
    assert manifest["artifacts"]["decode"]["sha256"] == mac.sha256(decoded)
    record = json.loads(output.with_suffix(".provenance.json").read_text())
    assert record["input"]["path"] == str(picture if picture_matches else decoded)
    assert record["input"]["source"] == ("user-provided" if picture_matches else "resumed-bundle")
    assert record["model"]["parameters"]["texture_size"] == 64
    assert any("birefnet-general-lite" in row["component"] for row in record["components"])


@pytest.mark.parametrize("eventual_success", [False, True])
def test_mac_precap_retries_crash_empty_mesh_and_corrupt_output(
    eventual_success, monkeypatch, capsys
):
    vertices = Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = Tensor([[0, 1, 2], [0, 2, 1]])
    files = {}
    scratch = []

    def save(data, path):
        assert data == {"vertices": vertices, "faces": faces}
        path.write_bytes(b"fake serialized tensors")

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and not weights_only
        return files[path]

    def run(argv, *, check, capture_output, text):
        assert argv[:3] == [sys.executable, mac.__file__, "--precap-worker"]
        assert check and capture_output and text
        source, dest = map(Path, argv[3:5])
        assert source.is_file()
        assert argv[5] == "1"
        scratch.append(source.parent)
        if len(scratch) == 1:
            raise subprocess.CalledProcessError(11, argv)
        output_faces = {
            2: np.empty((0, 3), dtype=int),
            3: [[0, 1, 99]],
            4: [[0, 1, 2]],
        }[len(scratch)]
        files[dest] = (vertices, Tensor(output_faces))
        return subprocess.CompletedProcess(argv, 0)

    torch = ModuleType("torch")
    torch.save = save
    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(subprocess, "run", run)
    if eventual_success:
        actual_vertices, actual_faces = mac._precap_subprocess(vertices, faces, pre_cap=1)
        assert actual_vertices is vertices
        np.testing.assert_array_equal(actual_faces.numpy(), [[0, 1, 2]])
        assert len(scratch) == 4
    else:
        with pytest.raises(RuntimeError, match="after 3 attempts.*out-of-range faces"):
            mac._precap_subprocess(vertices, faces, pre_cap=1, max_attempts=3)
        assert len(scratch) == 3
    assert not any(path.exists() for path in scratch)
    out = capsys.readouterr().out
    assert "crash/failure" in out
    assert "no faces" in out
    assert "out-of-range faces" in out
    assert ("succeeded on attempt 4" in out) is eventual_success


def test_mac_precap_worker_simplifies_and_saves_cpu_arrays(tmp_path, monkeypatch):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 1]], dtype=np.int64)
    src, dest = tmp_path / "in.pt", tmp_path / "out.pt"
    src.touch()
    simplified = []

    def load(path, *, map_location, weights_only):
        assert (path, map_location, weights_only) == (src, "cpu", False)
        return {"vertices": Tensor(vertices), "faces": Tensor(faces)}

    def simplify(v, f, ratio):
        assert v is vertices and f is faces and ratio == 0.5
        simplified.append(ratio)
        return v, f[:1]

    class OutputTensor(np.ndarray):
        def float(self):
            return self.astype(np.float32)

    def save(value, path):
        assert path == dest and simplified == [0.5]
        v, f = value
        assert v.dtype == np.float32 and f.dtype == np.int32
        np.testing.assert_array_equal(v, vertices)
        np.testing.assert_array_equal(f, [[0, 1, 2]])
        path.touch()

    torch = ModuleType("torch")
    torch.load = load
    torch.save = save
    torch.from_numpy = lambda data: data.view(OutputTensor)
    native = ModuleType("fast_simplification")
    native.simplify = simplify
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "fast_simplification", native)
    monkeypatch.setattr(sys, "argv", [mac.__file__, "--precap-worker", str(src), str(dest), "1"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(mac.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert dest.is_file()


@pytest.mark.parametrize("mode", ["keep", "discard", "resume", "failed-decode", "refuse-remover", "no-gpu", "no-weights"])
def test_cuda_trellis_writes_checkpoint_before_decode(
    mode, offline, tmp_path, monkeypatch
):
    vendor = tmp_path / "vendor"
    root = cuda.checkout(vendor)
    for loader in cuda.bria_patch.pipeline_files(root):
        loader.parent.mkdir(parents=True, exist_ok=True)
        loader.write_text(cuda.bria_patch.NEEDLE)
    cuda.bria_patch.apply(root)
    (vendor / cuda.BUILT_MARKER_NAME).write_text(json.dumps({"commit": "local-test-revision"}))

    weights = tmp_path / "weights"
    if mode != "no-weights":
        weights.mkdir()
        (weights / "pipeline.json").write_text("{}")
    monkeypatch.setenv("I2L_TRELLIS_MODEL_PATH", str(weights))
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("OPENCV_IO_ENABLE_OPENEXR", raising=False)

    image = tmp_path / "source.png"
    picture = Image.new("RGBA", (16, 16))
    picture.paste((10, 20, 30, 255), (4, 4, 12, 12))
    picture.save(image)
    output = tmp_path / "result/model.glb"
    checkpoint = output.with_name("model_latents.pt")
    mesh = trimesh.creation.box()
    phases = []

    class CudaTensor(Tensor):
        def cuda(self):
            return self

    class Sparse:
        def __init__(self, feats, coords):
            self.feats = feats
            self.coords = coords

        def replace(self, feats):
            return Sparse(feats, self.coords)

    shape = Sparse(CudaTensor([[1.0]]), CudaTensor([[0, 0, 0]]))
    texture = shape.replace(CudaTensor([[2.0]]))
    cached = {
        "shape_slat_feats": shape.feats, "coords": shape.coords,
        "tex_slat_feats": texture.feats, "res": 512,
        "pipeline_type": "512", "seed": 42, "images": [str(image)],
    }

    def save(bundle, path):
        assert path == checkpoint
        assert bundle == cached
        phases.append("save")
        path.write_bytes(b"synthetic checkpoint")

    class Pipeline:
        rembg_model = object() if mode == "refuse-remover" else None
        pbr_attr_layout = {"base_color": [0, 1, 2]}

        @staticmethod
        def from_pretrained(local):
            assert Path(local) == weights
            with pytest.raises(RuntimeError, match="must never be loaded"):
                rembg.BiRefNet()
            phases.append("load")
            return Pipeline()

        def cuda(self):
            phases.append("cuda")

        def preprocess_image(self, image):
            assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0
            return image

        def run(self, image, **kwargs):
            assert kwargs["seed"] == 42
            assert kwargs["return_latent"] is True and kwargs["preprocess_image"] is False
            assert kwargs["pipeline_type"] == "512"
            phases.append("sample")
            return self.decode_latent(shape, texture, 512), (shape, texture, 512)

        def decode_latent(self, shape_arg, texture_arg, resolution):
            assert shape_arg.feats is shape.feats
            assert texture_arg.feats is texture.feats
            assert resolution == 512
            assert checkpoint.read_bytes() == b"synthetic checkpoint"
            phases.append("decode")
            if mode == "failed-decode":
                raise RuntimeError("fixture decode failure")

            def simplify(limit):
                assert limit == cuda.space.NVDIFFRAST_FACE_LIMIT
                phases.append("simplify")

            return [SimpleNamespace(
                vertices=mesh.vertices, faces=mesh.faces, attrs=texture.feats,
                coords=shape.coords, simplify=simplify,
            )]

    def bake(**kwargs):
        assert kwargs["vertices"] is mesh.vertices
        assert kwargs["faces"] is mesh.faces
        assert kwargs["attr_volume"] is texture.feats
        assert kwargs["decimation_target"] == 17
        assert kwargs["texture_size"] == 64
        phases.append("bake")

        def export(path, *, extension_webp):
            assert path == str(output) and extension_webp
            mesh.export(path)

        return SimpleNamespace(export=export)

    def load(path, *, map_location, weights_only):
        assert map_location == "cpu" and not weights_only
        assert path == checkpoint
        assert path.read_bytes() == b"synthetic checkpoint"
        phases.append("resume")
        return cached

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: mode != "no-gpu", empty_cache=lambda: phases.append("free"))
    torch.save = save
    torch.load = load
    voxel = ModuleType("o_voxel")
    voxel.postprocess = SimpleNamespace(to_glb=bake)
    rembg = ModuleType("trellis2.pipelines.rembg")
    rembg.BiRefNet = type("BiRefNet", (), {"__module__": rembg.__name__})
    pipeline_module = ModuleType("trellis2.pipelines.trellis2_image_to_3d")
    pipeline_module.Trellis2ImageTo3DPipeline = Pipeline
    pipelines = ModuleType("trellis2.pipelines")
    pipelines.rembg = rembg
    sparse = ModuleType("trellis2.modules.sparse")
    sparse.SparseTensor = Sparse
    for dependency in (torch, voxel, pipelines, rembg, pipeline_module, sparse):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)

    args = [
        "--vendor-root", str(vendor), "--decimation-target", "17", "--texture-size", "64",
        "--resolution", "512", "--seed", "42",
    ]
    if mode == "resume":
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint")
        args.extend(["--from-latents", str(checkpoint), str(output)])
    else:
        args.extend([str(image), str(output)])
    if mode == "discard":
        args.append("--no-save-latents")

    if mode == "failed-decode":
        with pytest.raises(RuntimeError, match="fixture decode failure"):
            cuda.main(args)
        assert checkpoint.is_file()
        assert phases == ["load", "cuda", "sample", "save", "decode"]
        assert not output.exists()
        return

    if mode in {"refuse-remover", "no-gpu", "no-weights"}:
        error = {
            "refuse-remover": "background remover despite the patch",
            "no-gpu": "CUDA device",
            "no-weights": "need.*16.5 GB",
        }[mode]
        with pytest.raises(SystemExit, match=error):
            cuda.main(args)
        assert phases == (["load"] if mode == "refuse-remover" else [])
        assert not output.exists()
        return

    assert cuda.main(args) == 0
    expected = ["resume"] if mode == "resume" else ["sample", "save"]
    assert phases == ["load", "cuda", *expected, "decode", "simplify", "bake", "free"]
    assert checkpoint.exists() is (mode != "discard")
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert ("latents" in manifest["artifacts"]) is (mode == "keep")
    assert manifest["seed"] == 42
    assert manifest["artifacts"]["glb"]["sha256"] == mac.sha256(output)
    record = json.loads(output.with_suffix(".provenance.json").read_text())
    assert record["software"]["backend_revision"] == "local-test-revision"
    assert record["input"]["path"] == str(image)


@pytest.mark.parametrize("mode", ["keep", "discard", "resume", "bake-error"])
def test_mac_trellis_filters_decode_and_keeps_checkpoints_until_export(
    mode, offline, tmp_path, monkeypatch
):
    import os

    vendor = tmp_path / "vendor"
    pipeline_file = mac.space_patch.pipeline_path(vendor / "TRELLIS.2")
    pipeline_file.parent.mkdir(parents=True)
    pipeline_file.write_text(
        "class Fixture:\n    def setup(self, new_pipeline):\n        new_pipeline.rembg_model = None\n"
    )
    (vendor / "stubs").mkdir()
    snapshot = tmp_path / "empty-hub/models--microsoft--TRELLIS.2-4B/snapshots/test"
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.json").write_text("{}")

    source = tmp_path / "source.png"
    picture = Image.new("RGBA", (16, 16))
    picture.paste((10, 20, 30, 255), (4, 4, 12, 12))
    picture.save(source)
    output = tmp_path / "result/model.glb"
    checkpoint = output.with_name("model_latents.pt")
    decode_cache = output.with_name("model_decode.pt")
    mesh = trimesh.creation.box()
    events = []
    staged = {}

    class DeviceTensor(Tensor):
        def __init__(self, data, device="mps"):
            super().__init__(data)
            self.device = device

        def cpu(self):
            return self.to("cpu")

        def __getitem__(self, index):
            return DeviceTensor(self.values[index], self.device)

        def to(self, device):
            return DeviceTensor(self.values, device)

    class Sparse:
        def __init__(self, *, feats, coords):
            self.feats = feats
            self.coords = coords

        def replace(self, feats):
            return Sparse(feats=feats, coords=self.coords)

    shape = Sparse(feats=DeviceTensor([[1.0]]), coords=DeviceTensor([[0, 0, 0]]))
    texture = shape.replace(DeviceTensor([[2.0]]))
    origin = mac.bundle_origin(source, "birefnet-general-lite")

    def save(bundle, path):
        assert origin.items() <= bundle.items()
        assert bundle["pipeline_type"] == "512"
        assert bundle["seed"] == 42
        if path == checkpoint:
            events.append("save-latents")
            assert bundle["shape_slat_feats"].device == "cpu"
            assert bundle["tex_slat_feats"].device == "cpu"
        else:
            assert path == decode_cache
            events.append("save-decode")
            np.testing.assert_array_equal(bundle["faces"].numpy(), mesh.faces)
        staged[path] = bundle
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic checkpoint")

    def load(path, *, map_location, weights_only):
        assert (path, map_location, weights_only) == (checkpoint, "cpu", False)
        events.append("load-latents")
        return staged[path]

    class Pipeline:
        pbr_attr_layout = {"base_color": [0, 1, 2]}

        @staticmethod
        def from_pretrained(local, *, load_rembg):
            assert Path(local) == snapshot and not load_rembg
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"
            events.append("load")
            return Pipeline()

        def to(self, device):
            assert device == "mps"
            events.append("mps")

        def preprocess_image(self, image):
            assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0
            return image

        def run(self, image, **kwargs):
            assert kwargs["seed"] == 42
            assert kwargs["return_latent"] is True and kwargs["preprocess_image"] is False
            assert kwargs["pipeline_type"] == "512"
            events.append("sample")
            return self.decode_latent(shape, texture, 512), (shape, texture, 512)

        def decode_latent(self, shape_arg, texture_arg, resolution):
            assert resolution == 512
            assert shape_arg.feats.device == texture_arg.feats.device == "mps"
            assert checkpoint.is_file()
            events.append("decode")
            faces = np.vstack([mesh.faces, [[-1, 0, 1]]])
            return [SimpleNamespace(
                vertices=DeviceTensor(mesh.vertices),
                faces=DeviceTensor(faces),
                attrs=texture.feats,
                coords=shape.coords,
            )]

    def bake(**kwargs):
        np.testing.assert_array_equal(kwargs["faces"].numpy(), mesh.faces)
        assert kwargs["vertices"].device == kwargs["faces"].device == "cpu"
        assert kwargs["attr_volume"].device == kwargs["coords"].device == "cpu"
        assert kwargs["decimation_target"] == 17 and kwargs["texture_size"] == 64
        assert events[-1] == "free"
        events.append("bake")
        if mode == "bake-error":
            raise RuntimeError("fixture bake failure")

        def export(path, *, extension_webp):
            assert path == str(output) and extension_webp
            mesh.export(path)

        return SimpleNamespace(export=export)

    torch = ModuleType("torch")
    torch.mps = SimpleNamespace(
        manual_seed=lambda seed: events.append(("seed", seed)),
        empty_cache=lambda: events.append("free"),
    )
    torch.device = lambda device: device
    torch.as_tensor = lambda data: np.asarray(data)
    torch.save = save
    torch.load = load
    voxel = ModuleType("o_voxel")
    voxel.postprocess = SimpleNamespace(to_glb=bake)
    pipeline_module = ModuleType("trellis2.pipelines.trellis2_image_to_3d")
    pipeline_module.Trellis2ImageTo3DPipeline = Pipeline
    sparse = ModuleType("trellis2.modules.sparse")
    sparse.SparseTensor = Sparse
    mlx = ModuleType("image_to_3dlab.mlx_attention")

    def release():
        events.append("free-mlx")
        return {"released": 1, "peak": 10, "active": 0}

    mlx.release_memory = release
    flex_gemm = ModuleType("flex_gemm")
    for dependency in (torch, voxel, pipeline_module, sparse, mlx, flex_gemm):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)

    args = [
        "--vendor-root", str(vendor), "--decimation-target", "17", "--texture-size", "64",
        "--resolution", "512", "--seed", "42", "--pre-cap", "25",
    ]
    if mode == "resume":
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"synthetic checkpoint")
        staged[checkpoint] = {
            "shape_slat_feats": shape.feats.cpu(),
            "tex_slat_feats": texture.feats.cpu(),
            "coords": shape.coords.cpu(),
            "res": 512,
            "seed": 42,
            "pipeline_type": "512",
            **origin,
        }
        args.extend(["--from-latents", str(checkpoint), str(output)])
    else:
        args.extend([str(source), str(output), "--matted-with", "birefnet-general-lite"])
    if mode in {"discard", "bake-error"}:
        args.append("--no-save-latents")
    if mode == "discard":
        args.append("--no-save-decode")

    if mode == "bake-error":
        with pytest.raises(RuntimeError, match="fixture bake failure"):
            mac.main(args)
        assert checkpoint.is_file() and decode_cache.is_file()
        assert not output.exists()
        assert events[-3:] == ["save-decode", "free", "bake"]
        return

    assert mac.main(args) == 0
    expected = ["load-latents"] if mode == "resume" else [
        ("seed", 42), "sample", "free-mlx", "save-latents",
    ]
    assert events[:2] == ["load", "mps"]
    assert events[2:] == [
        *expected, "decode", *([] if mode == "discard" else ["save-decode"]), "free", "bake",
    ]
    assert checkpoint.exists() is (mode != "discard")
    assert decode_cache.exists() is (mode != "discard")
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert ("latents" in manifest["artifacts"]) is (mode != "discard")
    assert ("decode" in manifest["artifacts"]) is (mode != "discard")
    assert manifest["seed"] == 42
    assert manifest["artifacts"]["glb"]["sha256"] == mac.sha256(output)
    record = json.loads(output.with_suffix(".provenance.json").read_text())
    assert record["input"]["path"] == str(source)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("device", "MPS backend not available"),
        ("precap", "fast_simplification not importable"),
        ("pipeline", "pipeline missing methods: ['get_cond']"),
        ("bake", "o_voxel.postprocess.to_glb not found"),
        ("attention", "expected 'sdpa'"),
        (None, "PASS: clean port present"),
    ],
)
def test_mac_check_validates_backend_contract(failure, message, offline, tmp_path, monkeypatch, capsys):
    import os

    vendor = tmp_path / "vendor"
    for location in (
        ".venv/bin/python", "TRELLIS.2/trellis2/representations/mesh/base.py",
        "deps/trellis2-apple/o-voxel/o_voxel/postprocess.py", "deps/mtlmesh/src/metal/remesh.metal",
    ):
        file = vendor / location
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    loader = mac.space_patch.pipeline_path(vendor / "TRELLIS.2")
    loader.parent.mkdir(parents=True)
    loader.write_text(
        "class Fixture:\n    def setup(self, new_pipeline):\n        new_pipeline.rembg_model = None\n"
    )
    (vendor / "stubs").mkdir()

    def device():
        assert os.environ["SPARSE_ATTN_BACKEND"] == "mlx"
        assert str(vendor / "stubs") in sys.path
        if failure == "attention":
            os.environ["ATTN_BACKEND"] = "changed-at-import"
        return failure != "device"

    torch = ModuleType("torch")
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=device))
    pipeline = ModuleType("trellis2.pipelines.trellis2_image_to_3d")
    pipeline.Trellis2ImageTo3DPipeline = SimpleNamespace(run=object(), decode_latent=object(),
                                                       preprocess_image=object())
    if failure != "pipeline":
        pipeline.Trellis2ImageTo3DPipeline.get_cond = object()
    voxel = ModuleType("o_voxel")
    voxel.postprocess = SimpleNamespace()
    if failure != "bake":
        voxel.postprocess.to_glb = object()
    for dependency in (torch, pipeline, voxel):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)
    monkeypatch.setitem(sys.modules, "fast_simplification",
                        None if failure == "precap" else ModuleType("fast_simplification"))

    assert mac.main(["--check", "--vendor-root", str(vendor), "--sparse-attn-backend", "mlx"]) == int(bool(failure))
    assert message in capsys.readouterr().out


@pytest.mark.parametrize("failure", [None, "bad-repo"])
def test_mac_revision_uses_only_its_vendor_checkout(failure, tmp_path, monkeypatch):
    checkout = tmp_path / "TRELLIS.2"
    (checkout / ".git").mkdir(parents=True)

    def run(argv, **kwargs):
        assert argv == ["git", "-C", str(checkout), "rev-parse", "HEAD"]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        if failure:
            raise subprocess.CalledProcessError(128, argv)
        return subprocess.CompletedProcess(argv, 0, stdout="commit-abcd\n")

    monkeypatch.setattr(mac.subprocess, "run", run)
    assert mac.backend_revision(tmp_path) == (None if failure else "commit-abcd")


def test_mac_bake_filters_sparse_corruption_after_precap(tmp_path, monkeypatch, capsys):
    mesh = trimesh.creation.box()
    vertices = Tensor(mesh.vertices)
    faces = Tensor(np.tile(mesh.faces, (100, 1)))
    volume, coords = Tensor([[1, 1, 1]]), Tensor([[0, 1, 2]])
    paths = []
    compressed = np.tile(mesh.faces, (84, 1))
    compressed[0, 0] = 999
    destination = tmp_path / "new/model.glb"

    def save(bundle, path):
        assert bundle == {"vertices": vertices, "faces": faces}
        path.touch()

    def load(path, *, map_location, weights_only):
        assert path.is_file() and map_location == "cpu" and not weights_only
        return vertices, Tensor(compressed)

    def launch(args, **kwargs):
        assert args[:3] == [sys.executable, mac.__file__, "--precap-worker"]
        assert Path(args[3]).is_file() and args[-1] == "1008"
        paths.append(Path(args[3]).parent)
        Path(args[4]).touch()
        return subprocess.CompletedProcess(args, 0)

    def bake(**kwargs):
        assert kwargs["vertices"] is vertices
        np.testing.assert_array_equal(kwargs["faces"].numpy(), compressed[1:])
        assert kwargs["attr_volume"] is volume and kwargs["coords"] is coords
        assert kwargs["decimation_target"] == 40 and kwargs["texture_size"] == 128

        def export(path, *, extension_webp):
            assert path == str(destination) and extension_webp
            mesh.export(path)

        return SimpleNamespace(export=export)

    torch = ModuleType("torch")
    torch.save = save
    torch.load = load
    torch.as_tensor = lambda mask, *, device: np.asarray(mask) if device == "cpu" else pytest.fail("device")
    voxel = ModuleType("o_voxel")
    voxel.postprocess = SimpleNamespace(to_glb=bake)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "o_voxel", voxel)
    monkeypatch.setattr(subprocess, "run", launch)
    _, artifacts = mac._bake_export(vertices, faces, volume, coords, {"base_color": [0, 1, 2]},
                                    512, destination, decimation_target=40, texture_size=128,
                                    pre_cap=1008)
    assert artifacts["glb"]["sha256"] == mac.sha256(destination)
    assert len(paths) == 1 and not paths[0].exists()
    assert "filtered 1 corrupt face(s)" in capsys.readouterr().out
