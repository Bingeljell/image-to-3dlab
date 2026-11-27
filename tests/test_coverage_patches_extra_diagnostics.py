"""Exercise diagnostic reporting and validation on CPU."""

import hashlib
import json
import os
import sys
from itertools import pairwise
from types import ModuleType, SimpleNamespace

import check_trellis_space_attention as attention
import check_trellis_space_dino as dino
import diagnose_remesh_kernel as remesh
import numpy as np
import pytest
import torch
from PIL import Image


def module(monkeypatch, name, **exports):
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            package = ModuleType(parent)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, parent, package)
    item = ModuleType(name)
    item.__dict__.update(exports)
    monkeypatch.setitem(sys.modules, name, item)


@pytest.fixture
def process_environment(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for key in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_CONV_BACKEND",
                "FLEX_GEMM_AUTOTUNE_CACHE_PATH", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


@pytest.mark.parametrize(("device", "available", "selected"), [
    ("auto", False, "cpu"), ("auto", True, "mps"), ("cpu", True, "cpu"),
    ("mps", False, None), ("mps", True, "mps"),
])
def test_dino_capture_writes_processed_image_and_tensor(
    tmp_path, capsys, process_environment, device, available, selected,
):
    changes = process_environment
    picture = tmp_path / "source.png"
    Image.new("RGBA", (3, 2), (1, 2, 3, 4)).save(picture)
    output = tmp_path / "report" / "result"
    tensor = torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)
    events = []

    class Pipeline:
        def preprocess_image(self, raw):
            assert raw.mode == "RGBA"
            return raw.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    class FeatureExtractor:
        def __init__(self, model, *, image_size):
            events.append((model, image_size))
            assert os.environ["HF_HUB_OFFLINE"] == "1"

        def to(self, actual):
            events.append(actual)

        def __call__(self, processed):
            assert len(processed) == 1 and processed[0].mode == "RGB"
            return tensor

    module(changes, "transformers", __version__="fixture")
    module(changes, "trellis2.modules.image_feature_extractor", DinoV3FeatureExtractor=FeatureExtractor)
    module(changes, "trellis2.pipelines.trellis2_image_to_3d", Trellis2ImageTo3DPipeline=Pipeline)
    changes.setattr(torch.backends.mps, "is_available", lambda: available)
    changes.setattr(sys, "argv", ["check", str(picture), "--root", str(tmp_path), "--device", device,
                                  "--resolution", "512", "--output", str(output)])
    if selected is None:
        with pytest.raises(RuntimeError, match="MPS is unavailable"):
            dino.main()
        assert events == []
        assert not output.parent.exists()
        return
    assert dino.main() == 0
    assert events == [("facebook/dinov3-vitl16-pretrain-lvd1689m", 512), selected]
    report = json.loads(output.with_suffix(".json").read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert report["device"] == selected
    assert report["resolution"] == 512
    features = torch.load(output.with_suffix(".pt"), weights_only=True)
    torch.testing.assert_close(features, tensor.float())
    assert report["feature_shape"] == [1, 3, 4]
    assert report["feature_mean"] == 5.5
    assert report["feature_sha256"] == hashlib.sha256(tensor.float().numpy().tobytes()).hexdigest()
    with Image.open(output.with_suffix(".png")) as image:
        assert image.mode == "RGB"
        assert report["processed_sha256"] == hashlib.sha256(np.asarray(image).tobytes()).hexdigest()


class VarLenTensor:
    def __init__(self, feats, layout):
        self.feats = feats
        self.layout = layout

    @staticmethod
    def layout_from_seqlen(lengths):
        ends = np.cumsum([0, *lengths])
        return [slice(a, b) for a, b in pairwise(ends)]


@pytest.mark.parametrize(("backend", "result"), [
    ("sdpa", "pass"), ("metal_flash", "pass"), ("sdpa", "mps-missing"),
    ("sdpa", "wrong-backend"), ("metal_flash", "self"), ("sdpa", "cross"),
])
def test_attention_gate_compares_packed_output_for_both_call_shapes(
    tmp_path, capsys, process_environment, backend, result,
):
    changes = process_environment
    calls = []

    def packed(q, *kv):
        values = []
        calls.append((len(kv), q.feats.shape[-1]))
        for i, bound in enumerate(q.layout):
            if kv:
                keys, vals = kv[0][i], kv[1][i]
                queries = q.feats[bound]
            else:
                queries, keys, vals = q.feats[bound].unbind(1)
            scores = torch.einsum("qhd,khd->hqk", queries, keys) / queries.shape[-1] ** .5
            values.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), vals))
        output = torch.cat(values)
        if result == ("cross" if kv else "self"):
            output = output + 1
        return SimpleNamespace(feats=output)

    module(changes, "trellis2.modules.sparse", VarLenTensor=VarLenTensor,
           config=SimpleNamespace(ATTN="wrong" if result == "wrong-backend" else backend))
    module(changes, "trellis2.modules.sparse.attention.full_attn", sparse_scaled_dot_product_attention=packed)
    changes.setattr(torch.backends.mps, "is_available", lambda: result != "mps-missing")
    tensor_to = torch.Tensor.to

    def cpu_to(tensor, destination, *args, **kwargs):
        assert destination == "mps"
        return tensor_to(tensor, "cpu", *args, **kwargs)

    changes.setattr(torch.Tensor, "to", cpu_to)
    changes.setattr(sys, "argv", ["check", "--backend", backend, "--root", str(tmp_path)])
    messages = {
        "mps-missing": "MPS is unavailable", "wrong-backend": "wrong sparse attention backend",
        "self": "self Metal attention parity failed", "cross": "cross Metal attention parity failed",
    }
    with torch.random.fork_rng(devices=[]):
        if result == "pass":
            assert attention.main() == 0
            printed = capsys.readouterr().out
            assert "self: max_error=" in printed and "cross: max_error=" in printed
            assert f"TRELLIS {backend} attention integration: PASS" in printed
        else:
            with pytest.raises(RuntimeError, match=messages[result]):
                attention.main()
    width = 128 if backend == "sdpa" else 64
    assert calls == ([] if result in {"mps-missing", "wrong-backend"} else [(0, width), (2, width)])


@pytest.fixture
def remesh_kernels(monkeypatch):
    data = SimpleNamespace(keys=None, miss=False, failure=False, sphere_failure=False)
    real_device = torch.device
    real_tensor, real_zeros = torch.tensor, torch.zeros

    def device(name):
        assert name == "mps"
        return real_device("cpu")

    def on_cpu(factory, *args, **kwargs):
        assert kwargs.pop("device") == "mps"
        return factory(*args, **kwargs)

    def initialize(resolution, capacity, allocate_on):
        assert allocate_on == real_device("cpu")
        return data, {"capacity": capacity, "resolution": resolution}

    def insert(table, params, coords, *resolutions):
        assert table is data
        assert params == {"capacity": 2 * len(coords), "resolution": resolutions[0]}
        assert len(torch.unique(coords, dim=0)) == len(coords)
        assert not coords[:, 0].any()
        data.keys = coords.clone()

    def lookup(table, params, coords, *resolutions):
        torch.testing.assert_close(data.keys, coords)
        if data.failure:
            raise RuntimeError("lookup kernel failed")
        result = torch.arange(len(coords), dtype=torch.int64)
        if data.miss:
            result[0] = 0xFFFFFFFF
        return result

    def run_sphere(vertices, faces, **kwargs):
        if data.sphere_failure:
            raise RuntimeError("remesh kernel failed")
        assert kwargs["scale"] == 1.2 and kwargs["resolution"] == 16
        assert vertices.shape[1] == 3 and faces.shape[1] == 3
        return vertices, faces

    module(monkeypatch, "cumesh", _C=SimpleNamespace(hashmap_insert_3d_idx_as_val=insert, hashmap_lookup_3d=lookup))
    module(monkeypatch, "cumesh.metal_remeshing", _init_hashmap=initialize, remesh_narrow_band_dc=run_sphere)
    monkeypatch.setattr(torch, "device", device)
    monkeypatch.setattr(torch, "tensor", lambda *args, **kwargs: on_cpu(real_tensor, *args, **kwargs))
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: on_cpu(real_zeros, *args, **kwargs))
    return data


@pytest.mark.parametrize("scenario", ["healthy", "miss", "kernel-errors", "skip-sphere"])
def test_remesh_cli_reports_kernel_result_or_error(remesh_kernels, monkeypatch, capsys, scenario):
    remesh_kernels.miss = scenario == "miss"
    remesh_kernels.failure = remesh_kernels.sphere_failure = scenario == "kernel-errors"
    argv = ["remesh", "--resolution", "16", "--keys", "32"]
    if scenario == "skip-sphere":
        argv += ["--skip-sphere"]
    monkeypatch.setattr(sys, "argv", argv)
    assert remesh.main() == 0
    report = capsys.readouterr().out
    if scenario == "kernel-errors":
        assert "FAILED: RuntimeError: lookup kernel failed" in report
        assert "FAILED: RuntimeError: remesh kernel failed" in report
    elif scenario == "miss":
        assert "of keys not found" in report and "a wireframe" in report
        assert "wrong_index" not in report
    else:
        assert "Hashmap is clean" in report and "wrong_index" in report
    assert ("TEST 2" in report) is (scenario != "skip-sphere")
    if scenario in {"healthy", "miss"}:
        assert "volume ratio: 1.0000" in report


def test_empty_lookup_has_no_miss_rate():
    assert remesh.summarise_lookup(torch.empty(0, dtype=torch.int64), 0) == {
        "queried": 0, "missing": 0, "negative": 0, "miss_rate": 0,
    }
