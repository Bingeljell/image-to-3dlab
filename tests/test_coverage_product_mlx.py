"""Check the tensor bridge on CPU with the macOS-only MLX kernel substituted."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from image_to_3dlab import mlx_attention as attention


@pytest.fixture
def mlx(monkeypatch):
    module = ModuleType("mlx.core")
    package = ModuleType("mlx")
    package.core = module
    module.float32 = np.float32
    module.float16 = np.float16
    # The dtype test only checks the hand-off. NumPy has no bfloat16 type.
    module.bfloat16 = object()
    module.array = np.array
    module.eval = lambda array: None
    module.invocations = []

    def scaled_attention(q, k, v, *, scale, mask):
        assert mask is None
        module.invocations.append((q.shape, k.shape, v.shape, scale, q.dtype))
        # The mathematical kernel is supplied by PyTorch on CPU at this boundary.
        return torch.nn.functional.scaled_dot_product_attention(
            *(torch.from_numpy(x).float() for x in (q, k, v)), scale=scale,
        ).numpy()

    module.fast = SimpleNamespace(scaled_dot_product_attention=scaled_attention)
    module.cache = 128
    module.get_active_memory = lambda: 24
    module.get_cache_memory = lambda: module.cache
    module.get_peak_memory = lambda: 180

    def clear_cache():
        module.cache = 0

    module.clear_cache = clear_cache
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", module)
    monkeypatch.delenv(attention.DTYPE_ENV, raising=False)
    return module


@pytest.mark.parametrize(("name", "attr"), [("FP32", "float32"), ("fp16", "float16"), ("bf16", "bfloat16")])
def test_dtype_environment_maps_to_runtime_type(mlx, monkeypatch, name, attr):
    monkeypatch.setenv(attention.DTYPE_ENV, name)
    assert attention.resolve_dtype() is getattr(mlx, attr)


def test_unsupported_dtype_names_the_setting(mlx):
    with pytest.raises(ValueError, match="expected one of fp32, fp16, bf16"):
        attention.resolve_dtype("nonsense")


def test_mlx_memory_release_reports_before_and_after(mlx):
    assert attention.mlx_available()
    assert attention.memory_snapshot() == {"active": 24, "cache": 128, "peak": 180}
    assert attention.release_memory() == {
        "cache_before": 128, "cache_after": 0, "released": 128, "active": 24, "peak": 180,
    }
    assert mlx.cache == 0


def test_attention_keeps_batch_boundaries_and_output_dtype(mlx):
    q = torch.zeros((3, 2, 4), dtype=torch.float64)
    k = torch.zeros((5, 2, 4), dtype=torch.float32)
    v = torch.arange(40, dtype=torch.float32).reshape(5, 2, 4)
    out = attention.varlen_attention(q, k, v, q_seqlen=[1, 2], kv_seqlen=[2, 3], scale=0.75)
    expected = torch.stack([v[:2].mean(0), v[2:].mean(0), v[2:].mean(0)]).to(q)
    torch.testing.assert_close(out, expected)
    assert out.is_contiguous()
    assert mlx.invocations == [
        ((1, 2, 1, 4), (1, 2, 2, 4), (1, 2, 2, 4), .75, np.dtype("float32")),
        ((1, 2, 2, 4), (1, 2, 3, 4), (1, 2, 3, 4), .75, np.dtype("float32")),
    ]


def test_single_attention_uses_feature_size_for_scale(mlx):
    q = torch.ones((2, 2, 4))
    result = attention.attention_single(q, q, q)
    torch.testing.assert_close(result, q)
    assert mlx.invocations[0][3:] == (0.5, np.dtype("float32"))


def test_attention_empty_sequences_and_mismatched_counts(mlx):
    q = torch.empty((0, 2, 4), dtype=torch.float64)
    out = attention.varlen_attention(q, q, q, [], [], dtype=mlx.float32)
    assert out.shape == q.shape and out.dtype == q.dtype
    with pytest.raises(ValueError, match="same number of sequences"):
        attention.varlen_attention(q, q, q, [], [1])
    assert not mlx.invocations
