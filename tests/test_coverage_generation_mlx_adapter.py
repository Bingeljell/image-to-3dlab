"""Exercise the patched TRELLIS MLX dispatch using CPU tensors."""

import importlib
import sys
from pathlib import Path

import pytest

from scripts import patch_trellis_mlx_attention as patch
from tests.test_patch_trellis_mlx_attention import CONFIG_SOURCE


@pytest.mark.parametrize("arity", [1, 2, 3])
def test_mlx_adapter_unpacks_views_before_calling_kernel(tmp_path, monkeypatch, arity):
    torch = pytest.importorskip("torch")
    from image_to_3dlab import mlx_attention
    from scripts.trellis_attention_compat import mlx_sparse_attention

    q = torch.arange(6).reshape(3, 1, 2).float()
    k, v = q + 10, q + 20
    inputs = {
        1: {"qkv": torch.stack([q, k, v], 1)},
        2: {"q": q, "kv": torch.stack([k, v], 1)},
        3: {"q": q, "k": k, "v": v},
    }[arity]
    package = tmp_path / "image_to_3dlab"
    package.mkdir()
    (package / "mlx_attention.py").touch()
    dispatch = tmp_path / "vendor/trellis2/full_attn.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    located = []

    def kernel(query, key, value, q_seqlen, kv_seqlen):
        for actual, expected in zip((query, key, value), (q, k, v), strict=True):
            torch.testing.assert_close(actual, expected)
        assert (q_seqlen, kv_seqlen) == ([1, 2], [1, 2])
        located.append(sys.path[0])
        return query + key + value

    monkeypatch.setattr(mlx_attention, "varlen_attention", kernel)
    expected = q + k + v
    for _ in range(2):
        result = mlx_sparse_attention(arity, inputs, [1, 2], [1, 2], dispatch)
        torch.testing.assert_close(result, expected)
    assert located == [str(tmp_path)] * 2
    assert sys.path.count(str(tmp_path)) == 1


def test_mlx_adapter_reports_checkout_location_when_repo_cannot_be_found(tmp_path):
    pytest.importorskip("torch")
    from scripts.trellis_attention_compat import mlx_sparse_attention

    dispatch = tmp_path / "full_attn.py"
    with pytest.raises(RuntimeError, match="could not locate image_to_3dlab") as error:
        mlx_sparse_attention(1, {}, [1], [1], dispatch)
    assert str(dispatch) in str(error.value)


def test_mlx_patch_installs_and_executes_tracked_helper(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from image_to_3dlab import mlx_attention

    repo_package = tmp_path / "image_to_3dlab"
    repo_package.mkdir()
    (repo_package / "mlx_attention.py").touch()
    root = tmp_path / "vendor"
    config = root / patch.TARGETS[0][0]
    attention = root / patch.TARGETS[1][0]
    config.parent.mkdir(parents=True)
    attention.parent.mkdir()
    config.write_text(CONFIG_SOURCE)
    attention.write_text(
        """from types import SimpleNamespace
config = SimpleNamespace(ATTN="mlx")

def run(qkv):
    num_all_args = 1
    q_seqlen = kv_seqlen = [2]
    if config.ATTN == 'metal_flash':
        return "metal fixture"
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
    return out
"""
    )
    assert patch.apply(root, check_only=True) == 2
    helper = root / "trellis2/_assetfurnace_attention.py"
    assert not helper.exists()
    assert patch.apply(root) == 2
    tracked = Path(patch.__file__).with_name("trellis_attention_compat.py")
    assert helper.read_bytes() == tracked.read_bytes()
    assert patch.apply(root) == 0
    helper.write_text("# stale helper")
    assert patch.apply(root, check_only=True) == 1
    assert helper.read_text() == "# stale helper"
    assert patch.apply(root) == 1
    assert helper.read_bytes() == tracked.read_bytes()
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.syspath_prepend(str(root))

    calls = []

    def kernel(q, k, v, q_seqlen, kv_seqlen):
        calls.append((q_seqlen, kv_seqlen))
        return (q - k) * v

    monkeypatch.setattr(mlx_attention, "varlen_attention", kernel)
    for name in tuple(sys.modules):
        if name == "trellis2" or name.startswith("trellis2."):
            monkeypatch.delitem(sys.modules, name)
    try:
        vendor = importlib.import_module("trellis2.modules.sparse.attention.full_attn")
        data = torch.arange(12).reshape(2, 3, 1, 2).float()
        q, k, v = data.unbind(1)
        torch.testing.assert_close(vendor.run(data), (q - k) * v)
        assert calls == [([2], [2])]
        vendor.config.ATTN = "metal_flash"
        assert vendor.run(data) == "metal fixture"
        vendor.config.ATTN = "unknown"
        with pytest.raises(ValueError, match="Unknown attention module: unknown"):
            vendor.run(data)
    finally:
        for name in tuple(sys.modules):
            if name == "trellis2" or name.startswith("trellis2."):
                sys.modules.pop(name)
