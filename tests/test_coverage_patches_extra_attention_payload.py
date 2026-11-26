"""Check the attention algorithms shipped into TRELLIS with CPU tensors."""

from types import SimpleNamespace

import pytest
import torch

from scripts import trellis_attention_compat as attention


def oracle(q, k, v):
    weights = torch.einsum("qhd,khd->hqk", q, k) / q.shape[-1] ** .5
    return torch.einsum("hqk,khd->qhd", weights.softmax(-1), v)


@pytest.fixture
def packed():
    rng = torch.Generator().manual_seed(39)
    return (torch.randn(5, 2, 4, generator=rng), torch.randn(7, 2, 4, generator=rng),
            torch.randn(7, 2, 3, generator=rng))


def test_mac_sdpa_chunks_queries_and_keeps_distinct_key_offsets(monkeypatch, packed):
    q, k, v = packed
    calls = []
    kernel = attention.scaled_dot_product_attention

    def record(q_part, k_part, v_part):
        calls.append(tuple(part.squeeze(0).transpose(0, 1) for part in (q_part, k_part, v_part)))
        return kernel(q_part, k_part, v_part)

    monkeypatch.setenv("SDPA_Q_CHUNK", "2")
    monkeypatch.setattr(attention, "scaled_dot_product_attention", record)
    result = attention.chunked_sdpa(q, k, v, [2, 0, 3], [1, 2, 4])
    expected = torch.cat([oracle(q[:2], k[:1], v[:1]), oracle(q[2:], k[3:], v[3:])])
    torch.testing.assert_close(result, expected)
    expected_calls = [(q[:2], k[:1], v[:1]), (q[2:4], k[3:], v[3:]), (q[4:], k[3:], v[3:])]
    for actual, slices in zip(calls, expected_calls, strict=True):
        for actual_slice, expected_slice in zip(actual, slices, strict=True):
            torch.testing.assert_close(actual_slice, expected_slice)


@pytest.mark.parametrize("arg_count", [1, 2, 3])
def test_unpack_handles_full_qkv_packed_kv_and_separate_inputs(arg_count):
    q, k, v = torch.arange(12, dtype=torch.float32).reshape(3, 1, 4), torch.ones(3, 1, 4), torch.zeros(3, 1, 4)
    qkv = torch.stack((q, k, v), dim=1) if arg_count == 1 else None
    kv = torch.stack((k, v), dim=1) if arg_count == 2 else None
    inputs = [{"qkv": qkv}, {"q": q, "kv": kv}, {"q": q, "k": k, "v": v}][arg_count - 1]
    result = attention.unpack_attention(arg_count, 1, inputs)
    for got, expected in zip(result, (q, k, v), strict=True):
        torch.testing.assert_close(got, expected)


def test_metal_attention_passes_prefixes_and_rejects_large_heads(packed):
    q, k, v = packed
    def kernel(qq, kk, vv, q_prefix, kv_prefix, max_q, max_kv, scale):
        assert qq is q and kk is k and vv is v
        assert q_prefix.tolist() == [0, 2, 5]
        assert kv_prefix.tolist() == [0, 3, 7]
        assert q_prefix.dtype == torch.int32 and q_prefix.device == q.device
        assert (max_q, max_kv, scale) == (3, 4, .5)
        return torch.cat([oracle(q[:2], k[:3], v[:3]), oracle(q[2:], k[3:], v[3:])])

    result = attention.metal_attention(q, k, v, [2, 3], [3, 4], q.device, kernel)
    assert result.shape == (5, 2, 3)
    with pytest.raises(RuntimeError, match="supports fast heads only through 64"):
        attention.metal_attention(torch.ones(5, 2, 128), k, v, [2, 3], [3, 4], q.device, kernel)


def cpu_kernel(q, k, v, *, causal):
    assert not causal
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def test_aule_dense_preserves_batch_head_and_output_width(packed):
    q, k, v = packed
    q, k, v = torch.stack((q, q * 2)), torch.stack((k, k * 2)), torch.stack((v, v * -1))
    result = attention.aule_dense(q, k, v, cpu_kernel)
    torch.testing.assert_close(result, torch.stack([oracle(q[0], k[0], v[0]), oracle(q[1], k[1], v[1])]))


@pytest.mark.parametrize("lengths_are_tensors", [False, True])
def test_aule_sparse_handles_distinct_window_pairs(packed, lengths_are_tensors):
    q, k, v = packed
    lengths = ([2, 3], [3, 4])
    if lengths_are_tensors:
        lengths = tuple(torch.tensor(part) for part in lengths)
    result = attention.aule_packed(q, k, v, *lengths, cpu_kernel)
    torch.testing.assert_close(result, torch.cat([oracle(q[:2], k[:3], v[:3]), oracle(q[2:], k[3:], v[3:])]))


@pytest.mark.parametrize("mps", [True, False])
def test_mac_device_selection_and_dino_model_lifecycle(monkeypatch, mps):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert attention.accelerator_device().type == ("mps" if mps else "cuda")
    assert attention.model_device(torch.nn.Linear(2, 2)).type == "cpu"
    for wrapped in (True, False):
        layers = [object()]
        model = SimpleNamespace(model=SimpleNamespace(layer=layers)) if wrapped else SimpleNamespace(layer=layers)
        assert attention.dino_layers(model) is layers


@pytest.mark.parametrize("cuda", [False, True])
def test_mac_only_clears_cuda_cache_when_available(monkeypatch, cuda):
    cleared = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cleared.append(True))
    attention.empty_cuda_cache()
    assert cleared == ([True] if cuda else [])


def use_vendor_package(monkeypatch, path):
    import sys
    from types import ModuleType

    package = ModuleType("trellis2")
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, "trellis2", package)
    monkeypatch.setitem(sys.modules, "trellis2._assetfurnace_attention", None)
    sys.modules.pop("trellis2._assetfurnace_attention")


@pytest.mark.parametrize("mlx_order", ["after-core", "before-core", "legacy"])
def test_mac_patch_migrates_inline_attention_and_invokes_copied_helper(tmp_path, monkeypatch, packed, mlx_order):
    import runpy
    import sys

    import patch_trellis_mlx_attention as mlx
    import patch_trellis_space_core as space

    path = tmp_path / "trellis2/modules/sparse/attention/full_attn.py"
    path.parent.mkdir(parents=True)
    path.write_text('''from types import SimpleNamespace

def run(q, k, v, q_seqlen, kv_seqlen, backend):
    config = SimpleNamespace(ATTN=backend)
    num_all_args = 3
    if config.ATTN == "flash_attn":
        raise AssertionError("wrong backend")
    elif config.ATTN == 'sdpa':
        # image-to-3dlab: evaluate each packed sequence independently with
        raise AssertionError("old inline implementation")
    elif config.ATTN == 'metal_flash':
        raise AssertionError("old inline implementation")
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
    return out
''')
    config = path.parent.parent / "config.py"
    config.write_text(f"from typing import Literal\nATTN: Literal{mlx.BACKEND_LIST_OLD}\n"
                      f"BACKENDS = {mlx.BACKEND_LIST_OLD}\n")
    if mlx_order == "before-core":
        assert mlx.apply(tmp_path) == 2
    elif mlx_order == "legacy":
        config.write_text(config.read_text().replace(mlx.BACKEND_LIST_OLD, mlx.BACKEND_LIST_NEW))
        path.write_text(path.read_text().replace(mlx.DISPATCH_ANCHOR, mlx.LEGACY_DISPATCH))
    assert space.patch_sparse_attention(tmp_path)
    if mlx_order == "after-core":
        assert mlx.apply(tmp_path) == 2
    elif mlx_order == "legacy":
        assert mlx.LEGACY_DISPATCH in path.read_text()
        assert mlx.apply(tmp_path) == 1
    copied = tmp_path / space.ATTENTION_HELPER_TARGET
    assert copied.read_bytes() == space.ATTENTION_HELPER_SOURCE.read_bytes()
    assert not space.patch_sparse_attention(tmp_path)
    assert not space.patch_sparse_config(tmp_path)
    assert mlx.apply(tmp_path) == 0
    use_vendor_package(monkeypatch, copied.parent)
    fake_model = tmp_path / "image_to_3dlab" / "mlx_attention.py"
    fake_model.parent.mkdir()
    fake_model.touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    calls = []

    def varlen_attention(q, k, v, q_seqlen, kv_seqlen):
        calls.append((q_seqlen, kv_seqlen))
        return v

    monkeypatch.setitem(sys.modules, "image_to_3dlab.mlx_attention", SimpleNamespace(varlen_attention=varlen_attention))
    dispatch = runpy.run_path(str(path))["run"]
    q, k, v = packed
    result = dispatch(q, k, v, [2, 3], [3, 4], "sdpa")
    torch.testing.assert_close(result, torch.cat([oracle(q[:2], k[:3], v[:3]), oracle(q[2:], k[3:], v[3:])]))
    assert dispatch(q, k, v, [2, 3], [3, 4], "mlx") is v
    assert calls == [([2, 3], [3, 4])]


def test_rocm_patch_installs_helpers_and_dispatch_calls_actual_copied_code(tmp_path, monkeypatch, packed):
    import runpy
    import sys

    import patch_trellis_rocm as rocm
    from test_patch_trellis_rocm import UPSTREAM_SOURCES

    for name, text in UPSTREAM_SOURCES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    dense = tmp_path / rocm.DENSE_ATTN
    dense.write_text('''from types import SimpleNamespace

def run(q, kv):
    config = SimpleNamespace(BACKEND="aule")
    num_all_args = 2
    if config.BACKEND == "flash_attn":
        raise AssertionError("wrong backend")
    else:
        raise ValueError(f"Unknown attention module: {config.BACKEND}")
    return out
''')
    lib = tmp_path / rocm.DEVICE_LIB_FILES[0]
    lib.write_text('''import os
_rocm = os.getenv("ROCM_PATH")
flags = [
    f"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}",
]
''')
    rocm.apply(tmp_path)
    assert rocm.is_patched(tmp_path)
    copied = tmp_path / rocm.ATTENTION_HELPER_TARGET
    assert copied.read_bytes() == rocm.ATTENTION_HELPER_SOURCE.read_bytes()
    bitcode = lib.with_name(rocm.BITCODE_HELPER_NAME)
    assert bitcode.read_bytes() == rocm.BITCODE_HELPER_SOURCE.read_bytes()
    use_vendor_package(monkeypatch, copied.parent)
    monkeypatch.setitem(sys.modules, "aule", SimpleNamespace(flash_attention_triton=cpu_kernel))
    q, k, v = packed
    # Packed KV requires a shared head width.
    v = torch.nn.functional.pad(v, (0, 1))
    result = runpy.run_path(str(dense))["run"](q.unsqueeze(0), torch.stack((k, v), dim=1).unsqueeze(0))
    torch.testing.assert_close(result, oracle(q, k, v).unsqueeze(0))

    sdk = tmp_path / "sdk" / "amdgcn" / "bitcode"
    sdk.mkdir(parents=True)
    monkeypatch.setenv("ROCM_PATH", str(sdk.parent.parent))
    monkeypatch.delenv("HIP_DEVICE_LIB_PATH", raising=False)
    assert runpy.run_path(str(lib))["flags"] == [f"--rocm-device-lib-path={sdk}"]
    assert not any(changed for _, changed in rocm.apply(tmp_path))
    helpers = [(copied, rocm.ATTENTION_HELPER_SOURCE)] + [
        ((tmp_path / name).with_name(rocm.BITCODE_HELPER_NAME), rocm.BITCODE_HELPER_SOURCE)
        for name in rocm.DEVICE_LIB_FILES
    ]
    for helper, source in helpers:
        helper.unlink()
        assert not rocm.is_patched(tmp_path)
        rocm.apply(tmp_path)
        assert rocm.is_patched(tmp_path)
        helper.write_bytes(b"# stale helper\n")
        assert not rocm.is_patched(tmp_path)
        rocm.apply(tmp_path)
        assert helper.read_bytes() == source.read_bytes()
        assert rocm.is_patched(tmp_path)
        assert not any(changed for _, changed in rocm.apply(tmp_path))


def test_window_helpers_use_actual_packed_sequences():
    q = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4)
    k, v = q * .5, q * .25
    lengths = torch.tensor([3, 1])
    expected = torch.cat([oracle(q[:3], k[:3], v[:3]), oracle(q[3:], k[3:], v[3:])])
    torch.testing.assert_close(attention.aule_window_self(torch.stack((q, k, v), dim=1), lengths, cpu_kernel),
                               expected)
    cross = attention.aule_window_cross(
        q, torch.stack((k, v), dim=1), lengths, torch.tensor([1, 3]), cpu_kernel,
    )
    torch.testing.assert_close(cross, torch.cat([oracle(q[:3], k[:1], v[:1]), oracle(q[3:], k[1:], v[1:])]))


def test_rocm_upgrades_previous_inline_attention_and_bitcode_adapters():
    import ast

    import patch_trellis_rocm as rocm

    dense = next(spec for spec in rocm.PATCHES if spec.file == rocm.DENSE_ATTN)
    legacy = (
        "    elif config.BACKEND == 'aule':\n"
        f"        # {rocm.MARKER} {rocm.AULE_PROOF}\n"
        "        out = legacy_attention()\n" + dense.needle + "\n"
    )
    result, changed = rocm.patch_text(legacy, dense, "dense attention")
    assert changed and "legacy_attention" not in result
    assert "aule_dense(q, k, v, flash_attention_triton)" in result
    assert rocm.patch_text(result, dense, "dense attention") == (result, False)

    spec = rocm.DEVICE_LIB_PATCHES[0]
    legacy = (
        f"flags = [\n    {rocm.MARKER} {rocm.DEVICE_LIB_PROOF}\n"
        "    'flag' or (_ for _ in ()).throw(RuntimeError(\n"
        "        'Set HIP_DEVICE_LIB_PATH to the folder holding ocml.bc.'))),\n"
        "]\n"
    )
    result, changed = rocm.patch_text(legacy, spec, "bitcode")
    assert changed and "legacy" not in result
    assert rocm.BITCODE_LINK in result
    ast.parse(result)
    assert rocm.patch_text(result, spec, "bitcode") == (result, False)


def test_metal_dispatch_loads_only_the_optional_kernel_for_supported_heads(monkeypatch, packed):
    import sys

    calls = []

    def kernel(*args):
        calls.append(args[3].tolist())
        q, k, v = args[:3]
        return oracle(q, k, v)

    metal = SimpleNamespace(sparse_attention_fwd=kernel)
    monkeypatch.setitem(sys.modules, "flex_gemm", SimpleNamespace(kernels=SimpleNamespace(metal=metal)))
    q, k, v = packed
    result = attention.metal_attention(q, k, v, [5], [7], q.device)
    torch.testing.assert_close(result, oracle(q, k, v))
    assert calls == [[0, 5]]


def test_mac_pipeline_upgrades_device_alias_and_calls_copied_cache_guard(tmp_path, monkeypatch):
    import runpy

    import patch_trellis_space_core as space

    pipeline = tmp_path / "trellis2/pipelines"
    pipeline.mkdir(parents=True)
    base = pipeline / "base.py"
    base.write_text('''import torch

class Device:
    def cuda(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda")
        self.to(device)
''')
    model = pipeline / "trellis2_image_to_3d.py"
    model.write_text('''import torch

def free():
    pass
'''.replace("    pass", "    if True:\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()"))
    assert space.patch_pipeline_device(tmp_path)
    assert not space.patch_pipeline_device(tmp_path)
    assert space.install_attention_helper(tmp_path)
    use_vendor_package(monkeypatch, pipeline.parent)
    seen = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: seen.append("empty_cache"))
    device = runpy.run_path(str(base))["Device"]()
    device.to = seen.append
    device.cuda()
    runpy.run_path(str(model))["free"]()
    assert seen == [torch.device("mps"), "empty_cache"]


@pytest.mark.parametrize("variant", ["duplicated", "missing-end"])
def test_rocm_legacy_attention_upgrade_rejects_unrecognized_block(variant):
    import patch_trellis_rocm as rocm

    spec = next(spec for spec in rocm.PATCHES if spec.file == rocm.DENSE_ATTN)
    source = (
        "    elif config.BACKEND == 'aule':\n"
        f"        # {rocm.MARKER} {rocm.AULE_PROOF}\n"
        "        out = unrecognized()\n"
    )
    source = (source + spec.needle) * 2 if variant == "duplicated" else source
    with pytest.raises(RuntimeError, match="ambiguous|boundaries changed"):
        rocm.patch_text(source, spec, "old attention")


def test_mac_legacy_attention_upgrade_leaves_ambiguous_file_untouched(tmp_path):
    import patch_trellis_space_core as space

    path = tmp_path / space.SPARSE_ATTENTION_TARGET
    path.parent.mkdir(parents=True)
    source = "        # image-to-3dlab: evaluate each packed sequence independently with\n"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="older patch is ambiguous"):
        space.patch_sparse_attention(tmp_path)
    assert path.read_text() == source
    assert not (tmp_path / space.ATTENTION_HELPER_TARGET).exists()


def test_mac_legacy_attention_upgrade_refuses_unknown_adjacent_branch(tmp_path):
    import patch_trellis_space_core as space

    path = tmp_path / space.SPARSE_ATTENTION_TARGET
    path.parent.mkdir(parents=True)
    source = """    elif config.ATTN == 'sdpa':
        # image-to-3dlab: evaluate each packed sequence independently with
        out = old_sdpa()
    elif config.ATTN == 'metal_flash':
        out = old_metal()
    elif config.ATTN == 'local_backend':
        out = local_attention()
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")
"""
    path.write_text(source)
    with pytest.raises(RuntimeError, match="older patch boundaries changed"):
        space.patch_sparse_attention(tmp_path)
    assert path.read_text() == source
    assert not (tmp_path / space.ATTENTION_HELPER_TARGET).exists()
