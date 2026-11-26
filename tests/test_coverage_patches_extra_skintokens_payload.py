"""Test the SkinTokens fallback against CPU attention and the installed adapter."""

import importlib
import sys

import patch_skintokens_portable as portable
import pytest
import torch
from test_patch_skintokens_portable import _upstream_tree

from scripts.skintokens_attention_compat import flash_attn_func


@pytest.fixture
def vendor(tmp_path, monkeypatch):
    root = _upstream_tree(tmp_path)
    model = root / "src/model"
    for name in ("tokenrig.py", "skin_vae_model.py"):
        path = model / name
        fixture = "import torch\n" + portable.FLASH_IMPORT
        if name == "tokenrig.py":
            fixture += '\nsettings = dict(attn_implementation="flash_attention_2")\n'
        fixture += "\n@torch.autocast(device_type='cuda', dtype=torch.bfloat16)\ndef unused():\n    pass\n"
        if name == "skin_vae_model.py":
            fixture += "\n@torch.autocast('cuda', dtype=torch.bfloat16)\ndef other_unused():\n    pass\n"
        path.write_text(fixture)
    monkeypatch.syspath_prepend(str(root / "src"))
    for name in ("flash_attn_interface", "flash_attn", "flash_attn.flash_attn_interface"):
        monkeypatch.setitem(sys.modules, name, None)
    yield root
    for name in tuple(sys.modules):
        if name == "model" or name.startswith("model."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize(("heads", "kv_heads", "causal", "scale"), [
    (4, 2, True, None), (2, 2, False, 0.2),
])
def test_skin_tokens_patched_dispatch_uses_copied_cpu_fallback(vendor, heads, kv_heads, causal, scale):
    portable.patch(vendor)
    helper = vendor / "src/model/_assetfurnace_attention.py"
    assert helper.read_bytes() == portable.HELPER_SOURCE.read_bytes()
    for name in ("model.tokenrig", "model.skin_vae_model"):
        module = importlib.import_module(name)
        assert module.flash_attn_func.__module__ == "model._assetfurnace_attention"

        rng = torch.Generator().manual_seed(3)
        q = torch.randn(2, 5, heads, 8, generator=rng)
        k = torch.randn(2, 5, kv_heads, 8, generator=rng).double()
        v = torch.randn(2, 5, kv_heads, 8, generator=rng).double()
        actual, lse = module.flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        ks = k.float().repeat_interleave(heads // kv_heads, dim=2)
        vs = v.float().repeat_interleave(heads // kv_heads, dim=2)
        scores = torch.einsum("blhd,bmhd->bhlm", q, ks) * (scale or 8 ** -0.5)
        if causal:
            scores = scores.masked_fill(torch.ones((5, 5), dtype=torch.bool).triu(1), -torch.inf)
        expected = torch.einsum("bhlm,bmhd->blhd", scores.softmax(dim=-1), vs)
        torch.testing.assert_close(actual, expected)
        tracked, tracked_lse = flash_attn_func(q, k, v, causal=causal, softmax_scale=scale)
        torch.testing.assert_close(tracked, expected)
        assert lse is tracked_lse is None and actual.is_contiguous()


def test_skin_tokens_check_refuses_stale_helper_and_patch_restores_it(vendor):
    portable.patch(vendor)
    helper = vendor / "src/model/_assetfurnace_attention.py"
    helper.write_text("# stale\n")
    with pytest.raises(SystemExit, match="not patched"):
        portable.patch(vendor, check=True)
    assert portable.patch(vendor)
    assert helper.read_bytes() == portable.HELPER_SOURCE.read_bytes()
    assert portable.patch(vendor, check=True) == []
    assert portable.patch(vendor) == []


def test_skintokens_cli_checks_without_writing_then_reinstalls_helper(vendor, capsys):
    args = ["--root", str(vendor)]
    assert portable.main([*args, "--check"]) == 1
    assert "not patched:" in capsys.readouterr().err
    assert portable.main(args) == 0
    assert "SkinTokens: patched" in capsys.readouterr().out
    assert portable.main([*args, "--check"]) == 0
    assert "patch present" in capsys.readouterr().out
    assert portable.main(args) == 0
    assert "already patched" in capsys.readouterr().out


def test_skintokens_refuses_modified_legacy_attention_patch(vendor):
    target = vendor / "src/model/tokenrig.py"
    original = target.read_text()
    added = original + portable.LEGACY_FLASH_MARKER + "\n" + portable.LEGACY_FLASH_END
    target.write_text(added)
    with pytest.raises(ValueError, match="previous attention patch has changed"):
        portable.patch(vendor)
    assert target.read_text() == added
    assert not (vendor / portable.HELPER_TARGET).exists()
