"""The SkinTokens portability patch: every anchor applies, re-running is a no-op, and the
attention fallback computes what flash-attn would."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import patch_skintokens_portable as portable


def _upstream_tree(root: Path) -> Path:
    """A tree holding exactly upstream's anchor text, one file per edited path."""
    by_path: dict[str, list[str]] = {}
    for edit in portable.EDITS:
        by_path.setdefault(edit.path, []).append(edit.old)
    for path, anchors in by_path.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("import torch\n" + "\n".join(anchors) + "\n")
    return root


def test_patch_applies_then_is_idempotent(tmp_path):
    root = _upstream_tree(tmp_path)
    changed = portable.patch(root)
    assert set(changed) == {e.path for e in portable.EDITS}
    snapshot = {p: (root / p).read_text() for p in changed}
    assert portable.patch(root) == []
    assert {p: (root / p).read_text() for p in changed} == snapshot
    portable.patch(root, check=True)  # raises if anything is missing


def test_no_cuda_hardcoding_survives(tmp_path):
    root = _upstream_tree(tmp_path)
    portable.patch(root)
    for path in {e.path for e in portable.EDITS}:
        text = (root / path).read_text()
        assert "'cuda'" not in text
        assert 'v.to("cuda")' not in text
        assert 'attn_implementation="flash_attention_2"' not in text


def test_check_reports_an_unpatched_tree(tmp_path):
    root = _upstream_tree(tmp_path)
    with pytest.raises(SystemExit):
        portable.patch(root, check=True)
    assert portable.main(["--root", str(root), "--check"]) == 1


def test_changed_upstream_fails_loudly(tmp_path):
    root = _upstream_tree(tmp_path)
    (root / "demo.py").write_text("import torch\n# rewritten upstream\n")
    with pytest.raises(ValueError, match="anchor not found"):
        portable.patch(root)


def test_sdpa_fallback_matches_reference_attention():
    """The injected fallback, run as shipped, against plain softmax attention, including
    grouped-query attention (fewer key/value heads than query heads)."""
    torch = pytest.importorskip("torch")
    namespace: dict = {}
    # Force the fallback branch: neither flash-attn module exists in this venv.
    exec(portable.FLASH_IMPORT_PORTABLE, namespace)  # noqa: S102
    flash = namespace["flash_attn_func"]
    torch.manual_seed(0)
    b, length, heads, kv_heads, dim = 2, 7, 4, 2, 8
    q = torch.randn(b, length, heads, dim)
    k = torch.randn(b, length, kv_heads, dim)
    v = torch.randn(b, length, kv_heads, dim)
    out, lse = flash(q, k, v, causal=True)
    assert lse is None
    assert out.shape == (b, length, heads, dim)
    kr = k.repeat_interleave(heads // kv_heads, dim=2)
    vr = v.repeat_interleave(heads // kv_heads, dim=2)
    scores = torch.einsum("blhd,bmhd->bhlm", q, kr) / dim ** 0.5
    mask = torch.ones(length, length, dtype=torch.bool).triu(1)
    weights = scores.masked_fill(mask, float("-inf")).softmax(-1)
    expected = torch.einsum("bhlm,bmhd->blhd", weights, vr)
    assert torch.allclose(out, expected, atol=1e-5)


@pytest.mark.skipif(not (portable.DEFAULT_ROOT / "demo.py").exists(),
                    reason="vendor/SkinTokens is not cloned")
def test_installed_clone_is_patched():
    portable.patch(portable.DEFAULT_ROOT, check=True)
