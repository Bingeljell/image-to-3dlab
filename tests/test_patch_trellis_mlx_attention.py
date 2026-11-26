"""Tests for the MLX sparse-attention patch.

These import the real patch functions rather than re-deriving them, so what is tested is
what ships. The three properties that matter for any patch script here: the anchor must be
present or it fails closed, re-running must be a no-op, and the injected code must import
everything it uses rather than assuming the host file's imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import patch_trellis_mlx_attention as patch

CONFIG_SOURCE = """import os
ATTN = 'flash_attn'

def set_backend():
    if env is not None and env in ['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']:
        ATTN = env

def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']):
    ATTN = backend
"""

FULL_ATTN_SOURCE = '''    elif config.ATTN == 'metal_flash':
        out = something()
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")

    if s is not None:
        return s.replace(out)
'''


def test_config_patch_adds_mlx_to_both_allow_lists():
    patched, changed = patch.patch_config(CONFIG_SOURCE)
    assert changed
    # Both the runtime env check and the Literal annotation must accept the new name,
    # otherwise SPARSE_ATTN_BACKEND=mlx is silently ignored.
    assert patched.count("'mlx'") == 2
    assert "Literal['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa', 'mlx']" in patched


def test_config_patch_is_idempotent():
    once, first = patch.patch_config(CONFIG_SOURCE)
    twice, second = patch.patch_config(once)
    assert first and not second
    assert once == twice


def test_config_patch_fails_closed_when_anchor_missing():
    with pytest.raises(RuntimeError, match="allow-list"):
        patch.patch_config("ATTN = 'flash_attn'\n")


def test_full_attn_patch_inserts_branch_before_the_guard():
    patched, changed = patch.patch_full_attn(FULL_ATTN_SOURCE)
    assert changed
    assert "elif config.ATTN == 'mlx':" in patched
    # The unknown-backend guard must survive, still last.
    assert patched.count("Unknown attention module") == 1
    assert patched.index("config.ATTN == 'mlx'") < patched.index("Unknown attention module")


def test_full_attn_patch_is_idempotent():
    once, first = patch.patch_full_attn(FULL_ATTN_SOURCE)
    twice, second = patch.patch_full_attn(once)
    assert first and not second
    assert once == twice


def test_full_attn_patch_fails_closed_when_anchor_missing():
    with pytest.raises(RuntimeError, match="dispatch"):
        patch.patch_full_attn("    elif config.ATTN == 'sdpa':\n        pass\n")


def test_injected_branch_imports_what_it_uses():
    """A patched file must not rely on the host file's imports.

    This cost a real generation run once: injected code referenced a name the host file
    did not import, and the failure only surfaced at the end of a 16-minute run.
    """
    branch = patch.DISPATCH_NEW
    assert "from trellis2._assetfurnace_attention import mlx_sparse_attention" in branch
    assert "out = mlx_sparse_attention(" in branch


def test_injected_branch_is_syntactically_valid_python():
    import ast
    # The branch is indented one level and starts with elif, so it only parses when
    # given a matching if at the same indentation.
    module = "def _host():\n    if False:\n        pass\n" + patch.DISPATCH_NEW
    ast.parse(module)


def test_full_attn_patch_upgrades_exact_legacy_dispatch():
    legacy = FULL_ATTN_SOURCE.replace(patch.DISPATCH_ANCHOR, patch.LEGACY_DISPATCH)
    changed, did = patch.patch_full_attn(legacy)
    assert did
    assert changed == patch.patch_full_attn(FULL_ATTN_SOURCE)[0]
