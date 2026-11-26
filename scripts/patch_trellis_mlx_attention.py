#!/usr/bin/env python3
"""Add an ``mlx`` sparse-attention backend to a vendored TRELLIS.2 checkout.

TRELLIS.2-4B has a head dimension of 128. PyTorch's MPS backend has no fused attention
kernel at any head dimension, and Pedro's Metal kernel stops at 64, so the model falls back
to unfused SDPA that materialises the full score matrix. Measured 2026-09-20 at the real
Stage-3 shape (9,801 tokens, 12 heads, head dim 128), attention is 93.2% of sampling time.

MLX has a fused kernel that handles 128-wide heads: 3.2x faster than torch MPS SDPA at
fp32, 13.8x at fp16, and still 2.83x / 9.35x once tensors are round-tripped through host
memory, which is what the implementation does.

This patch is deliberately additive. It adds a new backend name and a new dispatch branch;
it does not touch the existing ``sdpa`` path, which stays the default. Select the new one
with ``SPARSE_ATTN_BACKEND=mlx``.

The attention itself lives in ``image_to_3dlab/mlx_attention.py`` so it can be unit-tested;
only a thin dispatch branch is injected here.

Usage:
    python scripts/patch_trellis_mlx_attention.py [--root vendor/trellis-space-mac/TRELLIS.2]
    python scripts/patch_trellis_mlx_attention.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-space-mac" / "TRELLIS.2"
HELPER_SOURCE = Path(__file__).with_name("trellis_attention_compat.py")
HELPER_TARGET = "trellis2/_assetfurnace_attention.py"
ATTENTION_TARGET = "trellis2/modules/sparse/attention/full_attn.py"

BACKEND_LIST_OLD = "['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa']"
BACKEND_LIST_NEW = "['xformers', 'flash_attn', 'flash_attn_3', 'metal_flash', 'sdpa', 'mlx']"

DISPATCH_ANCHOR = """    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")"""

LEGACY_DISPATCH = '''    elif config.ATTN == 'mlx':
        # image-to-3dlab: route attention through MLX's fused Metal kernel.
        #
        # torch MPS has no fused attention kernel and Pedro's Metal kernel supports head
        # dimensions only through 64, while TRELLIS.2-4B uses 128. The unfused fallback
        # dominates sampling: 93.2% of Stage-3 step time, measured 2026-09-20.
        #
        # The real work lives in image_to_3dlab/mlx_attention.py so it is importable and
        # unit-tested; this branch is only dispatch. The vendored tree has no guaranteed
        # sys.path entry for the repository, so walk up from this file to find it. That
        # holds as long as vendor/ stays inside the repo, and fails loudly if it does not.
        import sys as _sys
        from pathlib import Path as _Path

        _root = None
        for _parent in _Path(__file__).resolve().parents:
            if (_parent / 'image_to_3dlab' / 'mlx_attention.py').is_file():
                _root = _parent
                break
        if _root is None:
            raise RuntimeError(
                "mlx attention backend: could not locate image_to_3dlab starting from "
                f"{__file__}. The vendored checkout must live inside the repository."
            )
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from image_to_3dlab.mlx_attention import varlen_attention as _varlen_attention

        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        out = _varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    else:
        raise ValueError(f"Unknown attention module: {config.ATTN}")'''


DISPATCH_NEW = """    elif config.ATTN == 'mlx':
        from trellis2._assetfurnace_attention import mlx_sparse_attention

        out = mlx_sparse_attention(num_all_args, locals(), q_seqlen, kv_seqlen, __file__)
""" + DISPATCH_ANCHOR


def replace_exact(source: str, old: str, new: str, *, count: int, label: str) -> tuple[str, bool]:
    """Replace an audited anchor, or fail rather than produce a partial patch."""
    if new in source:
        return source, False
    found = source.count(old)
    if found != count:
        raise RuntimeError(f"{label}: expected {count} source anchors, found {found}")
    return source.replace(old, new, count), True


def patch_config(source: str) -> tuple[str, bool]:
    """Add 'mlx' to both backend allow-lists.

    The bare list and the Literal[...] annotation share the same bracketed text, so one
    replacement covers both occurrences.
    """
    return replace_exact(
        source,
        BACKEND_LIST_OLD,
        BACKEND_LIST_NEW,
        count=2,
        label="sparse attention backend allow-list",
    )


def patch_full_attn(source: str) -> tuple[str, bool]:
    """Insert the mlx dispatch branch ahead of the unknown-backend guard."""
    anchor = LEGACY_DISPATCH if LEGACY_DISPATCH in source else DISPATCH_ANCHOR
    return replace_exact(
        source,
        anchor,
        DISPATCH_NEW,
        count=1,
        label="sparse attention dispatch",
    )


TARGETS = (
    ("trellis2/modules/sparse/config.py", patch_config),
    (ATTENTION_TARGET, patch_full_attn),
)


def apply(root: Path, *, check_only: bool = False) -> int:
    changed = 0
    helper = root / HELPER_TARGET
    payload = HELPER_SOURCE.read_bytes()
    helper_current = helper.is_file()
    for relative, transform in TARGETS:
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"error: missing {path}; is --root a TRELLIS.2 checkout?")
        source = path.read_text()
        patched, did = transform(source)
        state = "would patch" if check_only else "patched"
        needs_helper = relative == ATTENTION_TARGET and not helper_current
        if did or needs_helper:
            changed += 1
            if not check_only:
                if needs_helper:
                    helper.write_bytes(payload)
                if did:
                    path.write_text(patched)
            print(f"{state}: {relative}")
        else:
            print(f"already applied: {relative}")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="TRELLIS.2 checkout to patch")
    parser.add_argument("--check", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args(argv)
    changed = apply(args.root.resolve(), check_only=args.check)
    print(f"{changed} file(s) {'would change' if args.check else 'changed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
