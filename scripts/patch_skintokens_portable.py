#!/usr/bin/env python3
"""Let SkinTokens (the auto-rigger) run on an Apple Silicon Mac, and on NVIDIA without flash-attn.

    python scripts/patch_skintokens_portable.py               # patches vendor/SkinTokens
    python scripts/patch_skintokens_portable.py --check       # exit 1 if any edit is missing

**Why.** Upstream assumes an NVIDIA card with flash-attn: it hard-codes `"cuda"` as the
device and autocast target, imports flash-attn with no fallback, and asks transformers for
`flash_attention_2`. None of that exists on a Mac, and flash-attn is a long compile on
NVIDIA. Every edit below keeps upstream's behaviour where its assumption holds and falls
back to PyTorch's own attention (SDPA) where it does not. Proven on a Mac 2026-10-04: a
40k-face character rigs in ~70 s, 7-8 GB peak.

`vendor/` is git-ignored, so this lives here and is re-applied after any re-clone.
Upstream is pinned (see `bootstrap_autorig.py`), so every anchor is exact. Safe to run
repeatedly: an edit already in place is skipped.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "SkinTokens"

DEVICE = '("cuda" if torch.cuda.is_available() else "mps")'
# flash_attention_2 only when flash-attn is installed, whatever the card: transformers
# raises at model build time if it is asked for and missing.
ATTN = ('("flash_attention_2" if __import__("importlib.util").util.find_spec("flash_attn") '
        'else "sdpa")')

# Upstream's flash-attn import, and the same import with a PyTorch fallback. Identical in
# tokenrig.py and skin_vae_model.py. The fallback keeps flash-attn's [B, L, H, D] layout
# and its (out, lse) return shape, so no caller changes.
FLASH_IMPORT = """\
try:
    from flash_attn_interface import flash_attn_func # type: ignore
except Exception as e:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
    def flash_attn_func(*args, **kwargs):
        res = _flash_attn_func(*args, **kwargs)
        return res, None
"""
FLASH_IMPORT_PORTABLE = """\
try:
    from flash_attn_interface import flash_attn_func # type: ignore
except Exception as e:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func
        def flash_attn_func(*args, **kwargs):
            res = _flash_attn_func(*args, **kwargs)
            return res, None
    except Exception:
        # i2l-portable: no flash-attn (e.g. Apple Silicon). Same [B, L, H, D] layout via SDPA.
        import torch.nn.functional as _F
        def flash_attn_func(q, k, v, causal=False, softmax_scale=None, **kwargs):
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            if k.shape[1] != q.shape[1]:  # grouped-query attention
                rep = q.shape[1] // k.shape[1]
                k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            v = v.to(q.dtype); k = k.to(q.dtype)
            out = _F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=softmax_scale)
            return out.transpose(1, 2).contiguous(), None
"""


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str  # every occurrence of `old` is replaced


EDITS: tuple[Edit, ...] = (
    # demo.py: the entry point we call. It moves batches to "cuda" and uses a worker
    # process, which on macOS cannot share MPS tensors (_share_filename_ RuntimeError).
    Edit("demo.py", "import requests\nfrom torch import Tensor\n",
         "import requests\nimport torch\nfrom torch import Tensor\n"),
    Edit("demo.py", "        num_workers=1,\n",
         "        num_workers=1 if torch.cuda.is_available() else 0,\n"),
    Edit("demo.py", 'k: v.to("cuda") if isinstance(v, Tensor) else v',
         f"k: v.to({DEVICE}) if isinstance(v, Tensor) else v"),
    # FLASH3 asks nvidia-smi for the card name, which fails without one.
    Edit("src/model/michelangelo/utils/misc.py",
         'self.available = "H100" in get_device_type()',
         'self.available = torch.cuda.is_available() and "H100" in get_device_type()'),
    Edit("src/model/skin_vae/autoencoders/FSQ.py",
         "partial(autocast, 'cuda', enabled = False)",
         f"partial(autocast, {DEVICE}, enabled = False)"),
    Edit("src/model/skin_vae_model.py", FLASH_IMPORT, FLASH_IMPORT_PORTABLE),
    Edit("src/model/skin_vae_model.py",
         "@torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
         f"@torch.autocast(device_type={DEVICE}, dtype=torch.bfloat16)"),
    Edit("src/model/skin_vae_model.py",
         "@torch.autocast('cuda', dtype=torch.bfloat16)",
         f"@torch.autocast({DEVICE}, dtype=torch.bfloat16)"),
    Edit("src/model/tokenrig.py", FLASH_IMPORT, FLASH_IMPORT_PORTABLE),
    Edit("src/model/tokenrig.py",
         'attn_implementation="flash_attention_2"', f"attn_implementation={ATTN}"),
    Edit("src/model/tokenrig.py",
         "@torch.autocast(device_type='cuda', dtype=torch.bfloat16)",
         f"@torch.autocast(device_type={DEVICE}, dtype=torch.bfloat16)"),
    Edit("src/server/spec.py", "    device='cuda',\n", f"    device={DEVICE},\n"),
    Edit("src/server/spec.py", '_attn_implementation="flash_attention_2"',
         f"_attn_implementation={ATTN}"),
)


def apply_edit(text: str, edit: Edit) -> tuple[str, str]:
    """(new text, "applied" | "present"). Raises if neither the anchor nor the edit is there.

    "present" means every occurrence was already replaced; a half-patched file (some
    occurrences left) still has the anchor, so it is finished rather than skipped.
    """
    found = text.count(edit.old)
    if found == 0:
        if edit.new in text:
            return text, "present"
        raise ValueError(f"{edit.path}: anchor not found, upstream changed: {edit.old[:60]!r}")
    return text.replace(edit.old, edit.new), "applied"


def patch(root: Path = DEFAULT_ROOT, check: bool = False) -> list[str]:
    """Apply (or with check=True, only verify) every edit. Returns the files that changed."""
    texts: dict[str, str] = {}
    changed: set[str] = set()
    for edit in EDITS:
        text = texts.get(edit.path)
        if text is None:
            text = (root / edit.path).read_text()
        new_text, state = apply_edit(text, edit)
        if state == "applied":
            if check:
                raise SystemExit(f"not patched: {edit.path}")
            changed.add(edit.path)
        texts[edit.path] = new_text
    for path in sorted(changed):
        (root / path).write_text(texts[path])
    return sorted(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--check", action="store_true", help="verify only, change nothing")
    args = parser.parse_args(argv)
    try:
        changed = patch(args.root, check=args.check)
    except SystemExit as missing:
        print(missing, file=sys.stderr)
        return 1
    if args.check:
        print("SkinTokens: portable patch present")
    elif changed:
        print("SkinTokens: patched " + ", ".join(changed))
    else:
        print("SkinTokens: already patched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
