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
import hashlib
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
        # i2l-portable: no flash-attn (e.g. Apple Silicon).
        from ._assetfurnace_attention import flash_attn_func
"""

HELPER_SOURCE = Path(__file__).with_name("skintokens_attention_compat.py")
HELPER_TARGET = "src/model/_assetfurnace_attention.py"
LEGACY_FLASH_START = "try:\n    from flash_attn_interface import flash_attn_func # type: ignore\n"
LEGACY_FLASH_MARKER = "        # i2l-portable: no flash-attn (e.g. Apple Silicon). Same"
LEGACY_FLASH_END = "            return out.transpose(1, 2).contiguous(), None\n"
LEGACY_FLASH_DIGEST = "d3093e6b314b80983635ceb2236b9edd7b3737fec7d931ae2a3d4ec5f5d3425e"


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
    # Fixed ports can sit inside a range Windows reserves for Hyper-V/WSL (#193), and then
    # the Blender server cannot bind. The rig job passes free ones; unset keeps upstream's.
    Edit("src/server/spec.py", "PORT = 59875\n",
         'PORT = int(os.environ.get("SKINTOKENS_PORT", "59875"))\n'),
    Edit("src/server/spec.py", "BPY_PORT = 59876\n",
         'BPY_PORT = int(os.environ.get("SKINTOKENS_BPY_PORT", "59876"))\n'),
)


def apply_edit(text: str, edit: Edit) -> tuple[str, str]:
    """(new text, status). Raises when a known upstream or patched anchor is missing.

    "present" means every occurrence was already replaced; a half-patched file (some
    occurrences left) still has the anchor, so it is finished rather than skipped.
    """
    if edit.new == FLASH_IMPORT_PORTABLE and LEGACY_FLASH_MARKER in text:
        marker = text.index(LEGACY_FLASH_MARKER)
        start = text.rfind(LEGACY_FLASH_START, 0, marker)
        end = text.find(LEGACY_FLASH_END, marker)
        if start >= 0 and end >= 0:
            end += len(LEGACY_FLASH_END)
            if hashlib.sha256(text[start:end].encode()).hexdigest() == LEGACY_FLASH_DIGEST:
                return text[:start] + edit.new + text[end:], "upgrade"
        raise ValueError(f"{edit.path}: previous attention patch has changed")
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
        if state == "applied" and check:
            raise SystemExit(f"not patched: {edit.path}")
        if state != "present" and not check:
            changed.add(edit.path)
        texts[edit.path] = text if check else new_text
    if check and not any(FLASH_IMPORT_PORTABLE in text for text in texts.values()):
        return []  # Exact older inline patch still runs without the copied helper.
    helper = root / HELPER_TARGET
    expected = HELPER_SOURCE.read_bytes()
    try:
        installed = helper.read_bytes()
    except OSError:
        installed = None
    if installed != expected:
        if check:
            raise SystemExit(f"not patched: {HELPER_TARGET}")
        helper.write_bytes(expected)
    for path in sorted(changed):
        (root / path).write_text(texts[path])
    if installed != expected:
        changed.add(HELPER_TARGET)
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
