#!/usr/bin/env python3
"""Make Tencent's Hunyuan3D-2.1 survive on an AMD card, without touching what NVIDIA runs.

Upstream is NVIDIA-first, but almost all of it is plain PyTorch: the shape stage has no
compiled code at all, and the paint stage's one CUDA extension is 127 lines that ROCm's
torch builds with `hipcc` on its own. What is left is three things this script fixes.

**1. A crash in the scheduler.** `index_for_timestep` looks a timestep up with exact float
equality, `(schedule_timesteps == timestep).nonzero()`, and then reads `indices[pos]`. When
no entry matches exactly the tensor is empty and that read raises IndexError -- minutes into
a generation, with the model already loaded. It is a bug on any card; it shows up here
because nothing pins the arithmetic to one order. Nearest-match instead of equality.

**2. The paint stage outgrowing a 20 GB card.** Upstream folds the PBR channels and the
views into the batch dimension and attends over all of them at once, so memory grows as the
square of the token count. On a consumer card that is the difference between running and not.
Chunking the query axis is the same maths, done in pieces.

**3. A hardcoded `cuda:0`.** It resolves to the GPU under ROCm, so it is not what breaks a
run here, but it silently defeats CPU offload, which is the other lever on a small card.

Safe to run twice: a patched file is recognised and left alone. Every anchor is asserted, so
an upstream change fails loudly instead of producing a checkout that half-works.

    python scripts/patch_hunyuan_rocm.py                # vendor/hunyuan-rocm/Hunyuan3D-2.1
    python scripts/patch_hunyuan_rocm.py --root <checkout>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "hunyuan-rocm" / "Hunyuan3D-2.1"

MARKER = "# image-to-3dlab:"


@dataclass(frozen=True)
class Patch:
    """One anchored replacement, and how to tell afterwards that it is in place.

    `proof` is a phrase that appears only in the replacement and never in the original.
    That is what makes "is it patched?" answerable: testing for the *absence* of the anchor
    does not work when the replacement contains the anchor, which the import patch's does.
    """

    file: str
    needle: str
    replacement: str
    proof: str
    expected: int = 1
    # True when the replacement still contains the needle, which it does when the patch
    # prepends a line to an existing import. "Is the anchor gone?" is the wrong question
    # for those, and asking it anyway made every patched checkout read as unpatched.
    needle_survives: bool = False


# --- 1. The scheduler's exact float lookup -------------------------------------------------

SCHEDULER_FILE = "hy3dshape/hy3dshape/schedulers.py"
SCHEDULER_NEEDLE = "        indices = (schedule_timesteps == timestep).nonzero()"
SCHEDULER_REPLACEMENT = (
    f"        {MARKER} nearest match, not float equality: upstream reads indices[pos]\n"
    "        # straight after, which raises IndexError once no entry matches exactly.\n"
    "        indices = (schedule_timesteps - timestep).abs().argmin().reshape(1)"
)

# The same line appears twice, in the flow-matching and the ddim scheduler. Both are patched
# by replacement, which is why this is a count and not a flag.
SCHEDULER_EXPECTED = 2


# --- 2. Chunked attention in the paint stage ----------------------------------------------

ATTN_FILE = "hy3dpaint/hunyuanpaintpbr/unet/attn_processor.py"
ATTN_NEEDLE = (
    "        hidden_states = F.scaled_dot_product_attention(\n"
    "            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False\n"
    "        )"
)
ATTN_REPLACEMENT = (
    f"        {MARKER} chunk the sequence axis. Upstream attends over every view and every\n"
    "        # PBR channel in one go, so memory goes as the square of the token count and a\n"
    "        # 20 GB card runs out. Same maths, done in pieces.\n"
    "        #\n"
    "        # query, key and value are all (batch, heads, sequence, head_dim): the tokens\n"
    "        # are dim -2 and dim 0 is the batch. Chunking dim 0 instead looks right and\n"
    "        # leaves key and value at full batch, where SDPA broadcasts the output batch\n"
    "        # back up and the pieces get welded end to end.\n"
    "        chunk = int(os.environ.get('HY3D_SDPA_QUERY_CHUNK', '0') or 0)\n"
    "        if chunk > 0 and query.shape[-2] > chunk:\n"
    "            parts = []\n"
    "            for i in range(0, query.shape[-2], chunk):\n"
    "                # The mask is (batch, heads, q_len, k_len); its query axis is dim -2\n"
    "                # and has to be sliced with the queries or the shapes stop agreeing.\n"
    "                mask = None if attention_mask is None else attention_mask[..., i:i + chunk, :]\n"
    "                parts.append(F.scaled_dot_product_attention(\n"
    "                    query[..., i:i + chunk, :], key, value,\n"
    "                    attn_mask=mask, dropout_p=0.0, is_causal=False))\n"
    "            hidden_states = torch.cat(parts, dim=-2)\n"
    "        else:\n"
    "            hidden_states = F.scaled_dot_product_attention(\n"
    "                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False\n"
    "            )"
)
ATTN_IMPORT_NEEDLE = "import torch.nn.functional as F"
ATTN_IMPORT_REPLACEMENT = ATTN_IMPORT_NEEDLE + "\nimport os  # image-to-3dlab: the chunk size"


# --- 3. The hardcoded device --------------------------------------------------------------

DEVICE_NEEDLE = (
    '            processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c")'
    '.to("cuda:0")'
)
DEVICE_REPLACEMENT = (
    "            # image-to-3dlab: follow the tensor rather than naming a device, so CPU\n"
    '            # offload still works. ".to(\'cuda:0\')" resolves under ROCm but\n'
    "            # overrides offloading, which is the other lever on a small card.\n"
    '            processed_hs = rearrange(pbr_hs, "b n_pbrs n l c -> (b n_pbrs n) l c")'
)


PATCHES = (
    # The scheduler line is in two schedulers and only one of them may move, so it is
    # replaced wherever it appears and the count asserted.
    Patch(SCHEDULER_FILE, SCHEDULER_NEEDLE, SCHEDULER_REPLACEMENT,
          proof="nearest match, not float equality", expected=SCHEDULER_EXPECTED),
    Patch(ATTN_FILE, ATTN_IMPORT_NEEDLE, ATTN_IMPORT_REPLACEMENT,
          proof="import os  # image-to-3dlab", needle_survives=True),
    Patch(ATTN_FILE, ATTN_NEEDLE, ATTN_REPLACEMENT,
          proof="chunk the sequence axis"),
    Patch(ATTN_FILE, DEVICE_NEEDLE, DEVICE_REPLACEMENT,
          proof="follow the tensor rather than naming a device"),
)


def patch_text(source: str, spec: Patch, label: str) -> tuple[str, bool]:
    """Return the patched text and whether it changed.

    The proof is checked *first*, before the anchor. The import patch keeps its own anchor
    in its replacement, so testing the anchor first made that patch re-apply on every run
    and stack another `import os` each time.

    Raises when neither the proof nor the anchor is there, which is the signal that upstream
    moved and the patch wants re-reading before it is trusted again.
    """
    if spec.proof in source:
        return source, False
    if spec.needle not in source:
        raise RuntimeError(f"{label}: the anchor for this patch is gone; upstream changed, "
                           "re-check the patch before running it")
    return source.replace(spec.needle, spec.replacement), True


def patch_file(path: Path, spec: Patch) -> bool:
    """Patch one file. Returns whether it changed, and asserts the anchor count."""
    text = path.read_text()
    found = text.count(spec.needle)
    if found and found != spec.expected:
        raise RuntimeError(f"{path}: expected {spec.expected} occurrence(s) of this patch's "
                           f"anchor, found {found}; re-check the patch before running it")
    patched, changed = patch_text(text, spec, str(path))
    if changed:
        path.write_text(patched)
    return changed


def is_patched(root: Path = DEFAULT_ROOT) -> bool:
    """True when every patch's proof is in the file and no unpatched anchor is left.

    Both halves matter. The proof alone would pass if upstream's own text happened to
    contain the phrase; the anchor alone would never pass for the import patch, which
    keeps its anchor on purpose.
    """
    try:
        for spec in PATCHES:
            text = (root / spec.file).read_text()
            if spec.proof not in text:
                return False
            if not spec.needle_survives and spec.needle in text:
                return False
    except OSError:
        return False
    return True


def patch(root: Path = DEFAULT_ROOT) -> int:
    """Apply every patch. Returns how many files changed, and is safe to run twice."""
    if not root.is_dir():
        raise SystemExit(f"No Hunyuan3D-2.1 checkout at {root}. Run the bootstrap first.")
    changed: set[str] = set()
    for spec in PATCHES:
        path = root / spec.file
        if not path.is_file():
            raise SystemExit(f"{path} is missing; this is not a Hunyuan3D-2.1 checkout.")
        if patch_file(path, spec):
            changed.add(spec.file)
    return len(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the Hunyuan3D-2.1 checkout (default: vendor/hunyuan-rocm)")
    parser.add_argument("--check", action="store_true",
                        help="Only report whether the patches are in place.")
    args = parser.parse_args(argv)

    if args.check:
        print("patched" if is_patched(args.root) else "not patched")
        return 0 if is_patched(args.root) else 1
    changed = patch(args.root)
    print(f"{'patched' if changed else 'already patched'}: {args.root} ({changed} file(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())