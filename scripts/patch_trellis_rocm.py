#!/usr/bin/env python3
"""Make TRELLIS.2 run on an AMD card: an attention backend, and a build that compiles.

Three separate things stop this route, and all three are fixed here.

**Attention.** TRELLIS.2's sparse attention (`SPARSE_ATTN_BACKEND`) has three backends and
all three need a compiled CUDA extension: xformers, flash-attn and flash-attn-3. So on a
card without them there is nothing to select and the run dies before the first forward
pass. This adds `aule` (AuleAttention, https://github.com/xenn0010/Aule-Attention) as a
fourth. It is a Triton flash attention with no extension to compile, ships as a
pure-Python wheel, and runs on ROCm — where it was checked against torch's own SDPA and
agrees to 5e-4. The edit is the dispatch branch upstream lacks, in the sparse, windowed and
dense attention paths.

**The device bitcode.** The ROCm ports of nvdiffrast and CuMesh hardcode the ROCm 10
device-bitcode path `<root>/lib/llvm/amdgcn/bitcode`. ROCm 7.x keeps it at
`<root>/amdgcn/bitcode`, so the build stops with `cannot find ROCm device library`. This
makes each honour `HIP_DEVICE_LIB_PATH` (hipcc's own variable) and then whichever of the two
layouts the install actually has.

**The C++ standard.** torch 2.14's headers need C++20, and FlexGEMM and o-voxel are pinned
at upstream commits that say `-std=c++17`, so the compile dies inside
`c10/util/intrusive_ptr.h` on `std::strong_ordering` before reaching a line of the
extension's own. For those two that is a three-word change rather than two more
third-party checkouts, and both cards build them from identical source.

**The Python floor.** Both dmonkman ports declare `requires-python = ">=3.12"`, which stops
uv installing them on the 3.11 this repo uses everywhere. Neither upstream sets that floor —
it exists because their published wheels are cp312.

Deliberately *not* copied from the community forks: those wrap every existing backend in a
bare `except Exception` and fall back to SDPA, and they replace the `raise ValueError` for
an unknown backend with a silent fallback. A kernel that fails there would quietly produce
a worse mesh and nobody would know. Unknown backends still raise; a missing `aule` import
raises.

    python scripts/patch_trellis_rocm.py --root vendor/trellis-rocm
    python scripts/patch_trellis_rocm.py --check --root vendor/trellis-rocm
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "trellis-rocm"
CHECKOUT = "TRELLIS.2"
EXTENSIONS = "extensions"

MARKER = "# image-to-3dlab:"


@dataclass(frozen=True)
class Patch:
    """One anchored replacement, and how to tell afterwards that it is in place.

    `proof` is a phrase that appears only in the replacement and never in the original.
    That is what makes "is it patched?" answerable: testing for the *absence* of the anchor
    does not work when the replacement contains the anchor.
    """

    file: str
    needle: str
    replacement: str
    proof: str
    expected: int = 1
    # True when the replacement still contains the needle. "Is the anchor gone?" is the
    # wrong question for those, and asking it anyway made every patched checkout read as
    # unpatched.
    needle_survives: bool = False
    # When set, the replacement's %%INDENT%% becomes whatever whitespace sits before the
    # anchor on its line, and %%DEEP%% is one further level (always four spaces). The four
    # forks nest the same flag at different depths — nvdiffrast wraps it in `if _rocm:` — so
    # one fixed indentation is a syntax error in three of them.
    indent_to_anchor: bool = False


# --- 1. The aule attention backend ---------------------------------------------------------
#
# The config files both validate the backend name against a list before letting it through;
# an unlisted name is silently ignored and the default (flash_attn) stands, which fails later
# and much less legibly. Both the list and the Literal are extended, so the patch is not
# half of what it looks like it is.

SPARSE_CONFIG = f"{CHECKOUT}/trellis2/modules/sparse/config.py"
SPARSE_CONFIG_NEEDLE = (
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3']")
SPARSE_CONFIG_REPLACEMENT = (
    "env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'aule']")
SPARSE_LITERAL_NEEDLE = "def set_attn_backend(backend: Literal['xformers', 'flash_attn']):"
SPARSE_LITERAL_REPLACEMENT = (
    "def set_attn_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', "
    "'aule']):")

DENSE_CONFIG = f"{CHECKOUT}/trellis2/modules/attention/config.py"
DENSE_CONFIG_NEEDLE = (
    "env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']")
DENSE_CONFIG_REPLACEMENT = (
    "env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive', "
    "'aule']")
DENSE_LITERAL_NEEDLE = "def set_backend(backend: Literal['xformers', 'flash_attn']):"
DENSE_LITERAL_REPLACEMENT = (
    "def set_backend(backend: Literal['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', "
    "'naive', 'aule']):")

AULE_PROOF = "AuleAttention, a Triton flash attention"

# Dense attention: [N, L, H, C] tensors, one unbind per packing, then straight through.
# `aule` takes and returns [N, H, L, C], hence the two permutes.
DENSE_ATTN = f"{CHECKOUT}/trellis2/modules/attention/full_attn.py"
DENSE_ATTN_NEEDLE = (
    "    else:\n"
    "        raise ValueError(f\"Unknown attention module: {config.BACKEND}\")"
)
DENSE_ATTN_REPLACEMENT = (
    f"    elif config.BACKEND == 'aule':\n"
    f"        # {MARKER} {AULE_PROOF}: it needs no CUDA extension to compile, so it is\n"
    "        # the one backend that also runs on ROCm. q/k/v arrive as [N, L, H, C].\n"
    "        from aule import flash_attention_triton\n"
    "        if num_all_args == 1:\n"
    "            q, k, v = qkv.unbind(dim=2)\n"
    "        elif num_all_args == 2:\n"
    "            k, v = kv.unbind(dim=2)\n"
    "        q = q.permute(0, 2, 1, 3)\n"
    "        k = k.permute(0, 2, 1, 3)\n"
    "        v = v.permute(0, 2, 1, 3)              # [N, H, L, C]\n"
    "        out = flash_attention_triton(q, k, v, causal=False)\n"
    "        out = out.permute(0, 2, 1, 3)           # [N, L, H, C]\n"
    "\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown attention module: {config.BACKEND}\")"
)

# Sparse attention: q/k/v arrive packed into one [T, H, C] tensor with a per-sequence
# length list, which is what flash_attn's varlen kernels take. Those have no ROCm build, so
# the branch runs each sequence on its own and stitches the results back together. Same
# attention, done in pieces.
SPARSE_ATTN = f"{CHECKOUT}/trellis2/modules/sparse/attention/full_attn.py"
SPARSE_ATTN_NEEDLE = (
    "    else:\n"
    "        raise ValueError(f\"Unknown attention module: {config.ATTN}\")"
)
SPARSE_ATTN_REPLACEMENT = (
    f"    elif config.ATTN == 'aule':\n"
    f"        # {MARKER} {AULE_PROOF}: no CUDA extension to compile, so it also runs on\n"
    "        # ROCm. There is no varlen kernel to call, so each sequence in the packed\n"
    "        # [T, H, C] tensors is attended on its own and the pieces concatenated.\n"
    "        # q_seqlen and kv_seqlen are plain lists by this point (see above).\n"
    "        from aule import flash_attention_triton\n"
    "        if num_all_args == 1:\n"
    "            q, k, v = qkv.unbind(dim=1)\n"
    "        elif num_all_args == 2:\n"
    "            k, v = kv.unbind(dim=1)\n"
    "        outs, q_off, kv_off = [], 0, 0\n"
    "        for qn, kn in zip(q_seqlen, kv_seqlen):\n"
    "            q_i = q[q_off:q_off + qn].transpose(0, 1).unsqueeze(0)     # [1, H, qn, C]\n"
    "            k_i = k[kv_off:kv_off + kn].transpose(0, 1).unsqueeze(0)   # [1, H, kn, C]\n"
    "            v_i = v[kv_off:kv_off + kn].transpose(0, 1).unsqueeze(0)   # [1, H, kn, Co]\n"
    "            out_i = flash_attention_triton(q_i, k_i, v_i, causal=False)[0]  # [H, qn, Co]\n"
    "            outs.append(out_i.transpose(0, 1))                         # [qn, H, Co]\n"
    "            q_off += qn\n"
    "            kv_off += kn\n"
    "        out = torch.cat(outs, dim=0)   # [T_Q, H, Co]\n"
    "\n"
    "    else:\n"
    "        raise ValueError(f\"Unknown attention module: {config.ATTN}\")"
)


# --- 2. The ROCm device bitcode path -------------------------------------------------------
#
# One anchor, identical in all four forks. The replacement stays inside the list literal the
# anchor sits in, so a statement (and hence a helper function) cannot go here; comments and
# implicit string concatenation can.

DEVICE_LIB_NEEDLE = (
    "f\"--rocm-device-lib-path={os.path.join(_rocm,'lib','llvm','amdgcn','bitcode')}\",")
# Plain concatenation rather than an f-string: a `{...}` expression cannot be split across
# implicitly concatenated f-strings, and the first attempt at this broke all three files.
# Stays inside the list literal the anchor sits in, so a statement cannot go here either.
DEVICE_LIB_REPLACEMENT = (
    "%%INDENT%%%%MARK%% ROCm 7.x keeps the device bitcode in <root>/amdgcn/bitcode;\n"
    "%%INDENT%%# ROCm 10 (TheRock) moved it under lib/llvm. These forks hardcode\n"
    "%%INDENT%%# the ROCm 10 path, so on 7.x the build stops with \"cannot find ROCm\n"
    "%%INDENT%%# device library\". HIP_DEVICE_LIB_PATH (hipcc's own variable) wins,\n"
    "%%INDENT%%# then whichever layout this install actually has.\n"
    "%%INDENT%%\"--rocm-device-lib-path=\" + (\n"
    "%%INDENT%%%%DEEP%%os.environ.get('HIP_DEVICE_LIB_PATH') or next(\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%(p for p in (os.path.join(_rocm,'lib','llvm',"
    "'amdgcn','bitcode'),\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%%%DEEP%%os.path.join(_rocm,'amdgcn','bitcode'))\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%%%DEEP%%if os.path.isdir(p)), None)\n"
    "%%INDENT%%%%DEEP%%# Neither layout: say so here, not as a bare --rocm-device-lib-path=\n"
    "%%INDENT%%%%DEEP%%# and a compile error that never mentions the bitcode. A generator's\n"
    "%%INDENT%%%%DEEP%%# throw() is how an expression raises inside a list literal.\n"
    "%%INDENT%%%%DEEP%%or (_ for _ in ()).throw(RuntimeError(\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%'ROCm device bitcode not found under ' + repr(_rocm) +\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%' (looked in lib/llvm/amdgcn/bitcode and amdgcn/bitcode). '\n"
    "%%INDENT%%%%DEEP%%%%DEEP%%'Set HIP_DEVICE_LIB_PATH to the folder holding ocml.bc.'))),\n"
)
DEVICE_LIB_PROOF = "ROCm 7.x keeps the device bitcode"

# The two extensions this route takes from dmonkman's ROCm ports rather than upstream.
# nvdiffrast because upstream has no HIP path at all; CuMesh because its sources need a
# hipify (cuda:: to hip, cub:: to hipcub::, eight .cu files rewritten) that is machine
# generated and not something to hand-edit here. Both hardcode the ROCm 10 bitcode path.
DEVICE_LIB_FILES = (
    f"{EXTENSIONS}/nvdiffrast/setup.py",
    f"{EXTENSIONS}/CuMesh/setup.py",
)


# Windowed sparse attention has an if/elif for xformers and flash_attn and no else, so any
# other backend dies on an UnboundLocalError (`attn_func_args`, then `out`) that names
# neither the backend nor the file. TRELLIS.2's own models use full attention, so this is
# not on the default path -- which is exactly why it would surface late and confusingly.
# Each gets an `aule` branch done the same way as the full sparse one (each window on its
# own), and an `else: raise` so an unknown backend says what it is.
WINDOWED_ATTN = f"{CHECKOUT}/trellis2/modules/sparse/attention/windowed_attn.py"
UNKNOWN_ATTN = "    else:\n        raise ValueError(f\"Unknown attention module: {config.ATTN}\")\n"

WINDOW_PARTITION_NEEDLE = (
    "            'max_seqlen': torch.max(seq_lens)\n"
    "        }\n"
    "\n"
    "    return fwd_indices, bwd_indices, seq_lens, attn_func_args"
)
WINDOW_PARTITION_PROOF = "aule windows need no packed-kernel arguments"
WINDOW_PARTITION_REPLACEMENT = (
    "            'max_seqlen': torch.max(seq_lens)\n"
    "        }\n"
    "    elif config.ATTN == 'aule':\n"
    f"        # {MARKER} {WINDOW_PARTITION_PROOF}: each is attended on its own.\n"
    "        attn_func_args = {}\n"
    + UNKNOWN_ATTN +
    "\n"
    "    return fwd_indices, bwd_indices, seq_lens, attn_func_args"
)

WINDOW_SELF_NEEDLE = (
    "        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)"
    "  # [M, H, C]\n"
    "\n"
    "    out = out[bwd_indices]      # [T, H, C]"
)
WINDOW_SELF_PROOF = "windowed self attention, one window at a time"
WINDOW_SELF_REPLACEMENT = (
    "        out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv_feats, **attn_func_args)"
    "  # [M, H, C]\n"
    "    elif config.ATTN == 'aule':\n"
    f"        # {MARKER} {WINDOW_SELF_PROOF}, as the full sparse branch does.\n"
    "        from aule import flash_attention_triton\n"
    "        q, k, v = qkv_feats.unbind(dim=1)                                # [M, H, C]\n"
    "        outs, off = [], 0\n"
    "        for n in seq_lens.tolist():\n"
    "            q_i = q[off:off + n].transpose(0, 1).unsqueeze(0)            # [1, H, n, C]\n"
    "            k_i = k[off:off + n].transpose(0, 1).unsqueeze(0)\n"
    "            v_i = v[off:off + n].transpose(0, 1).unsqueeze(0)\n"
    "            out_i = flash_attention_triton(q_i, k_i, v_i, causal=False)[0]  # [H, n, C]\n"
    "            outs.append(out_i.transpose(0, 1))                           # [n, H, C]\n"
    "            off += n\n"
    "        out = torch.cat(outs, dim=0)                                     # [M, H, C]\n"
    + UNKNOWN_ATTN +
    "\n"
    "    out = out[bwd_indices]      # [T, H, C]"
)

WINDOW_CROSS_NEEDLE = (
    "        )  # [M, H, C]\n"
    "\n"
    "    out = out[q_bwd_indices]      # [T, H, C]"
)
WINDOW_CROSS_PROOF = "windowed cross attention, one window pair at a time"
WINDOW_CROSS_REPLACEMENT = (
    "        )  # [M, H, C]\n"
    "    elif config.ATTN == 'aule':\n"
    f"        # {MARKER} {WINDOW_CROSS_PROOF}.\n"
    "        from aule import flash_attention_triton\n"
    "        k, v = kv_feats.unbind(dim=1)                                    # [M, H, C]\n"
    "        outs, q_off, kv_off = [], 0, 0\n"
    "        for qn, kn in zip(q_seq_lens.tolist(), kv_seq_lens.tolist()):\n"
    "            q_i = q_feats[q_off:q_off + qn].transpose(0, 1).unsqueeze(0)  # [1, H, qn, C]\n"
    "            k_i = k[kv_off:kv_off + kn].transpose(0, 1).unsqueeze(0)      # [1, H, kn, C]\n"
    "            v_i = v[kv_off:kv_off + kn].transpose(0, 1).unsqueeze(0)\n"
    "            out_i = flash_attention_triton(q_i, k_i, v_i, causal=False)[0]  # [H, qn, C]\n"
    "            outs.append(out_i.transpose(0, 1))                           # [qn, H, C]\n"
    "            q_off += qn\n"
    "            kv_off += kn\n"
    "        out = torch.cat(outs, dim=0)                                     # [M, H, C]\n"
    + UNKNOWN_ATTN +
    "\n"
    "    out = out[q_bwd_indices]      # [T, H, C]"
)

AULE_PATCHES = (
    Patch(SPARSE_CONFIG, SPARSE_CONFIG_NEEDLE, SPARSE_CONFIG_REPLACEMENT,
          proof="'flash_attn_3', 'aule']"),
    Patch(SPARSE_CONFIG, SPARSE_LITERAL_NEEDLE, SPARSE_LITERAL_REPLACEMENT,
          proof="'flash_attn_3', 'aule']):"),
    Patch(DENSE_CONFIG, DENSE_CONFIG_NEEDLE, DENSE_CONFIG_REPLACEMENT,
          proof="'naive', 'aule']"),
    Patch(DENSE_CONFIG, DENSE_LITERAL_NEEDLE, DENSE_LITERAL_REPLACEMENT,
          proof="'naive', 'aule']):"),
    # These two keep upstream's `else: raise ValueError` in the replacement — an unknown
    # backend must still raise rather than fall back silently — so the anchor survives.
    Patch(DENSE_ATTN, DENSE_ATTN_NEEDLE, DENSE_ATTN_REPLACEMENT, proof=AULE_PROOF,
          needle_survives=True),
    Patch(SPARSE_ATTN, SPARSE_ATTN_NEEDLE, SPARSE_ATTN_REPLACEMENT, proof=AULE_PROOF,
          needle_survives=True),
    Patch(WINDOWED_ATTN, WINDOW_PARTITION_NEEDLE, WINDOW_PARTITION_REPLACEMENT,
          proof=WINDOW_PARTITION_PROOF),
    Patch(WINDOWED_ATTN, WINDOW_SELF_NEEDLE, WINDOW_SELF_REPLACEMENT, proof=WINDOW_SELF_PROOF),
    Patch(WINDOWED_ATTN, WINDOW_CROSS_NEEDLE, WINDOW_CROSS_REPLACEMENT,
          proof=WINDOW_CROSS_PROOF),
)

DEVICE_LIB_PATCHES = tuple(
    Patch(path, DEVICE_LIB_NEEDLE, DEVICE_LIB_REPLACEMENT, proof=DEVICE_LIB_PROOF,
          indent_to_anchor=True)
    for path in DEVICE_LIB_FILES
)

# --- 3. C++20, which torch 2.14's headers require ------------------------------------------
#
# FlexGEMM and o-voxel are pinned at upstream commits that compile with `-std=c++17`, and
# torch 2.14's headers `#error` below C++20: the build fails inside c10/util/intrusive_ptr.h
# on `std::strong_ordering` before it reaches a line of the extension's own. Every community
# ROCm fork fixes this, and for these two it is the *only* thing they change. So it is a
# three-word change here instead of two more third-party checkouts, and these two builds on
# AMD come from the same pinned source as the NVIDIA build.
#
# CuMesh is absent: its ROCm port already compiles as C++20.
CXX17_NEEDLE = "-std=c++17"
CXX17_REPLACEMENT = "-std=c++20"
CXX17_PROOF = "-std=c++20"

# (file, how many occurrences upstream has).
CXX20_FILES = (
    (f"{EXTENSIONS}/FlexGEMM/setup.py", 3),
    (f"{CHECKOUT}/o-voxel/setup.py", 2),
)

CXX20_PATCHES = tuple(
    Patch(path, CXX17_NEEDLE, CXX17_REPLACEMENT, proof=CXX17_PROOF, expected=count)
    for path, count in CXX20_FILES
)

# --- 4. The Python floor the ROCm ports invented --------------------------------------------
#
# Both dmonkman ports declare `requires-python = ">=3.12"`, and uv refuses to install them on
# the 3.11 this route (and every other route here) uses. Upstream declares no floor at all --
# nvdiffrast's v0.4.0 setup.py has no python_requires, and CuMesh's says >=3.8 -- so the floor
# is theirs, and it exists because their published wheels are cp312.
REQUIRES_PYTHON_NEEDLE = 'requires-python = ">=3.12"'
REQUIRES_PYTHON_REPLACEMENT = 'requires-python = ">=3.9"'
REQUIRES_PYTHON_PROOF = 'requires-python = ">=3.9"'

REQUIRES_PYTHON_PATCHES = tuple(
    Patch(path, REQUIRES_PYTHON_NEEDLE, REQUIRES_PYTHON_REPLACEMENT,
          proof=REQUIRES_PYTHON_PROOF)
    for path in (f"{EXTENSIONS}/nvdiffrast/pyproject.toml", f"{EXTENSIONS}/CuMesh/pyproject.toml")
)

PATCHES = AULE_PATCHES + DEVICE_LIB_PATCHES + CXX20_PATCHES + REQUIRES_PYTHON_PATCHES


def anchor_indent(source: str, needle: str) -> str:
    """The whitespace before the anchor on its line."""
    for line in source.splitlines():
        if needle in line:
            return line[: line.index(needle)]
    raise RuntimeError("the anchor for this patch is gone; upstream changed, re-check the "
                       "patch before running it")


def patch_text(source: str, spec: Patch, label: str) -> tuple[str, bool]:
    """Return the patched text and whether it changed.

    The proof is checked *first*, before the anchor, so a second run is a no-op.

    Raises when neither the proof nor the anchor is there, which is the signal that upstream
    moved and the patch wants re-reading before it is trusted again.
    """
    if spec.proof in source:
        return source, False
    if spec.needle not in source:
        raise RuntimeError(f"{label}: the anchor for this patch is gone; upstream changed, "
                           "re-check the patch before running it")
    replacement = spec.replacement
    if spec.indent_to_anchor:
        indent = anchor_indent(source, spec.needle)
        replacement = (replacement.replace("%%MARK%%", MARKER)
                       .replace("%%INDENT%%", indent)
                       .replace("%%DEEP%%", "    "))
    return source.replace(spec.needle, replacement), True


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


def missing_targets(root: Path = DEFAULT_ROOT) -> list[str]:
    """Files this patch expects that are not there. An empty list is a readable "done"."""
    missing = {spec.file for spec in PATCHES if not (root / spec.file).is_file()}
    return sorted(missing)


def is_patched(root: Path = DEFAULT_ROOT) -> bool:
    """True when every patch's proof is in its file and no unpatched anchor is left.

    Both halves matter. The proof alone would pass if upstream's own text happened to
    contain the phrase; the anchor alone would never pass for a patch whose replacement
    still contains it.
    """
    for spec in PATCHES:
        path = root / spec.file
        if not path.is_file():
            return False
        text = path.read_text()
        if spec.proof not in text:
            return False
        if not spec.needle_survives and spec.needle in text:
            return False
    return True


def apply(root: Path = DEFAULT_ROOT) -> list[tuple[Patch, bool]]:
    """Apply every patch. Returns [(patch, changed)], so a caller can report honestly.

    One entry per *patch*, not per file: the two config files take two patches each, and
    keying by filename collapsed those into one line and undercounted the work done.
    """
    missing = missing_targets(root)
    if missing:
        raise RuntimeError(f"{root}: cannot patch, these files are not there: "
                           + ", ".join(missing))
    return [(spec, patch_file(root / spec.file, spec)) for spec in PATCHES]


def describe(root: Path = DEFAULT_ROOT) -> str:
    """One line per patch: present, and whether it is in place yet."""
    lines = []
    for spec in PATCHES:
        path = root / spec.file
        if not path.is_file():
            lines.append(f"  MISSING  {spec.file}")
            continue
        text = path.read_text()
        state = "applied" if spec.proof in text else "NOT applied"
        lines.append(f"  {state:10s} {spec.file}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the trellis-rocm vendor root (holds TRELLIS.2/ and extensions/)")
    parser.add_argument("--check", action="store_true",
                        help="report what is patched and change nothing")
    args = parser.parse_args(argv)

    if args.check:
        print(describe(args.root))
        print("all patches applied" if is_patched(args.root) else "not all patches applied")
        return 0 if is_patched(args.root) else 1

    missing = missing_targets(args.root)
    if missing:
        print(f"Nothing patched. {args.root} does not have: " + ", ".join(missing))
        print("Run scripts/bootstrap_trellis_rocm.py first; it clones and builds them.")
        return 1
    changed = apply(args.root)
    applied = sum(1 for _, was in changed if was)
    print(f"image-to-3dlab TRELLIS.2 ROCm patches: {applied} of {len(changed)} changed")
    for spec, was in changed:
        print(f"  {'patched' if was else 'already':10s} {spec.file} ({spec.proof})")
    if not is_patched(args.root):
        print("FAILED: something did not take. Do not run a generation on this checkout.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())