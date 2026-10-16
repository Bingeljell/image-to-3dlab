#!/usr/bin/env python3
"""Let Stable Fast 3D compile its GPU texture baker on an AMD card.

`texture_baker/setup.py` decides whether to build the GPU kernels with

    use_cuda = use_cuda and CUDA_HOME is not None

but `CUDA_HOME` comes from `torch.utils.cpp_extension`, and on a ROCm build of PyTorch it
is **None** -- ROCm puts `ROCM_HOME` there instead. So the line goes false, `baker_kernel.cu`
is left out of the source list, and the build succeeds while quietly producing a CPU-only
baker. Nothing errors; the extension simply has no GPU kernels in it.

Everything else upstream needed is already there: `setup.py` has an `is_hip_extension`
branch that skips `cudart` and `c10_cuda`, and `torch.utils.cpp_extension.CUDAExtension`
dispatches to `hipcc` on its own when `torch.version.hip` is set. So this is the whole port.

Safe to run twice: a patched file is recognised and left alone, and the anchor is asserted
so an upstream change fails loudly rather than half-applying.

    python scripts/patch_sf3d_linux.py
    python scripts/patch_sf3d_linux.py --root <stable-fast-3d checkout>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "vendor" / "stable-fast-3d"


@dataclass(frozen=True)
class Patch:
    """One anchored replacement, and a phrase that appears only once it is in place.

    The proof, not the anchor's absence, is what says "patched": a second patch may keep
    its own anchor in its replacement, and then "is the anchor gone?" is the wrong
    question. See `patch_hunyuan_rocm.Patch` for the same shape.
    """

    file: str
    needle: str
    replacement: str
    proof: str
    expected: int = 1


# The bug. CUDA_HOME is None on ROCm, so the GPU kernels are dropped from the build.
BAKER_FILE = "texture_baker/setup.py"
BAKER_NEEDLE = "    use_cuda = use_cuda and CUDA_HOME is not None"
BAKER_REPLACEMENT = (
    "    # image-to-3dlab: a ROCm build of torch leaves CUDA_HOME None and puts ROCM_HOME in\n"
    "    # its place, so this silently dropped baker_kernel.cu and built a CPU-only baker.\n"
    "    use_cuda = use_cuda and (CUDA_HOME is not None or torch.version.hip is not None)"
)

# `rembg[gpu]` pulls onnxruntime-gpu, which is a CUDA wheel and reports no device on an
# AMD card, so the CPU wheel goes in its place.
#
# And a newer rembg than upstream's 2.0.57. image_to_3dlab/sf3d_backend.py cuts the subject
# out with image_to_3dlab/matte.py, which asks rembg for BiRefNet-lite -- and 2.0.57 has no
# BiRefNet session, so it quietly handed back u2net (downloaded unasked, and worse on thin
# parts) while the provenance said BiRefNet. 2.0.69 is what requirements.txt uses; [cpu]
# because from 2.0.5x on plain rembg no longer brings onnxruntime. Checked to resolve
# against SF3D's numpy 1.26.4.
REMBG_PIN = "rembg[cpu]==2.0.69; sys_platform != 'darwin'"
REQUIREMENTS_FILE = "requirements.txt"
REQUIREMENTS_NEEDLE = "rembg[gpu]==2.0.57; sys_platform != 'darwin'"
REQUIREMENTS_REPLACEMENT = (
    "# image-to-3dlab: plain rembg, not [gpu]: onnxruntime-gpu is a CUDA wheel and cannot\n"
    "# load on an AMD card. 2.0.69 for BiRefNet, which image_to_3dlab/matte.py asks for.\n"
    + REMBG_PIN
)
# A checkout patched before the version bump has the comment (so the patch above calls it
# done) but still pins 2.0.57. This moves that line on.
OLD_PATCHED_PIN = "rembg==2.0.57; sys_platform != 'darwin'"

PATCHES = (
    Patch(BAKER_FILE, BAKER_NEEDLE, BAKER_REPLACEMENT,
          proof="or torch.version.hip is not None"),
    Patch(REQUIREMENTS_FILE, REQUIREMENTS_NEEDLE, REQUIREMENTS_REPLACEMENT,
          proof="plain rembg, not [gpu]"),
    Patch(REQUIREMENTS_FILE, OLD_PATCHED_PIN, REMBG_PIN, proof=REMBG_PIN),
)


def patch_text(source: str, spec: Patch, label: str) -> tuple[str, bool]:
    """Return the patched text and whether it changed.

    The proof is checked first: the baker patch keeps its anchor inside its replacement, so
    testing the anchor first would make it re-apply on every run.
    """
    if spec.proof in source:
        return source, False
    if spec.needle not in source:
        raise RuntimeError(f"{label}: the anchor for this patch is gone; upstream changed, "
                           "re-check the patch before running it")
    return source.replace(spec.needle, spec.replacement), True


def patch_file(path: Path, spec: Patch) -> bool:
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
    """True when every patch's proof is in its file."""
    try:
        for spec in PATCHES:
            if spec.proof not in (root / spec.file).read_text():
                return False
    except OSError:
        return False
    return True


def patch(root: Path = DEFAULT_ROOT) -> int:
    """Apply every patch. Returns how many files changed, and is safe to run twice."""
    if not root.is_dir():
        raise SystemExit(f"No stable-fast-3d checkout at {root}. Run the bootstrap first.")
    changed: set[str] = set()
    for spec in PATCHES:
        path = root / spec.file
        if not path.is_file():
            raise SystemExit(f"{path} is missing; this is not a stable-fast-3d checkout.")
        if patch_file(path, spec):
            changed.add(spec.file)
    return len(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="the stable-fast-3d checkout (default: vendor/stable-fast-3d)")
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