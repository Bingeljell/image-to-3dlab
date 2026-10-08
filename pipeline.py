#!/usr/bin/env python3
"""Image-to-3D Lab command-line entry point."""

# ✝︎ b'tzelem Elohim

# These must be set before torch/SF3D is imported.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# ROCm's caching allocator walks past a 2 GB allocation on some cards and dies inside
# PyTorch rather than in the kernel, where the split size is the fix. Only a default: a
# user who knows their card needs something else keeps their own value.
# HSA_OVERRIDE_GFX_VERSION is deliberately not set: it tells the driver to pretend a card
# is a different one, which is a last resort for a card ROCm's runtime does not know, and
# a wrong guess produces kernels the hardware cannot run. Set it yourself if you need it.
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128")

from image_to_3dlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())