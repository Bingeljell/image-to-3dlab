#!/usr/bin/env python3
"""Image-to-3D Lab command-line entry point."""

# This must be set before torch/SF3D is imported.
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from image_to_3dlab.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
