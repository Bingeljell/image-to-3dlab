"""Locate the device bitcode directory for the ROCm extension builds."""

from __future__ import annotations

import os
from pathlib import Path


def device_lib_path(rocm: str) -> str:
    override = os.environ.get("HIP_DEVICE_LIB_PATH")
    if override:
        return override
    root = Path(rocm)
    locations = (root / "lib/llvm/amdgcn/bitcode", root / "amdgcn/bitcode")
    for folder in locations:
        if folder.is_dir():
            return str(folder)
    raise RuntimeError(
        f"ROCm device bitcode not found under {rocm!r} "
        "(looked in lib/llvm/amdgcn/bitcode and amdgcn/bitcode). "
        "Set HIP_DEVICE_LIB_PATH to the folder holding ocml.bc."
    )
