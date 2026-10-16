#!/usr/bin/env python3
"""What differs between the TRELLIS.2 routes, in one place, so the two cannot drift.

TRELLIS.2 runs on an NVIDIA card (`trellis_cuda_generate.py`) and on an AMD card
(`trellis_rocm_generate.py`). Everything they share -- the pipeline calls, the demo settings,
the decode and bake, the manifest and provenance shapes -- lives in `trellis_cuda_generate.py`
and is imported by the AMD script, exactly as that script imports from the Mac one.

What is genuinely per-card is small: where the checkout lives, which attention backend has to
be selected before any TRELLIS import, and what the run manifest should say about itself.
That is this dataclass, and nothing else.

The one difference worth reading twice: the AMD route sets no allocator configuration. On
ROCm, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` produced silent NaNs rather than an
error, so it is deliberately absent there rather than forgotten.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Route:
    """One card's worth of differences. Everything else is shared code."""

    #: Prefix for this route's progress lines, e.g. "[trellis-rocm] matted with ...".
    name: str
    #: The script a manifest should name as its generator.
    generator: str
    #: The vendor root `bootstrap_*.py` builds into.
    vendor: Path
    #: The bootstrap a user is told to run.
    bootstrap: str
    #: Pinned *before* any TRELLIS import: upstream reads these at import time.
    backend_env: dict[str, str]
    #: Other environment this route needs, or deliberately does not.
    extra_env: dict[str, str] = field(default_factory=dict)
    #: What torch calls the device on this stack. ROCm's torch says "cuda" too, which is
    #: correct and not a copy-paste slip: that is the API name both vendors implement.
    device: str = "cuda"
    #: The manifest's description of where the model came from.
    port: str = ""
    #: The provenance record's route id, which sorts output folders.
    provenance_route: str = ""
    #: Written by the bootstrap when the build finished.
    marker_name: str = ".i2l-build-complete"

    def environ(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Set this route's variables, and report what it set.

        Hard-assigned, never inherited: a stale value in the viewer's environment once
        selected a backend that did not exist.
        """
        env = dict(os.environ if base is None else base)
        env.update(self.backend_env)
        env.update(self.extra_env)
        # Upstream's example.py and app.py set both.
        env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        return env


CUDA = Route(
    name="trellis-cuda",
    generator="trellis_cuda_generate.py",
    vendor=REPO / "vendor" / "trellis-cuda",
    bootstrap="scripts/bootstrap_trellis_cuda.py",
    backend_env={
        "ATTN_BACKEND": "flash_attn",
        "SPARSE_ATTN_BACKEND": "flash_attn",
        "SPARSE_CONV_BACKEND": "flex_gemm",
    },
    extra_env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    port="microsoft/TRELLIS.2 (official, CUDA)",
    provenance_route="nvidia-cuda",
)

ROCM = Route(
    name="trellis-rocm",
    generator="trellis_rocm_generate.py",
    vendor=REPO / "vendor" / "trellis-rocm",
    bootstrap="scripts/bootstrap_trellis_rocm.py",
    backend_env={
        # AuleAttention, not flash-attn: a Triton flash attention with no CUDA extension to
        # compile. See scripts/patch_trellis_rocm.py for the dispatch that selects it.
        "ATTN_BACKEND": "aule",
        "SPARSE_ATTN_BACKEND": "aule",
        "SPARSE_CONV_BACKEND": "flex_gemm",
    },
    extra_env={},  # no expandable_segments: silent NaNs on ROCm, not a crash
    port="microsoft/TRELLIS.2 (official, ROCm)",
    provenance_route="amd-rocm",
)