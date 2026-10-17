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

#: Every route's vendor folder name. A weights tree under either serves both routes, since
#: they load the same files; viewer/backend_catalog.py counts the same list.
ROUTE_DIRS = ("trellis-cuda", "trellis-rocm")
LOCAL_MODEL_DIR = "TRELLIS.2-4B"


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
    #: Variables this route must not inherit from the shell that started it.
    unset_env: tuple[str, ...] = ()
    #: What torch calls the device on this stack. ROCm's torch says "cuda" too, which is
    #: correct and not a copy-paste slip: that is the API name both vendors implement.
    device: str = "cuda"
    #: The manifest's description of where the model came from.
    port: str = ""
    #: The provenance record's route id, which sorts output folders.
    provenance_route: str = ""
    #: Written by the bootstrap when the build finished.
    marker_name: str = ".i2l-build-complete"
    #: The Hugging Face repo `pipeline.from_pretrained` is given. Upstream loads it by id
    #: and falls back to a local directory when one holds `pipeline.json`, so this can be
    #: pointed at a directory of the same layout.
    model_repo: str = "microsoft/TRELLIS.2-4B"

    def model_path(self) -> str:
        """Where to load the weights from: a local directory if there is one, else the repo.

        Two ways to have a local one. `I2L_TRELLIS_MODEL_PATH` names it outright; otherwise
        this route's own `vendor/<route>/models/TRELLIS.2-4B` is used when it exists, then the
        other route's. The override exists for machines whose weights are somewhere else
        entirely -- a shared store, an offline box -- where re-downloading 16 GB is not on.
        The default is a convention, not a requirement: the Hugging Face cache works as it
        always did.

        The Setup card counts the same places (`viewer/backend_catalog.py`), so a route
        whose weights are local does not read as partly downloaded on the machine that
        built it.
        """
        named = os.environ.get("I2L_TRELLIS_MODEL_PATH")
        if named:
            return named
        for local in self.local_trees():
            if (local / "pipeline.json").is_file():
                return str(local)
        return self.model_repo

    def local_trees(self) -> list[Path]:
        """This route's own weights folder, then the other route's.

        The card counts both, so the run must too: otherwise a machine whose weights sit
        under the other route reads "present" and then downloads 16 GB anyway.
        """
        others = [self.vendor.parent / name for name in ROUTE_DIRS if name != self.vendor.name]
        return [root / "models" / LOCAL_MODEL_DIR for root in [self.vendor, *others]]

    def environ(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Set this route's variables, and report what it set.

        With no `base` this writes through to `os.environ`, which is the point: TRELLIS reads
        these at import time in a module loaded later, so a copy that nothing points at
        changes nothing. An explicit `base` is copied instead, so a caller passing a dict
        keeps ownership of it.

        The backends are hard-assigned, never inherited: a stale value in the viewer's
        environment once selected a backend that did not exist.
        """
        env = os.environ if base is None else dict(base)
        env.update(self.backend_env)
        # Defaults, not rules: a value the user set for their own card is kept.
        for key, value in self.extra_env.items():
            env.setdefault(key, value)
        for key in self.unset_env:
            env.pop(key, None)
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
    # And not inherited either: a shell that exports it for an NVIDIA habit would bring
    # the same NaNs back with nothing in this repo setting it.
    unset_env=("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF"),
    port="microsoft/TRELLIS.2 (official, ROCm)",
    provenance_route="amd-rocm",
)