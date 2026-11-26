"""Select Pixal3D checkpoints and enforce the background-remover licence guard."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ENV_VAR = "PIXAL3D_SKIP_MODELS"
LOW_VRAM_ENV = "PIXAL3D_LOW_VRAM"
SUBSTITUTE = "ZhengPeng7/BiRefNet"
FORBIDDEN_PREFIX = "briaai/"
HELPER_NAME = "_assetfurnace_pixal3d_compat.py"


def remaining_models(names: list[str], skip: str, *, announce: bool = False) -> list[str]:
    """The models left after applying a comma-separated skip list."""
    dropped = {name.strip() for name in skip.split(",") if name.strip()}
    if not dropped:
        return names
    if announce:
        print(f"[i2l_model_subset] not loading: {sorted(dropped)}", flush=True)
    return [name for name in names if name not in dropped]


def model_names_for_run(names: list[str]) -> list[str]:
    """Drop selected checkpoints at pipeline import, before any weights are read."""
    return remaining_models(names, os.environ.get(ENV_VAR, ""), announce=True)



def low_vram_requested(args_value: bool, env_value: str | None) -> bool:
    """Preserve the explicit option and recognize the environment's off values."""
    return args_value or (env_value is not None and env_value.strip() not in ("", "0", "false", "False"))


def low_vram_for_run(args) -> bool:
    return low_vram_requested(getattr(args, "low_vram", False), os.environ.get(LOW_VRAM_ENV))

def permissive_model_name(model_name: str) -> str:
    """The model the loader may use, given the identity named in its config."""
    return SUBSTITUTE if model_name.startswith(FORBIDDEN_PREFIX) else model_name


def guard_rembg_model(model_name: str) -> str:
    """Substitute the permissive remover before invoking the vendor model loader."""
    approved = permissive_model_name(model_name)
    if approved != model_name:
        print(f"[i2l_rembg_licence] refusing {model_name}; using {approved} instead", flush=True)
    return approved


def helper_matches(source: Path, target: Path) -> bool:
    try:
        return target.with_name(HELPER_NAME).read_bytes() == source.read_bytes()
    except OSError:
        return False


def install_helper(source: Path, target: Path) -> bool:
    if helper_matches(source, target):
        return False
    target.with_name(HELPER_NAME).write_bytes(source.read_bytes())
    return True


def replace_payload(source: str, anchor: str, adapter: str, marker: str,
                    legacy_end: str, legacy_digest: str, error: str) -> str:
    """Accept one audited anchor or migrate the exact previous inline payload."""
    if source.count(anchor) != 1:
        raise SystemExit(error)
    if anchor + adapter in source:
        return source
    before, _, tail = source.partition(anchor)
    if marker in source:
        old, end, rest = tail.partition(legacy_end)
        if not end or hashlib.sha256((old + end).encode()).hexdigest() != legacy_digest:
            raise SystemExit(error)
        tail = rest
    return before + anchor + adapter + tail
