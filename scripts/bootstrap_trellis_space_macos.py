#!/usr/bin/env python3
"""Bootstrap TRELLIS.2 on macOS from Microsoft's pinned Space source.

No Shiv source tree is cloned or installed.  Pedro's pinned repositories supply
Metal operators, then image-to-3dlab's audited core/backend patches are replayed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "audit/trellis-port/upstreams.lock.json"
REQUIREMENTS = REPO / "audit/trellis-port/requirements-macos.in"
DEFAULT_ROOT = REPO / "vendor/trellis-space-mac"

TARGETS = {
    "microsoft_hf_space": "TRELLIS.2",
    "utils3d": "deps/utils3d",
    "pedro_mtlbvh": "deps/mtlbvh",
    "pedro_mtldiffrast": "deps/mtldiffrast",
    "pedro_mtlgemm": "deps/mtlgemm",
    "pedro_mtlmesh": "deps/mtlmesh",
    "pedro_trellis2_apple": "deps/trellis2-apple",
}


def run(*args: str | Path, cwd: Path | None = None, env: dict | None = None) -> None:
    command = [str(arg) for arg in args]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def output(*args: str | Path, cwd: Path | None = None) -> str:
    return subprocess.check_output([str(arg) for arg in args], cwd=cwd, text=True).strip()


def clone_at(url: str, commit: str, target: Path) -> None:
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", url, target)
    head = output("git", "rev-parse", "HEAD", cwd=target)
    if head != commit:
        # Never reset a dirty source tree.  A fresh/clean tree may move to the pin.
        if output("git", "status", "--short", cwd=target):
            raise RuntimeError(f"{target} is dirty at {head}; expected {commit}")
        run("git", "fetch", "origin", commit, cwd=target)
        run("git", "checkout", "--detach", commit, cwd=target)
    actual = output("git", "rev-parse", "HEAD", cwd=target)
    if actual != commit:
        raise RuntimeError(f"failed to pin {target}: {actual} != {commit}")


def apply_patches(root: Path) -> None:
    run(
        sys.executable,
        REPO / "scripts/patch_trellis_space_core.py",
        "--root",
        root / "TRELLIS.2",
    )
    run(
        sys.executable,
        REPO / "scripts/patch_trellis_metal_backends.py",
        "--root",
        root,
    )


def install(root: Path, python: str) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the reproducible Mac environment")
    venv = root / ".venv"
    if not (venv / "bin/python").is_file():
        run(uv, "venv", venv, "--python", python)

    env = os.environ.copy()
    env.setdefault("MACOSX_DEPLOYMENT_TARGET", "12.0")
    if Path("/Applications/Xcode.app/Contents/Developer").is_dir():
        env.setdefault("DEVELOPER_DIR", "/Applications/Xcode.app/Contents/Developer")

    interpreter = venv / "bin/python"
    run(uv, "pip", "install", "--python", interpreter, "-r", REQUIREMENTS, env=env)
    run(uv, "pip", "install", "--python", interpreter, root / "deps/utils3d", env=env)
    for dependency in (
        "mtlbvh",
        "mtldiffrast",
        "mtlmesh",
        "mtlgemm",
    ):
        run(
            uv,
            "pip",
            "install",
            "--python",
            interpreter,
            "--no-build-isolation",
            root / "deps" / dependency,
            env=env,
        )
    run(
        uv,
        "pip",
        "install",
        "--python",
        interpreter,
        "--no-build-isolation",
        root / "deps/trellis2-apple/o-voxel",
        env=env,
    )


def bootstrap(root: Path, *, skip_install: bool, python: str) -> None:
    lock = json.loads(LOCK.read_text())["upstreams"]
    for name, relative in TARGETS.items():
        spec = lock[name]
        clone_at(spec["url"], spec["commit"], root / relative)
    apply_patches(root)
    if not skip_install:
        install(root, python)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    bootstrap(args.root.resolve(), skip_install=args.skip_install, python=args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
