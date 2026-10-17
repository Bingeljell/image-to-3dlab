#!/usr/bin/env python3
"""Bootstrap TRELLIS.2 on macOS from Microsoft's pinned Space source.

No Shiv source tree is cloned or installed.  Pedro's pinned repositories supply
Metal operators, then image-to-3dlab's audited core/backend patches are replayed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host  # noqa: E402, RUF100

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

#: The bootstrap to run instead, per non-Apple card. Someone who runs this on the wrong
#: machine should be told what to run, not left with a compiler error about `xcrun`.
ELSEWHERE = {
    host.NVIDIA: "scripts/bootstrap_trellis_cuda.py",
    host.AMD: "scripts/bootstrap_trellis_rocm.py",
}


def wrong_machine(build_target: str | None = None, card: str | None = None,
                   family: str | None = None) -> str | None:
    """Why this script cannot run here, or None when it can.

    Everything below `install()` builds Metal operators through `xcrun`, which exists only
    on macOS. Without this check the script clones four repositories and builds one
    dependency before failing, so the failure arrives minutes in and reads as a broken
    checkout rather than a machine that cannot host it.
    """
    target = host.build_target() if build_target is None else build_target
    if target == "macos-arm64":
        return None
    card = host.host_platform() if card is None else card
    family = host.os_family() if family is None else family
    reason = ("This is the Apple Silicon port: it builds Metal operators with xcrun, which "
              "exists only on macOS.")
    script = ELSEWHERE.get(card)
    if script:
        return f"{reason} On this {'AMD' if card == host.AMD else 'NVIDIA'} card use:\n    python {script}"
    return f"{reason} This machine is {family}, and TRELLIS.2 has no route for it. Nothing downloaded."


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
    problem = wrong_machine()
    if problem:
        print(problem)
        return 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    bootstrap(args.root.resolve(), skip_install=args.skip_install, python=args.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
