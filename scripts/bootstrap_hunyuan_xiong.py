#!/usr/bin/env python3
"""Set up Hunyuan3D-MLX (Xiong, full pipeline): both Python environments, then the weights.

    python scripts/bootstrap_hunyuan_xiong.py          # says what it will do, then asks
    python scripts/bootstrap_hunyuan_xiong.py --yes    # non-interactive (Setup & Status)

Shape and paint each have their own venv (`hunyuan_mlx/shape`, `hunyuan_mlx/paint`). Setup
used to run only the weight download, so a machine with the shape venv but no paint venv
finished "done" and still showed "not installed" (issue #97). Each venv is made with
`uv sync` only when it is missing; then the default route's weights are fetched.

Nothing is fetched without an explicit yes: AGENTS.md forbids weight downloads the user
has not chosen. The Hunyuan weights are not licensed in the EU, the UK or South Korea.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HALVES = ("shape", "paint")
WEIGHTS_GB = 13.7  # Hunyuan3D-2 shape (5.0) + Xiong's paint-large (8.7), the catalogue's figures


def venv_python(root: Path) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def plan(repo: Path, uv: str) -> list[tuple[list[str], Path]]:
    """(command, working directory) for every step still needed, in order."""
    steps = [([uv, "sync"], repo / "hunyuan_mlx" / half)
             for half in HALVES if not venv_python(repo / "hunyuan_mlx" / half).exists()]
    # Explicitly the default route: without --model this fetches all three shape
    # checkpoints, 23 GB where the default route needs 5.
    steps.append(([str(venv_python(repo / "hunyuan_mlx" / "shape")),
                   str(repo / "hunyuan_mlx" / "download_weights.py"), "--model", "2.0"], repo))
    return steps


def announcement(steps: list[tuple[list[str], Path]]) -> str:
    builds = [f"hunyuan_mlx/{cwd.name}" for cmd, cwd in steps if cmd[1:] == ["sync"]]
    lines = ["Hunyuan3D-MLX (Xiong, full pipeline), default route (shape 2.0 + PBR paint)"]
    if builds:
        lines.append(f"  build   uv sync in {', '.join(builds)}")
    lines.append(f"  fetch   about {WEIGHTS_GB} GB of weights from Hugging Face (skips what is there)")
    lines.append("  licence Tencent Hunyuan Community License: not licensed in the EU, UK or South Korea")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--yes", action="store_true", help="do not ask (the caller already did)")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv is required: https://docs.astral.sh/uv/")
    steps = plan(REPO, uv)
    print(announcement(steps), flush=True)
    if not args.yes and input("Go ahead? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise SystemExit("Nothing done.")
    for command, cwd in steps:
        print(f"$ {' '.join(command)}  (in {cwd})", flush=True)
        subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
