#!/usr/bin/env python3
"""Give a rigged humanoid GLB's bones an engine's names (Unity Humanoid), without Blender.

The Rig step does this itself when "Bone names" is set to Unity Humanoid; this is the same for a
model rigged before, or rigged elsewhere with SkinTokens' `bone_N` names. Body bones get the
engine's names, hair, capes and gear keep theirs, and no vertex keeps more than four weights.
See `image_to_3dlab/bone_names.py`.

    python scripts/name_bones.py output/animate/<run>/<name>_rigged.glb named.glb
    python scripts/name_bones.py output/animate/<run>/<name>_idle.glb named_idle.glb --preset unity
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_to_3dlab.bone_names import PRESETS, describe, name_bones


def main() -> int:  # pragma: no cover - thin CLI over the tested module
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="rigged (or animated) GLB")
    parser.add_argument("output", type=Path, help="GLB to write with engine bone names")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="unity",
                        help="whose names to use (default: unity, Unity's Humanoid)")
    args = parser.parse_args()
    if args.output.resolve() != args.source.resolve():
        shutil.copyfile(args.source, args.output)
    try:
        report = name_bones(args.output, args.preset)
    except LookupError as exc:
        raise SystemExit(f"{args.source.name}: {exc}") from exc
    print(describe(report))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
