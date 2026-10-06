#!/usr/bin/env python3
"""Say whether a rigged GLB fits the preset moves (a humanoid), read without Blender.

It reads the skeleton straight from the GLB and asks the same bone mapper the moves use
(kimodo_retarget).

The auto-rig can rig anything; the preset moves only fit a humanoid: two legs and one
spine under the hips, a chest that forks into a neck and two arms. This says so plainly
right after rigging, instead of letting a move fail later.

    python scripts/rig_check.py output/animate/<run>/<name>_rigged.glb
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kimodo_retarget import map_skintokens_to_soma

Y_UP = np.array([0.0, 1.0, 0.0])  # glTF is Y-up

NOT_HUMANOID = ("Rigged. Preset moves are for humanoids only for now: four-legged and other creatures can be "
                "rigged, not animated, and the rig may be wonky. Download it to animate in Blender.")


def _gltf_json(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, _, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB file")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:
        raise ValueError(f"{path} has no JSON chunk first")
    return json.loads(data[20:20 + length])


def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _local(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T  # glTF stores column-major
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(*node.get("rotation", [0, 0, 0, 1])) * np.array(node.get("scale", [1, 1, 1]))
    m[:3, 3] = node.get("translation", [0, 0, 0])
    return m


def skeleton(path: Path) -> tuple[dict[str, str | None], dict[str, np.ndarray]]:
    """({joint: parent joint or None}, {joint: rest position in world space}) of the first skin."""
    doc = _gltf_json(path)
    nodes = doc.get("nodes", [])
    skins = doc.get("skins") or []
    if not skins or not skins[0].get("joints"):
        raise LookupError("no skeleton")
    joints = set(skins[0]["joints"])
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    world: dict[int, np.ndarray] = {}

    def world_of(i: int) -> np.ndarray:
        if i not in world:
            p = parent_of.get(i)
            world[i] = (world_of(p) if p is not None else np.eye(4)) @ _local(nodes[i])
        return world[i]

    def joint_parent(i: int) -> int | None:
        p = parent_of.get(i)
        while p is not None and p not in joints:
            p = parent_of.get(p)
        return p

    name = lambda i: nodes[i].get("name") or f"joint{i}"
    parents = {name(i): (name(p) if (p := joint_parent(i)) is not None else None) for i in joints}
    heads = {name(i): world_of(i)[:3, 3].copy() for i in joints}
    return parents, heads


def check(path: Path) -> dict:
    """{"humanoid": bool, "bones": n, "message": words for a person, "detail": why (for logs)}."""
    try:
        parents, heads = skeleton(path)
    except LookupError:
        return {"humanoid": False, "bones": 0, "message": "This model has no skeleton, so it cannot be animated.",
                "detail": "no skin in the GLB"}
    roots = [b for b, p in parents.items() if p is None]
    if len(roots) != 1:
        return {"humanoid": False, "bones": len(parents), "message": NOT_HUMANOID,
                "detail": f"expected one root bone, found {len(roots)}"}
    try:
        map_skintokens_to_soma(parents, heads, up=Y_UP)
    except (ValueError, KeyError, IndexError, StopIteration) as exc:
        return {"humanoid": False, "bones": len(parents), "message": NOT_HUMANOID, "detail": str(exc)}
    return {"humanoid": True, "bones": len(parents), "message": "Rigged. Ready for moves.", "detail": ""}


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("glb", type=Path)
    result = check(parser.parse_args().glb)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["humanoid"] else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
