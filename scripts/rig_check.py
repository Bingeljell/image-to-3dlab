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


FOOT_BAND = (0.02, 0.12)  # the slice the feet are looked for in: heights above the lowest point, shares of the height
FOOT_CELL = 0.015         # the footprint grid, a share of the height: gaps up to two cells (3 %) close, feet stand 5-14 % apart
FOOT_MIN_SHARE = 0.03     # a part with fewer of the slice's points is a crumb, not a foot
FOOT_MAX_SPAN = 0.45      # a foot is at most this share of the height across (an overseer's long feet 0.41; fused sides 0.8+)
FOOT_TWO_SHARE = 0.80     # the two biggest parts hold at least this share of the slice: a golem's knuckles beside its
                          # feet hold 10 %, a wolf's four paws split it four ways (the two biggest 59 %)
_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}


def mesh_points(path: Path) -> np.ndarray:
    """Every mesh's vertex positions in a GLB, world space (Y up), as one (N, 3) array.

    A skinned mesh is left where its vertices are, as glTF says its node's own transform does not apply."""
    data = Path(path).read_bytes()
    doc = _gltf_json(path)
    length = struct.unpack_from("<I", data, 12)[0]
    at = 20 + length
    binary = b""
    if at + 8 <= len(data):
        size, kind = struct.unpack_from("<II", data, at)
        if kind == 0x004E4942:
            binary = data[at + 8:at + 8 + size]
    nodes = doc.get("nodes", [])
    parent_of = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def world_of(i: int) -> np.ndarray:
        p = parent_of.get(i)
        return (world_of(p) if p is not None else np.eye(4)) @ _local(nodes[i])

    out = []
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        m = np.eye(4) if "skin" in node else world_of(i)
        for prim in doc["meshes"][node["mesh"]].get("primitives", []):
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            view = doc["bufferViews"][acc["bufferView"]]
            dtype = np.dtype(_COMPONENTS[acc["componentType"]])
            start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
            stride = view.get("byteStride") or 3 * dtype.itemsize
            raw = np.frombuffer(binary, np.uint8, count=stride * (acc["count"] - 1) + 3 * dtype.itemsize, offset=start)
            pts = np.lib.stride_tricks.as_strided(raw, (acc["count"], 3 * dtype.itemsize), (stride, 1))
            xyz = pts.copy().view(dtype).reshape(-1, 3).astype(np.float64)
            out.append(xyz @ m[:3, :3].T + m[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def _parts(grid: np.ndarray) -> np.ndarray:
    """Label the 8-connected parts of a boolean grid (0 = empty, 1.. = part)."""
    labels = np.zeros(grid.shape, int)
    count = 0
    for start in zip(*np.nonzero(grid)):
        if labels[start]:
            continue
        count += 1
        labels[start] = count
        todo = [start]
        while todo:
            i, j = todo.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if 0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] and grid[a, b] and not labels[a, b]:
                        labels[a, b] = count
                        todo.append((a, b))
    return labels


def stands_on_two_legs(points: np.ndarray) -> dict:
    """{"two_legs": bool, "feet": parts found, "detail": why}: does the model stand on two separate feet?

    The auto-rig is not deterministic: now and then it hangs one leg off the spine and leaves the bones
    unnamed, and the same model rigs fine the next time (two of seven of ArcherSoul's bosses on 2026-11-24).
    Rigging again is only worth it for a model that is meant to be a humanoid, so this looks at the mesh
    itself: a slice just above the floor, gridded from above, is mostly two parts, each no longer than a
    foot (a golem's knuckles may touch the floor beside them). A robe to the floor, feet together or a crate
    give one part, a four-legged creature four of a size (or two long ones when each side's legs touch), a
    bird's toes many small ones, so none of them is rigged twice. Measured on ArcherSoul's figures: all 31
    heroes and all 21 boss builds pass; of 12 pets only the two ghosts, of 12 birds the three that stand on
    two legs; no slime, cloud or prop."""
    p = np.asarray(points, float).reshape(-1, 3)
    if len(p) < 50:
        return {"two_legs": False, "feet": 0, "detail": "too few points"}
    y = p[:, 1]
    lo, height = float(y.min()), float(np.ptp(y))
    if height <= 0:
        return {"two_legs": False, "feet": 0, "detail": "flat"}
    band = p[(y >= lo + FOOT_BAND[0] * height) & (y <= lo + FOOT_BAND[1] * height)][:, [0, 2]]
    if len(band) < 10:
        return {"two_legs": False, "feet": 0, "detail": "nothing near the floor"}
    cell = FOOT_CELL * height
    ij = np.floor((band - band.min(0)) / cell).astype(int) + 1
    grid = np.zeros(ij.max(0) + 2, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    grown = grid.copy()   # one cell round each, so a sparse ring of vertices reads as one part
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            grown |= np.roll(np.roll(grid, di, 0), dj, 1)
    labels = _parts(grown)
    of_point = labels[ij[:, 0], ij[:, 1]]
    parts = []   # (points, span as a share of the height), biggest first
    for k in range(1, labels.max() + 1):
        mine = band[of_point == k]
        if len(mine) >= FOOT_MIN_SHARE * len(band):
            parts.append((len(mine), float(np.ptp(mine, axis=0).max()) / height))
    parts.sort(reverse=True)
    feet = parts[:2]
    share = sum(n for n, _ in feet) / len(band)
    two = len(feet) == 2 and share >= FOOT_TWO_SHARE and max(s for _, s in feet) <= FOOT_MAX_SPAN
    return {"two_legs": two, "feet": len(parts),
            "detail": f"{len(parts)} part(s) near the floor, the two biggest hold {share:.0%} and are at most "
                      f"{max((s for _, s in feet), default=0):.2f} of the height across"}


def check(path: Path) -> dict:
    """{"humanoid": bool, "bones": n, "message": words for a person, "detail": why (for logs)},
    plus "forward" (glTF space, Y up) for a humanoid: the way its feet point."""
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
        _, frame = map_skintokens_to_soma(parents, heads, up=Y_UP)
    except (ValueError, KeyError, IndexError, StopIteration) as exc:
        return {"humanoid": False, "bones": len(parents), "message": NOT_HUMANOID, "detail": str(exc)}
    # which way the feet point, on the ground: viewers turn the character to face the camera
    forward = [round(float(v), 4) for v in frame[:, 2]]
    return {"humanoid": True, "bones": len(parents), "message": "Rigged. Ready for moves.", "detail": "",
            "forward": forward}


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("glb", type=Path)
    result = check(parser.parse_args().glb)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["humanoid"] else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
