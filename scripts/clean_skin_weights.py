#!/usr/bin/env python3
"""Drop skin weights that come from bones far away in the skeleton, then rebalance.

    <python with bpy> scripts/clean_skin_weights.py --in rigged.glb --out rigged.glb

Auto-riggers sometimes give a vertex a share of a bone on the other side of the body: a
generated martial artist's right fingertips carried weight from his right toe. At rest
nothing shows; once the hand rises and the toe stays down, those vertices stretch between
the two into spikes. Real blends sit close together in the tree (finger to finger, arm to
chest, spine to leg are all within a few hops), so a weight whose bone is more than
`MAX_HOPS` bones from the vertex's main bone is an error, not a blend, unless that bone
also sits right beside the vertex: a hand hanging against the thigh is nearer the thigh
bone than its own, and dropping its hand weight glued it to the leg.

A point above the head joint whose main bone is an arm or a hand is given wholly to the head.
A fingertip left inside an ear is nearer that finger than the head, so the hop rule
keeps it and a walk drags the ear out. A real hand is not above the head, so it stays,
and the shoulder stays a shoulder. A body that is not two-legged is left as the hop clean
left it.

The rule is pure (`tree_hops`, `prune_weights`, `retie_head_weight`); only `main()` needs `bpy`.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

# Finger tip to a neighbouring finger's tip is 4 hops through the knuckles and the hand.
MAX_HOPS = 4
# A bone this close to a vertex (beyond its nearest bone), as a share of the skeleton's
# height, is never dropped however far away it is in the tree: a hand beside the thigh
# stays a hand, while a toe is half the body away from a fingertip.
CLOSE_SHARE = 0.05


def tree_hops(parents: dict[str, str | None]) -> dict[str, dict[str, int]]:
    """Bones apart, for every pair, walking the tree in both directions."""
    links: dict[str, list[str]] = {b: [] for b in parents}
    for bone, parent in parents.items():
        if parent is not None:
            links[bone].append(parent)
            links[parent].append(bone)
    hops = {}
    for start in parents:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            bone = queue.popleft()
            for nxt in links[bone]:
                if nxt not in seen:
                    seen[nxt] = seen[bone] + 1
                    queue.append(nxt)
        hops[start] = seen
    return hops


def _segment_distance(point, head, tail) -> float:
    import numpy as np

    axis = tail - head
    length2 = float(axis @ axis)
    t = 0.0 if length2 == 0 else min(1.0, max(0.0, float((point - head) @ axis) / length2))
    return float(np.linalg.norm(point - (head + t * axis)))


def nearest_bone(point, segments: dict) -> str:
    """The bone whose head-to-tail segment passes closest to `point` (numpy 3-vectors)."""
    return min(segments, key=lambda bone: _segment_distance(point, *segments[bone]))


def close_bones(point, segments: dict, slack: float) -> set[str]:
    """Bones whose segment passes within `slack` of the nearest one's distance to `point`."""
    distances = {bone: _segment_distance(point, *ends) for bone, ends in segments.items()}
    nearest = min(distances.values())
    return {bone for bone, d in distances.items() if d <= nearest + slack}


def prune_weights(weights: dict[str, float], hops: dict[str, dict[str, int]], anchor: str,
                  max_hops: int = MAX_HOPS, close: set[str] = frozenset()) -> dict[str, float]:
    """One vertex's {bone: weight}, without bones far from `anchor`, the bone the vertex
    sits nearest to. Where a vertex is decides, not its biggest weight: some fingertips
    were weighted wholly to the toe. Bones in `close` (right beside the vertex) are always
    kept. Unchanged when nothing is dropped; renormalised to the original total otherwise;
    all of it to `anchor` when nothing near is left."""
    if not weights:
        return weights
    near = hops.get(anchor, {})
    kept = {b: w for b, w in weights.items() if b in close or near.get(b, max_hops + 1) <= max_hops}
    if len(kept) == len(weights):
        return weights
    total = sum(weights.values())
    if not kept:
        return {anchor: total}
    scale = total / sum(kept.values())
    return {b: w * scale for b, w in kept.items()}


def arm_below_shoulder(role: str) -> bool:
    """The upper arm, forearm, hand and fingers. The shoulder stays a shoulder."""
    return "Hand" in role or role.endswith("Arm")


def retie_head_weight(weights, point, head, head_position, hands, up) -> dict:
    """One vertex's {bone: weight}. An arm-owned point above the head joint goes to `head`.

    `hands` is every bone from the upper arm down, including the fingers. `up` is the up
    axis in the same space as `point` and `head_position`. Unchanged weights come back
    as the same dict.
    """
    if not weights:
        return weights
    main = max(weights, key=weights.get)
    if main not in hands:
        return weights
    import numpy as np

    above = float((np.asarray(point) - np.asarray(head_position)) @ np.asarray(up))
    if above <= 0.0:
        return weights
    return {head: sum(weights.values())}


def head_and_hands(parents, heads, up) -> tuple[str | None, set[str]]:
    """The head bone and the arm from the upper arm down, or `(None, set())` otherwise."""
    from kimodo_retarget import map_skintokens_to_soma

    try:
        mapping, _frame = map_skintokens_to_soma(parents, heads, up)
    except ValueError:
        return None, set()
    head = next((bone for bone, role in mapping.items() if role == "Head"), None)
    hands = {bone for bone, role in mapping.items() if arm_below_shoulder(role)}
    return head, hands


def clean_mesh(mesh, arm, max_hops: int = MAX_HOPS) -> int:  # pragma: no cover - bpy
    """Apply `prune_weights`, then give an arm-owned point above the head to the head.
    Returns vertices changed."""
    # Damn, I'm good.
    import numpy as np
    from kimodo_retarget import Z_UP

    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    hops = tree_hops(parents)
    to_arm = arm.matrix_world.inverted() @ mesh.matrix_world
    segments = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm.data.bones}
    joint = {name: pair[0] for name, pair in segments.items()}
    head_bone, hand_bones = head_and_hands(parents, joint, Z_UP)
    ends = np.array([e for pair in segments.values() for e in pair])
    slack = CLOSE_SHARE * float(np.ptp(ends, axis=0).max())
    names = {g.index: g.name for g in mesh.vertex_groups}
    changed = 0
    for vertex in mesh.data.vertices:
        weights = {names[g.group]: g.weight for g in vertex.groups
                   if g.weight > 0 and names[g.group] in parents}
        point = np.array(to_arm @ vertex.co)
        anchor = nearest_bone(point, segments)
        cleaned = prune_weights(weights, hops, anchor, max_hops, close_bones(point, segments, slack))
        if head_bone is not None:
            cleaned = retie_head_weight(
                cleaned, point, head_bone, joint[head_bone], hand_bones, Z_UP
            )
        if cleaned is weights:
            continue
        changed += 1
        for bone in weights.keys() | cleaned.keys():
            group = mesh.vertex_groups.get(bone) or mesh.vertex_groups.new(name=bone)
            if bone in cleaned:
                group.add([vertex.index], cleaned[bone], "REPLACE")
            else:
                group.remove([vertex.index])
    return changed


# Blender's default icosphere name. Only this exact helper is removed, not every mesh
# whose name starts the same way. A `glTF_not_exported` glTF extra is the other mark.
HELPER_NAME = "Icosphere"


def is_export_helper(name: str, extras: dict | None = None) -> bool:
    """True for SkinTokens' helper sphere, not for a body that merely starts with the same letters."""
    if extras and extras.get("glTF_not_exported"):
        return True
    return name == HELPER_NAME


def main() -> None:  # pragma: no cover - needs bpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", required=True, type=Path, help="rigged GLB")
    ap.add_argument("--out", required=True, type=Path, help="cleaned GLB to write")
    ap.add_argument("--max-hops", type=int, default=MAX_HOPS)
    args = ap.parse_args()

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    for obj in list(bpy.data.objects):
        # A bpy Object is not iterable, so its custom properties come through `keys()`.
        extras = {k: obj[k] for k in obj.keys()} if hasattr(obj, "keys") else {}  # noqa: SIM118
        if obj.type == "MESH" and is_export_helper(obj.name, extras):
            bpy.data.objects.remove(obj, do_unlink=True)
    total = 0
    for arm in (o for o in bpy.data.objects if o.type == "ARMATURE"):
        for mesh in (o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm):
            total += clean_mesh(mesh, arm, args.max_hops)
    print(f"cleaned skin weights on {total} vertices")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB")


if __name__ == "__main__":
    main()
