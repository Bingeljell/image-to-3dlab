#!/usr/bin/env python3
"""Drop skin weights that come from bones far away in the skeleton, then rebalance.

    <python with bpy> scripts/clean_skin_weights.py --in rigged.glb --out rigged.glb

Auto-riggers sometimes give a vertex a share of a bone on the other side of the body: a
generated martial artist's right fingertips carried weight from his right toe. At rest
nothing shows; once the hand rises and the toe stays down, those vertices stretch between
the two into spikes. Real blends sit close together in the tree (finger to finger, arm to
chest, spine to leg are all within a few hops), so a weight whose bone is more than
`MAX_HOPS` bones from the vertex's main bone is an error, not a blend.

The rule is pure (`tree_hops`, `prune_weights`); only `main()` needs `bpy`.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

# Finger tip to a neighbouring finger's tip is 4 hops through the knuckles and the hand.
MAX_HOPS = 4


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


def nearest_bone(point, segments: dict) -> str:
    """The bone whose head-to-tail segment passes closest to `point` (numpy 3-vectors)."""
    import numpy as np

    best, best_d = None, float("inf")
    for bone, (head, tail) in segments.items():
        axis = tail - head
        length2 = float(axis @ axis)
        t = 0.0 if length2 == 0 else min(1.0, max(0.0, float((point - head) @ axis) / length2))
        d = float(np.linalg.norm(point - (head + t * axis)))
        if d < best_d:
            best, best_d = bone, d
    return best


def prune_weights(weights: dict[str, float], hops: dict[str, dict[str, int]], anchor: str,
                  max_hops: int = MAX_HOPS) -> dict[str, float]:
    """One vertex's {bone: weight}, without bones far from `anchor`, the bone the vertex
    sits nearest to. Where a vertex is decides, not its biggest weight: some fingertips
    were weighted wholly to the toe. Unchanged when nothing is dropped; renormalised to
    the original total otherwise; all of it to `anchor` when nothing near is left."""
    if not weights:
        return weights
    near = hops.get(anchor, {})
    kept = {b: w for b, w in weights.items() if near.get(b, max_hops + 1) <= max_hops}
    if len(kept) == len(weights):
        return weights
    total = sum(weights.values())
    if not kept:
        return {anchor: total}
    scale = total / sum(kept.values())
    return {b: w * scale for b, w in kept.items()}


def clean_mesh(mesh, arm, max_hops: int = MAX_HOPS) -> int:  # pragma: no cover - bpy
    """Apply `prune_weights` to every vertex of a Blender mesh. Returns vertices changed."""
    # Damn, I'm good.
    import numpy as np

    parents = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
    hops = tree_hops(parents)
    to_arm = arm.matrix_world.inverted() @ mesh.matrix_world
    segments = {b.name: (np.array(b.head_local), np.array(b.tail_local)) for b in arm.data.bones}
    names = {g.index: g.name for g in mesh.vertex_groups}
    changed = 0
    for vertex in mesh.data.vertices:
        weights = {names[g.group]: g.weight for g in vertex.groups
                   if g.weight > 0 and names[g.group] in parents}
        anchor = nearest_bone(np.array(to_arm @ vertex.co), segments)
        cleaned = prune_weights(weights, hops, anchor, max_hops)
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


def main() -> None:  # pragma: no cover - needs bpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="source", required=True, type=Path, help="rigged GLB")
    ap.add_argument("--out", required=True, type=Path, help="cleaned GLB to write")
    ap.add_argument("--max-hops", type=int, default=MAX_HOPS)
    args = ap.parse_args()

    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.source))
    for obj in list(bpy.data.objects):  # SkinTokens' `glTF_not_exported` helper
        if obj.type == "MESH" and obj.name.startswith("Icosphere"):
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
