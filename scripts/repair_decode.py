#!/usr/bin/env python3
"""Repair winding on a cached decode, before `to_glb` ever sees it.

    vendor/trellis-mac/.venv/bin/python scripts/repair_decode.py \\
        decode.pt decode_fixed.pt

**Why at the decode.** TRELLIS's decoder emits geometry with inconsistent face winding —
measured on the moss fox: `is_winding_consistent False`, 130,373 connected components,
straight out of the model and before `generate.py` touches anything. The repo had this
recorded as a defect of ours; it is not.

The reference pipeline hides it by taking `to_glb`'s Branch 2, where dual contouring
**rebuilds the topology from scratch** and emits every quad consistently oriented. That
branch produces a cage on this port, so we take Branch 1, which keeps the decoder's topology
and inherits its winding. `unify_face_orientations` runs there and cannot fully repair it.

So: repair the input instead. `fix_normals` orients each connected component by its own
signed volume, which is the right operation for a mesh in 130k pieces — a component whose
volume comes out negative is inside-out and gets flipped.

Vertices are merged by position first. Without that the mesh is split along every attribute
seam, components cannot see their neighbours, and orientation has no path to propagate
along — which is why repairing the *exported* mesh reversed only 376 of 282,610 faces and
still failed to converge.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def summarise(mesh) -> dict:
    """The numbers that say whether winding repair worked.

    `total_volume` is the sum of signed component volumes and cancels: a mesh half
    inside-out reads 0. `outward_volume` sums their magnitudes, so it only reaches the true
    enclosed volume once every component faces outward — which is the property we want.
    """
    import numpy as np

    verts, faces = mesh.vertices, mesh.faces
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    per_face = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    labels, count = component_labels(mesh)
    per_component = np.bincount(labels, weights=per_face, minlength=count)
    return {
        "faces": len(faces),
        "vertices": len(verts),
        "components": int(count),
        "inverted_components": int((per_component < 0).sum()),
        "total_volume": float(per_component.sum()),
        "outward_volume": float(np.abs(per_component).sum()),
    }


def component_labels(mesh):
    """Per-face connected-component labels, via face adjacency."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    n = len(mesh.faces)
    adjacency = mesh.face_adjacency
    if len(adjacency) == 0:
        return np.arange(n), n
    graph = sp.coo_matrix(
        (np.ones(len(adjacency)), (adjacency[:, 0], adjacency[:, 1])), shape=(n, n)
    )
    count, labels = connected_components(graph, directed=False)
    return labels, count


def repair(vertices, faces):
    """Merge by position, make each component consistent, then orient it outward.

    Returns (mesh, before, after).

    **Why not `trimesh.repair.fix_normals`.** Its inversion step is *global*: it sums the
    whole mesh's volume and flips everything if that total is negative. On a mesh in 130,373
    components the positives and negatives cancel, the total lands near zero, and it
    concludes there is nothing to do. Verified on two boxes with one inverted — total volume
    0.0 before and after, and `is_winding_consistent` reports True throughout because it only
    checks each component internally.

    So orientation has to be decided per component, by that component's own signed volume.
    """
    import numpy as np
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    before = summarise(mesh)

    # Step 1: make each component internally consistent so a component-wide flip is meaningful.
    trimesh.repair.fix_winding(mesh)

    # Step 2: flip whole components that enclose negative volume.
    labels, count = component_labels(mesh)
    verts, faces_arr = mesh.vertices, mesh.faces
    a, b, c = verts[faces_arr[:, 0]], verts[faces_arr[:, 1]], verts[faces_arr[:, 2]]
    per_face = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    per_component = np.bincount(labels, weights=per_face, minlength=count)

    flip = per_component[labels] < 0
    if flip.any():
        mesh.faces[flip] = mesh.faces[flip][:, ::-1]
    return mesh, before, summarise(mesh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("decode", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--export-glb", type=Path, help="also write a viewable GLB")
    args = parser.parse_args()

    import torch

    payload = torch.load(args.decode, weights_only=False)
    print(f"loaded {payload['faces'].shape[0]:,} faces", flush=True)

    started = time.time()
    mesh, before, after = repair(
        payload["vertices"].cpu().numpy(), payload["faces"].cpu().numpy()
    )
    elapsed = time.time() - started

    # Save BEFORE reporting. The repair costs ~5.5 minutes on a 12.8M-face decode, and a
    # KeyError in the summary print once threw all of it away.
    # The attribute volume is indexed by position, so merging vertices does not disturb it.
    # ascontiguousarray: flipping faces with [:, ::-1] can leave negative strides, which
    # torch.as_tensor refuses outright - and it would refuse them only after the repair has
    # already cost five minutes.
    import numpy as np

    payload["vertices"] = torch.as_tensor(
        np.ascontiguousarray(mesh.vertices), dtype=torch.float32
    )
    payload["faces"] = torch.as_tensor(
        np.ascontiguousarray(mesh.faces), dtype=torch.int32
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"wrote {args.output}", flush=True)
    if args.export_glb:
        args.export_glb.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(args.export_glb)
        print(f"wrote {args.export_glb}", flush=True)

    print(f"repaired in {elapsed:.0f}s", flush=True)
    for name, stats in (("before", before), ("after", after)):
        print(
            f"  {name:6s} faces={stats['faces']:>10,} "
            f"components={stats['components']:>8,} "
            f"inverted={stats['inverted_components']:>8,} "
            f"total_vol={stats['total_volume']:+.6f} "
            f"outward_vol={stats['outward_volume']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
