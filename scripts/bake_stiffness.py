#!/usr/bin/env python3
"""Bake foliage stiffness into a GLB as vertex colours, for engine-side wind.

A wind shader needs two things per vertex: whether it moves at all, and how floppy it
is. Both are derivable from painted labels -- floppiness is just distance to the
nearest rigid vertex -- but until they are written *into the asset* they exist only
inside our renderer, and an engine cannot see them.

This writes them into the mesh so SceneKit, RealityKit, or any engine can run the same
wave live:

    R = stiffness   0 = anchored, 1 = free tip
    G = category    0 = rigid body, 0.5 = foliage, 1.0 = flower
    B = phase       a stable per-clump offset so fronds do not move in lockstep

glTF carries this as COLOR_0, which every engine can read, so no custom extension is
needed. The engine's shader multiplies its wave by R and offsets it by B.

Note the trade: COLOR_0 is also what a renderer uses to tint the surface, so an asset
carrying stiffness this way should be rendered from its texture, not its vertex
colours. A custom attribute (_STIFFNESS) would avoid that but is not read by default
anywhere; this is the pragmatic choice.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

FOLIAGE = 0.5
FLOWER = 1.0
RIGID = 0.0


def classify(colours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split label colours into (moves, category value)."""
    red, green, blue = colours[:, 0], colours[:, 1], colours[:, 2]
    is_foliage = (green > red) & (green >= blue)
    is_flower = (blue > red) & (blue > green)
    category = np.where(is_flower, FLOWER, np.where(is_foliage, FOLIAGE, RIGID))
    return is_foliage | is_flower, category.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path, help="labelled .glb from project_labels.py")
    parser.add_argument("output", type=Path, help="destination .glb")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=None,
        help="textured .glb to bake onto; defaults to the labelled mesh itself",
    )
    parser.add_argument(
        "--clump",
        type=float,
        default=0.07,
        help="clump size for the phase channel, as a fraction of the asset",
    )
    args = parser.parse_args()

    labelled = trimesh.load(args.labels.expanduser().resolve(), force="mesh")
    label_colours = np.array(labelled.visual.vertex_colors)[:, :3].astype(np.int32)
    moves, category = classify(label_colours)

    target = labelled
    if args.mesh:
        target = trimesh.load(args.mesh.expanduser().resolve(), force="mesh")
        # Exporting through trimesh does not preserve vertex order, so the textured
        # mesh and the labelled mesh are matched by position rather than by index.
        _, nearest = cKDTree(labelled.vertices).query(target.vertices)
        moves = moves[nearest]
        category = category[nearest]

    vertices = target.vertices
    rigid = labelled.vertices[~classify(label_colours)[0]]
    if not len(rigid):
        raise SystemExit("error: no rigid vertices to anchor against")

    # Stiffness: distance to the nearest rigid vertex, normalised over the moving set.
    stiffness = np.zeros(len(vertices), dtype=np.float32)
    if moves.any():
        distance, _ = cKDTree(rigid).query(vertices[moves])
        peak = float(distance.max()) or 1.0
        eased = (distance / peak).astype(np.float32)
        stiffness[moves] = eased * eased * (3.0 - 2.0 * eased)

    # Phase: hash a coarse grid so each clump gets its own stable offset. Position-based
    # rather than random, so re-running produces the same asset.
    extent = float(np.ptp(vertices, axis=0).max())
    cell = np.floor(vertices / max(extent * args.clump, 1e-6)).astype(np.int64)
    clump = (cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663) ^ (cell[:, 2] * 83492791)
    phase = ((clump % 997) / 997.0).astype(np.float32)

    rgba = np.zeros((len(vertices), 4), dtype=np.uint8)
    rgba[:, 0] = np.clip(stiffness * 255, 0, 255).astype(np.uint8)
    rgba[:, 1] = np.clip(category * 255, 0, 255).astype(np.uint8)
    rgba[:, 2] = np.clip(phase * 255, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255

    baked = trimesh.Trimesh(
        vertices=vertices, faces=target.faces, vertex_colors=rgba, process=False
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    baked.export(output)

    print(
        f"BAKE:: {int(moves.sum())} moving of {len(vertices)} vertices | "
        f"mean stiffness {float(stiffness[moves].mean()) if moves.any() else 0:.3f} | "
        f"-> {output}"
    )
    print("       R=stiffness  G=category(0 rigid, .5 foliage, 1 flower)  B=phase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
