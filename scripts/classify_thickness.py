#!/usr/bin/env python3
"""Separate solid body from thin foliage by measuring local thickness.

**VERDICT (2026-08-09): this does not work on the moss fox. Use a painted mask.**
Kept because the measurement is sound and the negative result is worth having --
it rules out a whole family of "derive the labels automatically" ideas.

Measured on `output/repair/fox_repaired.glb`, thickness as a fraction of asset scale:

| region | p25 | median | p75 |
|---|---|---|---|
| tail (foliage)     | 0.0182 | **0.0391** | 0.0620 |
| feet/legs (solid)  | 0.0175 | **0.0286** | 0.1088 |
| torso (solid)      | 0.0123 | 0.1080 | 0.1873 |

**The tail measures THICKER than the legs.** The distributions do not merely overlap,
they are inverted for the pair that matters, so no threshold can separate them and no
amount of smoothing rescues it. Every threshold from 0.02 to 0.06 produced speckle
rather than regions -- confirmed visually across four renders, which is the only test
that would have caught it.

The cause is upstream and already documented in `docs/fidelity-explained.md`: TRELLIS
builds everything from a fixed-resolution grid, so a leaf is never a thin sheet. It is
a lumpy solid blade of roughly the same gauge as a leg. Thickness cannot distinguish
two things the generator built at the same gauge. Only the torso is genuinely thicker,
and nobody needed a classifier to find the torso.

What still works: `project_labels.py` with a hand-painted mask.

---

**Why this exists.** "No holes" is two goals wearing one coat. The torso, head and
legs should be genuinely watertight -- that is a real defect. The tail and ear leaves
are thin overlapping sheets: inherently open, impossible to make watertight without
welding them into a blob, and exactly the geometry that `remesh=True` shredded. Games
ship foliage as double-sided alpha cards for this reason. So the two classes want
opposite treatment, and something has to tell them apart.

`project_labels.py` already can -- from a hand-painted mask. That needs the artist at
a desk, a per-subject painting pass, and a view-angle solve. This measures the
distinction instead: **cast a ray inward from each face and see how far until it
exits.** A torso hits its far wall. A leaf card hits nothing, or its own back face
almost immediately.

A miss counts as ZERO thickness, not infinity. An open single-layer sheet is the
thinnest thing here, and calling a miss "infinitely thick" would label every leaf as
solid -- the exact inversion of the intent.

**Known limitation, stated up front:** a ray can escape through one of the mesh's own
holes and report solid geometry as thin. That is what `--cone-samples` is for: several
rays in a narrow cone, taking the median, so one escape is out-voted. Verify the split
by rendering it (`--output` writes a two-material GLB), never by the histogram alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def _cone_directions(
    normals: np.ndarray, samples: int, angle: float, seed: int
) -> list[np.ndarray]:
    """`samples` directions per face, spread up to `angle` radians around -normal."""
    inward = -normals
    if samples <= 1:
        return [inward]

    # A stable pair of tangents per face, so the cone is reproducible run to run.
    reference = np.tile(np.array([0.0, 0.0, 1.0]), (len(normals), 1))
    degenerate = np.abs(inward[:, 2]) > 0.9
    reference[degenerate] = np.array([1.0, 0.0, 0.0])
    tangent = np.cross(inward, reference)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-12
    bitangent = np.cross(inward, tangent)

    rng = np.random.default_rng(seed)
    directions = [inward]
    for _ in range(samples - 1):
        theta = rng.uniform(0.0, angle)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        offset = np.tan(theta) * (
            np.cos(phi) * tangent + np.sin(phi) * bitangent
        )
        direction = inward + offset
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-12
        directions.append(direction)
    return directions


def local_thickness(
    mesh: trimesh.Trimesh,
    cone_samples: int = 1,
    cone_angle: float = 0.2,
    seed: int = 0,
) -> np.ndarray:
    """Distance from each face to the surface behind it, 0 where the ray escapes."""
    centres = np.asarray(mesh.triangles_center)
    normals = np.asarray(mesh.face_normals)
    scale = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
    epsilon = scale * 1e-5

    def cast(direction: np.ndarray) -> np.ndarray:
        origins = centres + direction * epsilon
        locations, ray_index, _ = mesh.ray.intersects_location(
            origins, direction, multiple_hits=False
        )
        distances = np.zeros(len(centres))
        if len(ray_index):
            hit = np.linalg.norm(locations - origins[ray_index], axis=1)
            # A ray can hit several times; keep the nearest per face.
            order = np.argsort(-hit)
            distances[ray_index[order]] = hit[order]
        return distances

    per_sample = []
    for direction in _cone_directions(normals, cone_samples, cone_angle, seed):
        # Cast BOTH ways and keep the larger.
        #
        # The first version cast only along -normal and called a miss "thin". On this
        # mesh winding is inconsistent, so -normal is not reliably inward: 29.4% of
        # rays escaped and the result labelled the legs as foliage and the tail leaves
        # as solid -- it was measuring normal ORIENTATION, not thickness. Taking the
        # max over both directions makes the measurement orientation-free: for solid
        # geometry one ray crosses the body and the other escapes, so the max is the
        # true thickness; for an open sheet both escape and the max is still zero.
        per_sample.append(np.maximum(cast(direction), cast(-direction)))

    return np.median(np.vstack(per_sample), axis=0)


def classify(thickness: np.ndarray, threshold: float) -> np.ndarray:
    """True where a face is THIN (foliage), False where it is solid body."""
    return np.asarray(thickness) < threshold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--output", type=Path, help="write a two-material GLB for visual checking"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="thin/solid cut, as a fraction of the asset's largest dimension",
    )
    parser.add_argument("--cone-samples", type=int, default=5)
    parser.add_argument("--cone-angle", type=float, default=0.2)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh.expanduser().resolve(), force="mesh", process=False)
    scale = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
    thickness = local_thickness(mesh, args.cone_samples, args.cone_angle)
    thin = classify(thickness, args.threshold * scale)

    print(f"faces {len(mesh.faces):,}   asset scale {scale:.4f}")
    print(f"threshold {args.threshold:.4f} x scale = {args.threshold * scale:.5f}")
    print(f"  THIN  (foliage) {thin.sum():>8,}  {100 * thin.mean():5.1f}%")
    print(f"  SOLID (body)    {(~thin).sum():>8,}  {100 * (~thin).mean():5.1f}%")
    print(f"  escaped (0)     {(thickness == 0).sum():>8,}  "
          f"{100 * (thickness == 0).mean():5.1f}%")

    print("\nthickness distribution, as a fraction of asset scale:")
    for percentile in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{percentile:<3} {np.percentile(thickness, percentile) / scale:.5f}")

    if args.output:
        # Two meshes, two materials, so the split is visible in any renderer.
        parts = trimesh.Scene()
        for name, mask, colour in (
            ("solid", ~thin, [90, 110, 190, 255]),
            ("foliage", thin, [235, 140, 60, 255]),
        ):
            if not mask.any():
                continue
            piece = mesh.copy()
            piece.update_faces(mask)
            piece.remove_unreferenced_vertices()
            piece.visual = trimesh.visual.ColorVisuals(
                piece, face_colors=np.tile(colour, (len(piece.faces), 1))
            )
            piece.visual.material = trimesh.visual.material.PBRMaterial(
                baseColorFactor=colour, metallicFactor=0.0, roughnessFactor=0.75
            )
            parts.add_geometry(piece, geom_name=name)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        parts.export(output)
        print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
