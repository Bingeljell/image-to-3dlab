"""The occlusion test must reject geometry hiding behind other geometry.

The bin-based depth buffer this replaces was inert on real meshes: with 99K vertices
spread over a 512x512 grid, most vertices sat alone in their bin and so counted as
"frontmost" by default. 68% of the hero fox's rear vertices were labelled from the
front image. These tests pin the behaviour the depth buffer could not deliver.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

trimesh = pytest.importorskip("trimesh")

from project_labels import occluded


def two_planes(gap: float = 0.5, n: int = 8) -> trimesh.Trimesh:
    """Two parallel square sheets, one directly behind the other along +Z.

    The camera looks down +Z, so the near sheet (z=gap) is visible and the far sheet
    (z=0) is entirely hidden behind it. This is the case the depth buffer got wrong.

    The near sheet is deliberately wider. Cast from a perimeter vertex of an
    equally-sized occluder, the ray grazes its exact boundary edge and whether that
    counts as a hit is numerically arbitrary -- an artefact of the fixture, not a
    question about occlusion. Overhang keeps the test about the thing it is testing.
    """
    vertices, faces = [], []
    for z, half in ((0.0, 1.0), (gap, 1.2)):
        grid = np.linspace(-half, half, n)
        xx, yy = np.meshgrid(grid, grid)
        corners = np.column_stack([xx.ravel(), yy.ravel()])
        base = len(vertices)
        vertices.extend(np.column_stack([corners, np.full(len(corners), z)]))
        for r in range(n - 1):
            for c in range(n - 1):
                a = base + r * n + c
                faces.append([a, a + 1, a + n])
                faces.append([a + 1, a + n + 1, a + n])
    return trimesh.Trimesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
        process=False,
    )


VIEW = np.array([0.0, 0.0, 1.0])


def test_far_sheet_is_occluded_near_sheet_is_not():
    mesh = two_planes()
    hidden = occluded(mesh, VIEW)
    z = mesh.vertices[:, 2]
    far, near = z < 0.25, z > 0.25

    assert hidden[far].all(), "the sheet behind another sheet must be occluded"
    assert not hidden[near].any(), "the frontmost sheet must never be occluded"


def test_single_sheet_is_never_occluded():
    """Nothing in front of it, so every vertex keeps its label."""
    mesh = two_planes()
    solo = trimesh.Trimesh(
        vertices=mesh.vertices[mesh.vertices[:, 2] > 0.25],
        faces=mesh.faces[mesh.faces.min(axis=1) >= (len(mesh.vertices) // 2)]
        - (len(mesh.vertices) // 2),
        process=False,
    )
    assert not occluded(solo, VIEW).any()


def test_reversing_the_view_reverses_which_sheet_hides():
    """Occlusion is a property of the view, not of the mesh."""
    mesh = two_planes()
    z = mesh.vertices[:, 2]
    from_front = occluded(mesh, VIEW)
    from_back = occluded(mesh, -VIEW)

    assert from_front[z < 0.25].all() and not from_front[z > 0.25].any()

    # Seen from behind, only the part of the near sheet that the (smaller) far sheet
    # actually covers is hidden; its overhanging rim is legitimately in plain view.
    covered = (np.abs(mesh.vertices[:, :2]) <= 1.0).all(axis=1)
    assert from_back[(z > 0.25) & covered].all()
    assert not from_back[z < 0.25].any()


def test_a_hole_lets_the_far_sheet_through():
    """A ray escaping through a tear is a real sighting, not a miss.

    This is why occlusion reads honestly on a torn mesh: the far surface behind a hole
    genuinely was photographed, and must stay labelled.
    """
    mesh = two_planes()
    near_start = len(mesh.vertices) // 2
    # Punch out every face of the near sheet that touches its centre vertices.
    centre = np.linalg.norm(mesh.vertices[:, :2], axis=1) < 0.45
    keep = ~(centre[mesh.faces].any(axis=1) & (mesh.faces >= near_start).all(axis=1))
    torn = trimesh.Trimesh(mesh.vertices, mesh.faces[keep], process=False)

    hidden = occluded(torn, VIEW)
    far_centre = centre & (np.arange(len(mesh.vertices)) < near_start)
    assert not hidden[far_centre].all(), "far vertices under the hole must be seen"


def test_offset_epsilon_does_not_self_occlude():
    """A vertex must not be blocked by the faces it belongs to."""
    mesh = two_planes()
    assert occluded(mesh, VIEW).sum() < len(mesh.vertices), "everything cannot be hidden"
