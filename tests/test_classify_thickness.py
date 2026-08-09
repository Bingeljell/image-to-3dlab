"""Split a generated mesh into solid body and thin foliage by measuring thickness.

The motivation: "no holes" is really two goals. A torso should be watertight; leaves
and a bushy tail are thin overlapping sheets that are inherently open, and are exactly
what `remesh=True` shredded. Games ship foliage as double-sided cards. Telling the two
apart by hand needs a painted mask; this measures it instead.

The measurement: cast a ray inward from each face and see how far until it exits. A
torso hits its far wall; a leaf card hits nothing, or its own back face immediately.
"""

from __future__ import annotations

import numpy as np
import trimesh

from scripts.classify_thickness import classify, local_thickness


def test_a_solid_cube_measures_its_own_width():
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    thickness = local_thickness(cube)
    assert np.allclose(thickness, 2.0, atol=1e-3)


def test_a_thin_plate_measures_thin_on_its_broad_faces():
    """Only the BROAD faces of a thin plate read as thin -- its rim faces look along
    the plate and measure its full width.

    A box has 8 rim triangles against 4 broad ones, so the plain median says "thick".
    That is not a bug in the measurement, it is the geometry: a leaf's rim really is
    solid-looking edge-on. On a real leaf the rim is a negligible slice of the area,
    but it means the per-face split will show speckle along foliage edges.
    """
    plate = trimesh.creation.box(extents=(2.0, 2.0, 0.05))
    thickness = local_thickness(plate)

    assert (thickness < 0.1).sum() == 4, "the four broad triangles read as thin"
    # Area-weighted, the plate is overwhelmingly thin.
    areas = plate.area_faces
    assert areas[thickness < 0.1].sum() / areas.sum() > 0.9


def test_an_open_sheet_has_nothing_behind_it():
    """A single-layer card has no back face, so the ray escapes. That is THIN, not
    thick -- treating a miss as infinite thickness would call every leaf solid."""
    sheet = trimesh.Trimesh(
        vertices=[[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    thickness = local_thickness(sheet)
    assert np.all(thickness == 0.0)


def test_classify_separates_a_cube_from_a_plate():
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    plate = trimesh.creation.box(extents=(2.0, 2.0, 0.05))
    plate.apply_translation((6.0, 0, 0))
    combined = trimesh.util.concatenate([cube, plate])

    thickness = local_thickness(combined)
    thin = classify(thickness, threshold=0.5)

    # Faces 0-11 are the cube, 12-23 the plate.
    assert not thin[:12].any(), "the solid cube must not be called foliage"
    assert thin[12:].sum() >= 2, "the plate's broad faces must be called foliage"


def test_threshold_is_absolute_so_callers_scale_it_themselves():
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    thickness = local_thickness(cube)
    assert not classify(thickness, threshold=1.0).any()
    assert classify(thickness, threshold=3.0).all()


def test_cone_sampling_survives_a_hole():
    """A single ray can escape through a hole in an otherwise solid body and report
    it as thin. Averaging a small cone of rays should out-vote one escape."""
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    # Punch a hole: drop one face from the far wall the rays would hit.
    holed = trimesh.Trimesh(
        vertices=cube.vertices, faces=np.delete(cube.faces, 0, axis=0), process=False
    )
    single = local_thickness(holed, cone_samples=1)
    coned = local_thickness(holed, cone_samples=9, cone_angle=0.25)
    assert (coned == 0).sum() <= (single == 0).sum()
