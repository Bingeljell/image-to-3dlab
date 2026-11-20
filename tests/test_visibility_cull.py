"""Tests for visibility-based face culling.

These import the shipping helpers directly. The Blender rendering half is deliberately
thin and untested here; everything that can be got wrong in arithmetic lives in the pure
module and is checked below.
"""

import numpy as np
import pytest

from scripts.visibility_cull import (
    MAX_FACES,
    color_to_index,
    cull_report,
    fibonacci_directions,
    hidden_from_rays,
    index_to_color,
    visible_faces,
)


def test_directions_are_unit_length():
    dirs = fibonacci_directions(200)
    assert dirs.shape == (200, 3)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)


def test_directions_cover_the_sphere_evenly():
    """A lat/long grid clumps at the poles; the point of Fibonacci is that it does not."""
    dirs = fibonacci_directions(1000)
    # Mean of a uniform spherical distribution is the origin.
    np.testing.assert_allclose(dirs.mean(axis=0), 0.0, atol=0.05)
    # Every octant should be populated for a genuinely even spread.
    signs = {tuple(np.sign(d).astype(int)) for d in dirs if np.all(d != 0)}
    assert len({s for s in signs if 0 not in s}) == 8


def test_single_direction_is_allowed():
    assert fibonacci_directions(1).shape == (1, 3)


def test_zero_directions_is_rejected():
    with pytest.raises(ValueError):
        fibonacci_directions(0)


def test_index_colour_round_trip():
    idx = np.array([0, 1, 255, 256, 65535, 65536, 1_000_000])
    np.testing.assert_array_equal(color_to_index(index_to_color(idx)), idx)


def test_background_black_decodes_to_minus_one():
    """Reserving 0 is what stops an unrendered pixel from being read as face 0."""
    assert color_to_index(np.array([0, 0, 0])) == -1


def test_face_zero_is_not_background():
    assert not np.array_equal(index_to_color(0), np.array([0, 0, 0]))
    assert color_to_index(index_to_color(0)) == 0


def test_index_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        index_to_color(MAX_FACES)
    with pytest.raises(ValueError):
        index_to_color(-1)


def test_colour_to_index_rejects_bad_shape():
    with pytest.raises(ValueError):
        color_to_index(np.zeros((4, 4, 4), dtype=np.uint8))


def test_visible_faces_marks_only_what_was_seen():
    buffer = np.zeros((2, 2, 3), dtype=np.uint8)
    buffer[0, 0] = index_to_color(3)
    buffer[1, 1] = index_to_color(7)
    seen = visible_faces([buffer], face_count=10)
    assert seen.sum() == 2
    assert seen[3] and seen[7]


def test_visible_faces_unions_across_buffers():
    """A face seen from any one direction must survive, even if hidden in all others."""
    a = np.zeros((1, 1, 3), dtype=np.uint8)
    a[0, 0] = index_to_color(1)
    b = np.zeros((1, 1, 3), dtype=np.uint8)
    b[0, 0] = index_to_color(4)
    seen = visible_faces([a, b], face_count=6)
    assert seen[1] and seen[4]
    assert seen.sum() == 2


def test_all_background_culls_everything():
    seen = visible_faces([np.zeros((3, 3, 3), dtype=np.uint8)], face_count=5)
    assert not seen.any()


def test_indices_beyond_the_mesh_are_ignored():
    """A stray colour from compression or dithering must not index out of bounds."""
    buffer = np.zeros((1, 2, 3), dtype=np.uint8)
    buffer[0, 0] = index_to_color(2)
    buffer[0, 1] = index_to_color(9999)
    seen = visible_faces([buffer], face_count=5)
    assert seen[2] and seen.sum() == 1


def test_no_buffers_culls_everything():
    assert not visible_faces([], face_count=4).any()


def test_report_counts_correctly():
    seen = np.array([True, False, True, True])
    assert "3 of 4" in cull_report(seen) and "75.0%" in cull_report(seen)


# --- the inner wall: faces from which no ray reaches the outside -----------------------


def _shells(*radii):
    """A cast against concentric spheres: True when the ray meets one of them."""

    def cast(origin, direction, _face):
        o, d = np.asarray(origin, float), np.asarray(direction, float)
        b = o @ d
        for r in radii:
            disc = b * b - (o @ o - r * r)
            if disc >= 0 and (-b + np.sqrt(disc)) > 1e-9:
                return True
        return False

    return cast


def _on_sphere(radius, count=40):
    return fibonacci_directions(count) * radius


def test_a_wall_inside_another_never_sees_out_and_the_outer_one_does():
    """Pixal3D's double wall: the inner one is hidden from every direction, the outer is not."""
    dirs = fibonacci_directions(16)
    cast = _shells(1.0, 0.97)
    centres = np.vstack([_on_sphere(1.0), _on_sphere(0.97)])
    hidden = hidden_from_rays(centres, dirs, cast, eps=1e-6)
    assert not hidden[:40].any(), "the outer wall is what the camera sees"
    assert hidden[40:].all(), "the wall inside it is never seen from anywhere"


def test_a_face_is_kept_as_soon_as_one_ray_escapes():
    """A seen face costs one ray when its first direction gets out, so the test stays fast."""
    calls = []

    def cast(origin, direction, face):
        calls.append(face)
        return False

    hidden = hidden_from_rays(np.zeros((3, 3)), fibonacci_directions(16), cast, eps=1e-6)
    assert not hidden.any()
    assert calls == [0, 1, 2]


def test_faces_start_at_different_directions():
    """Each face starts at its own direction, so a whole region never tries the same blocked
    ray first."""
    first = {}

    def cast(origin, direction, face):
        first.setdefault(face, tuple(np.round(direction, 6)))
        return False

    hidden_from_rays(np.zeros((4, 3)), fibonacci_directions(4), cast, eps=1e-6)
    assert len(set(first.values())) == 4


def test_the_ray_starts_just_off_the_face_along_its_direction():
    seen = []

    def cast(origin, direction, face):
        seen.append(np.asarray(origin) - np.asarray([1.0, 2.0, 3.0]))
        return True

    dirs = fibonacci_directions(2)
    hidden_from_rays(np.array([[1.0, 2.0, 3.0]]), dirs, cast, eps=0.01)
    assert np.allclose(seen[0], dirs[0] * 0.01)


def test_no_faces_gives_an_empty_answer():
    assert hidden_from_rays(np.zeros((0, 3)), fibonacci_directions(4), _shells(1.0), 1e-6).shape == (0,)

