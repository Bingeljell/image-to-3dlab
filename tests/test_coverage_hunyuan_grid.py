"""Check shape-grid coordinate generation without instantiating the decoder."""

import numpy as np

from hunyuan_mlx.shape.hy3dmlx.grid import (
    default_chunks,
    dense_grid,
    near_surface_mask,
    octree_grid,
)


def test_dense_grid_uses_xyz_order_and_inclusive_custom_bounds():
    points = []

    def field(xyz):
        points.extend(xyz)
        return xyz[:, 0] + 2 * xyz[:, 1] + 3 * xyz[:, 2]

    grid, low, high, shape = dense_grid(field, bounds=[-2, 3, 1, 8, 5, 7], resolution=2)
    assert shape == grid.shape == (3, 3, 3)
    assert grid.dtype == np.float32
    np.testing.assert_array_equal(low, [-2, 3, 1])
    np.testing.assert_array_equal(high, [8, 5, 7])
    np.testing.assert_array_equal(points[0], low)
    np.testing.assert_array_equal(points[-1], high)
    np.testing.assert_array_equal(grid[:, 0, 0], [7, 12, 17])
    np.testing.assert_array_equal(grid[0, :, 0], [7, 9, 11])
    np.testing.assert_array_equal(grid[0, 0, :], [7, 16, 25])


def test_octree_refines_only_the_surface_and_leaves_far_cells_nan():
    queries = []

    def plane(xyz):
        queries.append(xyz)
        return 10 * xyz[:, 0]

    grid, low, high, shape = octree_grid(plane, bounds=4, resolution=32, min_resolution=8)
    np.testing.assert_array_equal(low, [-4, -4, -4])
    np.testing.assert_array_equal(high, [4, 4, 4])
    assert shape == (33, 33, 33)
    assert len(queries) == 3
    assert len(queries[0]) == 9 ** 3
    assert 0 < len(queries[-1]) < 33 ** 3
    assert (grid[16] == 0).all()
    assert (grid[15] == -2.5).all()
    assert (grid[17] == 2.5).all()
    assert np.isnan(grid[[0, -1]]).all()


def test_octree_uses_dense_grid_below_minimum_and_does_not_invent_surface():
    def constant(xyz):
        return np.full(len(xyz), 10, np.float32)

    coarse, *_ = octree_grid(constant, bounds=1, resolution=3, min_resolution=4)
    assert (coarse == 10).all()
    refined, *_ = octree_grid(constant, bounds=1, resolution=8, min_resolution=4)
    assert np.isnan(refined).all()


def test_band_marks_shifted_crossing_but_excludes_inactive_cells():
    field = np.array([10, 6, 4, np.nan, 2], dtype=np.float32).reshape(5, 1, 1)
    assert near_surface_mask(field, alpha=-5).reshape(-1).tolist() == [False, True, True, False, False]


def test_chunk_size_preserves_explicit_value_and_defaults_by_latent_count():
    assert default_chunks(512, None) == 8000
    assert default_chunks(2048, None) == 100000
    assert default_chunks(2048, 32) == 32
