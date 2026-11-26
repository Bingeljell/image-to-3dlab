"""Dense and adaptive sampling grids for the shape decoder."""

import numpy as np


def default_chunks(num_latents, num_chunks):
    if num_chunks is not None:
        return num_chunks
    return 8000 if num_latents <= 1024 else 100000


def _bbox(bounds):
    if isinstance(bounds, (int, float)):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
    return np.array(bounds[0:3], dtype=np.float32), np.array(bounds[3:6], dtype=np.float32)


def _dense_values(evaluate, bbox_min, bbox_max, resolution):
    side = resolution + 1
    axes = [np.linspace(low, high, side, dtype=np.float32)
            for low, high in zip(bbox_min, bbox_max)]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return evaluate(xyz).reshape((side,) * 3)


def dense_grid(evaluate, *, bounds, resolution):
    """Sample the decoder at every coordinate in the bounding box."""
    bbox_min, bbox_max = _bbox(bounds)
    grid = _dense_values(evaluate, bbox_min, bbox_max, int(resolution))
    return grid, bbox_min, bbox_max, grid.shape


def near_surface_mask(grid: np.ndarray, alpha: float) -> np.ndarray:
    """Cells bracketing the iso-surface: sign changes vs a 6-neighbor, or |value| < 0.95.
    NaN cells (inactive from a previous octree level) are excluded. Returns bool mask.
    """
    g = grid + alpha
    valid = ~np.isnan(g)
    s = np.sign(np.where(valid, g, 0.0))
    near = np.zeros(g.shape, dtype=bool)
    for axis in range(3):
        lo = [slice(None)] * 3; lo[axis] = slice(0, -1)
        hi = [slice(None)] * 3; hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        change = (s[lo] != s[hi]) & valid[lo] & valid[hi]
        near[lo] |= change
        near[hi] |= change
    near |= (np.abs(np.where(valid, g, 1e9)) < 0.95)
    near &= valid
    return near


def octree_grid(evaluate, *, bounds, resolution, mc_level=0.0, min_resolution=63):
    """Sample a coarse grid, then double resolution only near the surface."""
    from scipy import ndimage
    bbox_min, bbox_max = _bbox(bounds)
    bbox_size = bbox_max - bbox_min

    res_list = []
    r = int(resolution)
    while r >= min_resolution:
        res_list.append(r)
        r //= 2
    res_list = res_list[::-1] or [int(resolution)]  # [coarse, ..., fine]

    grid = _dense_values(evaluate, bbox_min, bbox_max, res_list[0])

    struct = np.ones((3, 3, 3), dtype=bool)
    for res in res_list[1:]:
        gs = res + 1
        step = bbox_size / res
        near = near_surface_mask(grid, mc_level)
        expand = 0 if res == res_list[-1] else 1
        for _ in range(expand):
            near = ndimage.binary_dilation(near, struct)
        cidx = np.argwhere(near) * 2  # upsample coarse->fine (x2)
        nxt = np.zeros((gs, gs, gs), dtype=bool)
        cidx = np.clip(cidx, 0, gs - 1)
        nxt[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = True
        for _ in range(2 - expand):
            nxt = ndimage.binary_dilation(nxt, struct)
        nidx = np.argwhere(nxt)  # active fine-grid voxels [P,3]
        pts = nidx.astype(np.float32) * step + bbox_min
        vals = evaluate(pts)
        grid = np.full((gs, gs, gs), np.nan, dtype=np.float32)
        grid[nidx[:, 0], nidx[:, 1], nidx[:, 2]] = vals
    return grid, bbox_min, bbox_max, grid.shape
