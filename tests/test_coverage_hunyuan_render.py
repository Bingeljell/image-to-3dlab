"""Exercise the portable renderer with tiny meshes and textures."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from hunyuan_mlx.paint.hy3dpaint_mlx import mesh_render
from hunyuan_mlx.paint.hy3dpaint_mlx.raster import cr_raster


@pytest.fixture
def triangle():
    vertices = np.array([[-1, -1, 0, 1], [1, -1, 0, 1], [-2, 2, 0, 2]], dtype=np.float32)
    return vertices, np.array([[0, 1, 2]])


def test_raster_corrects_barycentrics_for_homogeneous_coordinates(triangle):
    vertices, faces = triangle
    ids, bary = cr_raster.rasterize(vertices[None], faces, (5, 5))
    np.testing.assert_array_equal(ids[0], np.ones(5))
    assert ids[4, 4] == 0
    np.testing.assert_allclose(bary[1, 1], [4 / 7, 2 / 7, 1 / 7], atol=1e-7)
    attrs = np.array([[[0, 1], [7, 1], [14, 1]]])
    pixels = cr_raster.interpolate(attrs, ids, bary, faces)
    np.testing.assert_allclose(pixels[1, 1], [4, 1], atol=1e-7)
    np.testing.assert_array_equal(pixels[4, 4], [0, 0])


def test_nearer_face_wins_depth_and_first_face_wins_a_tie(triangle):
    vertices, faces = triangle
    nearer = vertices.copy()
    nearer[:, 2] = -0.5 * nearer[:, 3]
    face_ids, _ = cr_raster.rasterize(np.concatenate([vertices, nearer, nearer]),
                                      np.concatenate([faces, faces + 3, faces + 6]), 5)
    assert face_ids[1, 1] == 2


def test_degenerate_offscreen_and_subpixel_triangles_leave_clear_buffer():
    vertices = np.array([
        [0, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1],
        [4, 4, 0, 1], [5, 4, 0, 1], [4, 5, 0, 1],
        [0.1, 0.1, 0, 1], [0.2, 0.1, 0, 1], [0.1, 0.2, 0, 1],
    ])
    ids, bary = cr_raster.rasterize(vertices, np.arange(9).reshape(3, 3), 5)
    assert not ids.any()
    assert not bary.any()


def test_raster_dispatch_handles_an_unavailable_gpu(monkeypatch, triangle):
    from hunyuan_mlx.paint.hy3dpaint_mlx import raster

    def unavailable():
        raise RuntimeError("Metal device unavailable")

    monkeypatch.setattr(raster, "gpu_raster", SimpleNamespace(available=unavailable), raising=False)
    vertices, faces = triangle
    ids, bary = cr_raster.rasterize_backend(vertices, faces, 5, backend="auto")
    assert ids[1, 1] == 1
    assert bary[1, 1].sum() == pytest.approx(1)
    with pytest.raises(ValueError, match="unknown rasterizer backend"):
        cr_raster.rasterize_backend(vertices, faces, 5, backend="invalid")


@pytest.mark.parametrize("backend", ["auto", "gpu"])
def test_raster_passes_inputs_through_to_available_gpu(monkeypatch, triangle, backend):
    from hunyuan_mlx.paint.hy3dpaint_mlx import raster

    vertices, faces = triangle
    marker = object()

    def gpu_rasterize(pos, tri, resolution):
        assert pos is vertices and tri is faces and resolution == (3, 5)
        return marker

    monkeypatch.setattr(
        raster, "gpu_raster",
        SimpleNamespace(available=lambda: True, rasterize=gpu_rasterize),
        raising=False,
    )
    assert cr_raster.rasterize_backend(vertices, faces, (3, 5), backend=backend) is marker


@pytest.fixture
def planar_renderer(monkeypatch):
    monkeypatch.setattr(mesh_render, "_RASTER_BACKEND", "numpy")
    renderer = mesh_render.MeshRender()
    # A square in the input XY plane becomes a square facing the camera.
    renderer.load_mesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], [[0, 1, 2], [0, 2, 3]])
    renderer.set_uv([[0, 0], [1, 0], [1, 1], [0, 1]], renderer.pos_idx)
    return renderer


def test_combined_control_render_matches_both_maps(planar_renderer):
    renderer = planar_renderer
    normal, position = renderer.render_control(0, 0, 9, bg=(0, 0, 0))
    np.testing.assert_array_equal(normal, renderer.render_normal(0, 0, 9, bg=(0, 0, 0)))
    np.testing.assert_array_equal(position, renderer.render_position(0, 0, 9, bg=(0, 0, 0)))
    np.testing.assert_allclose(position[4, 4], [0.5, 0.5, 0.5], atol=1e-7)
    np.testing.assert_allclose(normal[4, 4], [0.5, 0, 0.5], atol=1e-7)
    np.testing.assert_array_equal(position[0, 0], [0, 0, 0])


def test_camera_target_maps_to_center_of_view():
    camera = mesh_render.get_mv_matrix(0, 0, 3, center=(1, 0, 2))
    target_in_view = camera @ [1, 0, 2, 1]
    np.testing.assert_allclose(target_in_view[:2], [0, 0], atol=1e-7)
    assert target_in_view[2] < 0
    pos = mesh_render.get_ortho_proj(scale=4, near=1, far=3)
    np.testing.assert_allclose(pos @ [2, -2, -1, 1], [1, -1, -1, 1])
    np.testing.assert_allclose(pos @ [-2, 2, -3, 1], [-1, 1, 1, 1])


def test_bake_blends_front_views_with_weights_and_discards_reverse_views(planar_renderer):
    renderer = planar_renderer
    colors = [np.broadcast_to(color, (17, 17, 3)).astype(np.float32) for color in np.eye(3)]
    blended, mask = renderer.bake(colors, [0, 0, 0], [0, 0, 180], texture_size=9, weights=[1, 3, 99])
    assert mask[4, 4]
    np.testing.assert_allclose(blended[mask], np.broadcast_to([0.25, 0.75, 0], blended[mask].shape))
    assert (blended[~mask] == 0).all()
    tex_pos, tex_normal, chart_mask = renderer.uv_rasterize(9)
    assert chart_mask[4, 4]
    np.testing.assert_allclose(tex_pos[4, 4], 0, atol=1e-7)
    np.testing.assert_allclose(tex_normal[4, 4], [0, -1, 0], atol=1e-7)
    textures, shared = renderer.bake_multi([colors[:1], colors[1:2]], [0], [0], texture_size=9)
    assert shared[4, 4]
    np.testing.assert_allclose(textures[0][4, 4], [1, 0, 0], atol=1e-7)
    np.testing.assert_allclose(textures[1][4, 4], [0, 1, 0], atol=1e-7)


def test_inpaint_preserves_clipped_texture_if_it_has_no_usable_holes():
    texture = np.full((2, 2, 3), [-0.5, 0.25, 1.5], dtype=np.float32)
    for mask in [np.ones((2, 2), bool), np.zeros((2, 2), bool)]:
        np.testing.assert_array_equal(
            mesh_render.MeshRender.inpaint(texture, mask),
            np.broadcast_to([0, 0.25, 1], texture.shape),
        )


@pytest.mark.parametrize("opencv_fails", [True, False])
def test_inpaint_fills_a_center_hole_from_surrounding_color(monkeypatch, opencv_fails):
    if opencv_fails:
        def unavailable(*args, **kwargs):
            raise cv2.error("unsupported image format")

        monkeypatch.setattr(cv2, "inpaint", unavailable)
    texture = np.ones((7, 7, 3), dtype=np.float32)
    texture[3, 3] = 0
    mask = np.ones((7, 7), bool)
    mask[3, 3] = False
    np.testing.assert_allclose(mesh_render.MeshRender.inpaint(texture, mask), 1)


def test_occluded_texel_uses_3d_neighbor_and_padding_stops_at_limit():
    rgb = np.zeros((1, 7, 3))
    rgb[0, 1] = [1, 0, 0]
    rgb[0, 3] = [0, 1, 0]
    positions = np.zeros_like(rgb)
    positions[0, [1, 2, 3], 0] = [1, 0.99, 0]
    chart = np.array([[False, True, True, True, False, False, False]])
    covered = np.array([[False, True, False, True, False, False, False]])
    out = mesh_render.MeshRender.inpaint_occlusion_aware(rgb, positions, chart, covered, pad_px=1)
    np.testing.assert_array_equal(out[0], [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0],
                                           [0, 1, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(mesh_render.MeshRender.inpaint_occlusion_aware(
        rgb, positions, chart, np.zeros_like(covered), pad_px=0)[chart], rgb[chart])
    np.testing.assert_array_equal(mesh_render.MeshRender.inpaint_occlusion_aware(
        rgb, positions, np.zeros_like(chart), np.zeros_like(covered)), rgb)


def test_direct_gpu_module_reports_missing_mlx(monkeypatch, triangle):
    import runpy
    import sys

    monkeypatch.setitem(sys.modules, "mlx", None)
    gpu = runpy.run_module("hunyuan_mlx.paint.hy3dpaint_mlx.raster.gpu_raster")
    assert gpu["available"]() is False
    vertices, faces = triangle
    with pytest.raises(RuntimeError, match="MLX not available"):
        gpu["rasterize"](vertices, faces, 5)


def test_camera_space_coordinates_keep_depth_and_project_only_image_axes(planar_renderer):
    renderer = planar_renderer
    camera, clip = renderer._mv(elev=0, azim=0)
    half = 1.15 / np.sqrt(8)
    expected_xy = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
    np.testing.assert_allclose(camera[:, :2], expected_xy)
    np.testing.assert_allclose(camera[:, 2], -1.45)
    np.testing.assert_allclose(clip[:, :2], expected_xy * 2 / 1.2)
    np.testing.assert_allclose(clip[:, 2], 0.45, atol=1e-7)
    np.testing.assert_array_equal(camera[:, 3], np.ones(4))
    np.testing.assert_array_equal(clip[:, 3], np.ones(4))
