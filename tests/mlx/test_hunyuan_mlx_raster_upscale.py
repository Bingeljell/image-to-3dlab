"""Exercise real MLX upscaling and compare Metal raster math with tiny CPU cases."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
utils = pytest.importorskip("mlx.utils")

from hunyuan_mlx.paint.hy3dpaint_mlx import realesrgan
from hunyuan_mlx.paint.hy3dpaint_mlx.raster import cr_raster, gpu_raster

pytestmark = pytest.mark.mlx_unit


def _zero_parameters(model):
    model.update(utils.tree_map(mx.zeros_like, model.parameters()))


def test_rrdb_nested_residual_adds_all_three_dense_block_updates():
    layer = realesrgan.RRDB(nf=1, gc=1)
    _zero_parameters(layer)
    for idx, block in enumerate((layer.rdb1, layer.rdb2, layer.rdb3), start=1):
        block.conv5.bias = mx.array([float(idx)])
    values = mx.array([[[[1.0], [2.0]]]])
    np.testing.assert_allclose(np.asarray(layer(values)), [[[[1.44], [2.64]]]], atol=1e-6)


@pytest.mark.parametrize("tile", [0, 2])
def test_rrdbnet_upsample_repeats_each_pixel_and_clips_at_unit_interval(tile):
    model = realesrgan.RRDBNet(num_block=1, nf=3, gc=1)
    _zero_parameters(model)
    for layer in (model.conv_first, model.conv_up1, model.conv_up2, model.conv_hr, model.conv_last):
        weights = np.zeros(layer.weight.shape, np.float32)
        weights[:, 1, 1, :] = np.eye(3)
        layer.weight = mx.array(weights)
    pixels = np.array([
        [[-1, 0.5, 2], [0.2, 0.4, 0.6], [0.8, 0.7, 0.6]],
        [[0.7, 0.2, 1], [0.3, -0.1, 0.7], [0.9, 0.8, 1]],
    ], dtype=np.float32)
    expected = np.repeat(np.repeat(np.clip(pixels, 0, 1), 4, axis=0), 4, axis=1)
    np.testing.assert_allclose(np.asarray(realesrgan.upscale(model, pixels, tile=tile)), expected, atol=1e-7)


def test_rrdb_loader_finds_converted_local_file_before_legacy_and_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert realesrgan.pick_rrdbnet_weights() is None
    with pytest.raises(FileNotFoundError, match="no RealESRGAN weights"):
        realesrgan.load_rrdbnet()
    legacy = tmp_path / realesrgan.RRDBNET_NPZ
    legacy.parent.mkdir(parents=True)
    legacy.touch()
    assert realesrgan.pick_rrdbnet_weights(tmp_path) == str(legacy)
    current = tmp_path / realesrgan.RRDBNET_MLX
    current.parent.mkdir(parents=True)
    current.touch()
    assert realesrgan.pick_rrdbnet_weights(tmp_path) == str(current)


@pytest.mark.mlx_metal
@pytest.mark.parametrize("batched", [False, True])
def test_metal_raster_matches_cpu_depth_and_perspective_weights(batched):
    assert mx.metal.is_available()
    mx.set_default_device(mx.gpu)
    vertices = np.array([
        [-1, -1, 0, 1], [1, -1, 0, 1], [-2, 2, 0, 2],
        [-1, -1, -0.5, 1], [1, -1, -0.5, 1], [-2, 2, -1, 2],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [3, 4, 5], [3, 4, 5]], dtype=np.int32)
    resolution = (5, 7) if batched else 5
    positions = vertices[None] if batched else vertices
    findices, bary = gpu_raster.rasterize(positions, faces, resolution)
    expected_ids, expected_bary = cr_raster.rasterize(positions, faces, resolution)
    np.testing.assert_array_equal(findices, expected_ids)
    np.testing.assert_allclose(bary, expected_bary, atol=1e-6)
    assert findices[1, 1] == 2
    np.testing.assert_allclose(bary[findices > 0].sum(axis=1), 1, atol=1e-6)
    empty_ids, empty_bary = gpu_raster.rasterize(positions, np.empty((0, 3), np.int32), resolution)
    assert empty_ids.shape == findices.shape and not empty_ids.any()
    assert empty_bary.shape == bary.shape and not empty_bary.any()


def test_metal_tensor_conversion_retains_values_when_already_in_mlx():
    values = mx.array([[1.0, 2.0]], dtype=mx.float32)
    converted = gpu_raster._to_mx(values, mx.float16)
    assert converted.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(converted), [[1.0, 2.0]])
