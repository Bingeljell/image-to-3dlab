"""Numerical checks for Hunyuan's MLX layers using only small synthetic tensors."""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hunyuan_mlx.paint.hy3dpaint_mlx import layers2d
from hunyuan_mlx.shape.hy3dmlx import layers

pytestmark = pytest.mark.mlx_unit



@pytest.mark.parametrize("activation", [layers.gelu_erf, layers.gelu_tanh])
def test_gelu_has_expected_values_and_preserves_float16(activation):
    points = mx.array([-2.0, 0.0, 2.0], dtype=mx.float16)
    result = activation(points)
    assert result.dtype == mx.float16
    np.testing.assert_allclose(np.asarray(result), [-0.0455, 0, 1.9545], atol=1e-3)


@pytest.mark.parametrize("affine", [False, True])
def test_layer_normalization_uses_row_statistics_and_optional_affine(affine):
    layer = layers.LayerNorm(2, eps=0, affine=affine)
    if affine:
        layer.weight = mx.array([2.0, 3.0])
        layer.bias = mx.array([4.0, -1.0])
    result = layer(mx.array([[1, 3], [10, 6]], dtype=mx.float16))
    assert result.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(result), [[2, 2], [6, -4]] if affine else [[-1, 1], [1, -1]])


def test_rms_normalization_scales_without_subtracting_input_mean():
    layer = layers.RMSNorm(2, eps=0)
    layer.scale = mx.array([2.0, 0.5])
    result = layer(mx.array([[3.0, 4.0]]))
    np.testing.assert_allclose(np.asarray(result), [[6 / math.sqrt(12.5), 2 / math.sqrt(12.5)]])


@pytest.mark.parametrize("scale", [None, 0.0])
def test_dot_product_attention_matches_two_key_softmax(scale):
    query = mx.array([[[[1.0, 0.0]]]])
    keys = mx.array([[[[1.0, 0.0], [0.0, 1.0]]]])
    values = mx.array([[[[2.0, 4.0], [8.0, 6.0]]]])
    result = layers.sdpa(query, keys, values, scale=scale)
    p = 1 / (1 + math.exp(-1 / math.sqrt(2))) if scale is None else 0.5
    np.testing.assert_allclose(np.asarray(result).ravel(), [2 * p + 8 * (1 - p), 4 * p + 6 * (1 - p)])


@pytest.mark.parametrize("dim", [4, 5])
def test_fractional_timestep_builds_cosine_then_sine_and_pads_odd_width(dim):
    t = layers.timestep_embedding(mx.array([0.0, 0.5]), dim=dim, max_period=4, time_factor=2)
    expected = [[1.0, 1.0, 0.0, 0.0], [math.cos(1), math.cos(0.5), math.sin(1), math.sin(0.5)]]
    if dim == 5:
        expected = [row + [0] for row in expected]
    np.testing.assert_allclose(np.asarray(t), expected, atol=1e-7)


def test_mlp_embedding_applies_both_projections_and_silu():
    layer = layers.MLPEmbedder(1, 1)
    layer.in_layer.weight = mx.array([[2.0]])
    layer.in_layer.bias = mx.array([1.0])
    layer.out_layer.weight = mx.array([[3.0]])
    layer.out_layer.bias = mx.array([-2.0])
    result = layer(mx.array([[1.0], [-0.5]]))
    np.testing.assert_allclose(np.asarray(result), [[9 / (1 + math.exp(-3)) - 2], [-2]])


@pytest.mark.parametrize(("logspace", "include_pi", "include_input"), [
    (True, False, True), (False, True, False),
])
def test_fourier_embedding_orders_features_per_input_axis(logspace, include_pi, include_input):
    embedder = layers.FourierEmbedder(3, include_pi, input_dim=2, include_input=include_input,
                                      logspace=logspace)
    values = [0.0, 0.25]
    freqs = np.array([1, 2, 4] if logspace else [1, 2.5, 4]) * (math.pi if include_pi else 1)
    phases = (np.array(values)[:, None] * freqs).ravel()
    expected = [*(values if include_input else []), *np.sin(phases), *np.cos(phases)]
    assert embedder.out_dim == len(expected)
    np.testing.assert_allclose(np.asarray(embedder(mx.array([values])))[0], expected, atol=1e-7)


def test_fourier_embedding_without_frequencies_keeps_coordinates():
    embedder = layers.FourierEmbedder(0, input_dim=2, include_input=False)
    xyz = mx.array([[[1.0, -2.0]]])
    assert embedder.out_dim == 2
    np.testing.assert_array_equal(np.asarray(embedder(xyz)), [[[1, -2]]])


def _set_linear(layer, weight, bias):
    layer.weight = mx.array(weight, dtype=mx.float32)
    layer.bias = mx.array(bias, dtype=mx.float32)


def _set_conv(layer, centre, bias=None):
    shape = layer.weight.shape
    kernel = np.zeros(shape, np.float32)
    kernel[:, shape[1] // 2, shape[2] // 2, :] = centre
    layer.weight = mx.array(kernel)
    layer.bias = mx.zeros((shape[0],)) if bias is None else mx.array(bias, dtype=mx.float32)


def test_group_norm_uses_spatial_and_within_group_dimensions():
    layer = layers2d.GroupNorm2d(2, 4, eps=0)
    layer.weight = mx.array([1.0, 2.0, 1.0, 0.5])
    layer.bias = mx.array([0.0, 1.0, 2.0, -1.0])
    values = mx.array([[[[1, 3, -1, 1], [3, 1, 1, -1]]]], dtype=mx.float16)
    result = layer(values)
    assert result.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(result), [[[[-1, 3, 1, -0.5], [1, -1, 3, -1.5]]]])


@pytest.mark.parametrize("padding", [0, 1])
def test_downsample_padding_selects_expected_stride_centres(padding):
    layer = layers2d.Downsample2D(1, out_channels=1 if padding == 0 else None, padding=padding)
    _set_conv(layer.conv, [[1]])
    pixels = mx.array(np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1))
    result = np.asarray(layer(pixels))[0, ..., 0]
    np.testing.assert_array_equal(result, [[5, 7], [13, 15]] if padding == 0 else [[0, 2], [8, 10]])


def test_nearest_upsample_then_convolution_duplicates_each_texel():
    layer = layers2d.Upsample2D(1, out_channels=2)
    _set_conv(layer.conv, [[1], [-1]])
    result = np.asarray(layer(mx.array([[[[1.0], [3.0]]]])))[0]
    expected = [[[1, -1], [1, -1], [3, -3], [3, -3]]] * 2
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("with_time", [False, True])
def test_resnet_residual_includes_conditioning_and_channel_projection(with_time):
    channels = 2 if with_time else 1
    layer = layers2d.ResnetBlock2D(1, channels, temb_channels=1 if with_time else None, groups=1)
    _set_conv(layer.conv1, [[0]] * channels)
    _set_conv(layer.conv2, np.eye(channels))
    x = mx.array([[[[2.0], [4.0]]]])
    if with_time:
        _set_linear(layer.time_emb_proj, [[1.0], [-1.0]], [0.0, 0.0])
        _set_conv(layer.conv_shortcut, [[2], [3]])
        result = np.asarray(layer(x, mx.array([[1.0]])))[0, 0]
        positive = 1 / (1 + math.exp(-1))
        negative = -1 / (1 + math.exp(1))
        np.testing.assert_allclose(result, [[4 + positive, 6 + negative],
                                           [8 + positive, 12 + negative]], atol=2e-6)
        np.testing.assert_array_equal(np.asarray(layer(x))[0, 0], [[4, 6], [8, 12]])
    else:
        np.testing.assert_array_equal(np.asarray(layer(x)), np.asarray(x))


def test_spatial_attention_returns_the_analytical_two_pixel_residual():
    layer = layers2d.VAEAttnBlock(1, groups=1, eps=0)
    for projection in (layer.to_q, layer.to_k, layer.to_v, layer.to_out):
        _set_linear(projection, [[1]], [0])
    mixed = math.tanh(1)
    result = layer(mx.array([[[[1.0], [3.0]]]]))
    np.testing.assert_allclose(np.asarray(result).ravel(), [1 - mixed, 3 + mixed], atol=1e-7)
