"""Tiny paint UNets exercise view conditioning, skip connections and voxel geometry."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
utils = pytest.importorskip("mlx.utils")

from hunyuan_mlx.paint.hy3dpaint_mlx import unet, unet2p5d, unet2p5d_pbr

pytestmark = pytest.mark.mlx_unit


def _zero_parameters(model):
    model.update(utils.tree_map(mx.zeros_like, model.parameters()))


def _set_centre(layer, matrix):
    weights = np.zeros(layer.weight.shape, dtype=np.float32)
    weights[:, weights.shape[1] // 2, weights.shape[2] // 2, :] = matrix
    layer.weight = mx.array(weights)


def _set_unet_skip_identity(model):
    """Route the original sample through the final skip and ignore other branches."""
    _zero_parameters(model)
    _set_centre(model.conv_in, np.eye(model.conv_in.weight.shape[0], model.conv_in.weight.shape[-1]))
    last = model.up_blocks[-1].resnets[-1].conv_shortcut
    channels = last.weight.shape[0]
    _set_centre(last, np.concatenate([np.zeros((channels, channels)), np.eye(channels)], axis=1))
    model.conv_norm_out.weight = mx.ones((channels,))
    _set_centre(model.conv_out, np.eye(channels))


def _normalized_silu(sample):
    spatial_channels = tuple(range(sample.ndim - 3, sample.ndim))
    normalized = ((sample - sample.mean(axis=spatial_channels, keepdims=True))
                  / np.sqrt(sample.var(axis=spatial_channels, keepdims=True) + 1e-5))
    return normalized / (1 + np.exp(-normalized))


@pytest.mark.parametrize("timestep", ["python", "scalar", "batch"])
def test_unet_recovers_early_spatial_features_through_skips(timestep):
    model = unet.UNet2DConditionModel(
        in_channels=2, out_channels=2, block_out_channels=(2, 2, 2), layers_per_block=1,
        cross_attention_dim=2, attention_head_dim=(1, 1, 1), norm_num_groups=1, num_class_embeds=2,
    )
    _set_unet_skip_identity(model)
    sample = np.arange(2 * 4 * 4 * 2, dtype=np.float32).reshape(2, 4, 4, 2)
    t = {"python": 1, "scalar": mx.array(1), "batch": mx.array([1, 2])}[timestep]
    class_labels = None if timestep == "batch" else mx.array([0, 1])
    output = model(mx.array(sample), t, mx.zeros((2, 1, 2)), class_labels=class_labels)
    np.testing.assert_allclose(np.asarray(output), _normalized_silu(sample), atol=3e-7)


@pytest.mark.parametrize("route", ["compute", "cached", "unconditional"])
def test_rgb_paint_preserves_view_order_and_reference_cache(route):
    model = unet2p5d.UNet2p5DConditionModel({
        "block_out_channels": [4, 4], "layers_per_block": 1,
        "attention_head_dim": [1, 1], "norm_num_groups": 1,
    }, max_num_ref=1, max_num_gen=2)
    _set_unet_skip_identity(model.unet)
    _zero_parameters(model.unet_dual)
    sample = np.arange(32, dtype=np.float32).reshape(1, 2, 2, 2, 4)
    sample[:, 1] = -sample[:, 1]
    ref = mx.zeros((1, 1, 2, 2, 4))
    reference_cache = model.compute_condition_embed(ref) if route == "cached" else None
    if reference_cache is not None:
        assert set(reference_cache) == {"down_0_0_0", "mid_0_0", "up_1_0_0", "up_1_1_0"}
        np.testing.assert_array_equal(np.asarray(reference_cache["mid_0_0"]), [[[0, 0, 0, 0]]])
    output = model(mx.array(sample), 1, mx.zeros((1, 1, 1024)), mx.zeros(sample.shape),
                   mx.zeros(sample.shape), ref, mx.array([[0, 1]]),
                   ref_scale=0 if route == "unconditional" else 1, condition_embed_dict=reference_cache)
    np.testing.assert_allclose(np.asarray(output), _normalized_silu(sample), atol=3e-7)


@pytest.mark.parametrize("tokenwise", [True, False])
def test_image_projection_keeps_tokens_in_order(tokenwise):
    model = unet2p5d_pbr.ImageProjModel(2, 2, 2)
    model.proj.weight = mx.array([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])
    model.proj.bias = mx.zeros((4,))
    image = mx.array([[[0., 2.], [3., 0.]]])
    if not tokenwise:
        image = image[:, 0]
    expected = [[[-1, 1], [1, -1], [1, -1], [-1, 1]]]
    np.testing.assert_allclose(np.asarray(model(image)), expected if tokenwise else [expected[0][:2]], atol=1e-5)


def test_voxel_grid_ignores_background_and_discards_thin_foreground():
    position = np.ones((1, 1, 8, 8, 3), dtype=np.float32)
    position[0, 0, 0:4, 0:4] = [0.25, 0.5, 0.75]
    position[0, 0, 4:8, 4:8] = [0, 0.25, 0.5]
    position[0, 0, 0, 4] = [0.5, 0.5, 0.5]
    result = unet2p5d_pbr.compute_voxel_indices(position, grid_res=2, voxel_res=9)
    np.testing.assert_array_equal(result, [[[2, 4, 6], [4, 4, 4], [0, 0, 0], [0, 2, 4]]])
    # A single foreground pixel in an 8x8 pooling window falls below the valid-sample threshold.
    position[:] = 1
    position[0, 0, 0, 0] = [0.5, 0.5, 0.5]
    np.testing.assert_array_equal(unet2p5d_pbr.compute_voxel_indices(position, 1, 9), [[[0, 0, 0]]])


@pytest.mark.parametrize("cached", [False, True])
def test_pbr_paint_routes_two_materials_and_projects_reference_conditions(cached):
    model = unet2p5d_pbr.UNet2p5DPBRConditionModel({
        "block_out_channels": [4, 4], "layers_per_block": 1,
        "attention_head_dim": [1, 1], "norm_num_groups": 1,
    })
    _set_unet_skip_identity(model.unet)
    _zero_parameters(model.unet_dual)
    sample = np.arange(2 * 8 * 8 * 4, dtype=np.float32).reshape(1, 2, 1, 8, 8, 4)
    sample[:, 1] = -sample[:, 1]
    normal = mx.zeros((1, 1, 8, 8, 4))
    pos = np.zeros((1, 1, 8, 8, 3), dtype=np.float32)
    features = mx.zeros((1, 1, 1536))
    cond = model.prepare(normal, features, pos, 8, 1) if cached else None
    if cond is not None:
        assert set(cond["rope"]) == {64, 16, 4, 1}
        np.testing.assert_array_equal(np.asarray(cond["rope"][1][0]), np.ones((2, 1, 64)))
        np.testing.assert_array_equal(np.asarray(cond["dino"]), np.zeros((1, 4, 1024)))
    output = model(mx.array(sample), 1, normal, normal, normal, features, pos,
                   ref_scale=0 if cached else 1, cond=cond)
    np.testing.assert_allclose(np.asarray(output), _normalized_silu(sample), atol=3e-7)


def test_plain_down_block_keeps_pre_stride_and_post_stride_states():
    block = unet.DownBlock2D(1, 1, temb_ch=None, n_layers=1, groups=1, add_down=True)
    _zero_parameters(block)
    _set_centre(block.downsamplers[0].conv, [[1]])
    sample = np.arange(16, dtype=np.float32).reshape(1, 4, 4, 1)
    output, saved = block(mx.array(sample), None)
    np.testing.assert_array_equal(np.asarray(saved[0]), sample)
    np.testing.assert_array_equal(np.asarray(saved[1]), sample[:, ::2, ::2])
    np.testing.assert_array_equal(np.asarray(output), sample[:, ::2, ::2])
