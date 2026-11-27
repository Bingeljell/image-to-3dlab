"""Tiny configured Hunyuan models with deterministic, synthetic parameters."""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
utils = pytest.importorskip("mlx.utils")

from hunyuan_mlx.paint.hy3dpaint_mlx import dinov2 as paint_dino
from hunyuan_mlx.paint.hy3dpaint_mlx import vae
from hunyuan_mlx.shape.hy3dmlx.models import dinov2 as shape_dino
from hunyuan_mlx.shape.hy3dmlx.models import dit, dit_plain, shape_vae

pytestmark = pytest.mark.mlx_unit


def zero_parameters(model):
    model.update(utils.tree_map(mx.zeros_like, model.parameters()))


@pytest.mark.parametrize("backend", [shape_dino, paint_dino], ids=["shape", "paint"])
@pytest.mark.parametrize("swiglu", [False, True], ids=["gelu", "swiglu"])
def test_dino_class_token_patch_order_and_feedforward_residual(backend, swiglu):
    encoder = backend.DinoImageEncoder(
        hidden=3, heads=1, layers=1, ffn_dim=1, swiglu=swiglu, patch=1, num_tokens=3,
    )
    model = encoder.model
    zero_parameters(model)
    model.embeddings.cls_token = mx.array([[[1.0, 1.0, 4.0]]])
    model.embeddings.position_embeddings = mx.array([[[0., 0., 0.], [0., 1., 0.], [1., 0., 0.]]])
    model.embeddings.patch_embeddings.projection.weight = mx.eye(3).reshape(3, 1, 1, 3)
    layer = model.encoder.layer[0]
    layer.layer_scale2.lambda1 = mx.ones((3,))
    if swiglu:
        layer.mlp.weights_in.bias = mx.array([1., 2.])
        layer.mlp.weights_out.weight = mx.array([[0.], [1.], [0.]])
        shift = 2 / (1 + math.exp(-1))
    else:
        layer.mlp.fc1.bias = mx.array([1.])
        layer.mlp.fc2.weight = mx.array([[0.], [1.], [0.]])
        shift = 0.8413447
    model.layernorm.weight = mx.ones((3,))
    image = mx.array([[[[2., 0., 0.], [0., 4., 0.]]]])
    pre_norm = np.array([[1, 1 + shift, 4], [2, 1 + shift, 0], [1, 4 + shift, 0]])
    expected = (pre_norm - pre_norm.mean(1, keepdims=True)) / np.sqrt(pre_norm.var(1, keepdims=True) + 1e-6)
    np.testing.assert_allclose(np.asarray(encoder(image))[0], expected, atol=1e-6)
    empty = encoder.unconditional_embedding(2, dtype=mx.float16)
    assert empty.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(empty), np.zeros((2, 3, 3)))


@pytest.mark.parametrize(("guidance", "use_guidance"), [(False, True), (True, False), (True, True)])
def test_double_and_single_stream_dit_keep_only_image_tokens(guidance, use_guidance):
    model = dit.Hunyuan3DDiT(
        in_channels=2, context_in_dim=2, hidden_size=2, num_heads=1, mlp_ratio=1,
        depth=1, depth_single_blocks=1, guidance_embed=use_guidance,
    )
    zero_parameters(model)
    model.latent_in.weight = mx.eye(2)
    model.final_layer.linear.weight = mx.eye(2)
    if use_guidance:
        model.guidance_in.out_layer.bias = mx.array([1., 0.])
        model.final_layer.adaLN_modulation[1].weight = mx.array([[2., 0.], [0., 0.], [0., 0.], [0., 0.]])
    values = mx.array([[[2., 0.], [-3., 3.]]])
    context = mx.ones((1, 3, 2))
    output = model(values, mx.array([0.5]), context, guidance=mx.array([3.]) if guidance else None)
    expected = np.array([[[1., -1.], [-1., 1.]]])
    if guidance and use_guidance:
        expected[..., 0] += 2 / (1 + math.exp(-1))
    np.testing.assert_allclose(np.asarray(output), expected, atol=1e-6)


@pytest.mark.parametrize("qk_norm", [False, True])
def test_plain_dit_removes_time_token_and_uses_skip_projection(qk_norm):
    model = dit_plain.HunYuanDiTPlain(
        in_channels=2, hidden_size=2, context_dim=2, depth=3, num_heads=1,
        qk_norm=qk_norm, qkv_bias=True, num_moe_layers=1, num_experts=2, moe_top_k=1,
    )
    zero_parameters(model)
    model.x_embedder.weight = mx.eye(2)
    # Flip the saved early image features to distinguish the skip from the later stream.
    model.blocks[2].skip_linear.weight = mx.array([[0., 1., 0., 0.], [1., 0., 0., 0.]])
    model.blocks[2].skip_norm.weight = mx.ones((2,))
    model.final_layer.norm_final.weight = mx.ones((2,))
    model.final_layer.linear.weight = mx.eye(2)
    latents = mx.array([[[1., 3.], [3., 0.]]])
    output = model(latents, mx.array([0.2]), mx.ones((1, 2, 2)))
    np.testing.assert_allclose(np.asarray(output), [[[1, -1], [-1, 1]]], atol=1e-6)


def test_top_one_expert_uses_unnormalized_router_score_plus_shared_expert():
    moe = dit_plain.MoEBlock(1, num_experts=2, top_k=1)
    zero_parameters(moe)
    moe.gate.weight = mx.array([[1.], [-1.]])
    moe.experts[0].net[2].bias = mx.array([2.])
    moe.experts[1].net[2].bias = mx.array([-4.])
    moe.shared_experts.net[2].bias = mx.array([1.])
    output = moe(mx.array([[[1.], [-2.]]]))
    expected = [[[1 + 2 / (1 + math.exp(-2))], [1 - 4 / (1 + math.exp(-4))]]]
    np.testing.assert_allclose(np.asarray(output), expected, atol=1e-6)


@pytest.mark.parametrize("post_norm", [False, True])
def test_shape_decoder_passes_queries_in_spatial_order_and_chunks(post_norm):
    model = shape_vae.ShapeVAE(
        num_latents=1, embed_dim=2, width=2, heads=1, num_decoder_layers=1,
        num_freqs=0, qk_norm=post_norm, geo_decoder_ln_post=post_norm,
    )
    zero_parameters(model)
    model.post_kl.weight = mx.eye(2)
    model.geo_decoder.query_proj.weight = mx.array([[1., 0., 0.], [0., 1., 0.]])
    model.geo_decoder.output_proj.weight = mx.array([[1., 0.]])
    if post_norm:
        model.geo_decoder.ln_post.weight = mx.ones((2,))
    context = model.decode(mx.array([[[3., 4.]]]))
    np.testing.assert_array_equal(np.asarray(context), [[[3, 4]]])
    points = np.array([[2, 0, 99], [-2, 0, -99], [0, 2, 0]], dtype=np.float32)
    expected = np.array([1, -1, -1]) / math.sqrt(1 + 1e-5) if post_norm else [2, -2, 0]
    np.testing.assert_allclose(model._decode_points(context, points, 2), expected, atol=1e-6)
    assert model._decode_points(context, np.empty((0, 3), np.float32), 2).size == 0
    if not post_norm:
        grid, low, high, shape = model.query_grid(context, bounds=1, octree_resolution=2, num_chunks=4)
        assert shape == (3, 3, 3)
        np.testing.assert_array_equal(grid[:, 0, 0], [-1, 0, 1])
        np.testing.assert_array_equal(low, [-1, -1, -1])
        np.testing.assert_array_equal(high, [1, 1, 1])


def test_paint_autoencoder_splits_quantized_moments_and_decodes_at_input_size():
    model = vae.AutoencoderKL.from_config({
        "in_channels": 2, "out_channels": 2, "latent_channels": 1,
        "block_out_channels": [2, 2], "layers_per_block": 1, "norm_num_groups": 1,
    })
    zero_parameters(model)
    model.quant_conv.bias = mx.array([2., -1.])
    model.decoder.conv_out.bias = mx.array([3., 4.])
    image = mx.zeros((1, 4, 4, 2))
    np.testing.assert_array_equal(np.asarray(model.encode_moments(image)),
                                  np.broadcast_to([2, -1], (1, 2, 2, 2)))
    latent = model.encode_mean(image)
    np.testing.assert_array_equal(np.asarray(latent), np.full((1, 2, 2, 1), 2))
    np.testing.assert_array_equal(np.asarray(model.decode(latent)),
                                  np.broadcast_to([3, 4], (1, 4, 4, 2)))


def test_plain_dit_timestep_uses_sin_then_cos_without_rescaling_and_pads_odd_width():
    value = dit_plain._timesteps(mx.array([0.5]), dim=5, max_period=4)
    expected = [[math.sin(0.5), math.sin(0.25), math.cos(0.5), math.cos(0.25), 0]]
    np.testing.assert_allclose(np.asarray(value), expected, atol=1e-7)


def test_shape_decoder_refines_a_plane_and_omits_far_field_queries():
    model = shape_vae.ShapeVAE(
        num_latents=1, embed_dim=2, width=2, heads=1, num_decoder_layers=0,
        num_freqs=0, qk_norm=False, geo_decoder_ln_post=False,
    )
    zero_parameters(model)
    model.geo_decoder.query_proj.weight = mx.array([[1., 0., 0.], [0., 0., 0.]])
    model.geo_decoder.output_proj.weight = mx.array([[10., 0.]])
    context = model.decode(mx.zeros((1, 1, 2)))
    sdf, low, high, shape = model.query_grid_octree(
        context, bounds=2, octree_resolution=16, min_resolution=8, num_chunks=32,
    )
    assert shape == (17, 17, 17)
    np.testing.assert_array_equal(low, [-2, -2, -2])
    np.testing.assert_array_equal(high, [2, 2, 2])
    np.testing.assert_array_equal(sdf[8], np.zeros((17, 17)))
    np.testing.assert_array_equal(sdf[7], np.full((17, 17), -2.5))
    np.testing.assert_array_equal(sdf[9], np.full((17, 17), 2.5))
    assert np.isnan(sdf[[0, -1]]).all()
