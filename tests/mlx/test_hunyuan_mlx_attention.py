"""Small attention matrices with closed-form expected outputs."""

import math

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from hunyuan_mlx.paint.hy3dpaint_mlx import attention, attention_pbr

pytestmark = pytest.mark.mlx_unit



def _set_linear(layer, value=0, bias=0):
    rows, cols = layer.weight.shape
    layer.weight = mx.array(np.eye(rows, cols, dtype=np.float32) * value)
    if hasattr(layer, "bias"):
        layer.bias = mx.full((rows,), bias, dtype=mx.float32)


def _uniform_attention(layer, scale):
    _set_linear(layer.to_q)
    _set_linear(layer.to_k)
    _set_linear(layer.to_v, scale)
    _set_linear(layer.to_out[0], 1)


def _disable_base_attention(layer):
    _uniform_attention(layer.attn1, 0)
    _uniform_attention(layer.attn2, 0)
    _set_linear(layer.ff.net[2])


@pytest.mark.parametrize(("flip", "dim"), [(True, 4), (False, 5)])
def test_diffusion_timestep_embedding_matches_known_phase(flip, dim):
    emb = attention.timestep_embedding(mx.array([1.0]), dim, flip_sin_to_cos=flip,
                                       downscale_freq_shift=1, max_period=2)
    sin = [math.sin(1), math.sin(0.5)]
    cos = [math.cos(1), math.cos(0.5)]
    expected = [*cos, *sin] if flip else [*sin, *cos, 0]
    np.testing.assert_allclose(np.asarray(emb)[0], expected, atol=1e-7)


def test_time_mlp_and_gated_feedforward_use_their_activations():
    x = mx.array([[0.0], [1.0]])
    timestep = attention.TimestepEmbedding(1, 1)
    _set_linear(timestep.linear_1, 2, -1)
    _set_linear(timestep.linear_2, 3, 2)
    np.testing.assert_allclose(np.asarray(timestep(x)).ravel(), [
        2 - 3 / (1 + math.exp(1)), 2 + 3 / (1 + math.exp(-1)),
    ])

    feed_forward = attention.FeedForward(1, mult=1)
    feed_forward.net[0].proj.weight = mx.array([[2.0], [1.0]])
    feed_forward.net[0].proj.bias = mx.array([1.0, 0.0])
    _set_linear(feed_forward.net[2], 2, -1)
    np.testing.assert_allclose(np.asarray(feed_forward(x)).ravel(), [-1, 6 * 0.8413447 - 1], atol=2e-6)


@pytest.mark.parametrize("cross", [False, True])
def test_multihead_attention_distinguishes_self_and_cross_context(cross):
    layer = attention.Attention(2, 2, 1, context_dim=1 if cross else None, bias=True)
    _set_linear(layer.to_q)
    _set_linear(layer.to_k)
    layer.to_v.weight = mx.array([[2.0], [-1.0]]) if cross else mx.eye(2)
    layer.to_v.bias = mx.zeros((2,))
    _set_linear(layer.to_out[0], 1, 1)
    x = mx.array([[[1.0, 2.0], [3.0, 8.0]]])
    result = layer(x, mx.array([[[2.0], [4.0], [6.0]]]) if cross else None)
    np.testing.assert_allclose(np.asarray(result), [[[9, -3], [9, -3]]] if cross else [[[3, 6], [3, 6]]])


def test_reference_and_multiview_attention_use_their_own_scales():
    layer = attention.BasicTransformerBlock(2, 1, 2, 2, eps=0, use_ma=True, use_ra=True, layer_name="block")
    _disable_base_attention(layer)
    _uniform_attention(layer.attn_refview, 1)
    _uniform_attention(layer.attn_multiview, 1)
    x = mx.array([[[1.0, 3.0]], [[4.0, 0.0]]])
    cache = {}
    np.testing.assert_allclose(np.asarray(layer(x, xattn={"mode": "w", "condition_embed_dict": cache})),
                               np.asarray(x))
    np.testing.assert_allclose(np.asarray(cache["block"]), [[[-1, 1]], [[1, -1]]])
    np.testing.assert_allclose(np.asarray(layer(x)), np.asarray(x))
    cache["block"] = mx.array([[[2.0, 4.0], [4.0, 8.0]]])
    options = {"mode": "r", "num_in_batch": 2, "condition_embed_dict": cache, "ref_scale": 0.5}
    # Opposite normalized views cancel in the multiview path.
    np.testing.assert_allclose(np.asarray(layer(x, xattn=options)), [[[2.5, 6]], [[5.5, 3]]])
    paired = mx.array([[[1.0, 3.0]], [[0.0, 2.0]]])
    options["mva_scale"] = 2
    np.testing.assert_allclose(np.asarray(layer(paired, xattn=options)), [[[0.5, 8]], [[-0.5, 7]]])


@pytest.mark.parametrize("pbr", [None, {}])
def test_image_transformer_projects_spatial_tokens_and_keeps_residual(pbr):
    layer = attention.Transformer2DModel(2, 1, 2, 2, depth=2, groups=1, eps=0, pbr=pbr)
    _set_linear(layer.proj_in, 1)
    _set_linear(layer.proj_out, 2, 3)
    for block in layer.transformer_blocks:
        _disable_base_attention(block.transformer if pbr is not None else block)
    x = mx.array([[[[1.0, -1.0], [-1.0, 1.0]]]])
    np.testing.assert_allclose(np.asarray(layer(x, xattn={})), [[[[6, 0], [0, 6]]]])


def test_rotary_spatial_phases_follow_xyz_order_and_rotate_pairs():
    position = mx.array([[[1, 2, 0], [0, 0, 0]]])
    cos, sin = attention_pbr.get_3d_rotary_pos_embed(position, embed_dim=16, voxel_resolution=3, theta=8)
    phases = np.array([[[1, 1, .5, .5, .25, .25, 2, 2, 1, 1, .5, .5, 0, 0, 0, 0], [0] * 16]])
    np.testing.assert_allclose(np.asarray(cos), np.cos(phases), atol=1e-7)
    np.testing.assert_allclose(np.asarray(sin), np.sin(phases), atol=1e-7)
    pairs = np.tile([1., 0.], 8).reshape(1, 1, 1, 16)
    rotated = attention_pbr.apply_rotary_emb(mx.array(pairs), cos, sin)
    np.testing.assert_allclose(np.asarray(rotated)[0, 0, :, ::2], np.cos(phases)[0, :, ::2], atol=1e-7)
    np.testing.assert_allclose(np.asarray(rotated)[0, 0, :, 1::2], np.sin(phases)[0, :, 1::2], atol=1e-7)


def test_pbr_conditions_each_material_with_distinct_value_projections():
    layer = attention_pbr.PBRTransformerBlock(
        2, 1, 2, 2, use_ma=True, use_ra=True, use_mda=True, use_dino=True, layer_name="pbr",
    )
    layer.transformer.norm1.eps = 0
    _disable_base_attention(layer.transformer)
    base = layer.transformer.attn1
    _uniform_attention(base, 1)
    _set_linear(base.processor.to_q_mr)
    _set_linear(base.processor.to_k_mr)
    _set_linear(base.processor.to_v_mr, 2)
    _set_linear(base.processor.to_out_mr[0], 1)
    reference = layer.attn_refview
    _uniform_attention(reference, 1)
    _set_linear(reference.processor.to_v_mr, 3)
    _set_linear(reference.processor.to_out_mr[0], 1)
    _uniform_attention(layer.attn_multiview, 1)
    _uniform_attention(layer.attn_dino, 1)
    cache = {}
    dino = mx.array([[[8., 10.], [12., 14.]]])
    x = mx.array([[[1., 3.]], [[4., 0.]], [[3., 1.]], [[4., 2.]]])
    layer(x[:1], context=None, xattn={"mode": "w", "condition_embed_dict": cache, "dino": dino})
    np.testing.assert_allclose(np.asarray(cache["pbr"]), [[[-1, 1]]])

    cache["pbr"] = mx.array([[[2., 4.], [4., 8.]]])
    opts = {
        "mode": "r", "num_in_batch": 2, "n_pbr": 2, "condition_embed_dict": cache,
        "dino": dino, "ref_scale": 0.5, "mva_scale": 0.25,
    }
    exact = np.asarray(x) + np.array([[[-1, 1]], [[1, -1]], [[2, -2]], [[2, -2]]])
    exact += np.array([[[1.5, 3]], [[1.5, 3]], [[4.75, 8.75]], [[4.75, 8.75]]])
    exact += [10, 12]
    np.testing.assert_allclose(np.asarray(layer(x, context=None, xattn=opts)), exact)
    opts["rope_by_tokens"] = {2: (mx.ones((2, 2, 2)), mx.zeros((2, 2, 2)))}
    np.testing.assert_allclose(np.asarray(layer(x, context=None, xattn=opts)), exact)


def test_material_processor_can_load_only_query_and_key_projections():
    processor = attention_pbr.MaterialProc(dim=2, inner=1, kinds=["q"])
    assert set(processor.parameters()) == {"to_q_mr", "to_k_mr"}
    processor.to_q_mr.weight = mx.array([[1.0, -2.0]])
    processor.to_k_mr.weight = mx.array([[0.5, 0.0]])
    tokens = mx.array([[2.0, 1.0], [4.0, -1.0]])
    np.testing.assert_array_equal(np.asarray(processor.to_q_mr(tokens)), [[0.0], [6.0]])
    np.testing.assert_array_equal(np.asarray(processor.to_k_mr(tokens)), [[1.0], [2.0]])
