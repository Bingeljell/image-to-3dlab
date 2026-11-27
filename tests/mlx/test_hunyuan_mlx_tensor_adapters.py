"""Exercise tensor assignment, quantization and sampling on real MLX arrays."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from hunyuan_mlx.paint.hy3dpaint_mlx import convert as paint_convert
from hunyuan_mlx.shape.hy3dmlx import convert as shape_convert
from hunyuan_mlx.shape.hy3dmlx import sampler

pytestmark = pytest.mark.mlx_unit


def test_paint_converter_loads_numpy_convolution_and_mlx_bias_into_a_real_layer(capsys):
    module = nn.Conv2d(2, 1, kernel_size=2)
    # Only the upper-right filter element is nonzero, with different input-channel factors.
    weights = np.array([[[[0, 2], [0, 0]], [[0, -1], [0, 0]]]], dtype=np.float32)
    assert paint_convert.load_torch_weights(
        module,
        {"proj.weight": weights, "proj.bias": mx.array([0.5])},
        renames=[("proj.", "")], verbose=True,
    ) is module
    values = mx.array([[[[0., 0.], [3., 2.]], [[0., 0.], [0., 0.]]]])
    np.testing.assert_allclose(np.asarray(module(values)), [[[[4.5]]]])
    assert "loaded 2 tensors" in capsys.readouterr().out
    with pytest.raises(KeyError, match="missing parameter names.*bias"):
        paint_convert.load_torch_weights(module, {"weight": weights})


def test_shape_assign_installs_expected_tensors_and_counts_ignored_extra():
    module = nn.Linear(2, 1)
    matrix = mx.array([[1., -2.]])
    value = mx.array([0.25])
    items = [("weight", matrix), ("bias", value), ("unused", mx.ones((1,)))]
    assert shape_convert._assign(module, items, "test projection") == (2, 1)
    np.testing.assert_allclose(np.asarray(module(mx.array([[3., 2.]]))), [[-0.75]])


def test_shape_quantization_targets_wide_linears_and_keeps_excluded_layers():
    module = nn.Module()
    module.projection = nn.Linear(64, 1, bias=False)
    module.embeddings = nn.Linear(64, 1, bias=False)
    module.narrow = nn.Linear(2, 1, bias=False)
    module.normalization = nn.LayerNorm(64)
    # Endpoints are exactly quantizable at either four or eight bits.
    module.projection.weight = mx.array([[0.] * 32 + [1.] * 32])
    shape_convert._quantize(module, skip=("embeddings",), bits=4)
    assert isinstance(module.projection, nn.QuantizedLinear)
    assert isinstance(module.embeddings, nn.Linear)
    assert isinstance(module.narrow, nn.Linear)
    assert isinstance(module.normalization, nn.LayerNorm)
    np.testing.assert_allclose(np.asarray(module.projection(mx.ones((1, 64)))), [[32]])


@pytest.mark.parametrize("guidance", [False, True])
def test_shape_sampler_real_tensors_seed_and_weighted_velocity(guidance):
    shape, seed = (1, 2), 17
    latent = mx.random.normal(shape, dtype=mx.float32, key=mx.random.key(seed))
    steps = []
    visits = []

    def velocity(x, timestep):
        visits.append((x.shape, timestep.item()))
        if guidance:
            return mx.concatenate([3 * mx.ones_like(x[:1]), mx.ones_like(x[:1])], axis=0)
        return -2 * mx.ones_like(x)

    result = sampler.denoise(
        velocity, shape, num_inference_steps=3, guidance_scale=2.0, cfg=guidance,
        seed=seed, dtype=mx.float32 if guidance else None,
        sigmas=(np.array([0., 0.5]), np.array([0., 0.5, 1.])) if guidance else None,
        progress=lambda i, n: steps.append((i, n)),
    )
    assert result.dtype == mx.float32
    np.testing.assert_allclose(np.asarray(result), np.asarray(latent) + (5 if guidance else -2))
    shape_at_step = (2, 2) if guidance else shape
    assert visits == [(shape_at_step, 0.0), (shape_at_step, 0.5)]
    assert steps == [(1, 2 if guidance else 3), (2, 2 if guidance else 3)]
