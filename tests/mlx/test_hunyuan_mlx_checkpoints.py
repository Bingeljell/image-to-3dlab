"""Round-trip small, synthetic checkpoints through Hunyuan's actual loaders."""

import runpy
import sys

import numpy as np
import pytest
import yaml

mx = pytest.importorskip("mlx.core")
utils = pytest.importorskip("mlx.utils")

from hunyuan_mlx.paint.hy3dpaint_mlx import realesrgan
from hunyuan_mlx.shape.hy3dmlx import convert, pipeline
from hunyuan_mlx.shape.hy3dmlx.models import dinov2, dit, dit_plain, shape_vae

pytestmark = pytest.mark.mlx_unit


def _flat_fixture(model, prefix):
    return {prefix + key: mx.zeros_like(value)
            for key, value in utils.tree_flatten(model.parameters())}


@pytest.mark.parametrize("plain", [False, True])
def test_shape_loader_uses_checkpoint_config_and_strict_namespace_assignment(tmp_path, capsys, plain):
    kind = dit_plain.HunYuanDiTPlain if plain else dit.Hunyuan3DDiT
    extra = {"context_dim": 2, "num_moe_layers": 0} if plain else {
        "context_in_dim": 2, "depth_single_blocks": 0,
    }
    dit_params = {"in_channels": 2, "hidden_size": 2, "depth": 0, "num_heads": 1, **extra}
    vae_params = {"num_latents": 1, "embed_dim": 2, "width": 2, "heads": 1, "num_decoder_layers": 0}
    cfg = {
        "model": {"target": f"shape.models.{kind.__name__}", "params": dit_params},
        "vae": {"params": vae_params},
        "conditioner": {"params": {"main_image_encoder": {"kwargs": {"config": {
            "hidden_size": 2, "num_attention_heads": 1, "num_hidden_layers": 1, "patch_size": 2,
            "use_swiglu_ffn": False, "layer_norm_eps": 1e-5,
        }}}}},
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    fixture = _flat_fixture(
        dinov2.DinoImageEncoder(hidden=2, heads=1, layers=1, ffn_dim=3, swiglu=False, patch=2),
        convert.COND_PREFIX,
    )
    fixture.update(_flat_fixture(kind(**dit_params), "model."))
    fixture.update(_flat_fixture(shape_vae.ShapeVAE(**vae_params), "vae."))
    patch = convert.COND_PREFIX + "model.embeddings.patch_embeddings.projection.weight"
    channels_first = np.arange(24, dtype=np.float32).reshape((2, 3, 2, 2))
    fixture[patch] = mx.array(channels_first)
    fixture["vae.training.discarded"] = mx.ones((1,))
    fixture["other.sample"] = mx.ones((1,))
    model_file = tmp_path / "model.fp16.safetensors"
    mx.save_safetensors(str(model_file), fixture)

    dino, diffusion, vae, model_cfg = convert.load_models(
        str(tmp_path), dtype=mx.float16 if plain else None, quantize=4 if plain else None,
        verbose=plain,
    )
    assert isinstance(diffusion, kind)
    assert vae.latent_shape == (1, 2)
    assert model_cfg == cfg
    patch_conv = dino.model.embeddings.patch_embeddings.projection
    assert patch_conv.weight.dtype == (mx.float16 if plain else mx.float32)
    np.testing.assert_array_equal(np.asarray(patch_conv.weight), channels_first.transpose(0, 2, 3, 1))
    printed = capsys.readouterr().out
    assert ("4-bit DiT+DINO" in printed) == plain
    assert ("(+1)" in printed) == plain

    if not plain:
        model = pipeline.Hunyuan3DShapePipeline.from_pretrained(str(tmp_path))
        assert model.dtype == mx.float32
        assert isinstance(model.dit, kind)
        assert kind.__name__ in capsys.readouterr().out

    del fixture[patch]
    mx.save_safetensors(str(model_file), fixture)
    with pytest.raises(KeyError, match=r"dino: 1 missing keys.+patch_embeddings.projection.weight"):
        convert.load_models(str(tmp_path))


@pytest.mark.parametrize("native", [False, True], ids=["legacy-npz", "native-safetensors"])
def test_rrdb_loader_reads_each_layout_and_applies_the_same_filter(tmp_path, monkeypatch, native):
    network = realesrgan.RRDBNet(num_block=0, nf=2, gc=1)
    original_factory = realesrgan.RRDBNet
    monkeypatch.setattr(realesrgan, "RRDBNet", lambda: original_factory(num_block=0, nf=2, gc=1))
    params = {key: np.zeros(value.shape, np.float32) for key, value in utils.tree_flatten(network.parameters())}
    first = params["conv_first.weight"]
    first[:, 1, 1, :] = [[2, 1, 0], [-1, 0, 1]]
    params["conv_first.bias"] = np.array([0.5, -0.5], np.float32)
    if native:
        path = tmp_path / "tiny.safetensors"
        mx.save_safetensors(str(path), {k: mx.array(v) for k, v in params.items()})
    else:
        path = tmp_path / "tiny.npz"
        np.savez(path, **{k: v.transpose(0, 3, 1, 2) if v.ndim == 4 else v for k, v in params.items()})
    loaded = realesrgan.load_rrdbnet(str(path))
    result = loaded.conv_first(mx.array([[[[1, 2, 4]]]], dtype=mx.float32))
    np.testing.assert_allclose(np.asarray(result), [[[[4.5, 2.5]]]])


@pytest.mark.parametrize("fp16", [False, True])
def test_paint_checkpoint_converter_writes_layout_and_precision_without_changing_keys(
    tmp_path, monkeypatch, capsys, fp16,
):
    src, dst = tmp_path / "input.safetensors", tmp_path / "output.safetensors"
    array = np.arange(24, dtype=np.float32).reshape((2, 3, 2, 2))
    mx.save_safetensors(str(src), {
        "conv.weight": mx.array(array),
        "linear.weight": mx.array([[0.25, 0.75]]),
        "integer.buffer": mx.array([1, 2]),
    })
    argv = ["convert_weights.py", str(src), str(dst)]
    if fp16:
        argv.append("--fp16")
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_module("hunyuan_mlx.paint.scripts.convert_weights", run_name="__main__")
    output = mx.load(str(dst))
    assert set(output) == {"conv.weight", "linear.weight", "integer.buffer"}
    dtype = mx.float16 if fp16 else mx.float32
    assert output["conv.weight"].dtype == output["linear.weight"].dtype == dtype
    assert output["integer.buffer"].dtype == mx.int32
    np.testing.assert_array_equal(np.asarray(output["conv.weight"]), array.transpose(0, 2, 3, 1))
    np.testing.assert_array_equal(np.asarray(output["linear.weight"]), [[0.25, 0.75]])
    assert "converted 3 tensors" in capsys.readouterr().out


def test_paint_checkpoint_converter_is_importable_and_handles_already_half_weights(
    tmp_path, monkeypatch,
):
    from hunyuan_mlx.paint.scripts import convert_weights

    # Deliberately unusable as CLI arguments. The imported converter uses explicit paths.
    monkeypatch.setattr(sys, "argv", ["convert_weights.py"])
    src, dst = tmp_path / "source.safetensors", tmp_path / "result.safetensors"
    mx.save_safetensors(str(src), {"already_half": mx.array([[1.5]], dtype=mx.float16)})
    convert_weights.convert(str(src), str(dst), fp16=True)
    result = mx.load(str(dst))["already_half"]
    assert result.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(result), [[1.5]])
