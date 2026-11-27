"""Exercise checkpoint layout conversion with tiny, real arrays."""

import ast
import io
import re
import runpy
import sys

import numpy as np
import pytest

from hunyuan_mlx.paint.hy3dpaint_mlx.convert import parameter_names, remap_parameters
from hunyuan_mlx.paint.scripts import convert_realesrgan
from hunyuan_mlx.shape.hy3dmlx.convert import COND_PREFIX, load_config, split_checkpoint


def test_parameter_names_follows_nested_mapping_and_sequences():
    tree = {"norm": {"bias": 0}, "blocks": [{"weight": 1}, (2, {})], 4: {0: 3}}
    assert parameter_names(tree) == ["norm.bias", "blocks.0.weight", "blocks.1.0", "4.0"]
    assert parameter_names(9) == []


def test_paint_remaps_attention_names_and_transposes_convolution_only():
    conv = np.arange(24).reshape(2, 3, 2, 2)
    attention = np.array([[1, 2], [3, 4]])
    params = dict(remap_parameters(
        {"path.to_out.0.weight": attention, "stem.weight": conv},
        renames=[(".to_out.0.", ".to_out."), ("stem.", "input.")],
    ))
    assert set(params) == {"path.to_out.weight", "input.weight"}
    assert params["path.to_out.weight"] is attention
    assert params["input.weight"].shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(params["input.weight"][1, 1, 0], [14, 18, 22])


def test_shape_checkpoint_splits_prefixes_and_converts_dino_patch_axes():
    patch_key = "model.embeddings.patch_embeddings.projection.weight"
    patch = np.arange(24).reshape(2, 3, 2, 2)
    other = np.array([1, 2])
    dino, dit, vae = split_checkpoint([
        (COND_PREFIX + patch_key, patch),
        (COND_PREFIX + "bias", other),
        (COND_PREFIX + "patch_embeddings.projection.weight", other),
        ("model.x_embedder.weight", other),
        ("vae.decode.weight", other),
        ("training.optimizer", other),
    ])
    assert set(dino) == {patch_key, "bias", "patch_embeddings.projection.weight"}
    np.testing.assert_array_equal(dino[patch_key][1, 0, 1], [13, 17, 21])
    assert dino["bias"] is dino["patch_embeddings.projection.weight"] is other
    assert dit == {"x_embedder.weight": other}
    assert vae == {"decode.weight": other}


def test_shape_configuration_reads_actual_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("scheduler:\n  params:\n    pcm_timesteps: 12\n")
    assert load_config(str(tmp_path)) == {"scheduler": {"params": {"pcm_timesteps": 12}}}


@pytest.mark.parametrize("layout", ["params_ema", "params", "flat"])
def test_realesrgan_converts_local_checkpoint(tmp_path, monkeypatch, layout):
    import torch

    pth = tmp_path / "local.pth"
    out = tmp_path / "converted" / "out.npz"
    tensor = torch.tensor([[1., 2.]])
    tensors = {"example.weight": tensor}
    state = {
        "params_ema": {"params_ema": tensors, "params": {"unused": tensor}},
        "params": {"params_ema": {}, "params": tensors},
        "flat": tensors,
    }[layout]
    torch.save(state, pth)
    monkeypatch.setattr(sys, "argv", ["convert_realesrgan.py", "--pth", str(pth), "--out", str(out)])
    assert convert_realesrgan.main() == 0
    with np.load(out) as result:
        assert result.files == ["example.weight"]
        np.testing.assert_array_equal(result["example.weight"], [[1., 2.]])


def test_realesrgan_missing_checkpoint_requires_explicit_fetch(tmp_path, monkeypatch, capsys):
    import torch

    requests = []

    def retrieve(url, dest):
        requests.append((url, dest))
        torch.save({"value": torch.tensor([2.])}, dest)

    monkeypatch.setattr(convert_realesrgan.urllib.request, "urlretrieve", retrieve)
    out = tmp_path / "nested" / "result.npz"
    argv = ["convert_realesrgan.py", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    assert convert_realesrgan.main() == 1
    assert not requests and not out.exists()
    assert "64 MB" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*argv, "--fetch"])
    assert convert_realesrgan.main() == 0
    assert requests == [(convert_realesrgan.RELEASE_URL, out.parent / "RealESRGAN_x4plus.pth")]
    assert out.is_file()
    assert convert_realesrgan.main() == 0
    assert len(requests) == 1


def test_swift_marching_cubes_table_covers_all_cube_signs(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    runpy.run_module("hunyuan_mlx.shape.scripts.generate_mc_tables", run_name="__main__")
    text = output.getvalue()
    assert "public enum MCTables" in text
    edge_section, triangle_section = text.split("public static let triTable: [[Int]] = [\n")
    edges = [tuple(map(int, pair)) for pair in re.findall(r"\((\d+),(\d+)\)", edge_section)]
    assert len(set(edges)) == 12
    rows = [ast.literal_eval(line.strip().rstrip(",")) for line in triangle_section.splitlines()[:-2]]
    assert len(rows) == 256
    assert rows[0] == rows[255] == []
    assert len(rows[1]) == len(rows[254]) == 3
    for config, row in enumerate(rows):
        assert len(row) % 3 == 0
        for edge in row:
            low, high = edges[edge]
            assert bool(config & (1 << low)) != bool(config & (1 << high))


def test_shape_checkpoint_applies_target_dtype_to_each_retained_tensor():
    inputs = [
        ("model.weight", np.array([1.0])),
        ("metadata", np.array([5.0])),
        ("vae.bias", np.array([2.0])),
    ]
    dino, dit, vae = split_checkpoint(iter(inputs), as_array=lambda arr: arr.astype(np.float16))
    assert dino == {}
    assert dit["weight"].dtype == vae["bias"].dtype == np.float16
    np.testing.assert_array_equal(dit["weight"], [1])
    np.testing.assert_array_equal(vae["bias"], [2])


def test_remapping_accepts_a_tensor_conversion_boundary():
    params = dict(remap_parameters({"bias": [1, 2]}, as_array=np.asarray))
    np.testing.assert_array_equal(params["bias"], [1, 2])


def test_shape_weight_assignment_rejects_missing_or_incompatible_parameters():
    from hunyuan_mlx.shape.hy3dmlx.convert import checked_parameters

    expected = [("linear.weight", np.zeros((2, 3))), ("linear.bias", np.zeros(2))]
    weight, bias = np.ones((2, 3)), np.ones(2)
    params, extras = checked_parameters(expected, [
        ("linear.weight", weight), ("linear.bias", bias), ("training.step", np.ones(1)),
    ], "dit")
    assert [key for key, _ in params] == ["linear.weight", "linear.bias"]
    assert params[0][1] is weight and params[1][1] is bias
    assert extras == 1
    with pytest.raises(KeyError, match=r"dit: 1 missing keys.*linear.bias"):
        checked_parameters(expected, [("linear.weight", weight)], "dit")
    with pytest.raises(ValueError, match=r"dit.linear.weight: shape \(3, 2\) != expected \(2, 3\)"):
        checked_parameters(expected, [("linear.weight", weight.T), ("linear.bias", bias)], "dit")


@pytest.mark.parametrize("use_swiglu,kwargs,weight_name", [
    (None, True, "weights_out"),
    (True, True, "weights_out"),
    (False, False, "fc2"),
])
def test_dino_parameters_choose_weight_matching_feedforward_variant(use_swiglu, kwargs, weight_name):
    from hunyuan_mlx.shape.hy3dmlx.convert import dino_parameters

    encoder = {
        "hidden_size": 64, "num_attention_heads": 4, "num_hidden_layers": 2,
        "patch_size": 8, "layer_norm_eps": "0.00001",
    }
    if use_swiglu is not None:
        encoder["use_swiglu_ffn"] = use_swiglu
    config = {"conditioner": {"params": {"main_image_encoder": {"kwargs": {"config": encoder}}}}}
    weights = {f"model.encoder.layer.0.mlp.{weight_name}.weight": np.zeros((64, 96))}
    assert dino_parameters(config, weights) == {
        "hidden": 64, "heads": 4, "layers": 2, "ffn_dim": 96,
        "swiglu": kwargs, "patch": 8, "eps": 0.00001,
    }



def test_realesrgan_script_help_describes_local_path_and_explicit_fetch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["convert_realesrgan.py", "--help"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(convert_realesrgan.__file__, run_name="__main__")
    assert status.value.code == 0
    help_text = capsys.readouterr().out
    assert "--pth" in help_text and "--fetch" in help_text


def test_swift_marching_cubes_table_emits_empty_row_if_reference_lacks_surface(monkeypatch, capsys):
    from skimage import measure

    original = measure.marching_cubes

    def sample(vol, *args, **kwargs):
        if vol[0, 0, 0] < 0 and np.count_nonzero(vol < 0) == 1:
            raise RuntimeError("reference cannot extract this surface")
        return original(vol, *args, **kwargs)

    monkeypatch.setattr(measure, "marching_cubes", sample)
    runpy.run_module("hunyuan_mlx.shape.scripts.generate_mc_tables", run_name="__main__")
    rows = capsys.readouterr().out.split("public static let triTable: [[Int]] = [\n")[1].splitlines()
    assert ast.literal_eval(rows[1].strip().rstrip(",")) == []
    assert len(ast.literal_eval(rows[2].strip().rstrip(","))) == 3
