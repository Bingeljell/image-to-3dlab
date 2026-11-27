"""Run the same CPU tensor helpers imported by patched vendor packages."""

import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def test_multiview_average_slices_conditioning_and_keeps_single_view_fast(capsys):
    torch = pytest.importorskip("torch")
    from scripts.trellis_multiview_compat import (
        inject_sampler_multi_image,
        install_multi_image_hooks,
    )

    calls = []

    def denoise(model, x, t, cond, **kwargs):
        calls.append((model, t, cond.shape, kwargs))
        return x + cond.reshape(-1) + kwargs.get("neg_cond", 0)

    sparse = SimpleNamespace(_inference_model=denoise)
    shape = SimpleNamespace(_inference_model=denoise)
    pipeline = SimpleNamespace(sparse_structure_sampler=sparse, shape_slat_sampler=shape)
    pipeline.inject_sampler_multi_image = lambda name, mode: inject_sampler_multi_image(
        getattr(pipeline, name), mode
    )
    install_multi_image_hooks(pipeline)
    installed = sparse._inference_model
    install_multi_image_hooks(pipeline)
    assert sparse._inference_model is installed
    assert shape._inference_model is not denoise

    pos = torch.tensor([[[1.0]], [[3.0]], [[8.0]]])
    neg = torch.tensor([[[-1.0]], [[-2.0]], [[-3.0]]])
    out = installed("fixture", torch.zeros(1), 0.5, pos, neg_cond=neg, scale=2)
    torch.testing.assert_close(out, torch.tensor([[[2.0]]]))
    assert [c[3]["neg_cond"].item() for c in calls] == [-1, -2, -3]
    assert all(c[:3] == ("fixture", 0.5, (1, 1, 1)) and c[3]["scale"] == 2 for c in calls)
    torch.testing.assert_close(installed("fixture", torch.zeros(1), 0, pos), torch.tensor([4.0]))
    torch.testing.assert_close(installed("fixture", torch.zeros(1), 0, pos[:1]), torch.tensor([1.0]))
    assert installed._fusions == 2
    assert capsys.readouterr().out.count("fusing 3 views") == 1


def test_multiview_stochastic_cycles_and_shares_unbatched_negative_conditioning():
    torch = pytest.importorskip("torch")
    from scripts.trellis_multiview_compat import (
        inject_sampler_multi_image,
        install_multi_image_hooks,
    )

    shared_neg = torch.tensor([[[2.0]]])
    calls = []

    def denoise(model, x, t, cond, *, neg_cond=None):
        assert neg_cond is shared_neg
        calls.append(float(cond.item()))
        return cond + neg_cond

    sampler = SimpleNamespace(_inference_model=denoise)
    pipe = SimpleNamespace(tex_slat_sampler=sampler, multi_image_mode="stochastic")
    pipe.inject_sampler_multi_image = lambda name, mode: inject_sampler_multi_image(
        getattr(pipe, name), mode
    )
    install_multi_image_hooks(pipe)
    cond = torch.tensor([[[1.0]], [[3.0]]])
    for step, expected in enumerate((3, 5, 3)):
        value = sampler._inference_model(None, None, step, cond, neg_cond=shared_neg)
        assert value.item() == expected
    assert calls == [1, 3, 1]
    assert sampler._multi_image_per_view._fusions == 3


def test_multiview_conditioning_preserves_view_batch_and_stage_order():
    torch = pytest.importorskip("torch")
    from scripts.trellis_multiview_compat import prepare_conditioning

    calls = []
    original_seed = torch.random.get_rng_state()
    pipe = SimpleNamespace(
        preprocess_image=lambda item: f"{item} cut",
        get_cond=lambda photos, resolution: calls.append((photos.copy(), resolution)) or (photos, resolution),
    )
    try:
        images, low, high = prepare_conditioning(pipe, ("front", "back"), True, 16, "1024_cascade")
        assert images == ["front cut", "back cut"]
        assert low == (images, 512)
        assert high == (["front cut", "back cut"], 1024)
        assert calls == [low, high]
        assert torch.initial_seed() == 16
        assert prepare_conditioning(pipe, "front", False, 7, "512") == (["front"], (["front"], 512), None)
    finally:
        torch.random.set_rng_state(original_seed)


def test_hunyuan_helper_nearest_scheduler_index_for_inexact_match():
    torch = pytest.importorskip("torch")
    from scripts.hunyuan_rocm_compat import nearest_timestep_indices

    schedule = torch.tensor([1.0, 0.75, 0.5, 0.0])
    for value, index in ((0.499999, 2), (-0.1, 3), (1.0, 0)):
        result = nearest_timestep_indices(schedule, torch.tensor(value))
        assert result.shape == (1,) and result.item() == index


@pytest.mark.parametrize(("chunk", "expected_sizes"), [
    ("", [5]), ("0", [5]), ("1", [1, 1, 1, 1, 1]), ("2", [2, 2, 1]), ("6", [5]),
])
@pytest.mark.parametrize("masked", [False, True])
def test_hunyuan_helper_chunks_queries_without_changing_attention(chunk, expected_sizes, masked, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts.hunyuan_rocm_compat import chunked_attention

    monkeypatch.setenv("HY3D_SDPA_QUERY_CHUNK", chunk)
    generator = torch.Generator().manual_seed(19)
    query = torch.randn(2, 3, 5, 4, generator=generator)
    key = torch.randn(2, 3, 7, 4, generator=generator)
    value = torch.randn(2, 3, 7, 4, generator=generator)
    mask = torch.randn(2, 1, 5, 7, generator=generator) if masked else None
    kernel = torch.nn.functional.scaled_dot_product_attention
    expected = kernel(query, key, value, attn_mask=mask)
    calls = []

    def record(q_part, k_part, v_part, *, attn_mask, **options):
        assert k_part is key and v_part is value
        calls.append((q_part, attn_mask))
        return kernel(q_part, k_part, v_part, attn_mask=attn_mask, **options)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    torch.testing.assert_close(chunked_attention(query, key, value, mask), expected)
    assert [part.shape[-2] for part, _ in calls] == expected_sizes
    torch.testing.assert_close(torch.cat([part for part, _ in calls], dim=-2), query)
    if masked:
        torch.testing.assert_close(torch.cat([part_mask for _, part_mask in calls], dim=-2), mask)
    else:
        assert all(part_mask is None for _, part_mask in calls)


def test_installed_hunyuan_imports_the_exact_helper(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from scripts import hunyuan_rocm_compat
    from scripts import patch_hunyuan_rocm as patch
    from tests.test_patch_hunyuan_rocm import ATTN_SOURCE, SCHEDULER_SOURCE

    vendor = tmp_path / "deployed_hunyuan"
    for relative, source in [(patch.SCHEDULER_FILE, SCHEDULER_SOURCE), (patch.ATTN_FILE, ATTN_SOURCE)]:
        file = vendor / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source)
    assert patch.patch(vendor) == 2
    assert patch.patch(vendor) == 0
    for relative in [patch.SCHEDULER_FILE, patch.ATTN_FILE]:
        deployed = (vendor / relative).with_name("_i2l_rocm_compat.py")
        assert deployed.read_bytes() == Path(hunyuan_rocm_compat.__file__).read_bytes()
    monkeypatch.syspath_prepend(str(tmp_path))
    einops = ModuleType("einops")
    einops.rearrange = lambda *args: pytest.fail("not called by attention")
    monkeypatch.setitem(sys.modules, "einops", einops)
    try:
        scheduler = importlib.import_module("deployed_hunyuan.hy3dshape.hy3dshape.schedulers")
        indices = torch.tensor([0.0, 0.25, 0.5])
        assert scheduler.FlowMatch().index_for_timestep(0.250001, indices) == 1
        assert scheduler.DDIM().index_for_timestep(torch.tensor(0.49999), indices) == 2
        attention = importlib.import_module("deployed_hunyuan.hy3dpaint.hunyuanpaintpbr.unet.attn_processor")
        generator = torch.Generator().manual_seed(29)
        for attribute in ("query", "key", "value"):
            setattr(attention, attribute, torch.rand(1, 2, 3, 5, generator=generator))
        attention.attention_mask = torch.rand(1, 1, 3, 3, generator=generator)
        monkeypatch.setenv("HY3D_SDPA_QUERY_CHUNK", "2")
        expected = torch.nn.functional.scaled_dot_product_attention(
            attention.query, attention.key, attention.value, attn_mask=attention.attention_mask
        )
        torch.testing.assert_close(attention.compute(None), expected)
    finally:
        for module in tuple(sys.modules):
            if module == "deployed_hunyuan" or module.startswith("deployed_hunyuan."):
                sys.modules.pop(module)


@pytest.mark.parametrize("legacy", [False, True])
def test_installed_multiview_pipeline_calls_copied_helper(tmp_path, monkeypatch, legacy):
    torch = pytest.importorskip("torch")
    from scripts import patch_trellis_multiview as patch
    from scripts import trellis_multiview_compat

    root = tmp_path / "checkout"
    module_dir = root / "test_multiview_package"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").touch()
    module_file = module_dir / "pipeline.py"
    generate = root / "generate.py"
    conditioning = """        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)
        cond_512 = self.get_cond([image], 512)
        cond_1024 = self.get_cond([image], 1024) if pipeline_type != '512' else None"""
    old_methods = ""
    if legacy:
        old_methods = """    def install_multi_image_hooks(self, num_views: int) -> None:
        raise AssertionError("legacy hook still in use")

    def inject_sampler_multi_image(self, sampler_name, mode='multidiffusion'):
        def per_view():
            raise AssertionError("legacy prediction still in use")

"""
        conditioning = """        # Accept either one image or several views of the same subject.
        images = list(image) if isinstance(image, (list, tuple)) else [image]
        if preprocess_image:
            images = [self.preprocess_image(i) for i in images]
        torch.manual_seed(seed)
        cond_512 = self.get_cond(images, 512)
        cond_1024 = self.get_cond(images, 1024) if pipeline_type != '512' else None"""
    module_file.write_text(
        """import torch
from types import SimpleNamespace
from typing import Union
from PIL import Image

class Pipeline:
    def __init__(self):
        self.sparse_structure_sampler = SimpleNamespace(
            _inference_model=lambda model, x, t, cond, **kwargs: cond * 3
        )

    def preprocess_image(self, image):
        return image + 2

    def sample_sparse_structure(self, conditioning, resolution):
        return self.sparse_structure_sampler._inference_model(None, None, None, conditioning)

"""
        + old_methods
        + """    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        return torch.tensor(image, dtype=torch.float32).reshape(-1, 1, 1)

    def run(self, image, pipeline_type, preprocess_image=True, seed=11):
"""
        + conditioning
        + """
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            cond_512, ss_res
        )
        return coords, cond_1024
"""
    )
    generate.write_text(
        """import argparse
import os
from PIL import Image as PILImage

def main(pipeline, argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)
    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        return None
    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")
    return img
"""
    )
    if legacy:
        generate.write_text(generate.read_text().replace(
            """    img = PILImage.open(args.image)
    print(f"Input: {args.image} ({img.size[0]}x{img.size[1]})")""",
            """    img = [PILImage.open(path) for path in args.image]
    for path, view in zip(args.image, img):
        print(f"Input: {path} ({view.size[0]}x{view.size[1]})")
    if len(img) == 1:
        img = img[0]
    else:
        print(f"Conditioning on {len(img)} views (mode={args.multi_image_mode})")
        pipeline.multi_image_mode = args.multi_image_mode""",
        ))

    monkeypatch.setattr(patch, "PIPELINE", module_file)
    monkeypatch.setattr(patch, "GENERATE", generate)
    assert patch.main() == 0
    helper = module_dir / "_i2l_multiview.py"
    assert helper.read_bytes() == Path(trellis_multiview_compat.__file__).read_bytes()
    assert generate.with_name(helper.name).read_bytes() == helper.read_bytes()
    state = module_file.read_bytes(), generate.read_bytes(), helper.read_bytes()
    assert patch.main() == 0
    assert (module_file.read_bytes(), generate.read_bytes(), helper.read_bytes()) == state

    monkeypatch.syspath_prepend(str(root))
    pipeline = importlib.import_module("test_multiview_package.pipeline")
    try:
        previous = torch.random.get_rng_state()
        try:
            pipe = pipeline.Pipeline()
            result, high = pipe.run([1, 5], "1024_cascade")
            assert result.item() == 15
            assert high.flatten().tolist() == [3, 7]
            result, high = pipe.run(2, "512", preprocess_image=False)
            assert result.item() == 6 and high is None
        finally:
            torch.random.set_rng_state(previous)
    finally:
        for name in ("test_multiview_package.pipeline", "test_multiview_package._i2l_multiview",
                     "test_multiview_package"):
            sys.modules.pop(name, None)

    from PIL import Image

    first, second = root / "front.png", root / "back.png"
    Image.new("RGB", (3, 2)).save(first)
    Image.new("RGB", (4, 3)).save(second)
    cli = runpy.run_path(str(generate))["main"]
    settings = SimpleNamespace()
    with cli(settings, [str(first)]) as view:
        assert view.size == (3, 2)
    views = cli(settings, [str(first), str(second), "--multi-image-mode", "stochastic"])
    try:
        assert [view.size for view in views] == [(3, 2), (4, 3)]
        assert settings.multi_image_mode == "stochastic"
    finally:
        for view in views:
            view.close()
        sys.modules.pop("_i2l_multiview", None)
    assert cli(settings, [str(root / "missing.png")]) is None


def test_hunyuan_patch_upgrades_legacy_and_repairs_missing_helper(tmp_path):
    from scripts import patch_hunyuan_rocm as patch
    from tests.test_patch_hunyuan_rocm import ATTN_SOURCE, SCHEDULER_SOURCE

    paths = (tmp_path / patch.SCHEDULER_FILE, tmp_path / patch.ATTN_FILE)
    paths[0].parent.mkdir(parents=True)
    paths[1].parent.mkdir(parents=True)
    paths[0].write_text(SCHEDULER_SOURCE.replace(patch.SCHEDULER_NEEDLE, patch.SCHEDULER_LEGACY))
    paths[1].write_text(
        ATTN_SOURCE.replace(patch.ATTN_IMPORT_NEEDLE, patch.ATTN_IMPORT_LEGACY)
        .replace(patch.ATTN_NEEDLE, patch.ATTN_LEGACY)
        .replace(patch.DEVICE_NEEDLE, patch.DEVICE_REPLACEMENT)
    )
    assert patch.patch(tmp_path) == 2
    assert patch.is_patched(tmp_path)
    assert patch.SCHEDULER_LEGACY not in paths[0].read_text()
    assert patch.ATTN_LEGACY not in paths[1].read_text()

    helper = paths[0].with_name(patch.HELPER_NAME)
    helper.write_text("# stale copy")
    assert not patch.is_patched(tmp_path)
    assert patch.patch(tmp_path) == 1
    assert patch.is_patched(tmp_path)
    assert patch.patch(tmp_path) == 0

    source = paths[1].read_text()
    paths[1].write_text(source.replace(patch.ATTN_REPLACEMENT, "        hidden_states = alternate()"))
    assert not patch.is_patched(tmp_path)
    paths[1].write_text(source + "\n" + patch.ATTN_NEEDLE)
    assert not patch.is_patched(tmp_path)
    paths[1].write_text(source)


def test_hunyuan_patch_command_installs_helpers_in_selected_checkout(tmp_path, monkeypatch, capsys):
    from scripts import patch_hunyuan_rocm as patch
    from tests.test_patch_hunyuan_rocm import ATTN_SOURCE, SCHEDULER_SOURCE

    for relative, text in [(patch.SCHEDULER_FILE, SCHEDULER_SOURCE), (patch.ATTN_FILE, ATTN_SOURCE)]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    monkeypatch.setattr(sys, "argv", [patch.__file__, "--root", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(patch.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert "(2 file(s))" in capsys.readouterr().out
    assert patch.is_patched(tmp_path)


def test_multiview_patch_refuses_missing_checkout_and_drift(tmp_path, monkeypatch):
    from scripts import patch_trellis_multiview as patch

    pipeline = tmp_path / "pipeline.py"
    generate = tmp_path / "generate.py"
    monkeypatch.setattr(patch, "PIPELINE", pipeline)
    monkeypatch.setattr(patch, "GENERATE", generate)
    with pytest.raises(SystemExit, match="vendored TRELLIS.2 not found"):
        patch.main()
    pipeline.write_text("# upstream changed\n")
    with pytest.raises(RuntimeError, match="no import block"):
        patch.patch_contextmanager_import()
    with pytest.raises(RuntimeError, match="expected anchor not found in pipeline.py"):
        patch.patch_sampler_injection()
    assert pipeline.read_text() == "# upstream changed\n"
    moved_hook = """class Fixture:
    def install_multi_image_hooks(self, num_views: int) -> None:
        pass

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]], resolution: int, include_neg_cond: bool = True) -> dict:
        pass
"""
    pipeline.write_text(moved_hook)
    with pytest.raises(RuntimeError, match="unrecognized multi-view hook"):
        patch.patch_sampler_injection()
    assert pipeline.read_text() == moved_hook


def test_multiview_patch_removes_legacy_token_merge(tmp_path, monkeypatch):
    from scripts import patch_trellis_multiview as patch

    pipeline = tmp_path / "pipeline.py"
    original = """        cond = self.encoder(image)
        # Several views describe ONE object, so fold them into a single conditioning
        # sequence along the token axis: (B, N, D) -> (1, B*N, D). Leaving them batched
        # would ask the sampler for B separate objects, since it builds noise shaped
        # (num_samples, ...) and pairs it with the conditioning batch.
        if cond.ndim == 3 and cond.shape[0] > 1:
            cond = cond.reshape(1, -1, cond.shape[-1])
        return cond
"""
    pipeline.write_text(original)
    monkeypatch.setattr(patch, "PIPELINE", pipeline)
    assert patch.undo_token_concatenation()
    assert pipeline.read_text() == "        cond = self.encoder(image)\n        return cond\n"
    assert not patch.undo_token_concatenation()


def test_load_input_views_keeps_a_single_image_and_sets_mode_only_for_several(tmp_path, capsys):
    pytest.importorskip("torch")
    from PIL import Image

    from scripts.trellis_multiview_compat import load_input_views

    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for index, path in enumerate(paths, 1):
        Image.new("RGB", (index, index + 1)).save(path)
    pipeline = SimpleNamespace()
    with load_input_views(paths[:1], pipeline, "stochastic") as view:
        assert view.size == (1, 2)
        assert not hasattr(pipeline, "multi_image_mode")
    views = load_input_views(paths, pipeline, "stochastic")
    try:
        assert [view.size for view in views] == [(1, 2), (2, 3)]
        assert pipeline.multi_image_mode == "stochastic"
    finally:
        for view in views:
            view.close()
    assert "Conditioning on 2 views (mode=stochastic)" in capsys.readouterr().out


def test_report_missing_views_preserves_argument_order(tmp_path, capsys):
    pytest.importorskip("torch")
    from scripts.trellis_multiview_compat import report_missing_inputs

    present = tmp_path / "exists.png"
    present.touch()
    missing = [str(tmp_path / "last.png"), str(tmp_path / "first.png")]
    assert not report_missing_inputs([str(present)])
    assert capsys.readouterr().out == ""
    assert report_missing_inputs([missing[0], str(present), missing[1]])
    assert capsys.readouterr().out == f"Error: {', '.join(missing)} not found\n"
