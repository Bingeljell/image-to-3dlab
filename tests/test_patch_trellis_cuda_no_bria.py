"""The NVIDIA TRELLIS.2 BRIA guardrail: fires on upstream's exact line, runs twice safely,
and fails loudly when the anchor moves (a skipped patch here means BRIA gets downloaded)."""

from __future__ import annotations

import patch_trellis_cuda_no_bria as patch
import pytest

# Verbatim from microsoft/TRELLIS.2 @ 75fbf01, trellis2/pipelines/trellis2_image_to_3d.py.
UPSTREAM = '''\
        pipeline.image_cond_model = getattr(image_feature_extractor, args['image_cond_model']['name'])(**args['image_cond_model']['args'])
        pipeline.rembg_model = getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])

        pipeline.low_vram = args.get('low_vram', True)
'''


def _tree(tmp_path, text=UPSTREAM):
    root = tmp_path / "TRELLIS.2"
    for path in patch.pipeline_files(root):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_the_anchor_is_upstreams_exact_line():
    assert patch.NEEDLE in UPSTREAM


def test_patch_replaces_the_load_with_none(tmp_path):
    root = _tree(tmp_path)
    assert not patch.is_patched(root)
    changed = patch.apply(root)
    assert len(changed) == 2
    for path in patch.pipeline_files(root):
        text = path.read_text()
        assert patch.NEEDLE not in text
        assert "pipeline.rembg_model = None" in text
    assert patch.is_patched(root)


def test_running_twice_changes_nothing(tmp_path):
    root = _tree(tmp_path)
    patch.apply(root)
    before = [p.read_text() for p in patch.pipeline_files(root)]
    assert patch.apply(root) == []
    assert [p.read_text() for p in patch.pipeline_files(root)] == before


def test_a_moved_anchor_fails_loudly(tmp_path):
    root = _tree(tmp_path, "        pipeline.rembg_model = load_something_else()\n")
    with pytest.raises(RuntimeError, match="hook not found"):
        patch.apply(root)


def test_a_missing_checkout_is_not_patched(tmp_path):
    assert patch.is_patched(tmp_path / "nowhere") is False
    with pytest.raises(RuntimeError, match="not a TRELLIS.2 checkout"):
        patch.apply(tmp_path / "nowhere")


def test_one_unpatched_pipeline_is_enough_to_refuse(tmp_path):
    root = _tree(tmp_path)
    patch.apply(root)
    patch.pipeline_files(root)[1].write_text(UPSTREAM)
    assert patch.is_patched(root) is False


def test_the_cli_patches_a_given_root(tmp_path):
    root = _tree(tmp_path)
    assert patch.main(["--root", str(root)]) == 0
    assert patch.is_patched(root)


def test_a_leftover_constructor_beside_the_anchor_is_refused():
    text = UPSTREAM + "        x = getattr(rembg,  args['rembg_model']['name'])\n"
    with pytest.raises(RuntimeError, match="BRIA constructor still present"):
        patch.patch_source(text, "f")


def test_the_cli_fails_when_a_reformatted_constructor_remains(tmp_path, capsys):
    text = patch.REPLACEMENT + "\n        x = getattr(rembg,  args['rembg_model']['name'])\n"
    root = _tree(tmp_path, text)
    assert patch.is_patched(root) is False
    assert patch.main(["--root", str(root)]) == 1
    assert "BRIA" in capsys.readouterr().err
