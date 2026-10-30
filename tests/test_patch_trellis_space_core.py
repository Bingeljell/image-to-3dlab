"""Contracts for the official-Space-first Mac compatibility patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_space_core.py"
SPEC = importlib.util.spec_from_file_location("patch_trellis_space_core", SCRIPT)
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


def test_replacement_fails_closed_when_upstream_anchor_changes():
    try:
        patcher.replace_exact("upstream changed", "old", "new", count=1, label="gate")
    except RuntimeError as error:
        assert "expected 1 source anchors, found 0" in str(error)
    else:
        raise AssertionError("an unknown upstream shape must not be patched silently")


def test_replacement_is_idempotent():
    source, changed = patcher.replace_exact("old", "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", True)
    source, changed = patcher.replace_exact(source, "old", "new", count=1, label="gate")
    assert (source, changed) == ("new", False)


def test_sparse_attention_patch_uses_packed_metal_kernel():
    source = SCRIPT.read_text()
    assert "flex_gemm.kernels.metal.sparse_attention_fwd" in source
    assert "q_padded" not in source
    assert "kv_prefix" in source


def test_sdpa_preserves_packed_sequence_boundaries_and_metal_fails_fast_at_128():
    source = SCRIPT.read_text()
    assert "for q_length, kv_length in zip(q_seqlen, kv_seqlen)" in source
    assert "F.scaled_dot_product_attention" in source
    assert "if q.shape[-1] > 64" in source
    assert "Use the sdpa backend for TRELLIS.2-4B" in source


def test_core_patch_does_not_change_model_or_sampler_equations():
    source = SCRIPT.read_text()
    forbidden_targets = (
        "samplers/",
        "models/sparse_structure_flow.py",
        "models/structured_latent_flow.py",
        "tex_slat_normalization",
        "shape_slat_normalization",
        "preprocess_image(self",
    )
    for target in forbidden_targets:
        assert target not in source


def test_the_patch_assigns_none_and_never_names_the_constructor():
    """The patcher source itself must not be able to spell the remover call."""
    source = SCRIPT.read_text()
    assert "new_pipeline.rembg_model = None" in source
    assert "getattr(rembg, args['rembg_model']['name'])" not in source


UPSTREAM = (
    '    def from_pretrained(path: str) -> "Trellis2ImageTo3DPipeline":\n'
    "        new_pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
)
OLD_OPTIONAL = (
    '    def from_pretrained(path: str, *, load_rembg: bool = True) '
    '-> "Trellis2ImageTo3DPipeline":\n'
    "        new_pipeline.rembg_model = (\n"
    "            getattr(rembg, args['rembg_model']['name'])(**args['rembg_model']['args'])\n"
    "            if load_rembg else None\n"
    "        )\n"
)


def _pipeline(root: Path) -> Path:
    return root / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"


def _tree(tmp_path: Path, source: str) -> Path:
    root = tmp_path / "TRELLIS.2"
    path = _pipeline(root)
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return root


def test_patch_disables_the_remover_and_is_idempotent(tmp_path):
    root = _tree(tmp_path, UPSTREAM)
    assert patcher.is_patched(root) is False
    assert patcher.patch_optional_rembg(root) is True
    text = _pipeline(root).read_text()
    assert "new_pipeline.rembg_model = None" in text
    assert "getattr(rembg, args['rembg_model']['name'])" not in text
    assert patcher.is_patched(root) is True
    assert patcher.patch_optional_rembg(root) is False


def test_old_optional_form_is_rewritten_to_none(tmp_path):
    root = _tree(tmp_path, OLD_OPTIONAL)
    assert patcher.is_patched(root) is False
    assert patcher.patch_optional_rembg(root) is True
    text = _pipeline(root).read_text()
    assert "new_pipeline.rembg_model = None" in text
    assert "getattr(rembg, args['rembg_model']['name'])" not in text
    assert patcher.is_patched(root) is True


def test_is_patched_is_false_while_the_constructor_remains(tmp_path):
    both = UPSTREAM + "        new_pipeline.rembg_model = None\n"
    root = _tree(tmp_path, both)
    assert patcher.is_patched(root) is False
