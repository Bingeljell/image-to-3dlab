"""Importing a TRELLIS patch must not edit a tree. The function edits a temp one."""

from __future__ import annotations

import importlib
import sys

BRIA_LINE = (
    "        pipeline.rembg_model = getattr(rembg, "
    "args['rembg_model']['name'])(**args['rembg_model']['args'])"
)


def test_the_bria_patch_edits_a_temp_file_and_can_run_twice(tmp_path):
    module = importlib.import_module("patch_trellis_no_bria")
    target = tmp_path / "pipeline.py"
    target.write_text(BRIA_LINE + "\n")
    module.patch(target)
    module.patch(target)
    text = target.read_text()
    assert "pipeline.rembg_model = None" in text
    assert "getattr(rembg" not in text


def test_the_quality_patch_edits_a_temp_file(tmp_path):
    module = importlib.import_module("patch_trellis_quality")
    target = tmp_path / "generate.py"
    target.write_text(
        module.SIZE_NEEDLE
        + module.OPTION_ANCHOR
        + module.CALL_NEEDLE
        + "\n"
        + module.PACK_ANCHOR
        + module.PACK_NEEDLE
        + "\n"
    )
    module.patch(target)
    text = target.read_text()
    assert "choices=[512, 1024, 2048]" not in text
    assert "--remesh" in text


def test_the_highpoly_patch_edits_a_temp_file(tmp_path):
    module = importlib.import_module("patch_trellis_highpoly")
    target = tmp_path / "generate.py"
    target.write_text(module.OPTION_ANCHOR + "\n" + module.NEEDLE + "\n")
    module.patch(target)
    assert "--dump-highpoly" in target.read_text()


def test_a_missing_remesh_checkpoint_patch_fails_its_check(tmp_path, monkeypatch):
    module = importlib.import_module("patch_ovoxel_remesh_checkpoints")
    target = tmp_path / "postprocess.py"
    target.write_text("def to_glb():\n    pass\n")
    monkeypatch.setattr(sys, "argv", ["patch", "--check", "--target", str(target)])
    assert module.main() != 0


def test_a_present_remesh_checkpoint_patch_passes_its_check(tmp_path, monkeypatch):
    module = importlib.import_module("patch_ovoxel_remesh_checkpoints")
    target = tmp_path / "postprocess.py"
    target.write_text(module.MARKER + "\n")
    monkeypatch.setattr(sys, "argv", ["patch", "--check", "--target", str(target)])
    assert module.main() == 0
