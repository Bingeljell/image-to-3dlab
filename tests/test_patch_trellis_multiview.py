"""Regression tests for the replayable TRELLIS multi-view patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_multiview.py"
SPEC = importlib.util.spec_from_file_location("patch_trellis_multiview", SCRIPT)
multiview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multiview)


def test_run_hook_is_not_confused_with_method_definition(monkeypatch, tmp_path):
    pipeline = tmp_path / "trellis2_image_to_3d.py"
    pipeline.write_text(
        """class Pipeline:
    def install_multi_image_hooks(self, num_views):
        pass

    def run(self, pipeline_type):
        images = [object()]
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure(
            None, ss_res
        )
"""
    )
    monkeypatch.setattr(multiview, "PIPELINE", pipeline)

    assert multiview.patch_run_wraps_samplers() is True
    once = pipeline.read_text()
    assert once.count("self.install_multi_image_hooks(len(images))") == 1

    assert multiview.patch_run_wraps_samplers() is False
    assert pipeline.read_text() == once
