#!/usr/bin/env python3
"""Contract tests for hunyuan_mlx/download_weights.py's paint path.

Mocks huggingface_hub and the converter subprocess, then asserts the download
plan: repository IDs, pinned revisions, allow-patterns, and local destinations.
Also asserts the torch+safetensors dependency gate fires BEFORE any download.
The weights directory is a temp tree, so no machine state or network is used.

Run with any python 3.10+:
    python3 tests/test_download_weights.py
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "download_weights", REPO / "hunyuan_mlx" / "download_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_hf_hub():
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = mock.Mock(return_value=None)
    hub.hf_hub_download = mock.Mock(return_value=None)
    return hub


class DownloadPlanTest(unittest.TestCase):
    def _run(self, deps_present=True, bin_present=True, npz_present=False):
        mod = load_module()
        hub = fake_hf_hub()
        recorded = {}

        def fake_run(cmd, **kw):
            recorded.setdefault("cmds", []).append(cmd)
            return mock.Mock(returncode=0)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        weights = Path(tmp.name) / "weights"
        weights.mkdir()
        if bin_present:
            unet_dir = weights / "hunyuan3d-paintpbr-v2-1" / "unet"
            unet_dir.mkdir(parents=True)
            (unet_dir / "diffusion_pytorch_model.bin").write_bytes(b"fake bin")
        if npz_present:
            rr_dir = weights / "realesrgan"
            rr_dir.mkdir(parents=True)
            (rr_dir / "rrdbnet.npz").write_bytes(b"fake npz")

        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                mock.patch.object(mod, "_converter_deps_present", return_value=deps_present), \
                mock.patch.object(mod, "PAINT_WEIGHTS", weights), \
                mock.patch.object(subprocess, "run", side_effect=fake_run):
            if deps_present:
                mod.download_paint()
            else:
                with self.assertRaises(SystemExit) as cm:
                    mod.download_paint()
                recorded["exit"] = cm.exception
        return mod, hub, recorded

    def test_every_fetch_is_pinned_with_expected_repos_and_paths(self):
        mod, hub, _ = self._run()

        hub.snapshot_download.assert_called_once()
        args, kwargs = hub.snapshot_download.call_args
        self.assertEqual(kwargs.get("repo_id") or (args[0] if args else None),
                         "tencent/Hunyuan3D-2.1")
        self.assertEqual(kwargs.get("revision"), mod.PAINT_REVISION_21)
        self.assertIn("hunyuan3d-paintpbr-v2-1/*", kwargs.get("allow_patterns", []))

        vae = [c for c in hub.hf_hub_download.call_args_list
               if "hunyuan3d-paint-v2-0" in str(c.kwargs) or
               (c.args and "hunyuan3d-paint-v2-0" in c.args[1])]
        self.assertEqual(len(vae), 2, "VAE config + safetensors must both be fetched")
        for c in vae:
            repo = c.kwargs.get("repo_id") or (c.args[0] if c.args else None)
            self.assertEqual(repo, "tencent/Hunyuan3D-2")
            self.assertEqual(c.kwargs.get("revision"), mod.SHAPE_VAE_REVISION)

        def _filename(c):
            return c.kwargs.get("filename") or (c.args[1] if len(c.args) > 1 else "")

        dino = [c for c in hub.hf_hub_download.call_args_list
                if _filename(c) == "model.safetensors"]
        self.assertEqual(len(dino), 1)
        drepo = dino[0].kwargs.get("repo_id") or (dino[0].args[0] if dino[0].args else None)
        self.assertEqual(drepo, "facebook/dinov2-giant")
        self.assertEqual(dino[0].kwargs.get("revision"), mod.DINO_REVISION)
        self.assertEqual(Path(dino[0].kwargs["local_dir"]).name, "dinov2")

    def test_converters_invoked_when_sources_present(self):
        _, _, recorded = self._run(bin_present=True)
        cmds = recorded.get("cmds", [])
        self.assertTrue(
            any(any("convert_unet_bin.py" in str(part) for part in c) for c in cmds),
            f"UNet converter not invoked: {cmds}")
        self.assertTrue(
            any(any("convert_realesrgan.py" in str(part) for part in c) for c in cmds),
            f"RealESRGAN converter not invoked although the npz was absent: {cmds}")

    def test_realesrgan_converter_skipped_when_npz_present(self):
        _, _, recorded = self._run(bin_present=False, npz_present=True)
        cmds = recorded.get("cmds", [])
        self.assertFalse(
            any(any("convert_realesrgan.py" in str(part) for part in c) for c in cmds),
            f"RealESRGAN converter ran despite existing npz: {cmds}")
        self.assertFalse(
            any(any("convert_unet_bin.py" in str(part) for part in c) for c in cmds),
            "UNet converter ran although no .bin was present")

    def test_dependency_gate_fires_before_any_download(self):
        mod, hub, recorded = self._run(deps_present=False)
        hub.snapshot_download.assert_not_called()
        hub.hf_hub_download.assert_not_called()
        self.assertIn("torch", str(recorded["exit"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
