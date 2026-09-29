#!/usr/bin/env python3
"""Regression tests for hunyuan_mlx/paint/scripts/convert_unet_bin.py.

Uses tiny torch checkpoints (KBs), so these prove CONVERSION LOGIC --
validation, atomic publication, source preservation -- not a real 4 GB
conversion. Run with a torch+safetensors python, e.g.:

    /Volumes/kokoro-work/venv-torchconv/bin/python tests/test_convert_unet_bin.py
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "hunyuan_mlx" / "paint" / "scripts" / "convert_unet_bin.py"

import torch  # noqa: E402
from safetensors.torch import save_file, load_file  # noqa: E402


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_unet_bin", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_bin(path: Path, n_tensors: int = 3) -> dict:
    sd = {f"w{i}": torch.randn(8, 8, dtype=torch.float16) for i in range(n_tensors)}
    torch.save(sd, path)
    return sd


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_converter()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.bin = self.dir / "model.bin"
        self.sd = make_bin(self.bin)
        self.out = self.dir / "model.safetensors"

    def test_missing_destination_converts_and_validates(self):
        self.mod.convert(self.bin, self.out)
        self.assertTrue(self.out.is_file())
        ok, why = self.mod._matches(load_file(str(self.out)), self.sd)
        self.assertTrue(ok, why)
        self.assertTrue(self.bin.is_file(), "source .bin was not preserved")
        self.assertEqual(list(self.dir.glob("*.convert-tmp")), [], "temp file left behind")

    def test_valid_destination_is_accepted_unchanged(self):
        save_file(self.sd, str(self.out))
        before = self.out.read_bytes()
        self.mod.convert(self.bin, self.out)
        self.assertEqual(self.out.read_bytes(), before, "valid destination was rewritten")
        self.assertTrue(self.bin.is_file())

    def test_truncated_destination_is_repaired(self):
        save_file(self.sd, str(self.out))
        raw = self.out.read_bytes()
        self.out.write_bytes(raw[: len(raw) - 512])  # corrupt the tail
        self.mod.convert(self.bin, self.out)
        ok, why = self.mod._matches(load_file(str(self.out)), self.sd)
        self.assertTrue(ok, f"truncated destination not repaired: {why}")
        self.assertTrue(self.bin.is_file())

    def test_same_shaped_wrong_values_are_repaired(self):
        wrong = {k: torch.full_like(v, 0.5) for k, v in self.sd.items()}
        save_file(wrong, str(self.out))
        self.mod.convert(self.bin, self.out)
        ok, why = self.mod._matches(load_file(str(self.out)), self.sd)
        self.assertTrue(ok, f"wrong-valued destination not repaired: {why}")

    def test_validation_failure_preserves_everything_and_cleans_temp(self):
        # destination must be INVALID so the flow actually reaches save_file
        # (a valid destination early-returns before any write)
        save_file({k: torch.zeros_like(v) for k, v in self.sd.items()}, str(self.out))
        dest_before = self.out.read_bytes()
        bin_before = self.bin.read_bytes()

        real_save_file = save_file

        def exploding_save_file(payload, path, **kw):
            # simulate a crash mid-write: garbage on disk, then a raise
            real_save_file(payload, path, **kw)
            Path(path).write_bytes(b"garbage")
            raise RuntimeError("injected write failure")

        with mock.patch("safetensors.torch.save_file", exploding_save_file):
            with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                self.mod.convert(self.bin, self.out)

        self.assertEqual(self.out.read_bytes(), dest_before, "destination disturbed")
        self.assertEqual(self.bin.read_bytes(), bin_before, "SOURCE .bin disturbed")
        self.assertEqual(list(self.dir.glob("*.convert-tmp")), [], "temp file left behind")


if __name__ == "__main__":
    unittest.main(verbosity=2)
