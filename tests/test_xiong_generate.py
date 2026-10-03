#!/usr/bin/env python3
"""Orchestration tests for scripts/hunyuan_mlx_xiong_generate.py.

Stubs the expensive stages (shape pipeline, remesh, paint subprocess) and runs
main() end-to-end through manifest writing. Two real bugs (referencing `pipe`
out of scope; reading `mesh` in the manifest after release) both shipped past
a syntax check -- this file executes main(), which is the point.

Stdlib only (no MLX/torch needed): the inline `import mlx.core as mx` is
satisfied by fakes. Run directly: python3 tests/test_xiong_generate.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "hunyuan_mlx_xiong_generate.py"

MLX_STUBS = {
    "mlx": None,  # filled in setUpClass
    "mlx.core": None,
    "mlx.metal": None,
}


def _make_mlx_stubs():
    metal = types.SimpleNamespace(clear_cache=mock.Mock())
    core = types.ModuleType("mlx.core")
    core.metal = metal
    mlx = types.ModuleType("mlx")
    mlx.core = core
    mlx.metal = metal
    mlx.__path__ = []  # mark as package so `import mlx.core` resolves via sys.modules
    return {"mlx": mlx, "mlx.core": core, "mlx.metal": metal}


class FakeMesh:
    """Stand-in for a trimesh.Mesh: counted vertices/faces + a real export."""

    def __init__(self, n_verts: int, n_faces: int):
        self.vertices = list(range(n_verts))
        self.faces = list(range(n_faces))

    def export(self, path: str) -> None:
        Path(path).write_text(f"fake obj {len(self.vertices)} {len(self.faces)}\n")


def load_script():
    spec = importlib.util.spec_from_file_location("xiong_generate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OrchestrationTest(unittest.TestCase):
    def _run_main(self, tmp: Path, paint_error: Exception | None):
        mod = load_script()
        calls = {}
        mesh = FakeMesh(1234, 5678)

        def fake_run_shape(*a, **k):
            calls["shape"] = True
            return mesh

        def fake_run_remesh(m, target, t0):
            calls["remesh"] = True
            return m, False

        def fake_run_paint(mesh_path, image, output, seed, res, steps, tex, t0):
            calls["paint"] = True
            if paint_error is not None:
                raise paint_error
            output.write_bytes(b"fake glb")

        argv = ["xiong_generate.py", "/fake/ref.png", str(tmp / "chest.glb"),
                "--seed", "42"]
        with mock.patch.dict(sys.modules, _make_mlx_stubs()), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(mod, "run_shape", fake_run_shape), \
                mock.patch.object(mod, "run_remesh", fake_run_remesh), \
                mock.patch.object(mod, "run_paint", fake_run_paint):
            mod.main()
        return calls

    def test_main_reaches_manifest_with_captured_counts(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            self._run_main(td, None)
            out = td / "chest.glb"
            obj = td / "chest_shape.obj"
            manifest_path = td / "chest.json"  # <stem>.json per production code
            self.assertTrue(out.is_file(), "output GLB not written")
            self.assertTrue(obj.is_file(), "shape OBJ not exported")
            m = json.loads(manifest_path.read_text())
            self.assertEqual(m["vertices"], 1234, "vertex count lost across the mesh release")
            self.assertEqual(m["faces"], 5678, "face count lost across the mesh release")
            self.assertEqual(m["parameters"]["seed"], 42)
            self.assertEqual(Path(m["output"]).name, "chest.glb")
            self.assertEqual(m["schema_version"], 1)

    def test_paint_failure_propagates_without_success_manifest(self):
        with tempfile.TemporaryDirectory() as td_s:
            td = Path(td_s)
            argv = ["xiong_generate.py", "/fake/ref.png", str(td / "chest.glb"), "--seed", "42"]
            with mock.patch.dict(sys.modules, _make_mlx_stubs()), \
                    mock.patch.object(sys, "argv", argv):
                mod = load_script()
                mesh = FakeMesh(10, 10)

                def boom(*a, **k):
                    raise RuntimeError("paint stage exploded")

                with mock.patch.object(mod, "run_shape", lambda *a, **k: mesh), \
                        mock.patch.object(mod, "run_remesh", lambda m, t, t0: (m, False)), \
                        mock.patch.object(mod, "run_paint", boom):
                    with self.assertRaisesRegex(RuntimeError, "paint stage exploded"):
                        mod.main()
            self.assertFalse((td / "chest.json").exists(),
                             "success manifest written despite paint failure")
            self.assertTrue((td / "chest_shape.obj").is_file(),
                            "shape OBJ should survive a paint failure")


if __name__ == "__main__":
    unittest.main(verbosity=2)
