"""Check re-bake argument mapping and preview geometry using real local meshes."""

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import trimesh

from scripts import trellis_rebake as rebake


class Array(np.ndarray):
    def clone(self):
        return self.copy()


@pytest.mark.parametrize("preview", [False, True])
def test_rebake_exports_cached_material_with_selected_geometry(preview, tmp_path, monkeypatch):
    cache = tmp_path / "candidate.material.pt"
    geometry = tmp_path / "reusable.geometry.pt"
    accepted = tmp_path / "accepted.glb"
    output = tmp_path / "deliveries/result.glb"
    mesh = trimesh.creation.box()
    mesh.apply_translation([1, 2, 3])
    mesh.export(accepted)
    cache.touch()
    geometry.touch()
    calls = []

    torch = ModuleType("torch")
    torch.float32 = np.float32
    torch.int32 = np.int32
    torch.as_tensor = lambda data, dtype: np.asarray(data, dtype=dtype).view(Array)
    material = np.asarray([[0.5, 0, 1]], dtype=np.float32)
    layout = {"base_color": [0, 1, 2]}
    points = np.asarray([[1, 2, 3]])
    base_faces = np.asarray([[0, 1, 2]])
    base_vertices = mesh.vertices[:3]
    base = {"vertices": base_vertices, "faces": base_faces, "attrs": material, "layout": layout,
            "coords": points, "voxel_size": 0.01}

    def load(path, *, weights_only):
        assert not weights_only
        calls.append(path)
        if path == cache:
            return {"geometry_ref": geometry.name, "attrs": material}
        assert path == geometry
        return base

    torch.load = load
    voxel = ModuleType("o_voxel")

    def bake(**kwargs):
        assert kwargs["attr_volume"] is material and kwargs["coords"] is points
        assert kwargs["attr_layout"] == layout and kwargs["voxel_size"] == 0.01
        assert kwargs["remesh"] is preview and kwargs["texture_size"] == 128
        assert kwargs["decimation_target"] == 18
        assert kwargs["remesh_band"] == 1.25 and kwargs["remesh_project"] == 0.5
        if preview:
            vertices = kwargs["vertices"]
            np.testing.assert_allclose(vertices.mean(0), [1, -3, 2])
            assert vertices.dtype == np.float32
            assert kwargs["faces"].dtype == np.int32
        else:
            assert kwargs["vertices"] is base_vertices
            assert kwargs["faces"] is base_faces

        def export(path, *, extension_webp):
            assert path == str(output)
            assert extension_webp is not preview
            mesh.export(path)

        return SimpleNamespace(export=export)

    voxel.postprocess = SimpleNamespace(to_glb=bake)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "o_voxel", voxel)
    argv = [
        rebake.__file__, str(cache), str(output),
        "--decimation-target", "18", "--texture-size", "128",
        "--remesh-band", "1.25", "--remesh-project", "0.5",
    ]
    if preview:
        argv += ["--geometry-glb", str(accepted), "--no-webp", "--remesh"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(rebake.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert calls == [cache, geometry]
    assert len(trimesh.load(output, force="mesh").faces) == len(mesh.faces)


def test_rebake_cli_reports_missing_inputs(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "decode.pt"
    output = tmp_path / "out.glb"

    def call(*options):
        monkeypatch.setattr(sys, "argv", [rebake.__file__, str(cache), str(output), *options])
        with pytest.raises(SystemExit) as exc:
            rebake.main()
        assert exc.value.code == 2
        assert not output.exists()
        return capsys.readouterr().err

    assert "does not exist" in call()
    cache.touch()
    torch = ModuleType("torch")

    def load(path, *, weights_only):
        assert Path(path) == cache and not weights_only
        return {}

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "o_voxel", ModuleType("o_voxel"))
    assert "geometry preview GLB does not exist" in call("--geometry-glb", str(tmp_path / "missing.glb"))
