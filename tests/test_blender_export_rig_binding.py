from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "blender_export_rig_binding.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_export_rig_binding", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_requires_explicit_typed_artifacts(tmp_path):
    module = load_module()
    asset = tmp_path / "source.glb"
    scene = tmp_path / "prepared.blend"
    sidecar = tmp_path / "prepared.rig.json"

    assert module.parse_args([
        "blender", "--", str(asset), str(scene), str(sidecar), "adapter.v1"
    ]) == (asset, scene, sidecar, "adapter.v1")


@pytest.mark.parametrize(
    "args",
    [
        ["--"],
        ["--", "source.obj", "scene.blend", "sidecar.rig.json", "adapter"],
        ["--", "source.glb", "scene.txt", "sidecar.rig.json", "adapter"],
        ["--", "source.glb", "scene.blend", "sidecar.json", "adapter"],
    ],
)
def test_parse_args_rejects_ambiguous_artifacts(args):
    with pytest.raises(SystemExit):
        load_module().parse_args(args)


def test_exporter_stamps_persistent_scene_and_bone_ids():
    source = MODULE.read_text()

    assert 'OBJECT_ID = "i2l_metarig_id"' in source
    assert 'BONE_ID = "i2l_bone_id"' in source
    assert "save_as_mainfile" in source
    assert "asset_fingerprint(scene_path)" in source
