from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "blender_rebind.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_rebind", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_args_keeps_inputs_and_outputs_distinct(tmp_path):
    values = [
        tmp_path / "source.glb", tmp_path / "corrected.rig.json",
        tmp_path / "result.glb", tmp_path / "result.blend",
        tmp_path / "result.rig.json", tmp_path / "report.json",
    ]

    assert load_module().parse_args(["blender", "--", *map(str, values)]) == tuple(values)


def test_parse_args_refuses_overwriting_an_input(tmp_path):
    source = tmp_path / "same.glb"
    with pytest.raises(SystemExit, match="distinct"):
        load_module().parse_args([
            "--", str(source), str(tmp_path / "in.rig.json"), str(source),
            str(tmp_path / "out.blend"), str(tmp_path / "out.rig.json"),
            str(tmp_path / "report.json"),
        ])


def test_refreshed_sidecar_promotes_targets_to_the_new_source_positions(tmp_path):
    module = load_module()
    glb = tmp_path / "result.glb"
    blend = tmp_path / "result.blend"
    glb.write_bytes(b"new glb")
    blend.write_bytes(b"new blend")
    sidecar = {
        "assetFingerprint": "old",
        "joints": {"elbow": {"position": [0, 0, 0]}},
        "corrections": {"elbow": {"targetPosition": [1, 2, 3]}},
        "binding": {"sceneFingerprint": "old"},
    }

    result = module.refreshed_sidecar(sidecar, glb, blend)

    assert result["joints"]["elbow"]["position"] == [1, 2, 3]
    assert result["corrections"] == {}
    assert result["assetFingerprint"] == "sha256:" + hashlib.sha256(b"new glb").hexdigest()
    assert result["binding"]["sceneFingerprint"] == (
        "sha256:" + hashlib.sha256(b"new blend").hexdigest()
    )
    assert sidecar["assetFingerprint"] == "old"


def test_worker_resolves_persistent_ids_before_diagnostic_names():
    source = MODULE.read_text()

    assert "bones_by_id.get(target.bone_id)" in source
    assert "expectedBoneName" in source
    assert 'weightStrategy": "voxel-proxy-transfer"' in source
    assert "transfer_weights(bpy, mesh, rig)" in source
