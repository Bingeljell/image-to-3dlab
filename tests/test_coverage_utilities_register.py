"""Record and query asset verdicts against a local register and tiny meshes."""

import datetime as dt
import hashlib
import json
import runpy
import sys

import pytest
import trimesh
from PIL import Image

from scripts import mark_asset


def test_asset_register_cli_appends_and_filters_without_losing_old_verdicts(monkeypatch, tmp_path, capsys):
    register = tmp_path / "notes" / "index.jsonl"
    asset = tmp_path / "cube.glb"
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0, 0]] * len(mesh.vertices),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (2, 2)), metallicFactor=0.75, doubleSided=False,
        ),
    )
    mesh.export(asset)
    sidecar = asset.with_suffix(".provenance.json")
    sidecar.write_text("{}")
    common = ["mark_asset.py", "--register", str(register)]
    monkeypatch.setattr("sys.argv", [*common, "--list"])
    assert mark_asset.main() == 0
    assert "no verdicts recorded yet" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", [*common, str(asset), "--verdict", "good",
                                      "--note", " stable topology ", "--tags", "hard edge,, color ",
                                      "--source", "front.png", "--culled"])
    assert mark_asset.main() == 0
    printed = capsys.readouterr().out
    assert "[culled] stable topology" in printed
    assert "12 faces" in printed
    first = json.loads(register.read_text())
    assert first["tags"] == ["color", "hard edge"]
    assert first["provenance"] == str(sidecar)
    assert first["asset_sha256"] == hashlib.sha256(asset.read_bytes()).hexdigest()
    assert first["measured"]["boundary_edges"] == 0
    assert first["measured"]["winding_consistent"]
    assert first["measured"]["metallicFactor"] == 0.75

    monkeypatch.setattr("sys.argv", [*common, str(asset), "--verdict", "bad",
                                      "--note", "muddy color", "--source", "back.png"])
    assert mark_asset.main() == 0
    assert "culling not stated" in capsys.readouterr().out
    assert first == mark_asset.load_register(register)[0]
    assert len(mark_asset.load_register(register)) == 2
    with register.open("a") as content:
        content.write("\n")
    monkeypatch.setattr("sys.argv", [*common, "--for", "front.png", "--verdict", "good"])
    assert mark_asset.main() == 0
    output = capsys.readouterr().out
    assert "stable topology" in output and "1 verdict(s)" in output
    assert "muddy color" not in output


def test_register_retains_a_verdict_when_mesh_measurement_fails(tmp_path):
    unreadable = tmp_path / "unreadable.glb"
    unreadable.write_bytes(b"bad GLB")
    entry = mark_asset.build_entry(
        unreadable, "mixed", "could not open", culled=False,
        now=dt.datetime(2023, 2, 1, tzinfo=dt.UTC),
    )
    assert entry["recorded_at"] == "2023-02-01T00:00:00+00:00"
    assert entry["provenance"] is None
    assert entry["asset_sha256"] == hashlib.sha256(b"bad GLB").hexdigest()
    assert "error" in entry["measured"]
    printed = mark_asset._format(entry)
    assert "[DOUBLE-SIDED] could not open" in printed
    assert entry["asset_sha256"][:12] in printed
    with pytest.raises(ValueError, match="verdict must be one of"):
        mark_asset.build_entry(unreadable, "maybe", "could not open")
    with pytest.raises(ValueError, match="note is required"):
        mark_asset.build_entry(unreadable, "mixed", "  ")


@pytest.mark.parametrize(("arguments", "message"), [
    ([], "needs an asset and --verdict"),
    (["missing.glb", "--verdict", "good"], "missing.glb does not exist"),
])
def test_register_cli_rejects_missing_inputs(monkeypatch, capsys, arguments, message):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr("sys.argv", ["mark_asset.py", *arguments])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(mark_asset.__file__, run_name="__main__")
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
