"""Run the real command-line parser, manifest handling and provenance writer."""

import json
import os
import runpy
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from glb_fixtures import valid_glb

from image_to_3dlab import cli, models_home, sf3d_backend, trellis_backend
from image_to_3dlab.comfyui_backend import ComfyUIClient

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("launch", ["import", "main"])
def test_entry_point_sets_model_environment_before_cli(tmp_path, monkeypatch, capsys, launch):
    monkeypatch.setenv(models_home.ENV, str(tmp_path))
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:256")
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "HF_HOME", "U2NET_HOME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--help"])

    if launch == "main":
        with pytest.raises(SystemExit) as exit:
            runpy.run_path(str(REPO / "pipeline.py"), run_name="__main__")
        assert exit.value.code == 0
        assert "--run-manifest" in capsys.readouterr().out
    else:
        assert runpy.run_path(str(REPO / "pipeline.py"))["main"] is cli.main
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert os.environ["PYTORCH_HIP_ALLOC_CONF"] == "max_split_size_mb:256"
    assert os.environ["HF_HOME"] == str(tmp_path / "huggingface")
    assert os.environ["U2NET_HOME"] == str(tmp_path / "u2net")


@pytest.fixture
def source(tmp_path):
    image = tmp_path / "in.png"
    image.write_bytes(b"picture")
    return image


@pytest.mark.parametrize(("opts", "message"), [
    (["--fast", "--foreground-ratio", "0"], "foreground-ratio"),
    (["--fast", "--texture-resolution", "-1"], "texture-resolution"),
    (["--trellis", "--trellis-bake-target-faces", "0"], "trellis-bake-target-faces"),
    ([], "exactly one mode"),
])
def test_invalid_flags_report_usage_error(source, tmp_path, capsys, opts, message):
    assert cli.main([str(source), "--output-dir", str(tmp_path / "out"), *opts]) == 2
    assert message in capsys.readouterr().err


def test_quality_mode_refuses_unknown_license_without_opt_in(source, tmp_path, capsys):
    assert cli.main([str(source), "--quality", "--output-dir", str(tmp_path / "out")]) == 2
    assert "license policy" in capsys.readouterr().err


def test_quality_mode_reports_invalid_workflow_file(source, tmp_path, capsys):
    assert cli.main([str(source), "--quality", "--workflow", str(tmp_path / "missing")]) == 2
    assert "could not read workflow" in capsys.readouterr().err


def test_fast_mode_cannot_use_multiple_views(source, capsys):
    assert cli.main([str(source), "--fast", "--view", str(source)]) == 2
    assert "multiple views are only supported by the trellis2 backend" in capsys.readouterr().err


def test_resolve_identity_without_selected_backend():
    assert cli._resolve_model_identity(Namespace(fast=False, trellis=False, quality=False)) is None


def test_workflow_ignores_non_nodes_and_non_string_checkpoint_names(tmp_path):
    workflow = tmp_path / "api.json"
    workflow.write_text(json.dumps({
        "comment": "ignored",
        "1": {"class_type": 3},
        "2": {"class_type": "CheckpointLoader", "inputs": {"ckpt_name": 42}},
        "3": {"class_type": "CheckpointLoader", "inputs": {"ckpt_name": "actual.safetensors"}},
    }))
    assert cli._extract_workflow_model_identity(workflow) == "actual.safetensors"


def write_manifest(tmp_path, backend, parameters, **extra):
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "model": {"backend": backend, "parameters": parameters},
        "input": {"path": "in.png"}, "output": {"directory": str(tmp_path / "out")}, **extra,
    }))
    return manifest


def test_quality_mode_requires_workflow_after_license_validation(source, tmp_path, capsys, monkeypatch):
    manifest = write_manifest(
        tmp_path, "hunyuan-comfyui", {},
        use_case="showcase", distribution="private", commercial_intent=False,
        license_policy={"allow_unverified": True},
    )

    def unexpected_generation(*args, **kwargs):
        pytest.fail("A missing workflow reached the generation backend")

    monkeypatch.setattr(ComfyUIClient, "generate", unexpected_generation)
    assert cli.main(["--run-manifest", str(manifest)]) == 1
    assert "--quality requires --workflow (ComfyUI API-format JSON)" in capsys.readouterr().err


@pytest.mark.parametrize(("backend", "extra", "message"), [
    ("pixal3d", {}, "not implemented yet"),
    ("hunyuan-comfyui", {"workflow": ""}, "workflow must be a path"),
])
def test_unsupported_manifest_cannot_start(source, tmp_path, capsys, backend, extra, message):
    manifest = write_manifest(tmp_path, backend, {}, **extra)
    assert cli.main(["--run-manifest", str(manifest)]) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_trellis_manifest_parameters_reach_backend_and_sidecar(source, tmp_path, monkeypatch):
    parameters = {
        "seed": 41, "steps": 8, "pipeline_type": "1024", "texture_size": 2048,
        "bake_target_faces": 9000, "pre_simplify_cap": 20000, "normalize_material": False,
        "material_mode": "matte", "trellis_remesh": True, "trellis_remesh_project": "0.25",
    }
    manifest = write_manifest(tmp_path, "trellis2", parameters)
    repo = tmp_path / "vendor"

    def worker(views, working, options):
        assert views == [source]
        assert options == trellis_backend.TrellisOptions(
            repo=repo, seed=41, steps=8, pipeline_type="1024", texture_size=2048,
            bake_target_faces=9000, pre_simplify_cap=20000, normalize_material=False,
            material_mode="matte", remesh=True, remesh_project=.25,
        )
        asset = working / "mesh.glb"
        asset.write_bytes(valid_glb())
        return trellis_backend.TrellisResult(asset, "metal-o-voxel")

    monkeypatch.setattr(trellis_backend, "generate_trellis", worker)
    assert cli.main(["--run-manifest", str(manifest), "--trellis-repo", str(repo)]) == 0
    sidecar, = (tmp_path / "out").rglob("*.provenance.json")
    info = json.loads(sidecar.read_text())["model"]["parameters"]
    assert info["trellis_remesh_project"] == .25
    assert info["texture_backend"] == "metal-o-voxel"


@pytest.mark.parametrize("crash", [False, True])
def test_sf3d_manifest_options_and_cpu_flag(source, tmp_path, monkeypatch, capsys, crash):
    manifest = write_manifest(tmp_path, "sf3d", {
        "texture_resolution": 256, "foreground_ratio": .5, "remesh": "triangle", "target_vertices": 400,
    })
    monkeypatch.delenv("SF3D_USE_CPU", raising=False)

    def worker(image, directory, options):
        assert image == source
        assert os.environ["SF3D_USE_CPU"] == "1"
        assert (options.texture_resolution, options.foreground_ratio) == (256, .5)
        assert (options.remesh, options.target_vertices) == ("triangle", 400)
        assert not options.cpu_fallback
        if crash:
            raise RuntimeError("worker crashed")
        asset = directory / "mesh.glb"
        asset.write_bytes(valid_glb())
        return asset

    monkeypatch.setattr(sf3d_backend, "generate_sf3d", worker)
    result = cli.main(["--run-manifest", str(manifest), "--cpu", "--no-cpu-fallback"])
    sidecars = list((tmp_path / "out").rglob("*.provenance.json"))
    if crash:
        assert result == 1
        assert "worker crashed" in capsys.readouterr().err
        assert not sidecars
    else:
        assert result == 0
        assert len(sidecars) == 1
        assert json.loads(sidecars[0].read_text())["model"]["parameters"]["target_vertices"] == 400
