"""A manifest runs through the real TRELLIS adapter, with only inference replaced."""

import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import trimesh
from PIL import Image

from image_to_3dlab import __version__
from image_to_3dlab.provenance import read_model_record
from image_to_3dlab.trellis_backend import read_glb

FIXTURES = Path(__file__).with_name("acceptance_cli_vendor")
pytestmark = pytest.mark.skipif(
    sys.platform not in {"linux", "darwin"}, reason="TRELLIS fixture uses a POSIX launcher"
)

PARAMETERS = {
    "seed": 143,
    "pipeline_type": "512",
    "steps": 9,
    "texture_size": 512,
    "bake_target_faces": 1200,
    "pre_simplify_cap": 2400,
    "trellis_remesh": True,
    "trellis_remesh_project": 0.25,
}


@pytest.fixture
def cli_run(tmp_path):
    checkout = tmp_path / "dépôt local"
    launcher = checkout / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    launcher.chmod(0o700)
    shutil.copyfile(FIXTURES / "trellis_generator.py", checkout / "generate.py")
    disabled = checkout / "TRELLIS.2/trellis2/pipelines/trellis2_image_to_3d.py"
    disabled.parent.mkdir(parents=True)
    disabled.write_text("pipeline.rembg_model = None\n", encoding="utf-8")

    work = tmp_path / "exécution isolée"
    work.mkdir()
    images = ["entrée devant.png", "entrée côté.png"]
    for image in images:
        Image.new("RGBA", (2, 2), (255, 0, 0, 127)).save(work / image)
    outputs = work / "modèles finaux"
    manifest = work / "plan d’essai.json"
    model_id = "microsoft/TRELLIS.2-4B"
    run = {
        "schema_version": 1,
        "input": {"views": images},
        "model": {"backend": "trellis2", "id": model_id, "parameters": PARAMETERS},
        "use_case": "game",
        "distribution": "worldwide",
        "commercial_intent": True,
        "output": {"directory": str(outputs)},
    }
    cache = tmp_path / "cache privé"
    cache.mkdir()
    invocation = work / "invocation.json"

    def invoke(*, has_weights=True, fail=False, hunyuan=False):
        if has_weights:
            snapshot = cache / "models--microsoft--TRELLIS.2-4B/snapshots/fixture"
            snapshot.mkdir(parents=True)
            (snapshot / "model.marker").touch()
        if hunyuan:
            run["input"] = {"path": images[0]}
            run["model"] = {"backend": "hunyuan-comfyui", "id": "tencent/Hunyuan3D-2.1"}
            run["commercial_intent"] = False
            run["workflow"] = "flux local.json"
            (work / run["workflow"]).write_text("{}", encoding="utf-8")
        manifest.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        env = {
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "HF_HOME": str(cache),
            "HF_HUB_CACHE": str(cache),
            "I3D_NO_USAGE": "1",
            "FIXTURE_INVOCATION": str(invocation),
            "FIXTURE_GENERATION_FAIL": str(int(fail)),
        }
        command = [
            sys.executable, str(FIXTURES / "run_pipeline.py"),
            "--run-manifest", str(manifest), "--trellis-repo", str(checkout),
        ]
        # The adapter launches the fixture executable. Reap both on a hung run.
        with subprocess.Popen(
            command, cwd=work, env=env, start_new_session=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate(timeout=5)
                raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    return invoke, manifest, outputs, checkout, invocation


def test_manifest_publishes_trellis_output_and_licence(cli_run):
    invoke, manifest, outputs, checkout, invocation = cli_run
    process = invoke()
    assert process.returncode == 0, process.stderr
    asset, = (outputs / "commercial").glob("*.glb")
    assert asset.with_suffix(".provenance.json").is_file()
    working = outputs / ".working"
    call = json.loads(invocation.read_text(encoding="utf-8"))
    assert call == {
        "argv": [
            str(working / "entrée devant_trellis_input_view0.png"),
            str(working / "entrée côté_trellis_input_view1.png"),
            "--seed", "143", "--output", str(working / "entrée devant_trellis2"),
            "--pipeline-type", "512", "--texture-size", "512", "--bake-target-faces",
            "1200", "--steps", "9", "--remesh", "--remesh-project", "0.25",
            "--remesh-band", "1.0",
        ],
        "cwd": str(checkout),
        "pre_cap": "2400",
    }
    document, _, _, version = read_glb(asset)
    assert version == 2
    assert document["asset"]["generator"] == f"AssetFurnace {__version__}"
    primitive, = document["meshes"][0]["primitives"]
    assert primitive.get("mode", 4) == 4
    assert document["accessors"][primitive["attributes"]["POSITION"]]["count"] == 3
    mesh = trimesh.load(asset, force="mesh", process=False)
    face, = mesh.faces
    assert {tuple(vertex) for vertex in mesh.vertices[face]} == {
        (0, 0, 0), (1, 0, 0), (0, 1, 0),
    }

    record = read_model_record(asset)
    assert record is not None
    assert record["schema_version"] == 1
    assert record["made_with"] == {
        "name": "AssetFurnace", "version": __version__, "url": "https://assetfurnace.com",
    }
    assert record["intent"] == {"use_case": "game", "distribution": "worldwide",
                                "commercial_intent": True}
    assert record["model"]["backend"] == "trellis2"
    assert record["model"]["identity"] == "microsoft/TRELLIS.2-4B"
    assert {key: record["model"]["parameters"][key] for key in PARAMETERS} == PARAMETERS
    assert record["license"]["name"] == "MIT (TRELLIS.2) + DINOv3 License"
    assert record["license"]["url"] == "https://huggingface.co/microsoft/TRELLIS.2-4B"
    components = {component["component"]: component for component in record["components"]}
    assert components["facebook/dinov3-vitl16-pretrain-lvd1689m"]["license"] == "DINOv3 License"
    assert components["BRIA RMBG-2.0"]["loaded"] is False
    assert record["output"]["classification"] == "commercial-clear"
    for field, path in {
        "input": manifest.parent / "entrée devant.png",
        "output": asset,
        "source_manifest": manifest,
    }.items():
        assert record[field]["path"] == str(path)
        assert record[field]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("case", "options", "status", "diagnostic", "started"),
    [
        ("empty-cache", {"has_weights": False}, 1,
         "TRELLIS.2 (--trellis (vendor/trellis-mac)) needs its weights (about 16.5 GB)", False),
        ("failed-child", {"fail": True}, 1, "TRELLIS generation failed with exit code 23", True),
        ("hunyuan-policy", {"has_weights": False, "hunyuan": True}, 2,
         "license policy: Hunyuan outputs are not cleared for worldwide game distribution", False),
    ],
    ids=["empty-cache", "failed-child", "hunyuan-policy"],
)
def test_manifest_failure_does_not_publish(cli_run, case, options, status, diagnostic, started):
    invoke, _, outputs, _, invocation = cli_run
    process = invoke(**options)
    assert process.returncode == status, process.stderr
    assert diagnostic in process.stderr
    assert invocation.exists() == started
    assert not list(outputs.rglob("*.provenance.json"))
    published = [path for path in outputs.rglob("*.glb") if ".working" not in path.relative_to(outputs).parts]
    assert published == []
    if case == "failed-child":
        assert len(list((outputs / ".working").glob("*.glb"))) == 1
    if case == "hunyuan-policy":
        assert not outputs.exists()
