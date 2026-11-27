"""Exercise the finish command with local mesh files and no Blender process."""

import json
import sys
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab.provenance import read_model_record
from scripts import retopo_repaint


@pytest.fixture
def finish_inputs(tmp_path):
    source = tmp_path / "input.glb"
    trimesh.creation.box(extents=(1, 2, 3)).export(source)
    blender = tmp_path / "blender"
    blender.touch()
    return source, tmp_path / "unused.png", tmp_path / "out.glb", blender


def test_finish_keeps_mesh_and_records_height_and_texture_budget(finish_inputs, monkeypatch):
    source, image, output, blender = finish_inputs
    source_record = output.with_name("source-license.json")
    source_record.write_text(json.dumps({"license": {"name": "Example license"}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
            "--keep-mesh", "--skip-paint", "--height-m", "4", "--source-record", str(source_record),
            "--skip-compress", "--drop-surface-maps",
        ],
    )
    assert retopo_repaint.main() == 0
    mesh = trimesh.load(output, force="mesh")
    assert len(mesh.faces) == 12
    assert mesh.extents[1] == pytest.approx(4)
    np.testing.assert_allclose(mesh.extents / mesh.extents[1], [.5, 1, 1.5])
    record = json.loads(output.with_suffix(".retopo-repaint.json").read_text())
    assert record["retopology"] is None
    assert record["full_detail"] is True
    assert record["stages"] == ["compress"]
    assert record["texture_budget"]["surface_maps"] == "drop"
    assert read_model_record(output)["source_record"]["license"]["name"] == "Example license"


@pytest.mark.parametrize("missing", ["input", "blender", "transforms"])
def test_finish_reports_missing_prerequisites(finish_inputs, monkeypatch, tmp_path, missing):
    source, image, output, blender = finish_inputs
    args = [
        "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
        "--skip-paint",
    ]
    if missing == "input":
        source.unlink()
        expected = "not found"
    elif missing == "blender":
        blender.unlink()
        expected = "Blender"
    else:
        image.touch()
        args.extend(["--views", str(tmp_path)])
        expected = "no transforms.json"
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit, match=expected):
        retopo_repaint.main()
    assert not output.exists()


@pytest.fixture
def photo_inputs(tmp_path):
    source = tmp_path / "triangle.glb"
    mesh = trimesh.Trimesh(
        vertices=[[-1, 0, -1], [1, 0, -1], [-1, 0, 1]], faces=[[0, 2, 1]], process=False
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0, 0], [1, 0], [0, 1]],
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (12, 12), "blue")
        ),
    )
    mesh.export(source)
    views = tmp_path / "views"
    views.mkdir()
    image = views / "view_000.png"
    Image.new("RGBA", (32, 32), "red").save(image)
    camera = np.eye(4)
    camera[2, 3] = 4
    back_camera = np.diag([-1, 1, -1, 1])
    back_camera[2, 3] = -4
    (views / "transforms.json").write_text(json.dumps({
        "camera_angle_x": 1.2,
        "frames": [
            {"file_path": image.name, "transform_matrix": back_camera.tolist(), "paint": "none"},
            {"file_path": image.name, "transform_matrix": camera.tolist()},
        ],
    }))
    return source, image, views


def test_finish_bakes_photo_fade_and_reuses_completed_stages(photo_inputs, tmp_path, monkeypatch):
    source, image, views = photo_inputs
    blender, output, folder = tmp_path / "blender", tmp_path / "final.glb", tmp_path / "steps"
    blender.touch()
    steps = retopo_repaint.step_paths(output, folder)
    commands = []

    def run(command, *, stdout, stderr, check):
        assert command[0] == str(blender)
        assert not check
        stage = Path(command[5]).name
        params = command[7:]
        commands.append(stage)
        if stage == "blender_retopo_bake.py":
            assert params[:4] == [str(source), str(steps["retopo"]), "50000", "16"]
            steps["retopo"].write_bytes(source.read_bytes())
        elif stage == "blender_bake_detail.py":
            assert params == [str(source), str(steps["photo"]), str(steps["baked"]), "16",
                              str(steps["photo_fade"])]
            with Image.open(steps["photo_fade"]) as mask:
                assert mask.size == (16, 16)
                assert mask.getextrema()[1] > 0
            steps["baked"].write_bytes(steps["photo"].read_bytes())
        else:
            assert stage == "blender_upright.py"
            Path(params[1]).write_bytes(Path(params[0]).read_bytes())
        return CompletedProcess(command, 0)

    args = [
        "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
        "--views", str(views), "--steps-dir", str(folder), "--skip-paint", "--skip-compress",
        "--no-nudge", "--no-edge-fade", "--atlas", "16", "--detail", "high", "--upright",
    ]
    monkeypatch.setattr(retopo_repaint.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", args)
    assert retopo_repaint.main() == 0
    assert commands == ["blender_retopo_bake.py", "blender_bake_detail.py", "blender_upright.py"]
    log = output.with_suffix(".retopo-repaint.json")
    first = json.loads(log.read_text())
    assert first["retopology"]["faces"] == 50000
    assert first["stages"] == ["retopologise", "calm", "orient", "photo", "bake"]
    assert first["bake"]["flat_where_photo"]
    assert first["photo"]["count"] == 2
    assert first["photo"]["texture_share"] > 0
    assert not first["photo"]["nudge"]
    assert not first["photo"]["edge_fade"]
    final_bytes = output.read_bytes()

    commands.clear()
    monkeypatch.setattr(sys, "argv", [*args[:-1], "--resume"])
    assert retopo_repaint.main() == 0
    assert commands == []
    assert output.read_bytes() == final_bytes
    resumed = json.loads(log.read_text())
    assert resumed["resumed"]
    assert resumed["calm"] == {"reused": True}
    assert resumed["orient"] == {"reused": True}
    assert resumed["photo"]["reused"]
    assert resumed["bake"]["flat_where_photo"]


def test_photo_cli_writes_texture_weights_and_keeps_faces(photo_inputs, tmp_path, monkeypatch):
    from image_to_3dlab.photo_paint import read_glb
    from scripts import photo_paint

    source, _image, views = photo_inputs
    output, weights = tmp_path / "painted.glb", tmp_path / "weights.png"
    monkeypatch.setattr(sys, "argv", [
        "photo_paint", str(source), str(views), str(output), "--weights", str(weights),
        "--no-align", "--no-local-colour", "--cutout-edge", "0", "--inner-edge", "0",
    ])
    assert photo_paint.main() == 0
    positions, uv, faces, texture = read_glb(output)
    original_positions, original_uv, original_faces, _ = read_glb(source)
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(uv, original_uv)
    np.testing.assert_array_equal(faces, original_faces)
    assert texture[..., 0].max() > 128
    assert Image.open(weights).getextrema()[1] > 128


def test_photo_cli_rejects_missing_model(tmp_path, monkeypatch):
    from scripts import photo_paint

    monkeypatch.setattr(sys, "argv", [
        "photo_paint", str(tmp_path / "missing.glb"), str(tmp_path), str(tmp_path / "painted.glb"),
    ])
    with pytest.raises(SystemExit, match="not found"):
        photo_paint.main()


@pytest.mark.parametrize("available,failed", [(False, False), (True, True), (True, False)])
def test_finish_selects_native_route_or_falls_back(
    finish_inputs, tmp_path, monkeypatch, capsys, available, failed
):
    source, image, output, blender = finish_inputs
    dump = tmp_path / "native.post.bin"
    dump.touch()
    calls = []
    steps = retopo_repaint.step_paths(output)

    def native(dump_path, original, target, faces, atlas, log):
        calls.append("native")
        assert (dump_path, original, target, faces, atlas, log) == (
            dump, source, steps["retopo"], 20000, 2048, steps["retopo_log"]
        )
        if failed:
            raise retopo_repaint.pixal3d_native.ReplayFailed("fixture failed")
        target.write_bytes(original.read_bytes())
        return {"route": "pixal3d", "faces": faces}

    def classic(command, *, stdout, stderr, check):
        calls.append("classic")
        assert Path(command[5]).name == "blender_retopo_bake.py"
        assert command[7:9] == [str(source), str(steps["retopo"])]
        steps["retopo"].write_bytes(source.read_bytes())
        return CompletedProcess(command, 0)

    monkeypatch.setattr(retopo_repaint.pixal3d_native, "replay_available", lambda: available)
    monkeypatch.setattr(retopo_repaint.pixal3d_native, "native_retopo", native)
    monkeypatch.setattr(retopo_repaint.subprocess, "run", classic)
    monkeypatch.setattr(sys, "argv", [
        "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
        "--native-dump", str(dump), "--skip-paint", "--skip-bake", "--skip-compress",
    ])
    assert retopo_repaint.main() == 0
    route = json.loads(output.with_suffix(".retopo-repaint.json").read_text())["retopology"]["route"]
    if not available:
        assert calls == ["classic"]
        assert route == "classic"
        assert "post-replay is not built" in capsys.readouterr().out
    elif failed:
        assert calls == ["native", "classic"]
        assert route == "classic"
        assert "fixture failed" in capsys.readouterr().out
    else:
        assert calls == ["native"]
        assert route == "pixal3d"
    assert len(trimesh.load(output, force="mesh").faces) == 12


@pytest.mark.parametrize("verified", [False, True])
def test_finish_records_paint_stage_licence_or_flags_missing_record(
    finish_inputs, monkeypatch, verified
):
    from types import SimpleNamespace

    from image_to_3dlab.provenance import write_hunyuan_sidecar

    source, image, output, blender = finish_inputs
    Image.new("RGB", (4, 4), "blue").save(image)
    steps = retopo_repaint.step_paths(output)
    steps["retopo"].write_bytes(source.read_bytes())
    calls = []

    def paint(model, picture, destination, seed, resolution, steps_count, size, started):
        calls.append((model, picture, destination))
        assert (seed, resolution, steps_count, size) == (7, 128, 3, 64)
        assert started > 0
        destination.write_bytes(model.read_bytes())
        if verified:
            write_hunyuan_sidecar(destination, picture, backend="paint", paint=True)

    monkeypatch.setitem(sys.modules, "hunyuan_mlx_xiong_generate", SimpleNamespace(run_paint=paint))
    monkeypatch.setattr(retopo_repaint, "host_platform", lambda: retopo_repaint.APPLE)
    monkeypatch.setattr(sys, "argv", [
        "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
        "--resume", "--skip-bake", "--skip-compress",
        "--paint-seed", "7", "--paint-res", "128", "--paint-steps", "3", "--paint-tex", "64",
    ])
    assert retopo_repaint.main() == 0
    assert calls == [(steps["retopo"], image, steps["painted"])]
    assert len(trimesh.load(output, force="mesh").faces) == 12
    component, = read_model_record(output)["component_records"]
    if verified:
        assert component["components"][0]["license"].startswith("Tencent")
    else:
        assert component["license"]["verified"] is False
        assert component["output"]["classification"] == "unverified"
    record = json.loads(output.with_suffix(".retopo-repaint.json").read_text())
    assert record["paint"] == {"seed": 7, "res": 128, "steps": 3, "texture": 64}


def test_finish_honours_staged_pixel_match_off_and_bakes_without_mask(
    finish_inputs, monkeypatch, tmp_path, capsys
):
    from runpy import run_path

    source, image, output, blender = finish_inputs
    Image.new("RGB", (4, 4)).save(image)
    views = tmp_path / "views"
    views.mkdir()
    (views / "transforms.json").write_text(json.dumps({"pixel_match": False}))
    steps = retopo_repaint.step_paths(output)
    steps["retopo"].write_bytes(source.read_bytes())
    calls = []

    def run(command, *, stdout, stderr, check):
        calls.append(command[7:])
        assert Path(command[5]).name == "blender_bake_detail.py"
        steps["baked"].write_bytes(steps["retopo"].read_bytes())
        return CompletedProcess(command, 0)

    monkeypatch.setattr(retopo_repaint.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [
        "retopo_repaint", str(source), str(image), str(output), "--blender", str(blender),
        "--views", str(views), "--skip-paint", "--resume",
    ])
    with pytest.raises(SystemExit) as result:
        run_path(retopo_repaint.__file__, run_name="__main__")
    assert result.value.code == 0
    assert calls == [[str(source), str(steps["retopo"]), str(steps["baked"]), "2048"]]
    assert "a four-view build leaves Pixel Match off" in capsys.readouterr().out
    record = json.loads(output.with_suffix(".retopo-repaint.json").read_text())
    assert record["photo"] is None
    assert not record["bake"]["flat_where_photo"]
    assert record["texture_budget"]["surface_maps"] == "keep"
