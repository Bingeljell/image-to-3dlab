"""Check Pixal3D's command, pre-flight refusals, and saved run records."""

import io
import json
import runpy
import sys
from functools import partial
from unittest.mock import Mock

import pixal3d_generate as pixal
import pytest
from PIL import Image

from tests.test_glb_turn import make_glb


@pytest.fixture
def pixal_files(tmp_path, monkeypatch):
    front = tmp_path / "front.png"
    cutout = Image.new("RGBA", (24, 24))
    cutout.paste((20, 70, 90, 255), (4, 4, 20, 20))
    cutout.save(front)
    cli = tmp_path / "trellis-cli"
    cli.write_bytes(pixal.SHAPE_MARKER + b" " + pixal.DUMP_MARKER)
    models = tmp_path / "models"
    models.mkdir()
    for i in range(9):
        (models / f"local-{i}.gguf").touch()
    output = tmp_path / "build" / "result.glb"
    argv = [str(front), str(output), "--cli", str(cli), "--models", str(models), "--steps", "12"]
    return front, cli, models, output, argv


@pytest.mark.parametrize(("fault", "text"), [
    ("sides", "give both sides"),
    ("missing", "not found:"),
    ("unsupported", "cannot repaint"),
    ("wrong framing", "give --back"),
    ("multi model", "multi-view weights"),
    ("single model", "not ready"),
])
def test_pixal_refuses_invalid_inputs_before_launch(pixal_files, monkeypatch, fault, text):
    front, cli, models, output, argv = pixal_files
    shape = front.with_name("past.shape.bin")
    if fault == "sides":
        argv += ["--right", str(front)]
    elif fault == "missing":
        argv += ["--back", str(front.parent / "missing.png")]
    elif fault in {"unsupported", "wrong framing"}:
        shape.touch()
        shape.with_name("past.provenance.json").write_text('{"parameters":{"views":"front + back"}}', encoding="utf-8")
        argv += ["--reuse-shape", str(shape)]
        if fault == "unsupported":
            cli.touch()
            cli.write_bytes(b"stock CLI")
    elif fault == "multi model":
        argv += ["--back", str(front)]
    else:
        (models / "local-0.gguf").unlink()

    worker = Mock(side_effect=AssertionError("invalid run started"))
    monkeypatch.setattr(pixal.subprocess, "Popen", worker)
    monkeypatch.setattr(sys, "argv", [pixal.__file__, *argv])
    with pytest.raises(SystemExit, match=text):
        pixal.main()
    worker.assert_not_called()
    assert not output.parent.exists()


@pytest.mark.parametrize("scenario", ["new shape", "skin", "ignored steps", "failed", "missing glb"])
def test_pixal_process_outcome_controls_record_and_auxiliary_files(
        pixal_files, tmp_path, monkeypatch, capsys, scenario):
    front, cli, models, output, argv = pixal_files
    argv.append("--no-matte")
    shape = tmp_path / "past.shape.bin"
    shape.write_bytes(b"reusable shape")
    if scenario == "skin":
        back = tmp_path / "back.png"
        Image.new("RGB", (24, 24), color=(99, 99, 99)).save(back)
        (models / "pixal3d-models.json").write_text('{"files":[{"name":"local-0.gguf"}]}', encoding="utf-8")
        argv += ["--reuse-shape", str(shape), "--back", str(back), "--gsh", "3"]
        shape.with_name("past.provenance.json").write_text('{"parameters":[]}', encoding="utf-8")
        shape.with_name("past.json").write_text('{"parameters":{"views":"front + back"}}', encoding="utf-8")

        def cut_out(image):
            assert image.mode == "RGB"
            with Image.open(front) as cut:
                return cut.copy(), "fixture"
        monkeypatch.setattr(pixal, "_rembg_remove", cut_out)
    if scenario == "ignored steps":
        flow = tmp_path / "flow_runner.cpp"
        flow.write_text(pixal.STEPS_MARKER)
        cli.touch()
        argv += ["--steps", "8"]
        monkeypatch.setattr(pixal, "resolve_steps", partial(pixal.resolve_steps, source=flow))

    killed = []

    def start(command, *, env, **kwargs):
        assert command[0] == str(cli) and command[-1] == str(output)
        assert command[command.index("--dump-post") + 1] == str(output.with_suffix(pixal.DUMP_SUFFIX))
        assert env[pixal.DUMP_ENV] == "1"
        lines = "ggml_cuda_init: noisy output\n[1/6] structure\n"
        if scenario == "ignored steps":
            assert env[pixal.STEPS_ENV] == "8"
            lines += "[flow] [..........] 0/12\n"
        if scenario == "skin":
            assert env[pixal.SHAPE_LOAD_ENV] == str(shape)
            assert pixal.SHAPE_SAVE_ENV not in env
            assert command[command.index("--gsh") + 1] == "3.0"
            staged = output.with_suffix(".mvinput")
            assert (staged / "transforms.json").is_file()
            assert output.with_name("result__back_matted.png").is_file()
        else:
            assert env[pixal.SHAPE_SAVE_ENV] == str(output.with_suffix(pixal.SHAPE_SUFFIX))

        class Process:
            stdout = io.StringIO(lines)

            @staticmethod
            def kill():
                killed.append(True)

            @staticmethod
            def wait():
                if scenario == "failed":
                    return 7
                if scenario != "missing glb":
                    output.write_bytes(make_glb([[0.1, 0.2, 0.3]]))
                if scenario == "new shape":
                    output.with_suffix(pixal.SHAPE_SUFFIX).write_bytes(b"shape")
                    output.with_suffix(pixal.DUMP_SUFFIX).write_bytes(b"dump")
                return 0

        return Process()

    monkeypatch.setattr(pixal.subprocess, "Popen", start)
    monkeypatch.setattr(sys, "argv", [pixal.__file__, *argv])
    manifest = output.with_suffix(".provenance.json")
    failures = {"ignored steps": "ignored PIXAL3D_STEPS", "failed": "exited with code 7",
                "missing glb": "without writing"}
    if scenario in failures:
        with pytest.raises(SystemExit, match=failures[scenario]):
            pixal.main()
        assert bool(killed) is (scenario == "ignored steps")
        assert not manifest.exists() and not output.exists()
        return
    if scenario == "new shape":
        with pytest.raises(SystemExit) as status:
            runpy.run_path(pixal.__file__, run_name="__main__")
        assert status.value.code == 0
    else:
        assert pixal.main() == 0
    report = json.loads(manifest.read_text())
    assert report["output"]["sha256"] == pixal.sha256_file(output)
    assert report["parameters"]["steps"] == 12
    if scenario == "new shape":
        assert report["shape"] == "result.shape.bin"
        assert report["post_dump"] == "result.post.bin"
    else:
        assert report["reused_shape"]["sha256"] == pixal.sha256_file(shape)
        assert report["parameters"]["views"] == "front + back"
        assert "post_dump" not in report
    log = capsys.readouterr().out
    assert "ggml_cuda_init" not in log
    assert "[1/6] structure" in log
