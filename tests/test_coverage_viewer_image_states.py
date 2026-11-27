"""Picture jobs with a tiny cache and a stand-in for the picture process."""

import json
from pathlib import Path
from unittest.mock import Mock

import image_api
import pytest
from PIL import Image

from tests.test_coverage_viewer_workers import Child


@pytest.fixture
def weight_cache(tmp_path, monkeypatch):
    cache = tmp_path / "weights"
    monkeypatch.setattr(image_api, "HF_HUB", cache)
    found = {}
    for key, (directory, filename) in image_api.FLUX.weight_files.items():
        path = cache / directory / "snapshots" / "revision" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test weights placeholder")
        found[key] = path
    return found


def test_image_weights_resolve_real_snapshot_layout(weight_cache):
    assert image_api.resolve_weights(image_api.FLUX) == weight_cache
    missing = next(iter(weight_cache.values()))
    missing.unlink()
    with pytest.raises(image_api.MissingWeights) as caught:
        image_api.resolve_weights(image_api.FLUX)
    assert missing.name in caught.value.missing


@pytest.mark.parametrize("scenario", ["original", "edit", "cancel", "broken"])
def test_picture_worker_handles_provenance_and_cleans_up(tmp_path, monkeypatch, weight_cache, scenario):
    manager = image_api.ImageJobManager(tmp_path)
    engine = image_api.FLUX
    settings = image_api.clean_settings({}, engine)
    reference = tmp_path / "reference.png"
    if scenario == "edit":
        Image.new("RGB", (16, 16), (1, 5, 20)).save(reference)
    job = manager.create("A prop", settings, engine=engine, reference=reference if scenario == "edit" else None)
    process = Child(b"\nprogress output\n", wait_error=OSError("child pipe lost") if scenario == "broken" else None)

    def generate(args, **kwargs):
        result = Path(args[args.index("-o") + 1])
        assert result == job.output_path
        if scenario == "edit":
            assert args[args.index("-r") + 1] == str(reference)
        else:
            assert "-r" not in args
        if scenario in {"original", "edit"}:
            Image.new("RGB", (16, 16), "red").save(result)
        elif scenario == "cancel":
            job.cancel_requested = True
        return process

    def reap(child, *, grace):
        assert child is process and grace == 5
        child.exited = True
        return "worker stopped"

    runner = Mock(side_effect=generate)
    monkeypatch.setattr(image_api.subprocess, "Popen", runner)
    monkeypatch.setattr(image_api.processes, "stop_and_reap", reap)
    monkeypatch.setattr(image_api, "host_platform", lambda: image_api.APPLE)
    image_api.run_job(job, manager)
    runner.assert_called_once()
    assert manager.active is None
    assert not (job.directory / "pid").exists()
    if scenario == "cancel":
        assert job.status == "cancelled"
        assert list(job.directory.iterdir()) == []
    elif scenario == "broken":
        assert job.status == "error"
        assert "child pipe lost" in job.error
        assert "cleanup warning: worker stopped" in job.log_lines
    else:
        assert job.status == "done"
        record = json.loads(job.output_path.with_suffix(".provenance.json").read_text(encoding="utf-8"))
        assert record["model"]["id"] == "flux2-klein-4b"
        assert record["license"]["name"]
        if scenario == "edit":
            assert record["reference"] == {"picture": reference.name, "mode": "edit (back view)"}
        else:
            assert "reference" not in record
        assert "brush" not in record


def test_picture_cancellation_only_signals_present_child(tmp_path, monkeypatch):
    manager = image_api.ImageJobManager(tmp_path)
    job = manager.create("A prop", {})
    job.process = Child(b"")
    signal = Mock()
    monkeypatch.setattr(image_api.processes, "terminate_group", signal)
    assert manager.cancel(job.id)
    signal.assert_called_once_with(job.process.pid)
    assert not manager.cancel(job.id)
    assert job.status == "cancelling"


def test_back_version_notes_distinguish_draws_and_redraws(tmp_path):
    path = tmp_path / "sample__back.png"
    record = path.with_suffix(".provenance.json")
    record.write_text("{}", encoding="utf-8")
    assert image_api.version_note(path) == "Drawn"
    record.write_text('{"reference":{"instruction":"show the handle"}}', encoding="utf-8")
    assert image_api.version_note(path) == "Drawn again: show the handle"


def test_picture_command_omits_mask_for_unmasked_init(tmp_path, weight_cache):
    settings = image_api.clean_settings({}, image_api.FLUX)
    command = image_api.build_command("a prop", settings, tmp_path / "out.png", weight_cache,
                                      engine=image_api.FLUX, init=tmp_path / "start.png")
    assert "-i" in command and "--mask" not in command


def test_preparing_brush_write_failure_releases_the_slot(tmp_path, monkeypatch):
    manager = image_api.ImageJobManager(tmp_path)
    source = tmp_path / "front.png"
    Image.new("RGB", (64, 64)).save(source)
    original = source.read_bytes()
    mask = Image.new("L", (64, 64), 255)

    def save(*args, **kwargs):
        raise OSError("cannot write a scratch image")

    monkeypatch.setattr(Image.Image, "save", save)
    with pytest.raises(OSError, match="scratch image"):
        image_api.create_brush_job(manager, source, mask, "blue paint")
    assert manager.active is None
    assert source.read_bytes() == original
