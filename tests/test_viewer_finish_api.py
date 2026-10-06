"""Tests for the browser finishing job.

Every setting here ends up as a subprocess argument, so the tests that matter are the ones
covering what a browser can send: out-of-range numbers, unknown keys, and wrong types. The
underlying scripts reject bad values with a SystemExit deep inside a Blender run, which
would surface to the user as "worker exited with code 1" several minutes later.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "viewer" / "finish_api.py"


def _load():
    spec = importlib.util.spec_from_file_location("finish_api", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


finish = _load()


def test_defaults_are_used_when_nothing_is_sent():
    settings = finish.normalise_settings({})
    assert settings == finish.DEFAULT_SETTINGS
    assert settings is not finish.DEFAULT_SETTINGS  # a copy, not the shared dict


def test_client_values_override_defaults():
    settings = finish.normalise_settings({"metallic": 0.648, "roughness": 0.686, "ior": 1.4})
    assert settings["metallic"] == pytest.approx(0.648)
    assert settings["roughness"] == pytest.approx(0.686)
    assert settings["ior"] == pytest.approx(1.4)
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("faces", 500), ("faces", 500000),      # blender_retopo_bake rejects both
        ("metallic", 1.5), ("roughness", -0.1),
        ("ior", 0.5), ("ior", 4.0),
        ("voxel", 0.5),                          # coarse enough to melt the subject
        ("texture_size", 8192), ("paint_steps", 0),
    ],
)
def test_out_of_range_values_are_rejected_before_a_subprocess_sees_them(key, value):
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({key: value})
    assert key in str(caught.value)


def test_a_non_numeric_value_is_rejected_with_its_key():
    with pytest.raises(ValueError) as caught:
        finish.normalise_settings({"faces": "lots"})
    assert "faces" in str(caught.value)


def test_unknown_keys_are_ignored_rather_than_forwarded():
    """A typo must not reach the command line as a stray flag."""
    settings = finish.normalise_settings({"facees": 1234, "metallic": 0.4})
    assert "facees" not in settings
    assert settings["faces"] == finish.DEFAULT_SETTINGS["faces"]
    assert settings["metallic"] == pytest.approx(0.4)


def test_integer_settings_stay_integers():
    """`--faces 40000.0` is not accepted by argparse's int type."""
    settings = finish.normalise_settings({"faces": 39999.6, "paint_res": 512.0})
    assert isinstance(settings["faces"], int)
    assert isinstance(settings["paint_res"], int)


def test_zero_voxel_is_allowed_because_it_means_skip_the_remesh():
    assert finish.normalise_settings({"voxel": 0})["voxel"] == 0


def test_the_command_carries_every_setting():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    settings = finish.normalise_settings({"metallic": 0.648, "faces": 39935})
    command = finish.build_command(job, settings)

    assert command[2].endswith("retopo_repaint.py")
    assert str(job.asset_path) in command
    assert str(job.image_path) in command
    assert str(job.result_glb) in command
    assert command[command.index("--metallic") + 1] == "0.648"
    assert command[command.index("--faces") + 1] == "39935"


def test_skip_flags_appear_only_when_asked():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    plain = finish.build_command(job, finish.normalise_settings({"skip_paint": False}))
    assert "--skip-paint" not in plain
    assert "--skip-compress" not in plain

    skipped = finish.build_command(job, finish.normalise_settings({"skip_paint": True}))
    assert "--skip-paint" in skipped
    assert "--skip-bake" not in plain
    no_bake = finish.build_command(job, finish.normalise_settings({"skip_bake": True}))
    assert "--skip-bake" in no_bake


def test_the_bake_has_a_band_between_repaint_and_compress():
    bands = finish.stage_bands(["retopologise", "repaint", "bake", "compress"])
    assert bands["repaint"][1] == pytest.approx(bands["bake"][0])
    assert bands["bake"][1] == pytest.approx(bands["compress"][0])


def test_the_worker_and_the_viewer_agree_on_the_stage_order():
    """One stage plan, imported from the worker, and the resume table must follow it."""
    plan = finish.worker_module().stage_plan(False, False, photo=True)
    assert [stage for stage, _ in finish.STAGE_ARTIFACTS] == plan
    assert set(plan) == set(finish.STAGE_WEIGHTS)


def _progress(stages=("retopologise", "repaint", "bake", "compress")):
    """A tracker on a clock we drive, so the ETAs are assertable rather than wall-clock."""
    now = {"t": 0.0}
    tracker = finish.FinishProgress(list(stages), clock=lambda: now["t"])
    return tracker, now


def test_stage_lines_become_progress_events():
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::repaint::Repainting from fox.png at 512px")
    assert event["phase"] == "repaint"
    assert event["message"] == "Repainting from fox.png at 512px"
    # The band's floor: entering a stage is 0% of it, not some fraction of the way in.
    assert event["overall_pct"] == pytest.approx(5.0)


def test_the_bands_cover_exactly_the_stages_that_will_run():
    full = finish.stage_bands(["retopologise", "repaint", "compress"])
    assert full["retopologise"][0] == 0.0
    assert full["compress"][1] == pytest.approx(100.0)
    # The repaint is ~95% of the wall clock, so it must own most of the bar.
    low, high = full["repaint"]
    assert high - low > 80

    without_paint = finish.stage_bands(["retopologise", "compress"])
    assert "repaint" not in without_paint
    assert without_paint["compress"][1] == pytest.approx(100.0)


def test_progress_never_goes_backwards_through_a_real_run():
    tracker, now = _progress()
    lines = [
        "I2L_STAGE::retopologise::Retopologising to 40,000 faces",
        "I2L_STAGE::repaint::Repainting from source.png at 512px",
        "controls + dino ready (5s)",
        "  step 1/15 37s",
        "  step 15/15 202s",
        "views decoded (211s)",
        "super-res x4 (283s, views -> 2048px)",
        "I2L_STAGE::compress::Compressing textures to 2048px",
    ]
    percentages = []
    for index, line in enumerate(lines):
        now["t"] = float(index * 20)
        event = tracker.feed(line)
        assert event is not None, line
        percentages.append(event["overall_pct"])
    assert percentages == sorted(percentages)
    assert percentages[0] == 0.0


def test_a_late_paint_mark_cannot_drag_the_bar_back():
    tracker, now = _progress()
    tracker.feed("I2L_STAGE::repaint::Repainting")
    now["t"] = 100.0
    ahead = tracker.feed("super-res x4 (283s)")["overall_pct"]
    behind = tracker.feed("  step 1/15 37s")["overall_pct"]
    assert behind == ahead


def test_denoising_steps_are_the_only_sub_stage_progress_there_is():
    tracker, now = _progress()
    tracker.feed("I2L_STAGE::repaint::Repainting")
    now["t"] = 60.0
    event = tracker.feed("  step 6/15 95s")
    assert (event["step"], event["total"]) == (6, 15)
    assert 0 < event["stage_pct"] < 100
    assert event["stage_eta_seconds"] > 0
    assert event["total_eta_seconds"] > 0


def test_a_stage_with_no_sub_progress_claims_none():
    # A row frozen at "0%" reads as a stall; retopology and compression report no
    # percentage at all rather than a fake one.
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::retopologise::Retopologising")
    assert "stage_pct" not in event
    assert "stage_eta_seconds" not in event


def test_the_workers_own_done_marker_is_not_the_jobs_done():
    # The bug this replaces: the worker's terminal marker reached the browser as a
    # phase-"done" event with no result_url, the browser closed its stream on it, and the
    # real completion event -- the only one carrying the artifact URLs -- never arrived.
    tracker, _ = _progress()
    tracker.feed("I2L_STAGE::compress::Compressing")
    assert tracker.feed("I2L_STAGE::done::Finished at 4.8 MB") is None
    assert tracker.feed("I2L_STAGE::done::Finished at 4.8 MB :: all good") is None


def test_an_eta_is_withheld_until_there_is_evidence_for_one():
    assert finish.remaining_seconds(elapsed=0.0, fraction=0.5) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=0.0) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=1.0) is None
    assert finish.remaining_seconds(elapsed=50.0, fraction=0.5) == 50.0


def test_ordinary_output_is_not_mistaken_for_a_stage():
    tracker, _ = _progress()
    assert tracker.feed("[compress] 32.0 -> 4.8 MB") is None
    assert tracker.feed("Blender quit") is None
    # A paint-shaped line outside the paint stage is still not progress.
    assert tracker.feed("  step 3/15 62s") is None


def test_a_message_containing_the_separator_survives():
    tracker, _ = _progress()
    event = tracker.feed("I2L_STAGE::compress::Compressing to 2048px :: fast path")
    assert event["message"] == "Compressing to 2048px :: fast path"


def test_resume_is_asked_for_explicitly():
    job = finish.FinishJob("0" * 32, Path("/tmp/finish-job"))
    settings = finish.normalise_settings({})
    assert "--resume" not in finish.build_command(job, settings)
    assert "--resume" in finish.build_command(job, settings, resume=True)


def _run_dir(root: Path, name: str, files: dict[str, str]) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    for filename, contents in files.items():
        (directory / filename).write_text(contents)
    return directory


def test_a_finished_run_is_described_from_disk_alone(tmp_path):
    # The job registry lives in the server's memory and dies with it; the directory is
    # what makes a finished run recoverable afterwards.
    directory = _run_dir(tmp_path, "fox__finish__20260921-122518", {
        "source.glb": "g", "source.png": "p", "settings.json": "{}",
        "result_retopo.glb": "r", "result_painted.glb": "a", "result.glb": "done",
        "result.retopo-repaint.json": '{"seconds": {"repaint": 324.5}}',
    })
    run = finish.describe_run(directory)
    assert run["finished"] is True
    assert run["stages_complete"] == ["retopologise", "repaint", "compress"]
    assert run["seconds"] == {"repaint": 324.5}
    assert run["resumable"] is False  # nothing left to resume


def test_a_run_that_died_after_the_repaint_is_resumable(tmp_path):
    directory = _run_dir(tmp_path, "fox__finish__20260921-122519", {
        "source.glb": "g", "source.png": "p", "settings.json": "{}",
        "result_retopo.glb": "r", "result_painted.glb": "a",
    })
    run = finish.describe_run(directory)
    assert run["stages_complete"] == ["retopologise", "repaint"]
    assert run["resumable"] is True
    assert run["result_url"] is None


def test_a_run_url_is_one_the_static_handler_can_actually_serve():
    # describe_run's links go straight into an <a href>, so they have to be repo-relative
    # paths the viewer's own file handler resolves -- not absolute filesystem paths.
    inside = finish.OUTPUT_ROOT / "fox__finish__20260921-122518" / "result.glb"
    assert finish.served_url(inside) == (
        "/output/finish/fox__finish__20260921-122518/result.glb"
    )
    assert finish.served_url(Path("/etc/passwd")) is None


def test_a_run_without_its_settings_is_not_offered_for_resume(tmp_path):
    # Resuming means reusing intermediates, which is only faithful with the settings that
    # produced them -- a directory from before settings.json cannot promise that.
    directory = _run_dir(tmp_path, "fox__finish__20260921-122520", {
        "source.glb": "g", "source.png": "p", "result_retopo.glb": "r",
    })
    assert finish.describe_run(directory)["resumable"] is False


def test_runs_are_listed_newest_first_and_strangers_are_ignored(tmp_path):
    import os
    for index, name in enumerate(["a__finish__20260921-120000", "b__finish__20260921-130000"]):
        directory = _run_dir(tmp_path, name, {"source.glb": "g"})
        os.utime(directory, (1000 + index, 1000 + index))
    (tmp_path / "not-a-run").mkdir()
    (tmp_path / "scratch__notfinish__20260921-120000").mkdir()
    listed = [run["directory"] for run in finish.list_runs(tmp_path)]
    assert listed == ["b__finish__20260921-130000", "a__finish__20260921-120000"]


@pytest.mark.parametrize("name", [
    "../../etc", "fox__finish__20260921-122518/../..", "nope", "", "fox__finish__2026",
])
def test_a_resume_name_that_is_not_a_run_directory_is_refused(tmp_path, name):
    with pytest.raises(RuntimeError):
        finish._run_directory(tmp_path, name)


def test_adopting_a_directory_reuses_its_recorded_settings(tmp_path):
    _run_dir(tmp_path, "fox__finish__20260921-122518", {
        "source.glb": "g", "source.png": "p",
        "settings.json": '{"faces": 12000, "paint_steps": 8}',
        "result_retopo.glb": "r",
    })
    manager = finish.FinishJobManager(tmp_path)
    job = manager.adopt("fox__finish__20260921-122518")
    assert job.resume is True
    assert job.settings["faces"] == 12000
    assert job.settings["paint_steps"] == 8
    assert "--resume" in finish.build_command(job, job.settings, resume=job.resume)


def test_adopting_a_directory_that_lost_its_source_is_refused(tmp_path):
    _run_dir(tmp_path, "fox__finish__20260921-122518", {"settings.json": "{}"})
    with pytest.raises(RuntimeError, match="source.glb"):
        finish.FinishJobManager(tmp_path).adopt("fox__finish__20260921-122518")


def test_job_ids_are_validated_before_lookup():
    manager = finish.FinishJobManager(Path("/tmp/finish-root"))
    assert manager.get("../../etc/passwd") is None
    assert manager.get("not-a-job-id") is None
    assert manager.get("0" * 32) is None  # well-formed but unknown


# --- The photo stage: match the upload to its Pixal3D run, copy its camera in -------------


def _pixal3d_run(root, glb_bytes=b"glb-bytes"):
    import hashlib
    run = root / "seb" / "robot__pixal3d__1"
    views = run / "robot__pixal3d__1.svviews"
    views.mkdir(parents=True)
    (views / "transforms.json").write_text('{"frames": []}')
    glb = run / "robot__pixal3d__1.glb"
    glb.write_bytes(glb_bytes)
    (run / "robot__pixal3d__1.json").write_text(json.dumps({
        "backend": "pixal3d",
        "output": {"path": str(glb), "sha256": hashlib.sha256(glb_bytes).hexdigest()},
    }))
    return views


def test_an_uploaded_pixal3d_model_finds_its_camera(tmp_path):
    views = _pixal3d_run(tmp_path)
    assert finish.find_source_views(b"glb-bytes", tmp_path) == views


def test_a_model_with_no_matching_run_gets_no_camera(tmp_path):
    _pixal3d_run(tmp_path)
    assert finish.find_source_views(b"edited-or-foreign", tmp_path) is None


def test_the_camera_is_copied_into_the_run_and_passed_to_the_worker(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert (job.views_dir / "transforms.json").is_file()
    command = finish.build_command(job, job.settings)
    assert command[command.index("--views") + 1] == str(job.views_dir)


def test_skipping_the_photo_leaves_the_camera_out(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {"skip_photo": True})
    assert not job.views_dir.exists()
    assert "--views" not in finish.build_command(job, job.settings)


def test_a_finished_run_offers_its_source_for_comparison(tmp_path):
    run = tmp_path / "robot__finish__20260928-120000"
    run.mkdir()
    for name in ("source.glb", "source.png", "result.glb"):
        (run / name).write_bytes(b"x")
    described = finish.describe_run(run)
    assert described["source_url"] is None or described["source_url"].endswith("source.glb")


# --- Run folder layout (0.3.5): the finished model is the only GLB at the top ----------

@pytest.mark.parametrize("faces, label", [
    (5000, "5k"), (40000, "40k"), (12500, "12.5k"), (800, "800"), (1000, "1k"),
])
def test_face_counts_are_named_the_way_people_say_them(faces, label):
    assert finish.faces_label(faces) == label


def test_the_final_model_is_named_after_the_asset_and_its_faces():
    assert finish.final_name("vanguard__pixal3d__finish__20260929-101010", 5000) == "vanguard_5k.glb"
    assert finish.final_name("fox__finish__20260921-122518", 40000) == "fox_40k.glb"


def test_a_new_run_keeps_inputs_and_steps_out_of_the_top_folder(tmp_path):
    job = finish.FinishJob("0" * 32, tmp_path / "vanguard__pixal3d__finish__20260929-101010")
    job.settings = finish.normalise_settings({"faces": 5000})

    assert not job.legacy
    assert job.result_glb.name == "vanguard_5k.glb"
    assert job.result_glb.parent == job.directory
    assert job.record_path.name == "vanguard_5k.retopo-repaint.json"
    assert job.asset_path == job.directory / "input" / "source.glb"
    assert job.image_path == job.directory / "input" / "source.png"
    assert job.views_dir == job.directory / "input" / "source.views"
    assert job.settings_path == job.directory / "input" / "settings.json"
    assert job.log_path == job.directory / "steps" / "run.log"
    assert job.stage_artifact("retopo") == job.directory / "steps" / "1_retopo.glb"
    assert job.stage_artifact("baked") == job.directory / "steps" / "4_baked.glb"
    assert job.stage_artifact(None) == job.result_glb


def test_a_new_run_tells_the_worker_where_the_steps_go(tmp_path):
    job = finish.FinishJob("0" * 32, tmp_path / "fox__finish__20260929-101010")
    command = finish.build_command(job, finish.normalise_settings({}))
    assert command[command.index("--steps-dir") + 1] == str(job.directory / "steps")


def test_an_old_flat_run_keeps_its_old_names_and_no_steps_folder(tmp_path):
    directory = _run_dir(tmp_path, "fox__finish__20260921-122518", {"source.glb": "g"})
    job = finish.FinishJob("0" * 32, directory)
    assert job.legacy
    assert job.result_glb == directory / "result.glb"
    assert job.record_path == directory / "result.retopo-repaint.json"
    assert job.stage_artifact("painted") == directory / "result_painted.glb"
    assert "--steps-dir" not in finish.build_command(job, finish.normalise_settings({}))


def test_a_new_run_is_described_and_resumable_from_disk(tmp_path):
    name = "vanguard__pixal3d__finish__20260929-101010"
    directory = tmp_path / name
    for relative, contents in {
        "input/source.glb": "g", "input/source.png": "p",
        "input/settings.json": json.dumps({"faces": 5000}),
        "steps/1_retopo.glb": "r", "steps/2_painted.glb": "a",
    }.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_text(contents)

    described = finish.describe_run(directory)
    assert described["stages_complete"] == ["retopologise", "repaint"]
    assert described["resumable"] is True
    assert described["finished"] is False

    (directory / "vanguard_5k.glb").write_text("done")
    described = finish.describe_run(directory)
    assert described["finished"] is True
    assert described["resumable"] is False
    assert described["result_url"] is None or described["result_url"].endswith("/vanguard_5k.glb")


def test_create_writes_inputs_into_the_input_folder(tmp_path):
    manager = finish.FinishJobManager(output_root=tmp_path, generated_root=tmp_path / "none")
    job = manager.create("vanguard__pixal3d.glb", b"glb", b"png", {"faces": 5000})
    assert (job.directory / "input" / "source.glb").read_bytes() == b"glb"
    assert (job.directory / "input" / "source.png").read_bytes() == b"png"
    assert (job.directory / "input" / "settings.json").is_file()
    assert job.result_glb.name == "vanguard_5k.glb"
    assert [p.name for p in job.directory.iterdir()] == ["input"]
    # And it resumes: the adopted job finds the same files.
    manager.finish(job)
    adopted = manager.adopt(job.directory.name)
    assert adopted.asset_path == job.asset_path
    assert adopted.result_glb == job.result_glb


# --- Finish on machines without MLX or without Blender --------------------------------

def test_the_repaint_is_forced_off_away_from_apple_silicon():
    settings = finish.normalise_settings({"skip_paint": False})
    assert finish.fit_to_machine(settings, platform="nvidia")["skip_paint"] is True
    assert finish.fit_to_machine(settings, platform="apple-silicon")["skip_paint"] is False


def test_capabilities_say_what_this_machine_can_do(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    mac = finish.capabilities("apple-silicon", find=lambda: exe, version=lambda _: (5, 2))
    assert mac["repaint"] and mac["ready"] and mac["repaint_note"] is None
    assert mac["blender_version"] == "5.2" and mac["blender_problem"] is None

    nvidia = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (5, 2))
    assert not nvidia["repaint"] and nvidia["ready"]
    assert "front exact" in nvidia["repaint_note"]


def test_capabilities_explain_a_missing_or_old_blender(tmp_path):
    missing = finish.capabilities("nvidia", find=lambda: None, version=lambda _: None)
    assert not missing["ready"] and "blender.org" in missing["blender_problem"]
    exe = tmp_path / "blender"
    exe.write_text("")
    old = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (3, 6))
    assert old["ready"] and "older than" in old["blender_problem"]


def test_an_nvidia_run_never_asks_the_worker_for_a_repaint(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "host_platform", lambda: "nvidia")
    manager = finish.FinishJobManager(output_root=tmp_path, generated_root=tmp_path / "none")
    job = manager.create("fox.glb", b"glb", b"png", {"skip_paint": False})
    assert job.settings["skip_paint"] is True
    assert "--skip-paint" in finish.build_command(job, job.settings)


def test_repaint_is_opt_in_so_finish_never_needs_the_hunyuan_paint_weights():
    """Quick finish by default (0.3.5): Pixal3D models arrive painted, Pixel Match keeps the
    front exact, and nobody should download 8.7 GB of Hunyuan paint to finish one."""
    assert finish.DEFAULT_SETTINGS["skip_paint"] is True
    assert finish.normalise_settings({})["skip_paint"] is True
    assert finish.normalise_settings({"skip_paint": False})["skip_paint"] is False


def test_the_finish_page_calls_it_pixel_match():
    from pathlib import Path
    html = Path("viewer/index.html").read_text()
    assert "Pixel Match" in html
    assert 'id="finish-repaint" type="checkbox" checked' not in html


def test_a_skipped_pixel_match_is_said_plainly_and_stays_on_screen(tmp_path):
    # Seen on a real NVIDIA pod: Pixel Match stayed ticked, no stage ran, and the only word
    # about it was a start-up line replaced a second later. The note rides on the finished
    # result, and says what Pixel Match needs.
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"some-other-model", b"png", {})
    note = finish.photo_note(job, job.settings)
    assert note and "Pixel Match" in note and "skipped" in note and "Pixal3D" in note
    assert finish.photo_note(job, {**job.settings, "skip_photo": True}) is None


def test_no_note_when_the_camera_is_found(tmp_path):
    _pixal3d_run(tmp_path / "generated")
    manager = finish.FinishJobManager(tmp_path / "finish", tmp_path / "generated")
    job = manager.create("robot.glb", b"glb-bytes", b"png", {})
    assert finish.photo_note(job, job.settings) is None


def test_capabilities_offer_a_blender_install_only_where_one_can_run():
    # Linux x86_64 with no Blender: the Setup card gets an Install Blender button.
    linux = finish.capabilities("nvidia", find=lambda: None, version=lambda _: None,
                                installable=lambda: True)
    assert linux["blender_installable"] is True
    mac = finish.capabilities("apple-silicon", find=lambda: None, version=lambda _: None,
                              installable=lambda: False)
    assert mac["blender_installable"] is False


def test_no_install_offered_once_blender_is_found(tmp_path):
    exe = tmp_path / "blender"
    exe.write_text("")
    found = finish.capabilities("nvidia", find=lambda: exe, version=lambda _: (4, 2),
                                installable=lambda: True)
    assert found["blender_installable"] is False


def test_a_finished_model_is_named_by_its_path_in_output(tmp_path, monkeypatch):
    # Create hands this to the next step; the done event's result_url is a download link, not a path
    monkeypatch.setattr(finish, "OUTPUT_ROOT", tmp_path / "output" / "finish")
    glb = tmp_path / "output" / "finish" / "knight__finish__20261006-220000" / "knight_40k.glb"
    assert finish.library_path(glb) == "finish/knight__finish__20261006-220000/knight_40k.glb"
    assert finish.library_path(tmp_path / "elsewhere.glb") is None
