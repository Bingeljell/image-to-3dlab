"""Tests for the text-to-image step in the viewer.

A generation run costs minutes, so everything cheap is checked here: that the command line
is the measured-fast one rather than the upstream default, that a prompt cannot smuggle
arguments, that progress lines parse, and that the non-commercial licence reaches the
sidecar. The pure functions are imported, never re-implemented.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

VIEWER = Path(__file__).resolve().parents[1] / "viewer"


def _load():
    sys.path.insert(0, str(VIEWER))
    spec = importlib.util.spec_from_file_location("image_api", VIEWER / "image_api.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


api = _load()
WEIGHTS = {
    "diffusion_model": Path("/w/dit.gguf"),
    "llm": Path("/w/llm.gguf"),
    "vae": Path("/w/vae.safetensors"),
}


def command_for(prompt="a fox", **overrides):
    settings = api.clean_settings(overrides)
    return api.build_command(prompt, settings, Path("/out/x.png"), WEIGHTS,
                             binary=Path("/bin/sd-cli"))


def test_defaults_are_the_measured_fast_ones():
    """cfg 6.0 (the upstream recipe) makes the model do two passes per step for nothing:
    17 minutes against 11 for an identical picture. 768/10 steps was 4m22s."""
    assert api.DEFAULTS["cfg_scale"] == 1.0
    assert api.DEFAULTS["steps"] == 10
    assert api.DEFAULTS["width"] == api.DEFAULTS["height"] == 768


def test_offload_to_cpu_is_not_used():
    """Measured slower AND higher peak RAM on unified memory, despite the upstream recipe
    recommending it. Do not reinstate without re-measuring."""
    assert "--offload-to-cpu" not in command_for()


def test_command_carries_all_three_weights():
    command = command_for()
    for flag, key in (("--diffusion-model", "diffusion_model"), ("--llm", "llm"),
                      ("--vae", "vae")):
        assert command[command.index(flag) + 1] == str(WEIGHTS[key])


def test_prompt_is_one_argument_not_shell_text():
    """The command is a list handed to Popen without a shell, so a prompt containing
    quotes, semicolons or flags stays a prompt."""
    nasty = 'a fox"; rm -rf / #  --steps 500 --seed 9'
    command = command_for(nasty)
    assert command[command.index("-p") + 1] == nasty
    assert command.count("--steps") == 1
    assert command[command.index("--steps") + 1] == "10"


def test_negative_prompt_only_appears_when_asked():
    assert "--negative-prompt" not in command_for()
    command = command_for(negative_prompt="blurry")
    assert command[command.index("--negative-prompt") + 1] == "blurry"
    # and the positive prompt survives the insertion
    assert command[command.index("-p") + 1] == "a fox"


@pytest.mark.parametrize("given,expected", [(700, 704), (769, 768), (10, 256), (9000, 1536)])
def test_sizes_are_snapped_to_a_multiple_of_32(given, expected):
    """sd.cpp fails deep in a run on a size it cannot use. A user typing 700 means 'about
    this big', so round rather than refuse."""
    assert api.clean_settings({"width": given})["width"] == expected
    assert expected % api.SIZE_STEP == 0


@pytest.mark.parametrize("field,given", [("steps", "lots"), ("cfg_scale", None),
                                         ("seed", "abc"), ("width", {})])
def test_rubbish_settings_fall_back_to_defaults(field, given):
    assert api.clean_settings({field: given})[field] == api.DEFAULTS[field]


def test_steps_are_capped():
    assert api.clean_settings({"steps": 9999})["steps"] == api.MAX_STEPS
    assert api.clean_settings({"steps": 0})["steps"] == 1


def test_unknown_sampler_falls_back():
    assert api.clean_settings({"sampler": "wishful"})["sampler"] == api.DEFAULTS["sampler"]
    assert api.clean_settings({"sampler": "heun"})["sampler"] == "heun"


def test_progress_line_parses_with_an_eta():
    event = api.parse_progress("  |=========>    | 7/10 - 13.38s/it")
    assert event["step"] == 7 and event["total_steps"] == 10
    assert event["percent"] == 70.0
    assert event["eta_seconds"] == pytest.approx(3 * 13.38, rel=1e-3)


def test_decode_and_completion_lines_parse():
    assert api.parse_progress("decode_first_stage completed, taking 133.59s")["phase"] == "decoding"
    finished = api.parse_progress("generate_image completed in 262.30s")
    assert finished["phase"] == "finished"
    assert finished["generate_seconds"] == 262.30


def test_ordinary_chatter_is_not_an_event():
    for line in ("[INFO ] loading model", "", "ggml_metal_init: found device"):
        assert api.parse_progress(line) is None


def test_slug_is_filesystem_safe():
    name = api.slug("Glitchkin Hummingbird: a sleek/fantasy bird!! 100%")
    assert "/" not in name and ":" not in name and "%" not in name
    assert name and not name.startswith("-") and not name.endswith("-")


def test_slug_survives_a_prompt_with_nothing_usable():
    assert api.slug("!!!???") == "image"


def test_provenance_records_the_non_commercial_restriction():
    """A PNG in a folder six months from now remembers nothing on its own."""
    record = api.provenance("a fox", api.clean_settings({}), 262.3, Path("/o/fox.png"))
    assert record["license"]["classification"] == "research-only"
    assert "Built with Qwen" == record["license"]["attribution"]
    assert record["prompt"] == "a fox"
    json.dumps(record)  # must survive being written to the sidecar


def test_provenance_says_the_output_is_yours_and_cites_qwen():
    """Qwen stated on 2026-09-21 that outputs are not licensed Materials. The sidecar says
    the picture is yours and cites the statement, so the record stands on its own."""
    note = api.provenance("a fox", api.clean_settings({}), 1.0, Path("/o/f.png"))[
        "license"]["inherited_by_derivatives"]
    assert note == api.OUTPUT_RIGHTS
    assert "yours" in note and "non-commercial" not in note.lower()
    assert "https://x.com/QwenDevs/status/2101917379785838660" in note


def test_output_goes_to_the_images_folder():
    """One folder for every picture; the licence travels in the provenance record."""
    assert api.OUTPUT_ROOT == api.REPO / "output" / "images"


def test_only_one_job_runs_at_a_time(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    first = manager.create("one", api.clean_settings({}))
    first.status = "running"
    with pytest.raises(RuntimeError, match="already being generated"):
        manager.create("two", api.clean_settings({}))
    manager.finish(first)
    assert manager.create("two", api.clean_settings({})) is not None


def test_a_finished_job_frees_the_slot(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("one", api.clean_settings({}))
    job.status = "done"
    manager.finish(job)
    assert manager.create("two", api.clean_settings({})) is not None


def test_missing_weights_names_what_is_missing():
    exc = api.MissingWeights(["qwen_image_2.1-Q8_0.gguf"])
    assert "qwen_image_2.1-Q8_0.gguf" in str(exc)
    assert "Setup & Status" in str(exc)


def test_job_describe_hides_the_result_until_it_exists(tmp_path):
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("one", api.clean_settings({}))
    assert job.describe()["result_url"] is None
    job.status = "done"
    assert job.describe()["result_url"].endswith("/result.png")


def test_job_describe_reports_progress_log_and_where_the_picture_lands(tmp_path):
    # the studio follows every step the same way: status, last event, log tail, and the file it made
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a knight", api.clean_settings({}))
    job.emit({"phase": "sampling", "percent": 40.0})
    job.log_lines.append("step 4/10")
    described = job.describe()
    assert described["last_event"]["percent"] == 40.0
    assert described["log_tail"] == "step 4/10"
    assert described["picture"] is None            # nothing on disk yet
    job.status = "done"
    picture = api.ImageJob.picture_path(job)
    assert picture == f"images/{job.directory.name}/{job.output_path.name}"
    assert job.describe()["picture"] == picture


def test_sidecar_names_each_weight_file_readably():
    """The first sidecar written recorded a Python tuple's repr as the value, which is
    not something a person opening the file six months later can use."""
    weights = api.provenance("a fox", api.clean_settings({}), 1.0,
                             Path("/o/x.png"))["model"]["weights"]
    for key in ("diffusion_model", "llm", "vae"):
        assert set(weights[key]) == {"cache_dir", "file"}
        assert weights[key]["file"].endswith((".gguf", ".safetensors"))
        assert "(" not in weights[key]["cache_dir"]


def _fake_sd_cli(tmp_path: Path, lines: list[str], then_sleep: float) -> list[str]:
    script = tmp_path / "fake_sd_cli.py"
    script.write_text(
        "import sys, time\n"
        f"for line in {lines!r}:\n"
        "    print(line, flush=True)\n"
        f"time.sleep({then_sleep})\n"
    )
    return [sys.executable, str(script)]


def test_a_cpu_only_run_on_an_nvidia_machine_is_stopped_with_the_fix(tmp_path, monkeypatch):
    """On the RunPod 4090 a CPU-only run sat at 1,700% CPU for minutes with no word why.
    The viewer must stop it the moment the CPU backend loads with no GPU before it."""
    monkeypatch.setattr(api, "host_platform", lambda: api.NVIDIA)
    command = _fake_sd_cli(tmp_path, [
        "load_backend: loaded CPU backend from /x/libggml-cpu-haswell.so",
    ], then_sleep=60)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    started = time.monotonic()
    api.run_job(job, manager, weights=WEIGHTS)
    assert time.monotonic() - started < 10, "the CPU run was left to finish"
    assert job.status == "error"
    assert "libegl1 libgl1" in job.error


def test_a_gpu_run_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "host_platform", lambda: api.NVIDIA)
    command = _fake_sd_cli(tmp_path, [
        "ggml_vulkan: Found 1 Vulkan devices:",
        "load_backend: loaded CPU backend from /x/libggml-cpu-haswell.so",
    ], then_sleep=0)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    # The fake writes no PNG, so the run ends in the ordinary "no output" error, not ours.
    assert "libegl1" not in (job.error or "")
