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
sys.path.insert(0, str(VIEWER.parent))
from image_to_3dlab.host import AMD as HOST_AMD
from image_to_3dlab.host import APPLE as HOST_APPLE
from image_to_3dlab.host import NVIDIA as HOST_NVIDIA


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
    assert api.DEFAULTS["steps"] == 8  # tested by eye against 10: nothing visible lost
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
    assert command[command.index("--steps") + 1] == str(api.DEFAULTS["steps"])


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


def test_provenance_records_the_licence_and_its_class():
    """A PNG in a folder six months from now remembers nothing on its own."""
    record = api.provenance("a fox", api.clean_settings({}), 262.3, Path("/o/fox.png"))
    assert record["license"]["classification"] == "commercial-conditional"
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


def _fake_sd_cli(tmp_path: Path, lines: list[str], then_sleep: float,
                 exit_code: int = 0) -> list[str]:
    script = tmp_path / "fake_sd_cli.py"
    script.write_text(
        "import sys, time\n"
        f"for line in {lines!r}:\n"
        "    print(line, flush=True)\n"
        f"time.sleep({then_sleep})\n"
        f"sys.exit({exit_code})\n"
    )
    return [sys.executable, str(script)]


def test_a_failed_run_says_why_not_just_the_exit_code(tmp_path, monkeypatch):
    """'sd-cli exited with code 2' is a riddle; the binary's own last line -- here a
    rejected flag, because vendor/sdcpp once held a LocalAI backend that only speaks
    -addr -- is the reason, and the image tab has no log panel to find it in.

    The GPU line is not decoration: a run that reaches its third step was on the GPU, and
    the no-GPU explanation is for the ones that were not."""
    command = _fake_sd_cli(tmp_path, [
        "load_backend: loaded ROCm backend from /x/libggml-hip.so",
        "load_backend: loaded CPU backend from /x/libggml-cpu.so",
        "|=====>    | 3/10 - 1.5s/it",
        "flag provided but not defined: -diffusion-model",
    ], then_sleep=0, exit_code=2)
    monkeypatch.setattr(api, "host_platform", lambda: HOST_AMD)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "error"
    assert "code 2" in job.error
    assert "flag provided but not defined: -diffusion-model" in job.error


def test_a_failed_run_with_only_progress_says_just_the_code(tmp_path, monkeypatch):
    """When the tail is nothing but progress chatter, there is no reason to quote."""
    command = _fake_sd_cli(tmp_path, ["|=====>    | 3/10 - 1.5s/it"],
                           then_sleep=0, exit_code=2)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "error"
    assert job.error == "sd-cli exited with code 2"


def test_a_cpu_only_run_on_an_nvidia_machine_is_stopped_with_the_fix(tmp_path, monkeypatch):
    """On the RunPod 4090 a CPU-only run sat at 1,700% CPU for minutes with no word why.
    The viewer must stop it the moment the CPU backend loads with no GPU before it."""
    monkeypatch.setattr(api, "host_platform", lambda: HOST_NVIDIA)
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
    monkeypatch.setattr(api, "host_platform", lambda: HOST_NVIDIA)
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


# --- AMD: the same silent fallback, and words the AMD user can act on ----------------------

# Lines as the ROCm build printed them on an RX 7900 XTX, 2026-10-07.
ROCM_OK = [
    "ggml_cuda_init: found 1 ROCm devices (Total VRAM: 20480 MiB):",
    "load_backend: loaded ROCm backend from /workspace/lab/vendor/sdcpp/libggml-hip.so",
    "load_backend: loaded CPU backend from /workspace/lab/vendor/sdcpp/libggml-cpu-haswell.so",
]
ROCM_CPU_ONLY = [
    "load_backend: loaded CPU backend from /workspace/lab/vendor/sdcpp/libggml-cpu-haswell.so",
]


def test_a_cpu_only_run_on_an_amd_card_is_stopped_too(tmp_path, monkeypatch):
    """The ROCm build loads its GPU backend at run time exactly as CUDA does, so it falls
    back to the CPU just as silently. Watcher or not, an AMD user watched the same idle card."""
    monkeypatch.setattr(api, "host_platform", lambda: HOST_AMD)
    command = _fake_sd_cli(tmp_path, ROCM_CPU_ONLY, then_sleep=60)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    started = time.monotonic()
    api.run_job(job, manager, weights=WEIGHTS)
    assert time.monotonic() - started < 10, "the CPU run was left to finish"
    assert job.status == "error"


def test_an_amd_fallback_is_told_about_rocm_and_not_about_nvidia(tmp_path, monkeypatch):
    """`nvidia-smi` and NVIDIA_DRIVER_CAPABILITIES are both useless to someone with an AMD
    card. Telling them to check them reads as the viewer having got the machine wrong."""
    monkeypatch.setattr(api, "host_platform", lambda: HOST_AMD)
    command = _fake_sd_cli(tmp_path, ROCM_CPU_ONLY, then_sleep=60)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert "rocminfo" in job.error
    assert "nvidia-smi" not in job.error


def test_an_amd_gpu_run_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "host_platform", lambda: HOST_AMD)
    command = _fake_sd_cli(tmp_path, ROCM_OK, then_sleep=0)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert "rocminfo" not in (job.error or "")


def test_the_mac_is_still_left_watched_over_nothing(tmp_path, monkeypatch):
    """The Mac build is Metal, compiled in, and never announces a backend. Watching it would
    stop every run on the CPU line, which sd-cli always prints."""
    monkeypatch.setattr(api, "host_platform", lambda: HOST_APPLE)
    command = _fake_sd_cli(tmp_path, ROCM_OK, then_sleep=0)
    monkeypatch.setattr(api, "build_command", lambda *a, **k: command)
    manager = api.ImageJobManager(output_root=tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status != "error" or "rocminfo" not in (job.error or "")


# --- edit mode: a reference picture, read by the vision file --------------------------------


def test_an_edit_carries_the_reference_and_the_vision_file():
    settings = api.clean_settings({})
    command = api.build_command("the back", settings, Path("/out/x.png"), WEIGHTS,
                                binary=Path("/bin/sd-cli"), reference=Path("/in/front.png"),
                                vision=Path("/w/mmproj.gguf"))
    assert command[command.index("-r") + 1] == "/in/front.png"
    assert command[command.index("--llm_vision") + 1] == "/w/mmproj.gguf"


def test_a_plain_picture_has_no_reference():
    assert "-r" not in command_for() and "--llm_vision" not in command_for()


def test_an_edit_without_the_vision_file_is_refused():
    with pytest.raises(ValueError, match="vision"):
        api.build_command("the back", api.clean_settings({}), Path("/out/x.png"), WEIGHTS,
                          binary=Path("/bin/sd-cli"), reference=Path("/in/front.png"))


def test_the_vision_file_is_the_one_beside_the_qwen_llm():
    directory, filename = api.VISION_FILE
    assert directory == api.WEIGHT_FILES["llm"][0]
    assert filename.startswith("mmproj") and filename.endswith(".gguf")


# --- back-picture jobs ------------------------------------------------------------------------


def _front(tmp_path):
    from PIL import Image
    folder = tmp_path / "an-orc__20261012-120000"
    folder.mkdir(parents=True)
    front = folder / "an-orc__20261012-120000.png"
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    rgba.paste((180, 60, 40, 255), (16, 8, 48, 60))
    rgba.save(front)
    return front


def test_a_back_job_lands_beside_its_front(tmp_path):
    manager = api.ImageJobManager(tmp_path)
    front = _front(tmp_path)
    job = api.create_back_job(manager, front, seed=3)
    assert job.output_path == front.with_name("an-orc__20261012-120000__back.png")
    assert job.prompt == api.back_view.BACK_PROMPT
    assert job.settings["width"] == 768 and job.settings["steps"] == 8 and job.settings["seed"] == 3
    assert job.reference.is_file() and job.reference.parent == front.parent


def test_a_back_jobs_command_is_an_edit_from_its_reference(tmp_path):
    manager = api.ImageJobManager(tmp_path)
    job = api.create_back_job(manager, _front(tmp_path), seed=3)
    command = api.job_command(job, WEIGHTS, vision=Path("/w/mmproj.gguf"), binary=Path("/bin/sd-cli"))
    assert command[command.index("-r") + 1] == str(job.reference)


def test_a_back_job_without_the_vision_file_asks_for_setup(tmp_path, monkeypatch):
    manager = api.ImageJobManager(tmp_path)
    job = api.create_back_job(manager, _front(tmp_path), seed=3)
    monkeypatch.setattr(api, "resolve_vision", lambda *a: None)
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "error" and job.events[-1].get("needs_setup")
    assert "mmproj" in job.error


def test_a_plain_jobs_command_has_no_reference(tmp_path):
    manager = api.ImageJobManager(tmp_path)
    job = manager.create("a fox", api.clean_settings({}))
    assert "-r" not in api.job_command(job, WEIGHTS, vision=None, binary=Path("/bin/sd-cli"))


def test_a_back_job_only_reads_pictures_inside_the_output_folder(tmp_path):
    images = tmp_path / "images"
    front = _front(images)
    assert api.back_source(tmp_path, f"images/{front.parent.name}/{front.name}") == front
    for bad in ("../secrets.png", "/etc/passwd", "images/nope.png", f"images/{front.parent.name}"):
        with pytest.raises(ValueError):
            api.back_source(tmp_path, bad)


# --- FLUX.2 klein, the default picture engine (2026-10-30) ---------------------------------

def _ready_only(*ids):
    return lambda engine: engine.id in ids


def test_flux_is_preferred_when_installed_and_named_when_nothing_is():
    assert api.pick_engine(ready=_ready_only("flux", "qwen")) is api.FLUX
    assert api.pick_engine(ready=_ready_only("qwen")) is api.QWEN
    # nothing set up: point at the smaller, Apache-licensed download
    assert api.pick_engine(ready=_ready_only()) is api.FLUX
    assert api.pick_engine("qwen", ready=_ready_only("flux")) is api.QWEN   # asked for is asked for
    with pytest.raises(ValueError, match="no picture engine 'dalle'"):
        api.pick_engine("dalle")


def test_flux_draws_at_1024_in_4_steps_and_8_when_the_picture_is_the_end_product():
    """Bake-off 2026-10-28: 4 steps ~62 s, 8 steps ~103 s for a near-identical picture."""
    for_3d = api.clean_settings({}, api.FLUX)
    alone = api.clean_settings({}, api.FLUX, picture_only=True)
    assert (for_3d["width"], for_3d["height"], for_3d["steps"], for_3d["cfg_scale"]) == (1024, 1024, 4, 1.0)
    assert alone["steps"] == 8
    assert api.clean_settings({"steps": 6}, api.FLUX, picture_only=True)["steps"] == 6   # a choice wins
    assert api.clean_settings({}, api.QWEN, picture_only=True)["steps"] == api.DEFAULTS["steps"]


def test_a_flux_edit_reads_the_reference_without_a_vision_projector():
    settings = api.clean_settings({}, api.FLUX)
    command = api.build_command("its back", settings, Path("/out/b.png"), WEIGHTS,
                                binary=Path("/bin/sd-cli"), reference=Path("/in/front.png"), engine=api.FLUX)
    assert command[command.index("-r") + 1] == "/in/front.png"
    assert "--llm_vision" not in command
    assert api.can_edit(api.FLUX)


def test_a_qwen_edit_still_needs_its_vision_projector():
    with pytest.raises(ValueError, match="vision projector"):
        api.build_command("its back", api.clean_settings({}), Path("/out/b.png"), WEIGHTS,
                          binary=Path("/bin/sd-cli"), reference=Path("/in/front.png"))


def test_a_flux_picture_records_its_apache_licence(tmp_path):
    record = api.provenance("a fox", api.clean_settings({}, api.FLUX), 60.0, tmp_path / "x.png", api.FLUX)
    assert record["model"]["id"] == "flux2-klein-4b"
    assert record["license"]["name"] == "Apache-2.0" and record["license"]["classification"] == "commercial-clear"
    assert record["license"]["attribution"] is None
    assert "commercial" in record["license"]["inherited_by_derivatives"]
    assert record["model"]["weights"]["llm"]["file"] == "Qwen3-4B-Q8_0.gguf"


def test_a_missing_flux_install_says_flux_not_qwen(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "HF_HUB", tmp_path)
    with pytest.raises(api.MissingWeights, match="FLUX.2"):
        api.resolve_weights(api.FLUX)


def test_a_flux_back_job_runs_flux_and_records_it(tmp_path):
    manager = api.ImageJobManager(tmp_path)
    job = api.create_back_job(manager, _front(tmp_path), seed=3, engine=api.FLUX)
    assert job.engine is api.FLUX and job.settings["width"] == 1024
    assert "--llm_vision" not in api.job_command(job, WEIGHTS, None, binary=Path("/bin/sd-cli"))
    assert job.describe()["engine"] == "flux"


def test_the_flux_files_are_the_ones_the_installer_fetches():
    sys.path.insert(0, str(VIEWER.parent / "scripts"))
    import bootstrap_flux_image as boot
    fetched = {Path(name).name for _, name, _ in boot.WEIGHTS}
    assert {name for _, name in api.FLUX.weight_files.values()} == fetched


# --- Drawing the back again (2026-10-30) -----------------------------------------------------

def test_a_redrawn_back_is_drawn_beside_the_old_and_swapped_in_only_when_done(tmp_path):
    front = _front(tmp_path)
    old = front.with_name(f"{front.stem}__back.png")
    old.write_bytes(b"old back")
    old.with_suffix(".provenance.json").write_text("{}")
    manager = api.ImageJobManager(tmp_path)
    job = api.create_back_job(manager, front, seed=9, engine=api.FLUX, instruction="  a red   cloak ")
    assert job.output_path.name.endswith("__back_new.png") and job.replaces == old
    assert job.prompt.endswith(" a red cloak") and job.prompt.startswith(api.back_view.BACK_PROMPT)
    job.output_path.write_bytes(b"new back")
    kept = api.take_place(job.output_path, old, stamp="20261030-200000")
    assert old.read_bytes() == b"new back" and kept.read_bytes() == b"old back"
    assert kept.name.endswith("__back_prev_20261030-200000.png")
    assert kept.with_suffix(".provenance.json").exists()   # the old back keeps its record


def test_a_first_back_goes_straight_to_its_place(tmp_path):
    job = api.create_back_job(api.ImageJobManager(tmp_path), _front(tmp_path), engine=api.FLUX)
    assert job.output_path.name.endswith("__back.png") and job.replaces is None
    assert job.prompt == api.back_view.BACK_PROMPT and job.instruction is None


def test_a_failed_redraw_leaves_the_old_back_where_it_was(tmp_path, monkeypatch):
    front = _front(tmp_path)
    old = front.with_name(f"{front.stem}__back.png")
    old.write_bytes(b"old back")
    manager = api.ImageJobManager(tmp_path)
    job = api.create_back_job(manager, front, engine=api.FLUX)

    class Failed:
        stdout = __import__("io").BytesIO(b"error: out of memory\n")
        def wait(self): return 1
        def poll(self): return 1
    monkeypatch.setattr(api.subprocess, "Popen", lambda *a, **k: Failed())
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "error" and old.read_bytes() == b"old back"


def test_the_instruction_is_trimmed_and_capped():
    long = "word " * 400
    assert len(api.back_prompt(long)) <= len(api.back_view.BACK_PROMPT) + 1 + api.MAX_INSTRUCTION
    assert api.back_prompt("") == api.back_view.BACK_PROMPT


# --- The back picture's brush (2026-10-30) ----------------------------------------------------

def _back(tmp_path, colour=(60, 120, 60)):
    from PIL import Image
    front = _front(tmp_path)
    back = front.with_name(f"{front.stem}__back.png")
    Image.new("RGB", (64, 64), colour).save(back)
    return back


def _stroke(box=(20, 20, 40, 40), size=(64, 64)):
    from PIL import Image, ImageDraw
    stroke = Image.new("L", size, 0)
    ImageDraw.Draw(stroke).rectangle(box, fill=255)
    return stroke


def test_the_paste_back_covers_everything_the_model_may_draw():
    """Pasting back less than the model drew cut the new satchel off in a hard line (2026-10-30)."""
    import numpy as np
    grown, feathered = api.brush_masks(_stroke((160, 160, 320, 320), (512, 512)), (256, 256))  # any stroke size
    assert grown.size == feathered.size == (256, 256)
    drawn, pasted = np.asarray(grown) == 255, np.asarray(feathered)
    assert (pasted[drawn] == 255).all()                                  # all of the drawn area, at full strength
    assert grown.getpixel((80 - api.BRUSH_GROW, 120)) == 255 and grown.getpixel((40, 120)) == 0
    assert 0 < feathered.getpixel((80 - api.BRUSH_GROW - 2, 120)) < 255  # soft just past it
    assert feathered.getpixel((40, 120)) == 0


def test_the_paste_back_keeps_every_pixel_outside_the_brush(tmp_path):
    from PIL import Image
    back = _back(tmp_path)
    drawn = tmp_path / "drawn.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(drawn)        # the model changed everything, as it does
    paste = tmp_path / "paste.png"
    _stroke().save(paste)
    api.paste_through(back, drawn, paste)
    with Image.open(drawn) as out:
        assert out.getpixel((2, 2)) == (60, 120, 60)              # outside: the original, exactly
        assert out.getpixel((30, 30)) == (200, 30, 30)            # inside: the new paint


def test_the_engine_is_shown_the_picture_with_the_painted_area_blanked(tmp_path):
    """Shown the old pixels, Flux drew them again: a strap stayed bare skin (2026-10-30)."""
    from PIL import Image
    back = _back(tmp_path)
    job = api.create_brush_job(api.ImageJobManager(tmp_path), back, _stroke(), "a leather strap")
    command = api.job_command(job, WEIGHTS, None, binary=Path("/bin/sd-cli"))
    assert command[command.index("-r") + 1] == str(job.reference) != str(job.init)
    with Image.open(job.reference) as guide, Image.open(job.mask) as mask:
        assert guide.size == mask.size == (256, 256)
        grey = (api.BRUSH_GUIDE_GREY,) * 3
        assert guide.getpixel((120, 120)) == grey                 # painted: blanked
        assert guide.getpixel((4, 4)) == (60, 120, 60)            # the rest: the picture itself


def test_a_brush_job_starts_from_the_back_with_its_mask_and_keeps_the_old(tmp_path):
    back = _back(tmp_path)
    job = api.create_brush_job(api.ImageJobManager(tmp_path), back, _stroke(), "  a red   satchel. ", seed=5)
    assert job.replaces == back and job.output_path.name.endswith("__back_new.png")
    assert job.engine is api.FLUX and job.settings["steps"] == api.BRUSH_STEPS
    # 64 px is below what the engine draws: it starts from a copy at 256, and pastes back at 64
    assert (job.settings["width"], job.settings["height"]) == (256, 256)
    assert job.init.name.endswith("__back_brush_init.png")
    from PIL import Image
    with Image.open(job.init) as init, Image.open(job.mask) as mask, Image.open(job.paste_mask) as paste:
        assert init.size == mask.size == (256, 256) and paste.size == (64, 64)
    assert job.prompt.startswith("a red satchel. Keep the style")
    command = api.job_command(job, WEIGHTS, None, binary=Path("/bin/sd-cli"))
    assert command[command.index("-i") + 1] == str(job.init)
    assert command[command.index("--mask") + 1] == str(job.mask)
    assert command[command.index("--strength") + 1] == str(api.BRUSH_STRENGTH)
    assert command[command.index("-r") + 1] == str(job.reference)  # Flux sees the picture it repaints, blanked


def test_an_empty_stroke_is_refused_before_anything_runs(tmp_path):
    from PIL import Image
    with pytest.raises(ValueError, match="paint over"):
        api.create_brush_job(api.ImageJobManager(tmp_path), _back(tmp_path), Image.new("L", (64, 64), 0), "x")


def test_a_finished_brush_swaps_in_only_the_painted_part(tmp_path, monkeypatch):
    from PIL import Image
    back = _back(tmp_path)
    manager = api.ImageJobManager(tmp_path)
    job = api.create_brush_job(manager, back, _stroke(), "a red satchel")

    class Done:
        def __init__(self, cmd, **k):
            Image.new("RGB", (64, 64), (200, 30, 30)).save(cmd[cmd.index("-o") + 1])
            self.stdout = __import__("io").BytesIO(b"generate_image completed in 1.0s\n")
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(api.subprocess, "Popen", Done)
    monkeypatch.setattr(api.usage_api, "count_done", lambda *a: None)
    api.run_job(job, manager, weights=WEIGHTS)
    assert job.status == "done" and job.output_path == back
    with Image.open(back) as out:
        assert out.getpixel((2, 2)) == (60, 120, 60) and out.getpixel((30, 30)) == (200, 30, 30)
    kept = list(back.parent.glob("*__back_prev_*.png"))
    assert len(kept) == 1
    record = json.loads(back.with_suffix(".provenance.json").read_text())
    assert record["brush"]["instruction"] == "a red satchel" and record["model"]["id"] == "flux2-klein-4b"



def test_a_back_already_at_a_drawable_size_is_brushed_as_it_is(tmp_path):
    from PIL import Image
    back = _back(tmp_path)
    Image.new("RGB", (512, 512), (60, 120, 60)).save(back)
    job = api.create_brush_job(api.ImageJobManager(tmp_path), back, _stroke(size=(512, 512)), "x")
    assert job.init == back and (job.settings["width"], job.settings["height"]) == (512, 512)



def test_a_see_through_back_is_brushed_on_grey_not_on_black(tmp_path):
    """Converting an RGBA cut-out to RGB shows what hides under the alpha (often black)."""
    from PIL import Image
    back = _back(tmp_path)
    cut = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    cut.paste((60, 120, 60, 255), (100, 100, 400, 400))
    cut.save(back)
    job = api.create_brush_job(api.ImageJobManager(tmp_path), back, _stroke(size=(512, 512)), "x")
    with Image.open(job.init) as init:
        assert init.mode == "RGB" and init.getpixel((5, 5)) == (api.back_view.GREY,) * 3
    drawn = tmp_path / "drawn.png"
    Image.new("RGB", (512, 512), (200, 30, 30)).save(drawn)
    api.paste_through(back, drawn, job.paste_mask)
    with Image.open(drawn) as out:
        assert out.getpixel((5, 5)) == (api.back_view.GREY,) * 3      # outside: grey, not black


def test_a_brush_refused_as_busy_leaves_the_running_brushs_files_alone(tmp_path):
    back = _back(tmp_path)
    manager = api.ImageJobManager(tmp_path)
    running = api.create_brush_job(manager, back, _stroke(), "a red satchel")
    paste_before = running.paste_mask.read_bytes()
    with pytest.raises(RuntimeError, match="already being generated"):
        api.create_brush_job(manager, back, _stroke((0, 0, 63, 63)), "a blue hat")
    assert running.paste_mask.read_bytes() == paste_before


def test_a_brushed_picture_names_the_one_it_replaced(tmp_path, monkeypatch):
    from PIL import Image
    back = _back(tmp_path)
    manager = api.ImageJobManager(tmp_path)
    job = api.create_brush_job(manager, back, _stroke(), "a red satchel")

    class Done:
        def __init__(self, cmd, **k):
            Image.new("RGB", (256, 256), (200, 30, 30)).save(cmd[cmd.index("-o") + 1])
            self.stdout = __import__("io").BytesIO(b"")
        def wait(self): return 0
        def poll(self): return 0
    monkeypatch.setattr(api.subprocess, "Popen", Done)
    monkeypatch.setattr(api.usage_api, "count_done", lambda *a: None)
    api.run_job(job, manager, weights=WEIGHTS)
    brush = json.loads(back.with_suffix(".provenance.json").read_text())["brush"]
    assert brush["replaced"].startswith(back.stem + "_prev_") and brush["strength"] == api.BRUSH_STRENGTH



def test_only_a_picture_file_in_output_can_be_a_back_source(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "notes.json").write_text("{}")
    with pytest.raises(ValueError, match="not a picture in output"):
        api.back_source(tmp_path, "a/notes.json")


def test_an_oversized_picture_is_refused_before_it_is_decoded():
    from PIL import Image
    with pytest.raises(ValueError, match="largest taken"):
        api.check_picture_size(Image.new("L", (api.MAX_PICTURE_SIDE + 1, 4)))
    api.check_picture_size(Image.new("L", (api.MAX_PICTURE_SIDE, 4)))



# --- Who redraws a back or brushes it (both engines, 2026-10-30) -------------------------------

def _drawn_by(path, engine):
    path.with_suffix(".provenance.json").write_text(json.dumps({"model": {"id": engine.licence_key}}))


def test_the_engine_that_drew_a_picture_is_read_from_its_record(tmp_path):
    picture = tmp_path / "p.png"
    picture.write_bytes(b"x")
    assert api.engine_of(picture) is None                       # an upload: no record
    _drawn_by(picture, api.QWEN)
    assert api.engine_of(picture) is api.QWEN


def test_a_redraw_uses_the_engine_that_drew_the_picture_when_it_can(tmp_path, monkeypatch):
    picture = tmp_path / "p.png"
    picture.write_bytes(b"x")
    _drawn_by(picture, api.QWEN)
    ready = {"flux": True, "qwen": True}
    monkeypatch.setattr(api, "usable", lambda engine, edit=False: ready[engine.id])
    monkeypatch.setattr(api, "pick_engine", lambda name=None, edit=False, ready=None: (
        api.ENGINES[name] if name else next(e for e in api.ENGINES.values() if ready_now[e.id])))
    ready_now = ready
    assert api.choose_engine(None, picture, edit=True) is api.QWEN             # the front's own
    assert api.choose_engine("flux", picture, edit=True) is api.FLUX           # asked by name wins
    assert api.choose_engine(None, picture, edit=True, prefer="flux") is api.FLUX
    ready["qwen"] = False                                                      # Qwen can't draw backs here
    assert api.choose_engine(None, picture, edit=True) is api.FLUX


@pytest.mark.parametrize("projector, sees", [(Path("/w/mmproj.gguf"), True), (None, False)])
def test_a_qwen_brush_sees_the_picture_when_its_projector_is_there(tmp_path, monkeypatch, projector, sees):
    """Flux and Qwen brush the same way; Qwen without its projector falls back to the words."""
    monkeypatch.setattr(api, "resolve_vision", lambda engine=api.QWEN: projector)
    back = _back(tmp_path)
    job = api.create_brush_job(api.ImageJobManager(tmp_path), back, _stroke(), "a red satchel", engine=api.QWEN)
    command = api.job_command(job, WEIGHTS, projector, binary=Path("/bin/sd-cli"))
    assert job.engine is api.QWEN and "--mask" in command
    assert ("-r" in command) is sees and ("--llm_vision" in command) is sees