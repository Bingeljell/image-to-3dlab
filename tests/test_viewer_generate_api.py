"""Cheap, torch-free tests for the viewer Generate API contract."""

from __future__ import annotations

import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "viewer" / "generate_api.py"
SPEC = importlib.util.spec_from_file_location("viewer_generate_api", MODULE_PATH)
api = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Register before exec: dataclasses (BackendSpec) resolve their module via
# sys.modules[cls.__module__] during class creation, on Python's own attribute lookup path
# for type introspection -- without this the module isn't findable yet and dataclass()
# raises AttributeError on 'NoneType' object has no attribute '__dict__'.
sys.modules["viewer_generate_api"] = api
SPEC.loader.exec_module(api)


def test_parse_tqdm_carriage_return_line():
    event = api.parse_tqdm_line("Sampling shape SLat:  33%|███▎ | 4/12 [07:59<15:58, 119.80s/it]")
    assert event == {
        "label": "Sampling shape SLat", "pct": 33, "step": 4, "total": 12,
        "elapsed": 479.0, "remain": 958.0, "s_per_it": 119.8,
    }


def test_shape_slat_passes_are_disambiguated(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb", {}, "trellis")
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 0) == "shape_slat_coarse"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 4) == "shape_slat_coarse"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 0) == "shape_slat_fine"
    assert api._phase_for_tqdm(job, "Sampling shape SLat", 4) == "shape_slat_fine"


@pytest.mark.parametrize("payload", [
    {"resolution": "2048"},
    {"texture_size": 512},
    {"decimation_target": 0},
    {"allow_rembg": "yes"},
])
def test_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api.validate_settings(payload)


def test_validate_settings_applies_demo_defaults():
    settings = api.validate_settings({})
    assert settings == api.DEFAULT_SETTINGS
    assert settings is not api.DEFAULT_SETTINGS


def test_overall_progress_is_stage_weighted():
    assert api._overall_pct("load", 100) == 14
    assert api._overall_pct("bake", 50) == 93


# --- backend env hygiene (the conv_none crash class) ---
def test_job_env_drops_backend_selection_keys(monkeypatch):
    for key in ("SPARSE_CONV_BACKEND", "ATTN_BACKEND", "SPARSE_ATTN_BACKEND",
                "FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
        monkeypatch.setenv(key, "stale-value")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "kept")
    env = api._job_env()
    for key in ("SPARSE_CONV_BACKEND", "ATTN_BACKEND", "SPARSE_ATTN_BACKEND",
                "FLEX_GEMM_AUTOTUNE_CACHE_PATH"):
        assert key not in env, f"{key} must be owned by the generator, not inherited"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["SOME_UNRELATED_VAR"] == "kept"  # everything else is inherited


def test_human_bytes_rounds():
    assert api._human_bytes(0) == "0 B"
    assert api._human_bytes(1023) == "1023 B"
    assert api._human_bytes(1536) == "1.5 KB"
    assert api._human_bytes(14 * 1024 ** 3) == "14.0 GB"


def test_weights_on_disk_reports_present_and_missing(tmp_path):
    present = tmp_path / "models--microsoft--TRELLIS.2-4B"
    (present / "snapshots").mkdir(parents=True)
    (present / "snapshots" / "model.safetensors").write_bytes(b"\x00" * 2048)
    result = api.weights_on_disk(cache_dir=tmp_path)
    trellis = result["models--microsoft--TRELLIS.2-4B"]
    assert trellis["present"] is True
    assert trellis["bytes"] == 2048
    assert trellis["human"] == "2.0 KB"
    dino = result["models--facebook--dinov3-vitl16-pretrain-lvd1689m"]
    assert dino["present"] is False


# --- setup runner (bootstrap via the web UI) ---
def test_setup_available_reports_missing_uv(monkeypatch):
    monkeypatch.setattr(api.shutil, "which", lambda name: None if name == "uv" else "/usr/bin/uv")
    ok, reason = api.setup_available()
    assert ok is False and "uv" in reason


def test_setup_available_reports_missing_bootstrap(monkeypatch, tmp_path):
    monkeypatch.setattr(api.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(api, "REPO", tmp_path)
    ok, reason = api.setup_available()
    assert ok is False and "bootstrap" in reason


# --- hunyuan-mlx-xiong shape-weights status: must match the {label, present, human}
# shape every backend's Setup-panel `weights` field uses (see weights_on_disk) --
# 2026-08-20: this used to be a flat {name: bool} dict, and the frontend's
# `Object.values(s.weights).map(repo => repo.present / repo.label)` silently rendered
# "undefined ... not on disk" for all three models regardless of what was actually
# downloaded, because a JS boolean has no .present/.label property ---
def test_hunyuan_xiong_shape_weights_status_reports_present_and_missing(monkeypatch, tmp_path):
    present_dir = tmp_path / "2.0"
    present_dir.mkdir()
    (present_dir / "model.safetensors").write_bytes(b"\x00" * 2048)
    missing_dir = tmp_path / "2.1"  # never created
    monkeypatch.setattr(api, "HUNYUAN_XIONG_SHAPE_MODELS", {"2.0": present_dir, "2.1": missing_dir})

    status = api._hunyuan_xiong_shape_weights_status()

    assert status["2.0"]["present"] is True
    assert status["2.0"]["bytes"] == 2048
    assert status["2.0"]["human"] == "2.0 KB"
    assert isinstance(status["2.0"]["label"], str) and status["2.0"]["label"]
    assert status["2.1"]["present"] is False
    assert status["2.1"]["bytes"] == 0
    assert isinstance(status["2.1"]["label"], str) and status["2.1"]["label"]


def test_hunyuan_xiong_readiness_weights_field_has_label_and_present(monkeypatch, tmp_path):
    """The exact shape the frontend indexes into -- a regression test for the undefined-row bug."""
    monkeypatch.setattr(api, "HUNYUAN_XIONG_SHAPE_MODELS", {"2.0": tmp_path / "nope"})
    result = api._hunyuan_xiong_readiness()
    entry = result["weights"]["2.0"]
    assert set(entry) >= {"label", "present", "human"}
    assert entry["present"] is False


def test_setup_run_exposes_the_job_sse_contract():
    run = api.SetupRun("0" * 32)
    run.status = "running"
    run.emit({"phase": "setup", "message": "+ git clone https://github.com/..."})
    event = run.events[-1]
    assert event["phase"] == "setup"
    assert event["message"].startswith("+ git clone")
    assert "elapsed_seconds" in event


def test_clean_port_build_present(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PYTHON", tmp_path / "no-such")
    monkeypatch.setattr(api, "WRAPPER", tmp_path / "no-such")
    assert api.clean_port_build_present() is False
    p = tmp_path / "exists"
    p.write_text("")
    monkeypatch.setattr(api, "PYTHON", p)
    monkeypatch.setattr(api, "WRAPPER", p)
    assert api.clean_port_build_present() is True


# --- silent-death diagnostics (exit code + RSS trajectory) ---
def test_signal_hint_decodes_kills():
    assert api._signal_hint(-9) == " — killed by SIGKILL"
    assert api._signal_hint(-15) == " — killed by SIGTERM"
    assert api._signal_hint(-11) == " — killed by SIGSEGV"
    assert api._signal_hint(1) == ""
    assert api._signal_hint(0) == ""


def test_process_rss_gb_parses_ps_output(monkeypatch):
    class FakeResult:
        stdout = "1048576\n"

    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: FakeResult())
    assert api._process_rss_gb(1234) == 1.0


def test_process_rss_gb_returns_zero_on_failure(monkeypatch):
    def boom(*a, **k):
        raise OSError("no such process")

    monkeypatch.setattr(api.subprocess, "run", boom)
    assert api._process_rss_gb(999999) == 0.0


# --- stage visibility: to_glb tqdm (it/s) and the decode/bake banner lines ---
def test_parse_tqdm_line_accepts_it_per_second_rate():
    event = api.parse_tqdm_line("Extracting GLB:  17%|█▋        | 1/6 [00:00<00:03,  1.59it/s]")
    assert event["label"] == "Extracting GLB"
    assert event["pct"] == 17 and event["step"] == 1 and event["total"] == 6
    assert event["remain"] == 3.0
    assert abs(event["s_per_it"] - 1.0 / 1.59) < 1e-6


def test_parse_tqdm_line_keeps_s_per_it_format():
    event = api.parse_tqdm_line("Sampling shape SLat:  33%|███▎ | 4/12 [07:59<15:58, 119.80s/it]")
    assert event["s_per_it"] == 119.8


def test_emit_banner_fires_decode_and_bake(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "trellis")
    api._emit_banner(job, "decode_latent (+face filter) done in 38.3s")
    assert job.events[-1]["phase"] == "decode"
    assert job.events[-1]["overall_pct"] == api._overall_pct("decode", 100)
    api._emit_banner(job, "bake (pre-cap + to_glb + export) done in 442.5s -> /x/out.glb")
    assert job.events[-1]["phase"] == "bake"


# --- job-status polling fallback: same payload shape as the SSE stream's own events, so
# a dropped/never-reconnected EventSource has something to recover from (2026-08-20: a
# fast SF3D run finished server-side with the browser never finding out) ---
def test_job_status_payload_reports_last_event(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "trellis")
    job.emit({"phase": "load", "overall_pct": 0, "message": "Starting"})
    job.emit({"phase": "done", "overall_pct": 100, "message": "Generation complete",
              "result_url": "/api/generate/x/result.glb"})
    job.status = "done"

    payload = api._job_status_payload(job)

    assert payload["status"] == "done"
    assert payload["last_event"]["phase"] == "done"
    assert payload["last_event"]["result_url"] == "/api/generate/x/result.glb"


def test_job_status_payload_handles_no_events_yet(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "trellis")
    payload = api._job_status_payload(job)
    assert payload == {"status": "queued", "last_event": None}


# --- backend registry ------------------------------------------------------------------

def test_backend_registry_has_all_four():
    assert set(api.BACKENDS) == {"trellis", "sf3d", "hunyuan-mlx", "hunyuan-mlx-xiong"}
    for spec in api.BACKENDS.values():
        assert spec.stages, f"{spec.id} must declare at least one stage"


def test_sf3d_build_args_matches_pipeline_cli(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb",
                   api._sf3d_validate_settings({}), "sf3d")
    args = api._sf3d_build_args(job)
    assert args[0] == "--fast"
    assert "--output-dir" in args and str(tmp_path) in args
    assert args[-1] == str(tmp_path / "input.png")


# 2026-08-20: the old version of this test hand-derived <stem>_sf3d.glb -- a fictional
# filename that matched _sf3d_finalize's own (wrong) assumption, not what pipeline.py's
# provenance system actually produces. It passed while every real SF3D run through the
# web UI failed with "generator exited 0 but produced no output file". Uses the real
# finalize_output() to build the fixture so this can't drift from reality the same way.
def test_sf3d_finalize_moves_output_into_place(tmp_path):
    from image_to_3dlab.provenance import LICENSES, finalize_output

    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb", {}, "sf3d")
    image = tmp_path / "input.png"
    image.write_bytes(b"fake-png")
    generated = tmp_path / "raw_generated.glb"
    generated.write_bytes(b"glb-bytes")
    finalize_output(
        generated=generated, image=image, output_root=tmp_path, backend="sf3d",
        profile=LICENSES["sf3d"], intent={}, parameters={}, source_manifest=None,
    )

    api._sf3d_finalize(job)

    assert job.output_path.read_bytes() == b"glb-bytes"
    assert job.manifest_path.is_file()
    assert json.loads(job.manifest_path.read_text())["model"]["backend"] == "sf3d"


def test_sf3d_finalize_is_a_noop_when_nothing_was_produced(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb", {}, "sf3d")
    api._sf3d_finalize(job)  # must not raise
    assert not job.output_path.exists()


@pytest.mark.parametrize("payload", [
    {"remesh": "invalid"},
    {"texture_resolution": 0},
    {"foreground_ratio": 1.5},
    {"foreground_ratio": 0},
])
def test_sf3d_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api._sf3d_validate_settings(payload)


def test_hunyuan_build_args_matches_wrapper_cli(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb",
                   api._hunyuan_validate_settings({}), "hunyuan-mlx")
    args = api._hunyuan_build_args(job)
    assert args[0] == str(tmp_path / "input.png")
    assert args[1] == str(tmp_path / "model.glb")
    assert "--octree-resolution" in args and "512" in args


@pytest.mark.parametrize("payload", [
    {"octree_resolution": 128},
    {"decimation_target": 0},
    {"decimation_target": 700_000},  # past the confirmed xatlas wall (500k-700k, 2026-08-18)
])
def test_hunyuan_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api._hunyuan_validate_settings(payload)


def test_hunyuan_xiong_build_args_matches_wrapper_cli(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb",
                   api._hunyuan_xiong_validate_settings({}), "hunyuan-mlx-xiong")
    args = api._hunyuan_xiong_build_args(job)
    assert args[0] == str(tmp_path / "input.png")
    assert args[1] == str(tmp_path / "model.glb")
    assert "--octree-resolution" in args and "512" in args
    assert "--quantize" in args and "8" in args  # default, see wrapper script's speed caveat


@pytest.mark.parametrize("payload", [
    {"octree_resolution": 128},
    {"quantize": 2},  # only 0 (off), 4, or 8 are real quantization levels here
    {"decimation_target": 0},
    {"decimation_target": 700_000},  # same shared xatlas wall as the other Hunyuan backend
])
def test_hunyuan_xiong_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api._hunyuan_xiong_validate_settings(payload)


@pytest.mark.parametrize("line,expected_phase,expected_pct", [
    # Real lines captured from output/hunyuan_mlx_zimeng_test/flicker_octree512/*.log,
    # 2026-08-18 -- the exact format both hunyuan_mlx_generate.py and run_paint_pbr.py print.
    ("shape generated (301s): 191099 verts, 382196 faces", "shape", 100),
    # Real lines captured 2026-08-19 from the shape stage itself (hunyuan_mlx_xiong_generate.py
    # / hy3dmlx pipeline.py) -- these were silently unrecognized before that date, so the
    # shape stage showed no progress in the browser for its whole (often multi-minute) run.
    ("loaded Hunyuan3DDiT [8-bit DiT+DINO]: dino 727 (+0), dit 656 (+0), vae 266 (+0)",
     "shape", 5),
    ("[vae] grid (513, 513, 513) (range -1.047..1.063, active 1.7%) in 34.5s", "shape", 80),
    ("[mesh] 395339 verts, 790626 faces, total 105.4s", "shape", 95),
    ("simplified to 500,000 faces (0.4s)", "remesh", 100),
    ("mesh at/under decimation target (382,196 <= 500,000); no remesh needed (0s)",
     "remesh", 100),
    ("mesh loaded: 500000 faces (2s)", "paint_setup", None),
    ("xatlas parametrize done (158s)", "paint_setup", None),
    ("controls + dino ready (159s)", "paint_setup", None),
    ("views decoded (370s)", "paint_finish", None),
    ("super-res x4 (454s, views -> 2048px)", "paint_finish", None),
    ("DONE 484s -> outputs/textured_mesh_pbr.glb (+ pbr_albedo/mr textures)",
     "paint_finish", 100),
])
def test_hunyuan_parse_line_real_captured_lines(tmp_path, line, expected_phase, expected_pct):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "hunyuan-mlx")
    api._hunyuan_parse_line(job, line)
    assert job.events, f"no event emitted for: {line!r}"
    event = job.events[-1]
    assert event["phase"] == expected_phase
    if expected_pct is not None:
        assert event.get("stage_pct") == expected_pct


def test_hunyuan_parse_line_step_progress(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "hunyuan-mlx")
    api._hunyuan_parse_line(job, "  step 8/15 178s")
    event = job.events[-1]
    assert event["phase"] == "paint_diffusion"
    assert event["step"] == 8 and event["total"] == 15
    assert event["stage_pct"] == round(8 / 15 * 100)


def test_hunyuan_parse_line_shape_denoise_progress(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "hunyuan-mlx-xiong")
    api._hunyuan_parse_line(job, "[denoise] 15/30")
    event = job.events[-1]
    assert event["phase"] == "shape"
    assert event["step"] == 15 and event["total"] == 30
    assert event["stage_pct"] == round(15 / 30 * 60)


def test_hunyuan_parse_line_shape_denoise_summary_is_not_mistaken_for_a_step(tmp_path):
    """The final `[denoise] N steps (...) in Xs` summary must not match the per-step regex --
    it has non-digit trailing content, but a careless prefix check could still catch it."""
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "hunyuan-mlx-xiong")
    api._hunyuan_parse_line(job, "[denoise] 30 steps (CFG) in 57.3s")
    assert job.events == []


def test_hunyuan_parse_line_ignores_unrecognized_lines(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "a.png", tmp_path / "m.glb", {}, "hunyuan-mlx")
    api._hunyuan_parse_line(job, "Fetching 5 files: 100%|##########| 5/5")
    assert job.events == []


# --- job folder naming ------------------------------------------------------------------

def test_job_folder_name_uses_requested_slug_when_given():
    assert api._job_folder_name("flicker", "trellis", "my run 1!") == "my-run-1"


def test_job_folder_name_falls_back_to_image_backend_timestamp():
    name = api._job_folder_name("3-4th-flicker-alpha", "hunyuan-mlx", None)
    assert name.startswith("3-4th-flicker-alpha__hunyuan-mlx__")
    assert "/" not in name and ".." not in name


def test_slugify_strips_unsafe_characters():
    assert api._slugify("../../etc/passwd") == "etc-passwd"
    assert api._slugify("   ") == "job"


# --- output directory + debug-file cleanup ------------------------------------------------

def test_resolve_output_base_defaults_to_repo_output():
    assert api._resolve_output_base(None) == (api.REPO / "output").resolve()
    assert api._resolve_output_base("  ") == (api.REPO / "output").resolve()


def test_resolve_output_base_accepts_subdir_inside_output():
    resolved = api._resolve_output_base("output/my-runs")
    assert resolved == (api.REPO / "output" / "my-runs").resolve()


@pytest.mark.parametrize("escape", ["../vendor", "/etc", "../../etc/passwd"])
def test_resolve_output_base_rejects_escape_attempts(escape):
    with pytest.raises(ValueError):
        api._resolve_output_base(escape)


def test_job_manager_matches_folder_and_file_name(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "OUTPUT_ROOT", tmp_path)
    jobs = api.JobManager()
    job = jobs.create(tmp_path / "in.png", {}, "hunyuan-mlx", "flicker", "my-run", tmp_path)
    assert job.directory.name == "my-run"
    assert job.output_path == job.directory / "my-run.glb"
    assert job.manifest_path == job.directory / "my-run.json"


def test_job_manager_disambiguates_colliding_folder_names(tmp_path):
    jobs = api.JobManager()
    job1 = jobs.create(tmp_path / "in.png", {}, "trellis", "flicker", "dup", tmp_path)
    jobs.active = None  # simulate job1 finishing so a second job can be created
    job2 = jobs.create(tmp_path / "in.png", {}, "trellis", "flicker", "dup", tmp_path)
    assert job1.directory != job2.directory
    assert job2.directory.name == "dup-2"
    assert job2.output_path.name == "dup-2.glb"


def test_cleanup_debug_files_keeps_only_output_glb(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "run.glb", {}, "trellis")
    job.output_path.write_bytes(b"glb")
    job.manifest_path.write_text("{}")
    (tmp_path / "run_latents.pt").write_bytes(b"x")
    (tmp_path / "run.log").write_text("log")
    api._cleanup_debug_files(job)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.glb"]


# --- alpha-transparency check: a missing Pillow install is a broken environment, not a
# fact about the image (2026-08-20: this silently reported "no alpha" for every image) ---

def test_image_has_transparent_alpha_raises_when_pillow_missing(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="Pillow is not installed"):
        api.image_has_transparent_alpha(tmp_path / "whatever.png")


def test_image_has_transparent_alpha_true_for_real_transparency(tmp_path):
    from PIL import Image

    path = tmp_path / "transparent.png"
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 0))
    img.save(path)
    assert api.image_has_transparent_alpha(path) is True


def test_image_has_transparent_alpha_false_for_fully_opaque_rgba(tmp_path):
    from PIL import Image

    path = tmp_path / "opaque.png"
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    img.save(path)
    assert api.image_has_transparent_alpha(path) is False


# --- "has alpha" is not "is cut out" (2026-09-03): a letterboxed image passed the alpha
# check on its bars alone, and its opaque backdrop was rebuilt as 3D geometry ---
def _save(tmp_path, name, alpha_rows):
    from PIL import Image
    import numpy as np

    a = np.zeros((64, 64, 4), dtype=np.uint8)
    a[..., :3] = 200
    a[..., 3] = alpha_rows
    path = tmp_path / name
    Image.fromarray(a, "RGBA").save(path)
    return path


def test_border_opaque_fraction_flags_letterboxed_upload(tmp_path):
    import numpy as np

    alpha = np.full((64, 64), 255, dtype=np.uint8)
    alpha[:8] = 0
    alpha[-8:] = 0          # transparent bars only; left/right edges still opaque
    path = _save(tmp_path, "letterboxed.png", alpha)
    assert api.image_has_transparent_alpha(path) is True   # old gate lets it through
    assert api.image_border_opaque_fraction(path) > api.UNCUT_BORDER_LIMIT


def test_border_opaque_fraction_passes_real_cutout(tmp_path):
    import numpy as np

    alpha = np.zeros((64, 64), dtype=np.uint8)
    alpha[8:-8, 8:-8] = 255
    path = _save(tmp_path, "cutout.png", alpha)
    assert api.image_border_opaque_fraction(path) == 0.0


def test_border_opaque_fraction_none_for_non_rgba(tmp_path):
    from PIL import Image

    path = tmp_path / "rgb.png"
    Image.new("RGB", (16, 16), (1, 2, 3)).save(path)
    assert api.image_border_opaque_fraction(path) is None


def test_border_opaque_fraction_none_when_unreadable(tmp_path):
    """Unmeasurable must not mean "blocked" -- the wrapper still enforces the same rule."""
    assert api.image_border_opaque_fraction(tmp_path / "missing.png") is None


def test_uncut_image_error_states_the_measurement(tmp_path):
    msg = api.uncut_image_error(0.39)
    assert "39%" in msg
    assert "transparent background" in msg


def test_image_has_transparent_alpha_false_for_rgb_no_alpha_channel(tmp_path):
    from PIL import Image

    path = tmp_path / "rgb.png"
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    img.save(path)
    assert api.image_has_transparent_alpha(path) is False


# --- pid-file lifecycle (the anchor for orphan reconciliation on server restart) -------

def test_pid_file_write_and_remove_roundtrip(tmp_path):
    api._write_pid_file(tmp_path, 4242)
    assert api._pid_file_path(tmp_path).read_text() == "4242"
    api._remove_pid_file(tmp_path)
    assert not api._pid_file_path(tmp_path).exists()


def test_remove_pid_file_is_a_noop_when_missing(tmp_path):
    api._remove_pid_file(tmp_path)  # must not raise


# --- graceful-shutdown counterpart to orphan reconciliation ----------------------------

class _FakeProcess:
    def __init__(self, pid, alive=True):
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def test_terminate_active_job_kills_the_running_process_group(tmp_path, monkeypatch):
    jobs = api.JobManager()
    monkeypatch.setattr(api, "JOBS", jobs)
    job = jobs.create(tmp_path / "in.png", {}, "trellis", "flicker", None, tmp_path)
    job.process = _FakeProcess(4242)
    killed = []
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    api._terminate_active_job()

    assert killed == [4242]


def test_terminate_active_job_is_a_noop_with_no_active_job(monkeypatch):
    jobs = api.JobManager()
    monkeypatch.setattr(api, "JOBS", jobs)
    killed = []
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    api._terminate_active_job()  # must not raise

    assert killed == []


def _valid_rig_sidecar(asset: bytes, scene: bytes) -> bytes:
    import hashlib

    return json.dumps({
        "schemaVersion": 1,
        "rigProfile": "test",
        "assetFingerprint": "sha256:" + hashlib.sha256(asset).hexdigest(),
        "coordinateSpace": "armature-local",
        "mirror": {"axis": "X", "origin": 0},
        "joints": {"joint": {
            "label": "Joint", "position": [0, 0, 0], "sourceBone": "DEF-joint",
            "parent": None, "mirrorOf": None,
        }},
        "corrections": {"joint": {
            "sourcePosition": [0, 0, 0], "targetPosition": [0, 1, 0],
            "delta": [0, 1, 0], "mirrored": False,
        }},
        "binding": {
            "adapter": "test",
            "sceneFingerprint": "sha256:" + hashlib.sha256(scene).hexdigest(),
            "metarigObjectId": "object",
            "joints": {"joint": {"targets": [
                {"boneId": "bone", "boneName": "bone", "endpoint": "head"}
            ]}},
        },
    }).encode()


def test_terminate_active_job_also_kills_a_running_rebind(tmp_path, monkeypatch):
    rig_jobs = api.RIG_JOBS.__class__(tmp_path)
    job = rig_jobs.create(
        "asset.glb", b"glb", b"blend", _valid_rig_sidecar(b"glb", b"blend")
    )
    job.process = _FakeProcess(5252)
    killed = []
    monkeypatch.setattr(api, "RIG_JOBS", rig_jobs)
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    api._terminate_active_job()

    assert killed == [5252]


def test_handler_exposes_typed_rig_rebind_routes():
    source = MODULE_PATH.read_text()

    assert '["api", "rig", "rebind"]' in source
    assert "_create_rig_job" in source
    assert "_rig_events" in source
    assert "_rig_status" in source
    assert "_rig_artifact" in source
    assert "_cancel_rig_job" in source


def test_terminate_active_job_is_a_noop_when_process_already_exited(tmp_path, monkeypatch):
    jobs = api.JobManager()
    monkeypatch.setattr(api, "JOBS", jobs)
    job = jobs.create(tmp_path / "in.png", {}, "trellis", "flicker", None, tmp_path)
    job.process = _FakeProcess(4242, alive=False)
    killed = []
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    api._terminate_active_job()

    assert killed == []


# --- startup reconciliation of pid files a dead server left behind ---------------------

def test_reconcile_annotates_and_clears_a_dead_orphan(tmp_path, monkeypatch):
    job_dir = tmp_path / "some-job"
    job_dir.mkdir()
    (job_dir / "pid").write_text("4242")
    (job_dir / "run.log").write_text("views decoded (100s)\n")
    monkeypatch.setattr(api.os, "killpg", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))

    touched = api._reconcile_orphaned_jobs(tmp_path)

    assert touched == ["some-job"]
    assert not (job_dir / "pid").exists()
    assert "died mid-run" in (job_dir / "run.log").read_text()


def test_reconcile_kills_and_annotates_a_live_orphan(tmp_path, monkeypatch):
    job_dir = tmp_path / "live-job"
    job_dir.mkdir()
    (job_dir / "pid").write_text("4242")
    (job_dir / "run.log").write_text("step 3/15\n")
    killed = []

    def fake_killpg(pid, sig):
        if sig == api.signal.SIGTERM:
            killed.append((pid, sig))
        # sig == 0 (the liveness probe) raises nothing -- simulates a live process group

    monkeypatch.setattr(api.os, "killpg", fake_killpg)

    touched = api._reconcile_orphaned_jobs(tmp_path)

    assert touched == ["live-job"]
    assert killed == [(4242, api.signal.SIGTERM)]
    assert "orphaned generation" in (job_dir / "run.log").read_text()
    assert not (job_dir / "pid").exists()


def test_reconcile_survives_a_missing_run_log(tmp_path, monkeypatch):
    job_dir = tmp_path / "no-log-job"
    job_dir.mkdir()
    (job_dir / "pid").write_text("4242")
    monkeypatch.setattr(api.os, "killpg", lambda pid, sig: (_ for _ in ()).throw(ProcessLookupError()))

    touched = api._reconcile_orphaned_jobs(tmp_path)

    assert touched == ["no-log-job"]
    assert not (job_dir / "pid").exists()


def test_reconcile_discards_an_unparseable_pid_file(tmp_path):
    job_dir = tmp_path / "bad-pid-job"
    job_dir.mkdir()
    (job_dir / "pid").write_text("not-a-pid")

    touched = api._reconcile_orphaned_jobs(tmp_path)

    assert touched == []
    assert not (job_dir / "pid").exists()


def test_reconcile_is_a_noop_with_no_leftover_jobs(tmp_path):
    assert api._reconcile_orphaned_jobs(tmp_path) == []


def test_trellis_build_args_skips_resume_caches_unless_debug(tmp_path):
    settings = api.validate_settings({})
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "run.glb", settings,
                   "trellis", debug=False)
    args = api._trellis_build_args(job)
    assert "--no-save-latents" in args and "--no-save-decode" in args
    job.debug = True
    args = api._trellis_build_args(job)
    assert "--no-save-latents" not in args and "--no-save-decode" not in args


# --- a failed run must say why, not just "exited with code 1" (2026-09-03) ---
def test_failure_reason_returns_the_generators_last_words():
    log = [
        "[rss 0.13 GB]",
        "Sampling shape SLat:  50%|#####     | 6/12 [01:40<01:38, 16.38s/it]",
        "[rss 0.21 GB]",
        "dog.png carries an alpha channel, but 39% of its outer border is still opaque.",
        "Fix: re-export the image with a transparent background.",
        "generator exited with code 1",
    ]
    reason = api.failure_reason(log)
    assert reason is not None
    assert "39%" in reason
    assert "transparent background" in reason
    assert "rss" not in reason and "Sampling" not in reason


def test_failure_reason_is_none_when_only_progress_noise():
    log = ["[rss 0.13 GB]", "Sampling shape SLat: 100%|##########| 12/12 [02:49<00:00]",
           "Loading TRELLIS.2 pipeline (load_rembg=False)...", "generator exited with code 1"]
    assert api.failure_reason(log) is None


def test_failure_reason_stops_at_the_noise_above_the_message():
    """Only the trailing block is the reason; earlier output is not dragged in."""
    log = ["an unrelated earlier line", "[rss 0.13 GB]", "the actual failure"]
    assert api.failure_reason(log) == "the actual failure"


def test_failure_reason_keeps_a_traceback_together():
    log = ["[rss 0.1 GB]", "Traceback (most recent call last):",
           '  File "x.py", line 1, in <module>', "RuntimeError: weights missing"]
    reason = api.failure_reason(log)
    assert reason.startswith("Traceback")
    assert reason.endswith("RuntimeError: weights missing")
