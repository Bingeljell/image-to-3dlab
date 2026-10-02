"""Cheap, torch-free tests for the viewer Generate API contract."""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import re
import sys
import types
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
    {"sparse_attn_backend": "metal_flash"},
    {"sparse_attn_backend": "fp16"},
    {"sparse_attn_backend": "sdpa-fp16"},
])
def test_validate_settings_rejects_invalid_values(payload):
    with pytest.raises(ValueError):
        api.validate_settings(payload)


def test_validate_settings_applies_demo_defaults():
    settings = api.validate_settings({})
    assert settings == api.DEFAULT_SETTINGS
    assert settings is not api.DEFAULT_SETTINGS


def test_default_attention_backend_is_the_stock_one():
    """mlx must be opt-in.

    It needs the vendored checkout patched and mlx installed in its venv, neither of which
    a fresh clone has. A default that fails on a clean machine is worse than a slower one.
    """
    assert api.DEFAULT_SETTINGS["sparse_attn_backend"] == "sdpa"
    assert api.validate_settings({})["sparse_attn_backend"] == "sdpa"


def test_mlx_attention_backend_is_accepted_and_reaches_the_wrapper():
    settings = api.validate_settings({"sparse_attn_backend": "mlx"})
    assert settings["sparse_attn_backend"] == "mlx"

    job = types.SimpleNamespace(
        image_path=Path("in.png"),
        output_path=Path("out.glb"),
        settings=settings,
        debug=True,
    )
    args = api._trellis_build_args(job)
    assert "--sparse-attn-backend" in args
    assert args[args.index("--sparse-attn-backend") + 1] == "mlx"


@pytest.mark.parametrize("choice,flag,dtype", [
    ("sdpa", "sdpa", None),
    ("mlx", "mlx", "fp32"),
    ("mlx-fp16", "mlx", "fp16"),
])
def test_attention_choice_splits_into_flag_and_precision(choice, flag, dtype):
    cli, env = api.attention_backend_spec(choice)
    assert cli == flag
    assert env.get("I2L_MLX_ATTN_DTYPE") == dtype


def test_mlx_precision_is_pinned_explicitly_not_left_to_the_environment():
    """Selecting mlx must set fp32 rather than inherit whatever the shell had.

    The precision changes what the run computes, so leaving it to an exported variable
    makes two identical-looking jobs produce different output.
    """
    assert "I2L_MLX_ATTN_DTYPE" in api.BACKEND_ENV_KEYS
    _, env = api.attention_backend_spec("mlx")
    assert env["I2L_MLX_ATTN_DTYPE"] == "fp32"


def test_fp16_choice_still_passes_mlx_to_the_wrapper():
    settings = api.validate_settings({"sparse_attn_backend": "mlx-fp16"})
    job = types.SimpleNamespace(
        image_path=Path("in.png"), output_path=Path("out.glb"),
        settings=settings, debug=True,
    )
    args = api._trellis_build_args(job)
    # The wrapper has no fp16 flag; precision travels by environment.
    assert args[args.index("--sparse-attn-backend") + 1] == "mlx"
    assert "mlx-fp16" not in args


def _fake_backend(tmp_path, *, patched: bool, package: bool):
    tmp_path.mkdir(parents=True, exist_ok=True)
    dispatch = tmp_path / "full_attn.py"
    dispatch.write_text(
        "elif config.ATTN == 'mlx':\n    pass\n" if patched else "elif config.ATTN == 'sdpa':\n    pass\n"
    )
    if package:
        (tmp_path / ".venv" / "lib" / "python3.11" / "site-packages" / "mlx").mkdir(parents=True)
    return tmp_path, dispatch


@pytest.mark.parametrize("patched,package,ready", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, False),
])
def test_mlx_attention_status_needs_both_the_patch_and_the_package(tmp_path, patched, package, ready):
    vendor, dispatch = _fake_backend(tmp_path, patched=patched, package=package)
    status = api.mlx_attention_status(vendor=vendor, dispatch=dispatch)
    assert status == {
        "patched": patched, "package": package, "ready": ready,
        "hint": status["hint"],
    }
    assert (status["hint"] is None) == ready


def test_mlx_attention_status_hint_names_the_missing_step(tmp_path):
    vendor, dispatch = _fake_backend(tmp_path, patched=False, package=True)
    assert "patch_trellis_mlx_attention" in api.mlx_attention_status(vendor=vendor, dispatch=dispatch)["hint"]

    vendor2, dispatch2 = _fake_backend(tmp_path / "b", patched=True, package=False)
    assert "uv pip install" in api.mlx_attention_status(vendor=vendor2, dispatch=dispatch2)["hint"]


def test_mlx_attention_status_survives_a_missing_checkout(tmp_path):
    status = api.mlx_attention_status(vendor=tmp_path / "gone", dispatch=tmp_path / "gone" / "x.py")
    assert status["ready"] is False
    assert status["hint"]


def test_unready_mlx_never_makes_the_machine_look_unready(monkeypatch):
    """Generation works without mlx; the default sdpa path needs none of it.

    Folding mlx readiness into the top-level `ready` flag would block the Generate button
    over an optional accelerator.
    """
    monkeypatch.setattr(api, "mlx_attention_status",
                        lambda *a, **k: {"patched": False, "package": False, "ready": False, "hint": "x"})
    monkeypatch.setattr(api, "clean_port_build_present", lambda: True)
    monkeypatch.setattr(api, "weights_on_disk", lambda *a, **k: {})
    status = api.setup_status()
    assert status["ready"] is True
    assert status["mlx_attention"]["ready"] is False


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
    tinyclip = result["models--wkcn--TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"]
    assert tinyclip["present"] is False


def test_trellis_input_advisor_runs_in_backend_environment(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    script = tmp_path / "advisor.py"
    image = tmp_path / "input.png"
    for path in (interpreter, script, image):
        path.write_bytes(b"x")
    monkeypatch.setattr(api, "PYTHON", interpreter)
    monkeypatch.setattr(api, "TINYCLIP_ADVISOR", script)

    class Result:
        stdout = json.dumps({"verdict": "likely_flat", "flat_risk": 0.91}) + "\n"

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return Result()

    monkeypatch.setattr(api.subprocess, "run", fake_run)
    result = api.run_trellis_input_advisor(image)

    assert result["verdict"] == "likely_flat"
    assert calls[0][0] == [str(interpreter), str(script), str(image)]
    assert calls[0][1]["timeout"] == api.TINYCLIP_TIMEOUT_SECONDS
    assert calls[0][1]["check"] is True


def test_trellis_input_advisor_rejects_malformed_output(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    script = tmp_path / "advisor.py"
    image = tmp_path / "input.png"
    for path in (interpreter, script, image):
        path.write_bytes(b"x")
    monkeypatch.setattr(api, "PYTHON", interpreter)
    monkeypatch.setattr(api, "TINYCLIP_ADVISOR", script)

    class Result:
        stdout = "not-json\n"

    monkeypatch.setattr(api.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="invalid JSON"):
        api.run_trellis_input_advisor(image)


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

def test_backend_registry_lists_every_backend():
    assert set(api.BACKENDS) == {
        "trellis", "sf3d", "hunyuan-mlx", "hunyuan-mlx-xiong", "pixal3d", "hunyuan-cuda",
    }
    for spec in api.BACKENDS.values():
        assert spec.stages, f"{spec.id} must declare at least one stage"


def test_every_stage_event_a_backend_emits_names_a_declared_stage(tmp_path):
    """`phase` is the row key the browser looks up, so it must be a declared stage id.

    Pixal3D shipped emitting `{"phase": "stage", "stage": "shape"}`, which left the panel
    frozen on "Choose an image to begin" while the job ran to completion.
    """
    lines = {
        "pixal3d": [
            "[1/6] Pixal3D multiview: load views",
            "[2/6] SS proj conditioning + flow",
            "[3/6] shape SLAT flow (LR 512 -> HR 1024 cascade)",
            "[4/6] FlexiDualGrid shape decode -> mesh @res1024",
            "[5/6] texture SLAT flow (HR 1024, NAF@1024) + PBR decode",
            "[6/6] write out.glb",
        ],
    }
    for backend_id, sample in lines.items():
        spec = api.BACKENDS[backend_id]
        job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb",
                      spec.validate_settings({}), backend_id)
        for line in sample:
            spec.parse_line(job, line)
        emitted = [event["phase"] for event in job.events]
        assert emitted, f"{backend_id} emitted no stage events"
        unknown = sorted(set(emitted) - set(spec.stages))
        assert not unknown, f"{backend_id} emitted phases that are not stages: {unknown}"


def test_pixal3d_progress_climbs_with_the_banners(tmp_path):
    spec = api.BACKENDS["pixal3d"]
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb",
                  spec.validate_settings({}), "pixal3d")
    for line in ("[2/6] SS", "[4/6] decode", "[6/6] write"):
        spec.parse_line(job, line)
    percentages = [event["overall_pct"] for event in job.events]
    assert percentages == sorted(percentages)
    assert percentages[-1] <= 99


def test_pixal3d_settings_reject_an_unavailable_resolution():
    """No res-512 texture flow, and trellis-cli --sv-image refuses 1536."""
    for res in (512, 1536):
        with pytest.raises(ValueError):
            api._pixal3d_validate_settings({"res": res})
    assert api._pixal3d_validate_settings({"res": 1024})["res"] == 1024


def test_pixal3d_page_offers_only_the_resolution_that_runs():
    page = (Path(__file__).resolve().parents[1] / "viewer" / "index.html").read_text()
    select = page.split('<select id="pixal3d-res">', 1)[1].split("</select>", 1)[0]
    assert re.findall(r"<option[^>]*>(\d+)</option>", select) == ["1024"]


def test_pixal3d_settings_reject_a_fov_given_in_degrees():
    """The gauge camera is radians; 20 would be a plausible-looking disaster."""
    with pytest.raises(ValueError):
        api._pixal3d_validate_settings({"fov": 20})


def test_pixal3d_build_args_carry_the_camera(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "input.png", tmp_path / "model.glb",
                  api._pixal3d_validate_settings({}), "pixal3d")
    args = api._pixal3d_build_args(job)
    assert args[:2] == [str(tmp_path / "input.png"), str(tmp_path / "model.glb")]
    assert "--fov" in args


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


def test_cleanup_debug_files_keeps_the_glb_and_its_record(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "run.glb", {}, "trellis")
    job.output_path.write_bytes(b"glb")
    job.manifest_path.write_text("{}")
    (tmp_path / "run_latents.pt").write_bytes(b"x")
    (tmp_path / "run.log").write_text("log")
    api._cleanup_debug_files(job)
    # The manifest is the run's record (for Pixal3D, its licence record), not debug output.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.glb", "run.json"]


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
    assert api._pid_file_path(tmp_path).read_text() == f"4242 {os.getpid()}"
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
    monkeypatch.setattr(api, "_process_group_alive", lambda pid: True)
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    touched = api._reconcile_orphaned_jobs(tmp_path)

    assert touched == ["live-job"]
    assert killed == [4242]
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


def test_pixal3d_steps_default_to_auto_and_leave_the_choice_to_the_wrapper():
    settings = api._pixal3d_validate_settings({})
    assert settings["steps"] == "auto"
    job = types.SimpleNamespace(image_path=Path("i.png"), output_path=Path("o.glb"),
                                settings=settings)
    assert "--steps" not in api._pixal3d_build_args(job)


def test_pixal3d_full_steps_are_passed_through():
    settings = api._pixal3d_validate_settings({"steps": "12"})
    job = types.SimpleNamespace(image_path=Path("i.png"), output_path=Path("o.glb"),
                                settings=settings)
    args = api._pixal3d_build_args(job)
    assert args[args.index("--steps") + 1] == "12"


@pytest.mark.parametrize("steps", [8, 0, "fast", None])
def test_pixal3d_steps_outside_the_offered_choices_are_refused(steps):
    with pytest.raises(ValueError):
        api._pixal3d_validate_settings({"steps": steps})


def test_reconcile_leaves_a_job_whose_owner_is_still_alive(tmp_path, monkeypatch):
    """A Finish run driven by another live process (a second viewer, a script) is not an
    orphan. Starting the viewer used to kill one mid-repaint (2026-09-29)."""
    job_dir = tmp_path / "someone-elses-job"
    job_dir.mkdir()
    (job_dir / "pid").write_text("4242 5151")
    killed = []
    monkeypatch.setattr(api, "_process_alive", lambda pid: pid == 5151)
    monkeypatch.setattr(api, "_process_group_alive", lambda pid: True)
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    assert api._reconcile_orphaned_jobs(tmp_path) == []
    assert killed == []
    assert (job_dir / "pid").read_text() == "4242 5151"


def test_reconcile_kills_a_live_job_whose_owner_died(tmp_path, monkeypatch):
    job_dir = tmp_path / "orphan"
    job_dir.mkdir()
    (job_dir / "pid").write_text("4242 5151")
    killed = []
    monkeypatch.setattr(api, "_process_alive", lambda pid: False)
    monkeypatch.setattr(api, "_process_group_alive", lambda pid: True)
    monkeypatch.setattr(api, "_killpg_if_alive", killed.append)

    assert api._reconcile_orphaned_jobs(tmp_path) == ["orphan"]
    assert killed == [4242]


def test_reconcile_notes_a_finish_run_in_its_steps_log(tmp_path, monkeypatch):
    job_dir = tmp_path / "fox__finish__20260929-101010"
    (job_dir / "steps").mkdir(parents=True)
    (job_dir / "steps" / "run.log").write_text("step 3/15\n")
    (job_dir / "pid").write_text("4242")
    monkeypatch.setattr(api, "_process_group_alive", lambda pid: False)

    api._reconcile_orphaned_jobs(tmp_path)
    assert "died mid-run" in (job_dir / "steps" / "run.log").read_text()


# --- TRELLIS.2 on NVIDIA: same id, Microsoft's own code, the Mac path unchanged ---
def test_trellis_on_a_mac_keeps_the_metal_port():
    spec = api.trellis_spec(api.APPLE)
    assert spec.wrapper == api.WRAPPER and spec.interpreter == api.PYTHON
    assert spec.requires_alpha is True
    assert spec.readiness is api.setup_status


def test_trellis_on_nvidia_runs_the_cuda_generator():
    spec = api.trellis_spec(api.NVIDIA)
    assert spec.id == "trellis"
    assert spec.wrapper == api.TRELLIS_CUDA_WRAPPER and spec.wrapper.is_file()
    assert spec.interpreter == api.TRELLIS_CUDA_PYTHON
    # It mattes with our own remover, so an opaque upload is fine.
    assert spec.requires_alpha is False
    assert spec.readiness is api.cuda_setup_status


def test_cuda_args_drop_the_mac_only_flags(tmp_path):
    settings = api.validate_settings({"allow_rembg": True, "sparse_attn_backend": "mlx"})
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb",
                  settings, "trellis")
    args = api._trellis_cuda_build_args(job)
    assert args[:2] == [str(tmp_path / "in.png"), str(tmp_path / "out.glb")]
    assert "--allow-rembg" not in args and "--sparse-attn-backend" not in args
    assert "--no-save-latents" in args
    job.debug = True
    assert "--no-save-latents" not in api._trellis_cuda_build_args(job)


def test_cuda_args_are_ones_the_generator_accepts(tmp_path):
    import trellis_cuda_generate

    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb",
                  api.validate_settings({}), "trellis")
    parsed = trellis_cuda_generate.parse_args(api._trellis_cuda_build_args(job))
    assert parsed.resolution == "1024" and parsed.save_latents is False


def _cuda_install(tmp_path, monkeypatch, *, built=True, patched=True):
    vendor = tmp_path / "trellis-cuda"
    python = vendor / ".venv" / "bin" / "python"
    marker = vendor / ".i2l-build-complete"
    if built:
        python.parent.mkdir(parents=True)
        python.write_text("")
        marker.write_text("{}")
    monkeypatch.setattr(api, "TRELLIS_CUDA_PYTHON", python)
    monkeypatch.setattr(api, "TRELLIS_CUDA_MARKER", marker)
    monkeypatch.setattr(api, "weights_on_disk", lambda *a, **k: {})
    monkeypatch.setattr(api, "trellis_cuda_bria_patched", lambda *a: patched)


def test_cuda_route_is_ready_when_built_and_patched(tmp_path, monkeypatch):
    _cuda_install(tmp_path, monkeypatch)
    assert api.cuda_setup_status()["ready"] is True


def test_cuda_route_is_not_ready_without_the_bria_patch(tmp_path, monkeypatch):
    _cuda_install(tmp_path, monkeypatch, patched=False)
    status = api.cuda_setup_status()
    assert status["ready"] is False and status["bria_patched"] is False
    assert "patch_trellis_cuda_no_bria.py" in status["build"]["hint"]


def test_cuda_route_not_built_points_at_its_bootstrap(tmp_path, monkeypatch):
    _cuda_install(tmp_path, monkeypatch, built=False)
    status = api.cuda_setup_status()
    assert status["ready"] is False
    assert "bootstrap_trellis_cuda.py" in status["build"]["hint"]


def test_bria_patch_probe_reads_the_real_checkout_rule(tmp_path):
    assert api.trellis_cuda_bria_patched(tmp_path / "missing") is False


def test_the_old_mac_setup_runner_refuses_other_machines():
    ok, reason = api.setup_available(api.NVIDIA)
    assert ok is False and "Setup & Status" in reason


def test_cleanup_keeps_the_provenance_sidecar(tmp_path):
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {}, "trellis")
    for name in ("out.glb", "out.provenance.json", "out.json", "out_latents.pt", "run.log"):
        (tmp_path / name).write_text("x")
    api._cleanup_debug_files(job)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "out.glb", "out.json", "out.provenance.json"]


def test_cleanup_keeps_pixal3d_licence_record_and_camera(tmp_path):
    # Seen on a real NVIDIA pod: with Debug off, a Pixal3D model kept only its .glb. Its
    # licence record is <name>.json and its camera is <name>.svviews/, so it shipped with
    # no provenance, and Finish silently skipped Pixel Match. Both travel with the model.
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {}, "pixal3d")
    for name in ("out.glb", "out.json", "input__matted.png", "run.log"):
        (tmp_path / name).write_text("x")
    (tmp_path / "out.svviews").mkdir()
    (tmp_path / "out.svviews" / "transforms.json").write_text("{}")
    api._cleanup_debug_files(job)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.glb", "out.json", "out.svviews"]
    assert (tmp_path / "out.svviews" / "transforms.json").is_file()


def test_nvidia_trellis_hides_the_mac_only_controls():
    assert set(api.trellis_spec(api.NVIDIA).hidden_fields) == {"generate-attention", "generate-rembg"}
    assert api.trellis_spec(api.APPLE).hidden_fields == ()


def test_hidden_fields_name_real_generate_controls():
    html = (Path(api.__file__).parent / "index.html").read_text()
    for field in api.trellis_spec(api.NVIDIA).hidden_fields:
        assert f'id="{field}"' in html


def test_hidden_rows_are_not_shown_anyway_by_their_own_display_rule():
    # `.field { display: grid }` beat the hidden attribute, so the Mac-only controls the
    # page had correctly marked hidden still showed on NVIDIA. Each row class used for a
    # hidden field needs an explicit [hidden] rule.
    css = (Path(api.__file__).parent / "styles" / "generate.css").read_text()
    assert ".field[hidden]" in css and ".check[hidden]" in css


def test_drop_prompt_follows_whether_the_route_needs_a_cut_out():
    js = (Path(api.__file__).parent / "modes" / "generate.js").read_text()
    assert "function updateDropPrompt()" in js
    assert js.count("updateDropPrompt();") >= 2  # on metadata load and on backend change


def test_catalog_names_the_setup_already_running(monkeypatch):
    monkeypatch.setattr(api, "running_payload", lambda: {"backend": "trellis"})
    assert api.catalog_payload()["running_setup"] == {"backend": "trellis"}
    monkeypatch.setattr(api, "running_payload", lambda: None)
    assert api.catalog_payload()["running_setup"] is None


def test_backend_dropdown_takes_its_names_from_the_server():
    # The option text was fixed in index.html, so NVIDIA showed "TRELLIS.2 (clean port)".
    js = (Path(api.__file__).parent / "modes" / "generate.js").read_text()
    assert "function applyBackendLabels()" in js and "applyBackendLabels();" in js


def test_learned_timings_never_dirty_a_tracked_file(tmp_path, monkeypatch):
    # Seen on a real pod: the viewer rewrote the tracked viewer/generate_baseline.json after
    # a generation, and the curl installer then refused every update ("local edits to
    # tracked files"). Learned timings go to a git-ignored file; the shipped one is read-only.
    shipped = tmp_path / "shipped.json"
    shipped.write_text('{"seconds": {"decode": 100.0, "load": 10.0}}')
    learned = tmp_path / "output" / ".generate_baseline.json"
    monkeypatch.setattr(api, "BASELINE_PATH", shipped)
    monkeypatch.setattr(api, "LEARNED_BASELINE_PATH", learned)
    assert api._baseline() == {"decode": 100.0, "load": 10.0}
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {}, "trellis")
    job.stage_durations = {"decode": 42.0}
    api._update_baseline(job)
    assert shipped.read_text() == '{"seconds": {"decode": 100.0, "load": 10.0}}'
    assert api._baseline() == {"decode": 42.0, "load": 10.0}


def test_learned_baseline_lives_under_the_ignored_output_folder():
    assert api.LEARNED_BASELINE_PATH.parent == api.REPO / "output"


# --- Hunyuan3D-2.1 on NVIDIA: Tencent's own code, NVIDIA only ---------------------------
def test_hunyuan_cuda_runs_the_cuda_generator():
    spec = api.BACKENDS["hunyuan-cuda"]
    assert spec.wrapper == api.HUNYUAN_CUDA_WRAPPER and spec.wrapper.is_file()
    assert spec.interpreter == api.HUNYUAN_CUDA_PYTHON
    assert spec.requires_alpha is False  # the generator mattes with our own remover


def test_hunyuan_cuda_defaults_match_the_generator():
    import hunyuan_cuda_generate as gen

    for key, value in api.HUNYUAN_CUDA_DEFAULT_SETTINGS.items():
        assert gen.DEFAULTS[key] == value, key


@pytest.mark.parametrize("payload", [{"octree_resolution": 1024}, {"max_num_view": 12},
                                     {"paint_resolution": 1024}, {"steps": 0},
                                     {"seed": "x"}, "not a dict"])
def test_hunyuan_cuda_rejects_bad_settings(payload):
    with pytest.raises(ValueError):
        api._hunyuan_cuda_validate_settings(payload)


def test_hunyuan_cuda_args_are_ones_the_generator_accepts(tmp_path):
    import hunyuan_cuda_generate as gen

    settings = api._hunyuan_cuda_validate_settings({"octree_resolution": 512})
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", settings,
                  "hunyuan-cuda")
    args = gen.parse_args(api._hunyuan_cuda_build_args(job))
    assert args.octree_resolution == 512 and args.output == tmp_path / "out.glb"


def test_hunyuan_cuda_progress_moves_through_its_stages(tmp_path):
    """The generator's own progress lines, through the shared Hunyuan parser."""
    job = api.Job("0" * 32, tmp_path, tmp_path / "in.png", tmp_path / "out.glb", {},
                  "hunyuan-cuda")
    for line in ("loaded shape pipeline in 12.0s",
                 "shape generated in 40.0s -> /x/shape.glb",
                 "mesh loaded; paint models ready in 30.0s",
                 "paint stage done in 90.0s -> /x/out.glb",
                 "DONE in 130.0s -> /x/out.glb"):
        api._hunyuan_parse_line(job, line)
    phases = [event["phase"] for event in job.events]
    assert phases == ["shape", "shape", "paint_setup", "paint_finish", "paint_finish"]
    assert job.events[-1]["overall_pct"] == 100


def test_hunyuan_cuda_is_not_ready_with_weights_missing(tmp_path, monkeypatch):
    """Upstream would fetch them unannounced on the first run; the button must not allow it."""
    for name in ("HUNYUAN_CUDA_MARKER", "HUNYUAN_CUDA_PYTHON"):
        path = tmp_path / name
        path.write_text("")
        monkeypatch.setattr(api, name, path)
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "_dir_state", lambda path: (False, 0))
    status = api._hunyuan_cuda_readiness()
    assert status["build"]["present"] is True
    assert status["ready"] is False
    assert "--weights-only" in status["build"]["hint"]


def test_hunyuan_cuda_is_ready_when_built_and_fetched(tmp_path, monkeypatch):
    for name in ("HUNYUAN_CUDA_MARKER", "HUNYUAN_CUDA_PYTHON"):
        path = tmp_path / name
        path.write_text("")
        monkeypatch.setattr(api, name, path)
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "_dir_state", lambda path: (True, 1))
    assert api._hunyuan_cuda_readiness()["ready"] is True


def test_hunyuan_cuda_is_not_ready_before_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "HUNYUAN_CUDA_MARKER", tmp_path / "absent")
    status = api._hunyuan_cuda_readiness()
    assert status["ready"] is False and "bootstrap_hunyuan_cuda.py" in status["build"]["hint"]


def test_hunyuan_cuda_fields_exist_in_the_page():
    html = (Path(api.__file__).parent / "index.html").read_text()
    assert 'value="hunyuan-cuda"' in html and 'data-backend="hunyuan-cuda"' in html
    for field in ("hycuda-seed", "hycuda-steps", "hycuda-octree", "hycuda-views",
                  "hycuda-paint-res"):
        assert f'id="{field}"' in html, field


def test_progress_streams_tell_proxies_not_to_buffer():
    # Seen through RunPod's proxy: progress arrived in batches, minutes behind, because
    # proxies (nginx, Cloudflare, RunPod) hold event streams back unless told not to.
    sent = []

    class Fake:
        def send_response(self, code):
            sent.append(("status", code))

        def send_header(self, name, value):
            sent.append((name, value))

        def end_headers(self):
            sent.append(("end",))

    api.Handler._start_event_stream(Fake())
    assert ("Content-Type", "text/event-stream") in sent
    assert ("X-Accel-Buffering", "no") in sent and sent[-1] == ("end",)


def test_generate_tab_rechecks_readiness_when_opened():
    # Seen on a real pod: setup finished, the user opened Generate 3D, and it still said
    # "not installed yet" with Generate greyed out until a reload, because readiness was
    # only checked on page load and on a model change.
    js = (Path(api.__file__).parent / "modes" / "generate.js").read_text()
    listener = js[js.index("addEventListener('viewer:modechange'"):]
    listener = listener[:listener.index("});")]
    assert "'generate'" in listener and "refreshSetup()" in listener


def test_backends_say_whether_they_run_on_this_machine(monkeypatch):
    # The Generate dropdown offered the Mac-only Hunyuan-MLX routes on an NVIDIA pod.
    monkeypatch.setattr(api.backend_catalog, "host_platform", lambda: api.NVIDIA)
    here = {b["id"]: b["runs_here"] for b in api.backends_payload()["backends"]}
    assert here["hunyuan-cuda"] is True and here["hunyuan-mlx-xiong"] is False
    monkeypatch.setattr(api.backend_catalog, "host_platform", lambda: api.APPLE)
    here = {b["id"]: b["runs_here"] for b in api.backends_payload()["backends"]}
    assert here["hunyuan-mlx-xiong"] is True and here["hunyuan-cuda"] is False


def test_dropdown_hides_routes_that_do_not_run_here():
    js = (Path(api.__file__).parent / "modes" / "generate.js").read_text()
    assert "option.hidden = meta.runs_here === false" in js


def test_sign_in_answers_with_status_or_a_plain_error(monkeypatch):
    monkeypatch.setattr(api.hf_api, "sign_in", lambda token: (
        {"signed_in": True, "user": "ada", "repos": []} if token == "good"
        else {"error": "Hugging Face refused that token."}))
    assert api.hf_sign_in_response({"token": "good"}) == (
        200, {"signed_in": True, "user": "ada", "repos": []})
    code, body = api.hf_sign_in_response({"token": "nope"})
    assert code == 422 and "error" in body
    assert api.hf_sign_in_response("not a dict")[0] == 400


def test_blender_install_is_refused_when_it_cannot_or_should_not_run():
    ok = {"blender_installable": True}
    assert api.blender_install_refusal(ok, generating=False, setting_up=False) is None
    assert api.blender_install_refusal(ok, generating=True, setting_up=False)[0] == 409
    assert api.blender_install_refusal(ok, generating=False, setting_up=True)[0] == 409
    code, message = api.blender_install_refusal(
        {"blender_installable": False}, generating=False, setting_up=False)
    assert code == 409 and "blender.org" in message


def test_blender_install_runs_the_bootstrap_with_yes():
    command = api.blender_install_command()
    assert command[-2].endswith("bootstrap_blender.py") and command[-1] == "--yes"


def test_a_setup_run_streams_its_progress_at_the_address_it_hands_out():
    # Found on a pod 2026-10-02: Install Blender (and the Mac Run setup button) hand out
    # /api/setup/run/<id>/events, but the route only matched four path parts, so the
    # page's progress feed was a 404 since the day it was written.
    import http.server
    import threading
    import urllib.request
    import uuid

    run_id = uuid.uuid4().hex
    run = api.SetupRun(run_id)
    run.emit({"phase": "setup", "message": "Downloading blender..."})
    run.status = "done"
    run.emit({"phase": "setup_done", "status": "done", "message": "exit 0"})
    api.SETUP_RUNS[run_id] = run
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/setup/run/{run_id}/events"
        # A live feed: read events until the last one, as the page's EventSource does,
        # rather than waiting for the connection to close.
        lines = []
        with urllib.request.urlopen(url, timeout=30) as response:
            assert response.status == 200
            for raw in response:
                lines.append(raw.decode())
                if '"setup_done"' in lines[-1]:
                    break
        body = "".join(lines)
        assert "Downloading blender" in body and '"setup_done"' in body
    finally:
        server.shutdown()
        server.server_close()
        api.SETUP_RUNS.pop(run_id, None)
