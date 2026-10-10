"""Tests for the weight-download runner.

The parts worth testing are the ones a user reads when something goes wrong: the progress
line, the stall verdict, and the sentence shown instead of "exited with code 1". The
subprocess itself is not exercised here; it downloads gigabytes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Plain import for the same reason `test_backend_catalog.py` uses one: `generate_api`
# imports `download_api`, so a hand-loaded second copy leaves the two holding different
# objects and a patch here never reaches the code being tested.
import download_api as dl


def test_progress_bars_are_stripped_to_their_last_frame():
    # tqdm redraws with \r, so everything before the last frame is a stale repaint that
    # would otherwise pile up in the browser log.
    assert dl.strip_ansi("10%|##        |\r50%|#####     |\r90%|######### |") == "90%|######### |"
    assert dl.strip_ansi("\x1b[32mdone\x1b[0m") == "done"
    assert dl.strip_ansi("  plain line  ") == "plain line"


def test_rate_needs_two_samples_before_it_claims_anything():
    assert dl.rate_and_eta([], 100) == (None, None)
    assert dl.rate_and_eta([(0.0, 0)], 100) == (None, None)


def test_rate_and_eta_are_measured_over_the_window():
    rate, eta = dl.rate_and_eta([(0.0, 0), (10.0, 1000)], remaining=2000)
    assert rate == 100.0
    assert eta == 20.0


def test_a_window_with_no_growth_reports_no_eta():
    rate, eta = dl.rate_and_eta([(0.0, 500), (10.0, 500)], remaining=2000)
    assert rate == 0.0
    assert eta is None


def _backend():
    return dl.BY_ID["pixal3d"]


def test_progress_never_claims_completion_before_the_process_exits():
    # The expected size is an estimate, so a download that overshoots it must not show
    # 100% while the process is still running; only a clean exit says done.
    event = dl.describe_progress(_backend(), present=10 ** 13, rate=1.0, eta=1.0, stalled=False)
    assert event["overall_pct"] == 99


def test_a_stall_is_named_rather_than_shown_as_slow_progress():
    event = dl.describe_progress(_backend(), present=1000, rate=0.0, eta=None, stalled=True)
    assert "stalled" in event["detail"]
    assert event["stalled"] is True


def test_progress_states_both_numbers_not_just_a_percentage():
    # The percentage inherits the catalogue's approximate sizes; the raw bytes do not.
    event = dl.describe_progress(_backend(), present=2 * 1024 ** 3, rate=None, eta=None,
                                 stalled=False)
    assert "2.0 GB of" in event["detail"]


def test_before_any_weights_arrive_the_step_is_shown_not_zero_bytes():
    # Pixal3D fetches a 674 MB build before its weights; "0 B of 8.4 GB" read as stuck.
    event = dl.describe_progress(_backend(), present=0, rate=None, eta=None, stalled=False,
                                 step="Downloading trellis-cuda12-linux-x64.tar.gz (674 MB)...")
    assert event["detail"] == "Downloading trellis-cuda12-linux-x64.tar.gz (674 MB)..."
    assert "0 B" not in event["detail"]


def test_once_weights_arrive_bytes_win_over_the_step():
    event = dl.describe_progress(_backend(), present=2 * 1024 ** 3, rate=None, eta=None,
                                 stalled=False, step="Fetching weights")
    assert "2.0 GB of" in event["detail"]


def test_no_stall_is_claimed_before_the_first_weight_byte():
    # A slow connection spends more than the stall window on the build download alone.
    assert dl.is_stalled(present=0, idle_seconds=10 * dl.STALL_SECONDS) is False
    assert dl.is_stalled(present=1, idle_seconds=dl.STALL_SECONDS + 1) is True
    assert dl.is_stalled(present=1, idle_seconds=dl.STALL_SECONDS - 1) is False


def test_a_gated_repo_is_explained_as_a_login_problem():
    message = dl._explain(1, ["Traceback", "401 Client Error: Unauthorized for url"])
    assert "login" in message and "retry" in message


def test_a_full_disk_is_explained_as_a_full_disk():
    assert "disk space" in dl._explain(1, ["OSError: [Errno 28] No space left on device"])


def test_a_network_failure_is_explained_as_one():
    assert "network" in dl._explain(1, ["ConnectionError: Max retries exceeded"])


def test_an_unrecognised_failure_still_quotes_the_last_line():
    message = dl._explain(2, ["something specific went wrong"])
    assert "code 2" in message and "something specific went wrong" in message


def test_every_catalogued_backend_either_has_a_command_or_says_it_has_none():
    # A Download button that 500s is worse than one that explains itself.
    for backend_id in dl.BY_ID:
        if backend_id in dl.COMMANDS:
            continue
        try:
            dl.start(backend_id)
        except RuntimeError as exc:
            assert "no automated setup" in str(exc)
        else:
            raise AssertionError(f"{backend_id} started without a command")


def test_an_unknown_backend_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        dl.start("not-a-backend")


def test_hunyuan_setup_builds_both_venvs_not_only_the_weights():
    """Weights alone left the paint venv unbuilt: "done", yet "not installed" (issue #97)."""
    command = dl.COMMANDS["hunyuan_xiong"]
    assert Path(command[1]).name == "bootstrap_hunyuan_xiong.py" and "--yes" in command


def test_removal_refuses_paths_outside_the_repo_and_cache(tmp_path, monkeypatch):
    """The guard that stops a bad catalogue entry deleting something else.

    Nothing in the shipped catalogue points outside, which is exactly why this needs a
    test: the day one does, it must fail loudly rather than run `rmtree` on it.
    """
    import dataclasses

    import pytest
    stray = tmp_path / "somewhere-else"
    stray.mkdir()
    (stray / "w.bin").write_bytes(b"x")
    real = dl.BY_ID["pixal3d"]
    # WeightSet is frozen, so build a stand-in rather than mutating the shipped one.
    rogue = dataclasses.replace(
        real, weights=(dataclasses.replace(real.weights[0], path=stray),),
    )
    monkeypatch.setitem(dl.BY_ID, "rogue", rogue)

    with pytest.raises(RuntimeError, match="refusing to delete"):
        dl.remove("rogue")
    assert stray.is_dir(), "the guard let a delete through"


def test_paths_inside_the_repo_or_cache_are_allowed():
    assert dl._inside_known_roots(dl.REPO / "vendor" / "x") is True
    assert dl._inside_known_roots(dl.HF_HUB_DIR / "models--a--b") is True
    assert dl._inside_known_roots(Path("/etc")) is False


def test_removing_an_unknown_backend_is_rejected():
    import pytest
    with pytest.raises(KeyError):
        dl.remove("not-a-backend")


def test_removal_is_refused_while_that_backend_is_downloading(monkeypatch):
    import pytest
    run = dl.DownloadRun(dl.BY_ID["pixal3d"])
    run.status = "running"
    monkeypatch.setitem(dl.DOWNLOADS, "pixal3d", run)
    with pytest.raises(RuntimeError, match="downloading right now"):
        dl.remove("pixal3d")


# --- Starting a command that is not there -----------------------------------------------
# A Windows user reported "[WinError 2] The system cannot find the file specified" after
# logging into Hugging Face. `Popen` had been handed `.venv/bin/python`, which on Windows
# is spelled `.venv\\Scripts\\python.exe`, and the raw OSError went straight to the browser.


def test_a_missing_venv_is_explained_rather_than_reported_as_errno_2(tmp_path):
    absent = tmp_path / "shape" / ".venv" / "bin" / "python"
    message = dl._missing_executable(str(absent))
    assert message is not None
    assert "uv sync" in message
    assert str(tmp_path / "shape") in message


def test_a_missing_program_names_itself():
    assert "not-a-real-program" in dl._missing_executable("not-a-real-program")


def test_a_command_that_exists_is_not_second_guessed(tmp_path):
    assert dl._missing_executable(sys.executable) is None
    assert dl._missing_executable("sh") is None or os.name == "nt"


def test_an_unsupported_machine_is_refused_before_anything_downloads(monkeypatch):
    """The refusal belongs here too: the API is what spends the bandwidth."""
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "host_platform", lambda: "other")
    with pytest.raises(RuntimeError) as raised:
        dl.start("hunyuan_xiong")
    assert "Apple Silicon" in str(raised.value)
    assert "Nothing has been downloaded" in str(raised.value)
    assert "hunyuan_xiong" not in dl.DOWNLOADS


def test_trellis_on_nvidia_runs_the_cuda_bootstrap_with_yes():
    command = dl.command_for("trellis", host=dl.NVIDIA)
    assert command[1].endswith("bootstrap_trellis_cuda.py")
    assert command[-1] == "--yes"


def test_trellis_on_a_mac_still_runs_the_metal_bootstrap():
    command = dl.command_for("trellis", host="apple-silicon")
    assert command == dl.COMMANDS["trellis"]
    assert command[1].endswith("bootstrap_trellis_space_macos.py")


def test_a_route_without_a_host_command_uses_the_shared_one():
    assert dl.command_for("pixal3d", host=dl.NVIDIA) == dl.COMMANDS["pixal3d"]
    assert dl.command_for("pixal3d", rebuild=True) == dl.REBUILDS["pixal3d"]
    assert dl.command_for("trellis", rebuild=True) is None


def test_the_command_follows_the_machine(monkeypatch):
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.NVIDIA)
    assert dl.command_for("trellis")[1].endswith("bootstrap_trellis_cuda.py")
    assert dl.building_label("trellis") == "building the CUDA version"
    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.APPLE)
    assert dl.building_label("trellis") == "building the Metal port"


def test_a_setup_run_on_nvidia_holds_the_cuda_command(monkeypatch):
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "host_platform", lambda: backend_catalog.NVIDIA)
    run = dl.DownloadRun(backend_catalog.BY_ID["trellis"])
    assert run.command[1].endswith("bootstrap_trellis_cuda.py")


def test_every_route_claiming_automated_setup_actually_has_a_command():
    """The flag and the command table are two halves of one fact; a drift shows the user
    a button that throws "has no automated setup yet" only after they click it."""
    import backend_catalog

    for backend in backend_catalog.CATALOG:
        has_command = backend.id in dl.COMMANDS
        assert backend.automated_setup is has_command, backend.id


def test_the_background_remover_is_removed_as_one_file_leaving_its_folder(tmp_path, monkeypatch):
    """BiRefNet-lite shares rembg's folder with u2net: Remove takes the file, not the folder."""
    import dataclasses

    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    lite = tmp_path / "birefnet-general-lite.onnx"
    lite.write_bytes(b"12345")
    (tmp_path / "u2net.onnx").write_bytes(b"keep me")
    real = dl.BY_ID["matte"]
    entry = dataclasses.replace(real, weights=(dataclasses.replace(real.weights[0], path=lite),))
    monkeypatch.setitem(dl.BY_ID, "matte", entry)

    result = dl.remove("matte")
    assert result["freed_bytes"] == 5
    assert not lite.exists()
    assert (tmp_path / "u2net.onnx").read_bytes() == b"keep me"


def test_rembgs_model_folder_is_a_known_place(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", str(tmp_path))
    assert dl._inside_known_roots(tmp_path / "birefnet-general-lite.onnx") is True


def test_the_remover_download_asks_nothing_twice():
    """--yes because the Setup & Status dialog already asked."""
    assert dl.COMMANDS["matte"][-1] == "--yes"
    assert dl.COMMANDS["matte"][1].endswith("bootstrap_matte.py")


def _pixal3d_tree(tmp_path, patched: bool, cli_newer: bool):
    source = tmp_path / "flow_runner.cpp"
    cli = tmp_path / "trellis-cli"
    source.write_text("i2l_steps" if patched else "plain")
    cli.write_text("binary")
    older, newer = 1_000_000, 2_000_000
    os.utime(source, (older, older) if cli_newer else (newer, newer))
    os.utime(cli, (newer, newer) if cli_newer else (older, older))
    return source, cli


def test_an_unpatched_pixal3d_build_is_offered_a_rebuild(tmp_path):
    source, cli = _pixal3d_tree(tmp_path, patched=False, cli_newer=True)
    assert "not patched" in dl.rebuild_reason("pixal3d", source, cli)


def test_a_patch_not_yet_compiled_in_is_offered_a_rebuild(tmp_path):
    source, cli = _pixal3d_tree(tmp_path, patched=True, cli_newer=False)
    assert "older" in dl.rebuild_reason("pixal3d", source, cli)


def test_a_patched_current_build_is_left_alone(tmp_path):
    source, cli = _pixal3d_tree(tmp_path, patched=True, cli_newer=True)
    assert dl.rebuild_reason("pixal3d", source, cli) is None


def test_nothing_installed_or_a_prebuilt_is_never_offered_a_rebuild(tmp_path):
    """No binary is Set up's job; no source means a prebuilt, which cannot be patched."""
    source, cli = _pixal3d_tree(tmp_path, patched=False, cli_newer=True)
    assert dl.rebuild_reason("pixal3d", source, tmp_path / "missing") is None
    assert dl.rebuild_reason("pixal3d", tmp_path / "missing", cli) is None


def test_backends_without_a_rebuild_command_are_never_offered_one(tmp_path):
    source, cli = _pixal3d_tree(tmp_path, patched=False, cli_newer=True)
    assert dl.rebuild_reason("hunyuan_xiong", source, cli) is None


def test_the_rebuild_command_recompiles_without_downloading_or_asking():
    command = dl.REBUILDS["pixal3d"]
    assert {"--build-only", "--rebuild", "--yes"} <= set(command)
    assert "--weights-only" not in command


def test_a_rebuild_runs_the_rebuild_command_not_the_setup_one():
    import backend_catalog

    run = dl.DownloadRun(backend_catalog.BY_ID["pixal3d"], rebuild=True)
    assert run.command == dl.REBUILDS["pixal3d"]
    assert dl.DownloadRun(backend_catalog.BY_ID["pixal3d"]).command == dl.COMMANDS["pixal3d"]


def test_a_rebuild_is_refused_for_a_backend_that_has_none(monkeypatch):
    import backend_catalog

    monkeypatch.setattr(backend_catalog, "host_platform", lambda: "apple")
    with pytest.raises(RuntimeError, match="no automated rebuild"):
        dl.start("hunyuan_xiong", rebuild=True)


def test_the_catalog_carries_a_rebuild_reason_only_where_the_backend_runs():
    import generate_api

    catalog = {"backends": [{"id": "pixal3d", "supported_here": True},
                            {"id": "sf3d", "supported_here": False}]}
    out = generate_api.with_rebuild_reasons(catalog, reason=lambda backend_id: "stale")
    assert out["backends"][0]["rebuild_reason"] == "stale"
    assert out["backends"][1]["rebuild_reason"] is None


def test_a_page_loaded_mid_setup_can_find_the_run_again(monkeypatch):
    # A setup runs 30-60 minutes; a refreshed or reopened Setup page must reattach to it
    # instead of showing idle "Set up" buttons that refuse with "already running".
    monkeypatch.setattr(dl, "DOWNLOADS", {})
    assert dl.running_payload() is None
    run = dl.DownloadRun(dl.BY_ID["pixal3d"])
    run.status = "running"
    dl.DOWNLOADS["pixal3d"] = run
    assert dl.running_payload() == {
        "backend": "pixal3d", "rebuild": False,
        "events_url": "/api/setup/pixal3d/events",
    }
    run.status = "done"
    assert dl.running_payload() is None


def test_a_download_timeout_buried_above_a_traceback_is_still_explained():
    # Seen on a real pod: uv's "Failed to download ... operation timed out" sat above a
    # 20-line Python traceback, so the user got the traceback's last line instead.
    log = ["  x Failed to download `nvidia-cudnn-cu12==9.10.2.21`",
           "  |-> Request failed after 3 retries", "  `-> operation timed out",
           "Traceback (most recent call last):"] + [f"  frame {i}" for i in range(20)] + [
           "subprocess.CalledProcessError: Command '[uv, pip, install]' returned 1."]
    message = dl._explain(1, log)
    assert "network" in message and "Set up again" in message


def test_finished_nvidia_trellis_setup_does_not_promise_a_later_download(monkeypatch):
    monkeypatch.setattr(dl, "_this_host", lambda: dl.NVIDIA)
    assert dl.done_detail(dl.BY_ID["trellis"], present=16 * 1024 ** 3).startswith("done ·")
    monkeypatch.setattr(dl, "_this_host", lambda: "apple")
    assert "first generation run" in dl.done_detail(dl.BY_ID["trellis"], present=0)


def test_hunyuan_cuda_sets_up_with_its_own_bootstrap_and_says_cuda():
    command = dl.command_for("hunyuan-cuda", host=dl.NVIDIA)
    assert command[1].endswith("bootstrap_hunyuan_cuda.py") and command[-1] == "--yes"
    assert dl.building_label("hunyuan-cuda", host=dl.NVIDIA) == "building the CUDA version"


def test_files_already_on_disk_do_not_make_a_build_look_stalled():
    # Seen on a real pod: the shared background remover (214 MB) was already there from
    # another route, so Hunyuan's rasterizer compile read "stalled" after 90 s. Only bytes
    # this setup has fetched count, and until there are some the step is shown.
    backend = dl.BY_ID["hunyuan-cuda"]
    line = dl.describe_progress(backend, present=214 * 1024 ** 2, rate=None, eta=None,
                                stalled=dl.is_stalled(0, 10 * dl.STALL_SECONDS),
                                step="Building custom-rasterizer", fetched=0)
    assert line["detail"] == "Building custom-rasterizer" and line["stalled"] is False


def _sharing(tmp_path, monkeypatch):
    """Two routes and the remover's own card, all declaring one remover file."""
    import dataclasses

    shared = tmp_path / "birefnet-lite.onnx"
    shared.write_bytes(b"x" * 10)
    own = tmp_path / "own-weights"
    own.mkdir()
    (own / "w.bin").write_bytes(b"y" * 20)
    real = dl.BY_ID["pixal3d"]
    lite = dataclasses.replace(real.weights[0], path=shared)
    mine = dataclasses.replace(real.weights[0], path=own)
    monkeypatch.setitem(dl.BY_ID, "route-a", dataclasses.replace(real, id="route-a", weights=(mine, lite)))
    monkeypatch.setitem(dl.BY_ID, "route-b", dataclasses.replace(real, id="route-b", weights=(lite,)))
    monkeypatch.setitem(dl.BY_ID, "remover", dataclasses.replace(
        dl.BY_ID["matte"], id="remover", weights=(lite,)))
    monkeypatch.setattr(dl, "_inside_known_roots", lambda path: True)
    return shared, own


def test_removing_a_route_keeps_files_another_route_uses(tmp_path, monkeypatch):
    # Seen on a real pod: Pixal3D offered "Delete 213.6 MB of weights", and that was the
    # background remover TRELLIS and Hunyuan cut out with.
    shared, own = _sharing(tmp_path, monkeypatch)
    result = dl.remove("route-a")
    assert not own.exists() and shared.is_file()
    assert result["freed_bytes"] == 20


def test_the_removers_own_card_can_still_remove_it(tmp_path, monkeypatch):
    shared, _ = _sharing(tmp_path, monkeypatch)
    dl.remove("remover")
    assert not shared.exists()
