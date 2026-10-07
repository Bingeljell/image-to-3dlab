"""Tests for the weights catalogue behind the onboarding screen.

`AGENTS.md` forbids downloading weights before the user has confirmed what they want, and
the screen that asks them is only as honest as the numbers it shows. Both bugs below were
real, found the day the module was written, and both make the progress readout lie rather
than crash, which is why they get tests rather than a comment.
"""

from __future__ import annotations

import sys
from pathlib import Path

# A plain import, which `tests/conftest.py` makes possible by putting `viewer/` on the
# path. Deliberately not a hand-load: `download_api`, `generate_api` and
# `audit_model_weights` all import this module normally, and a second copy under the same
# name is how a monkeypatch here stops reaching the code under test.
import backend_catalog as bc
import pytest


def test_every_backend_states_a_licence_and_links_to_it():
    # The licence is the user's business, but we owe them the name and a way to read it.
    for backend in bc.CATALOG:
        assert backend.license_name, backend.id
        assert backend.license_url.startswith("https://"), backend.id


def test_the_hunyuan_territorial_restriction_is_surfaced():
    # Not editorialising, but not hiding it either: downloading these in a restricted
    # region is a harm we would be causing.
    hunyuan = bc.BY_ID["hunyuan_xiong"]
    assert hunyuan.caveat is not None
    for region in ("EU", "UK", "South Korea"):
        assert region in hunyuan.caveat


def test_exactly_one_backend_is_recommended():
    recommended = [b for b in bc.CATALOG if b.rank == 1]
    assert len(recommended) == 1
    assert recommended[0].id == "pixal3d"


def test_no_weight_set_contains_another(tmp_path):
    """Nested paths double-count, and a backend then reports 200% downloaded.

    Hit for real: the bootstrap moves BiRefNet *into* `pixal3d-sv/`, so a second entry
    pointed at the parent counted the whole backend twice.
    """
    for backend in bc.CATALOG:
        paths = [w.path for w in backend.weights]
        for outer in paths:
            for inner in paths:
                if outer is inner:
                    continue
                assert outer not in inner.parents, f"{backend.id}: {inner} sits inside {outer}"


def test_symlinked_files_are_counted_once(tmp_path):
    """The Hugging Face cache links `snapshots/` at `blobs/`, so following both doubles
    every byte and a 15 GB backend reports 30 GB."""
    blobs, snaps = tmp_path / "blobs", tmp_path / "snapshots"
    blobs.mkdir()
    snaps.mkdir()
    (blobs / "weight.bin").write_bytes(b"x" * 1000)
    (snaps / "weight.bin").symlink_to(blobs / "weight.bin")

    present, total = bc._dir_state(tmp_path)
    assert present is True
    assert total == 1000, "the symlink was followed and counted a second time"


def test_links_to_a_shared_blob_store_outside_the_folder_are_counted(tmp_path):
    """Newer huggingface_hub keeps the bytes in a store shared across repos and only
    links to them from `models--*`. Skipping links then reported 13.4 GB as 120 B."""
    store = tmp_path / "hub" / "blobs" / "ab"
    store.mkdir(parents=True)
    (store / "hash1").write_bytes(b"x" * 1000)
    repo = tmp_path / "hub" / "models--org--name"
    snaps = repo / "snapshots" / "rev"
    snaps.mkdir(parents=True)
    (snaps / "weight.bin").symlink_to(store / "hash1")
    (snaps / "same_again.bin").symlink_to(store / "hash1")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_bytes(b"r" * 40)

    present, total = bc._dir_state(repo)
    assert present is True
    assert total == 1040, "the linked blob was skipped, or counted once per link"


def test_a_dangling_link_is_skipped(tmp_path):
    (tmp_path / "gone.bin").symlink_to(tmp_path / "nowhere")
    assert bc._dir_state(tmp_path) == (False, 0)


def test_a_missing_directory_is_missing_not_an_error(tmp_path):
    assert bc._dir_state(tmp_path / "absent") == (False, 0)


def test_an_interrupted_download_reads_as_partial_not_ready():
    # A directory that exists but holds a tenth of the bytes is a failed fetch. Calling it
    # ready is how someone debugs a backend that was never fully downloaded.
    expected = 10 * bc.GB
    w = bc._weights_state
    assert w([{"bytes_present": expected, "bytes_expected": expected}]) == "ready"
    assert w([{"bytes_present": expected // 10, "bytes_expected": expected}]) == "partial"
    assert w([{"bytes_present": 0, "bytes_expected": expected}]) == "missing"


def test_a_slightly_short_download_still_counts_as_ready():
    # The size constants are measured and approximate, so the floor has to tolerate drift;
    # an exact comparison made a fully installed TRELLIS.2 report "partial" at 92%.
    expected = 10 * bc.GB
    assert bc._weights_state(
        [{"bytes_present": int(expected * 0.9), "bytes_expected": expected}]) == "ready"


def test_onboarding_is_needed_only_when_nothing_is_ready(monkeypatch, tmp_path):
    status = bc.catalog_status()
    assert status["needs_onboarding"] is (status["ready_count"] == 0)

    # A fresh clone: no weights *and* nothing built. Weights alone are not enough to
    # decide this, since a built TRELLIS with no weights is usable.
    import dataclasses
    absent = tmp_path / "nothing-here"
    monkeypatch.setattr(bc, "_dir_state", lambda path: (False, 0))
    monkeypatch.setattr(bc, "CATALOG", tuple(
        dataclasses.replace(b, build_probes=(absent,)) for b in bc.CATALOG
    ))
    # Pinned to a supported host: on any other machine every backend reports
    # "unsupported", which is a different question from "not installed yet".
    empty = bc.catalog_status(host=bc.APPLE)
    assert empty["needs_onboarding"] is True
    assert empty["ready_count"] == 0
    # NVIDIA-only routes (Hunyuan3D-2.1) are "unsupported" on a Mac, not "missing".
    here = [b for b in empty["backends"] if b["supported_here"]]
    assert here and all(b["state"] == "missing" for b in here)
    # "build" where the viewer can do it, "manual" where it cannot. Both mean "not yet";
    # neither means "nothing to offer", which is what an unsupported machine gets.
    assert all(b["action"] in {"build", "manual"} for b in here)
    assert all(b["action"] == ("build" if b["automated_setup"] else "manual") for b in here)


def test_backends_are_listed_best_first():
    """Ranked routes come first, in order; unranked ones follow.

    `rank` has always been optional on Backend, but nothing was unranked until the
    text-to-image route arrived. It is not competing for "which 3D backend should a
    newcomer pick", so it has no rank and sorts to the end.
    """
    ranks = [b["rank"] for b in bc.catalog_status()["backends"]]
    ranked = [r for r in ranks if r is not None]
    assert ranked == sorted(ranked)
    assert ranks == ranked + [None] * (len(ranks) - len(ranked))


def test_an_unranked_backend_is_never_the_recommendation():
    for backend in bc.catalog_status()["backends"]:
        if backend["rank"] is None:
            assert backend["recommended"] is False


def test_image_and_3d_routes_are_distinguishable():
    """The Setup page groups them, and 'kind' is how it knows which is which."""
    kinds = {b["id"]: b["kind"] for b in bc.catalog_status()["backends"]}
    assert kinds["qwen-image"] == "image"
    assert kinds["matte"] == "tool"  # a helper every route uses, not a route itself
    assert all(k in {"image", "3d", "tool"} for k in kinds.values())
    assert "3d" in kinds.values()


def test_the_background_remover_is_one_announced_file(tmp_path, monkeypatch):
    entry = next(b for b in bc.catalog_status()["backends"] if b["id"] == "matte")
    (weights,) = entry["weights"]
    assert weights["bytes_expected"] == 224_005_088
    assert weights["source_url"].startswith("https://github.com/danielgatis/rembg/")
    assert weights["path"].endswith("birefnet-general-lite.onnx")


def test_a_single_file_counts_as_present_by_its_own_size(tmp_path):
    model = tmp_path / "model.onnx"
    assert bc._dir_state(model) == (False, 0)
    model.write_bytes(b"12345")
    assert bc._dir_state(model) == (True, 5)


def test_the_image_route_names_its_licence_and_says_the_pictures_are_yours():
    """Qwen confirmed outputs are the user's; the card names the licence, with no warning box."""
    entry = next(b for b in bc.catalog_status()["backends"] if b["id"] == "qwen-image")
    assert entry["license"]["name"] == "Qwen Research License"
    assert not entry.get("caveat")
    assert "yours" in entry["tradeoff"]


def test_sizes_are_stated_before_anything_is_fetched():
    for backend in bc.catalog_status()["backends"]:
        assert backend["bytes_expected"] > 0, backend["id"]
        assert backend["human_expected"].endswith(("MB", "GB")), backend["id"]


def test_human_bytes_reads_like_a_download_dialog():
    assert bc.human_bytes(0) == "0 B"
    assert bc.human_bytes(92 * 1024 ** 2) == "92.0 MB"
    assert bc.human_bytes(int(8.4 * bc.GB)) == "8.4 GB"


def test_trellis_setup_is_a_build_not_a_download():
    """Its bootstrap clones, patches and compiles; the weights come on first generation.

    Calling that a download makes the confirmation lie and makes byte progress meaningless
    — a healthy hour-long compile reported no growth and would have read as stalled.
    """
    assert bc.BY_ID["trellis"].setup_fetches_weights is False
    assert bc.BY_ID["pixal3d"].setup_fetches_weights is True
    assert bc.BY_ID["hunyuan_xiong"].setup_fetches_weights is True


def test_the_flag_reaches_the_browser():
    trellis = next(b for b in bc.catalog_status()["backends"] if b["id"] == "trellis")
    assert trellis["setup_fetches_weights"] is False


def test_a_built_trellis_with_no_weights_is_ready_not_missing():
    """Its bootstrap installs the code and fetches nothing; the weights come on first run.

    Reporting it as missing offered a Set up button that re-ran a finished bootstrap,
    which then died on its own already-applied patches (hit for real 2026-09-21 while
    testing against an empty HF_HOME).
    """
    none_present = [{"bytes_present": 0, "bytes_expected": 10 * bc.GB}]
    assert bc._state(none_present, built=True, setup_fetches=False) == "ready"
    assert bc._action(none_present, built=True, setup_fetches=False) == "none"


def test_an_unbuilt_backend_is_offered_a_build_whatever_its_weights():
    weights = [{"bytes_present": 10 * bc.GB, "bytes_expected": 10 * bc.GB}]
    assert bc._state(weights, built=False, setup_fetches=True) == "missing"
    assert bc._action(weights, built=False, setup_fetches=True) == "build"


def test_a_built_backend_missing_weights_is_offered_the_download():
    empty = [{"bytes_present": 0, "bytes_expected": 10 * bc.GB}]
    half = [{"bytes_present": 5 * bc.GB, "bytes_expected": 10 * bc.GB}]
    full = [{"bytes_present": 10 * bc.GB, "bytes_expected": 10 * bc.GB}]
    assert bc._action(empty, built=True, setup_fetches=True) == "download"
    assert bc._action(half, built=True, setup_fetches=True) == "resume"
    assert bc._action(full, built=True, setup_fetches=True) == "none"


def test_a_backend_with_no_declared_probes_is_never_called_unbuilt():
    assert all(b.build_present or b.build_probes for b in bc.CATALOG)


# --- Which machines a backend runs on ---------------------------------------------------
# Reported from a Windows user who got "[WinError 2] The system cannot find the file
# specified" after logging into Hugging Face. The viewer had offered them a Set up button
# for an MLX backend that cannot exist on their machine, and the failure surfaced as a raw
# OS error. NVIDIA support is coming, so support is per-backend data, not a Mac check.


def test_a_virtualenv_interpreter_is_named_the_way_this_os_names_it(monkeypatch):
    project = Path("/somewhere/shape")
    monkeypatch.setattr(bc.os, "name", "posix")
    assert bc.venv_python(project).as_posix().endswith(".venv/bin/python")
    monkeypatch.setattr(bc.os, "name", "nt")
    assert bc.venv_python(project).as_posix().endswith(".venv/Scripts/python.exe")


def test_every_backend_says_which_machines_it_runs_on():
    for backend in bc.CATALOG:
        assert backend.runs_on, backend.id
        assert all(p in bc.PLATFORM_LABELS for p in backend.runs_on), backend.id


def test_an_unsupported_machine_is_never_offered_a_download():
    """The whole point: no button, no bytes, and a sentence saying why."""
    for backend in bc.catalog_status(host="other")["backends"]:
        assert backend["supported_here"] is False, backend["id"]
        assert backend["state"] == "unsupported", backend["id"]
        assert backend["action"] == "none", backend["id"]
        assert backend["platform_note"], backend["id"]


def test_the_page_can_tell_a_wrong_machine_from_an_empty_one():
    """"Nothing installed" and "nothing installable" look identical in a list of states."""
    assert bc.catalog_status(host="other")["host"]["any_backend_runs_here"] is False
    assert bc.catalog_status(host=bc.APPLE)["host"]["any_backend_runs_here"] is True


def test_adding_a_cuda_route_is_a_one_string_change():
    """The gate must not be a Mac check, because NVIDIA support is coming.

    Declaring the platform on one backend is the whole change; nothing else should need
    editing for it to become installable on that machine.
    """
    import dataclasses
    cuda_pixal3d = dataclasses.replace(bc.BY_ID["pixal3d"], runs_on=(bc.APPLE, bc.NVIDIA))
    assert cuda_pixal3d.runs_here(bc.NVIDIA) is True
    assert cuda_pixal3d.describe(bc.NVIDIA)["state"] != "unsupported"
    assert bc.BY_ID["hunyuan_xiong"].runs_here(bc.NVIDIA) is False


# --- One catalogue, every route ----------------------------------------------------------
#
# The catalogue and the Generate tab's backend table grew apart: the tab offered sf3d and
# hunyuan-mlx, which the catalogue had never heard of, and spelled the Xiong route
# `hunyuan-mlx-xiong` where the catalogue said `hunyuan_xiong`. Nobody noticed until
# `/api/setup?backend=qwen-image` answered "unknown backend" for a route that plainly
# exists. These tests are the guard: one registry of routes, one way to look a route up.


def test_every_route_the_generate_tab_offers_is_in_the_catalogue():
    """The drift that started this: two lists of backends, only one of them complete."""
    offered = {"trellis", "sf3d", "hunyuan-mlx", "hunyuan-mlx-xiong", "pixal3d"}
    for backend_id in offered:
        assert bc.resolve(backend_id) is not None, backend_id


def test_a_route_is_found_by_either_spelling():
    assert bc.resolve("hunyuan-mlx-xiong") is bc.BY_ID["hunyuan_xiong"]
    assert bc.resolve("hunyuan_xiong") is bc.BY_ID["hunyuan_xiong"]


def test_an_id_nobody_uses_is_still_unknown():
    assert bc.resolve("nonesuch") is None
    assert bc.readiness("nonesuch") is None


def test_the_image_route_answers_the_readiness_question_like_any_other():
    """The reported bug: the image route was a backend everywhere except here."""
    payload = bc.readiness("qwen-image")
    assert payload is not None
    assert payload["backend"] == "qwen-image"
    assert set(payload) >= {"build", "weights", "missing_weights", "ready", "warning"}
    assert isinstance(payload["build"]["present"], bool)


def test_readiness_names_missing_weights_by_label_not_by_path():
    """A path is not an answer to "what is missing"; the label is what the card shows."""
    payload = bc.readiness("qwen-image")
    labels = {w["label"] for w in payload["weights"].values()}
    assert set(payload["missing_weights"]) <= labels


def test_a_route_that_cannot_run_here_is_never_reported_ready():
    payload = bc.readiness("pixal3d", host="other")
    assert payload["ready"] is False
    assert payload["build"]["hint"]


def test_there_is_only_ever_one_catalogue_module():
    """Two copies under one name is how a monkeypatch lands on the wrong object."""
    import backend_catalog

    assert backend_catalog is bc
    assert sys.modules["backend_catalog"] is bc


def test_a_mac_hunyuan_port_points_nvidia_at_the_nvidia_route():
    """The MLX Hunyuan ports are Mac-only; on NVIDIA the lab has Tencent's own code.

    A Linux user must not read "needs Apple Silicon" as the whole truth, nor be sent off
    to another repo for something this lab now runs.
    """
    twin = bc.BY_ID["hunyuan-cuda"]
    for backend_id in ("hunyuan_xiong", "hunyuan-mlx"):
        entry = bc.BY_ID[backend_id].describe(bc.NVIDIA)
        assert entry["supported_here"] is False
        assert entry["upstream"]["url"].startswith("https://github.com/"), backend_id
        assert twin.label in entry["platform_note"], backend_id
        assert "later" not in entry["platform_note"], backend_id


def test_hunyuan_cuda_is_nvidia_linux_only():
    entry = bc.BY_ID["hunyuan-cuda"]
    assert entry.runs_here(bc.NVIDIA, "linux")
    assert not entry.runs_here(bc.NVIDIA, "windows")
    assert not entry.runs_here(bc.APPLE, "darwin")
    assert "EU" in entry.caveat


def test_trellis_is_supported_on_nvidia_now_not_pointed_elsewhere():
    """TRELLIS.2 has its own NVIDIA route (Microsoft's code, built for CUDA), so an NVIDIA
    Linux machine gets a Set up button rather than a link to the official repo."""
    trellis = bc.BY_ID["trellis"]
    assert trellis.runs_on == (bc.APPLE, bc.NVIDIA)
    entry = trellis.describe(bc.NVIDIA)
    if trellis.excluded_os(bc.NVIDIA):  # this test machine is Windows
        pytest.skip("TRELLIS.2 on NVIDIA is Linux only")
    assert entry["supported_here"] is True
    assert entry["platform_note"] is None
    assert entry["install"] == "scripts/bootstrap_trellis_cuda.py"
    assert entry["upstream"]["url"] == "https://github.com/microsoft/TRELLIS.2"


def test_trellis_install_and_build_probe_differ_per_machine():
    trellis = bc.BY_ID["trellis"]
    assert trellis.install_for(bc.APPLE) == "viewer"
    assert trellis.install_for(bc.NVIDIA) == "scripts/bootstrap_trellis_cuda.py"
    mac = trellis.probes_for(bc.APPLE)
    nvidia = trellis.probes_for(bc.NVIDIA)
    assert mac == (bc.venv_python(bc.REPO / "vendor" / "trellis-space-mac"),)
    assert all("trellis-cuda" in str(p) for p in nvidia) and nvidia


def test_trellis_nvidia_build_is_judged_by_its_own_probe(monkeypatch, tmp_path):
    import dataclasses

    marker = tmp_path / ".i2l-build-complete"
    trellis = dataclasses.replace(bc.BY_ID["trellis"],
                                  build_probes_by_host={bc.NVIDIA: (marker,)})
    assert trellis.built_on(bc.NVIDIA) is False
    marker.write_text("{}")
    assert trellis.built_on(bc.NVIDIA) is True


def test_trellis_on_windows_nvidia_is_refused_with_a_reason():
    trellis = bc.BY_ID["trellis"]
    assert trellis.runs_here(bc.NVIDIA, "windows") is False
    assert trellis.runs_here(bc.NVIDIA, "linux") is True
    note = trellis._platform_note(bc.NVIDIA, "windows")
    assert "Windows" in note and "Linux" in note


def test_the_mac_trellis_route_is_unchanged():
    entry = bc.BY_ID["trellis"].describe(bc.APPLE)
    assert entry["supported_here"] is True
    assert entry["install"] == "viewer"
    assert entry["setup_fetches_weights"] is False


def test_readiness_hint_names_the_nvidia_installer(monkeypatch):
    if bc.BY_ID["trellis"].excluded_os(bc.NVIDIA):
        pytest.skip("TRELLIS.2 on NVIDIA is Linux only")
    monkeypatch.setattr(bc.Backend, "built_on", lambda self, host=None: False)
    payload = bc.readiness("trellis", host=bc.NVIDIA)
    assert "bootstrap_trellis_cuda.py" in payload["build"]["hint"]


def test_a_route_with_no_official_elsewhere_keeps_the_plain_note():
    entry = bc.BY_ID["pixal3d"].describe("other")
    assert entry["upstream"] is None
    assert entry["platform_note"].startswith("Needs ")


def test_trellis_warns_about_the_gated_dinov3_before_setup():
    caveat = bc.BY_ID["trellis"].caveat or ""
    assert "dinov3" in caveat.lower() and "Hugging Face sign-in" in caveat


def test_every_gated_repo_is_one_the_sign_in_check_asks_about():
    # The Setup page hides a gated warning only on the sign-in check's "yes"; a repo it never
    # asks about would keep its warning forever.
    import hf_api

    checked = {repo for repo, _ in hf_api.GATED}
    gated = {b.id: b.gated_repo for b in bc.CATALOG if b.gated_repo}
    assert gated == {"trellis": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                     "sf3d": "stabilityai/stable-fast-3d"}
    assert set(gated.values()) <= checked
    assert bc.BY_ID["trellis"].describe(bc.APPLE)["gated_repo"] == gated["trellis"]


def test_pixal3d_setup_counts_the_background_remover_it_now_installs():
    sources = [w.source for w in bc.BY_ID["pixal3d"].weights]
    assert any("BiRefNet" in s for s in sources)


def test_trellis_on_nvidia_is_described_as_what_it_is():
    # Seen on a real NVIDIA pod: the card said "clean port", "Slowest" and that the first
    # run stops after the 14 GB download; the confirmation promised a Metal build and no
    # downloads. On NVIDIA it is Microsoft's own code, and setup fetches the weights after
    # checking Hugging Face access first.
    entry = bc.BY_ID["trellis"].describe(bc.NVIDIA)
    assert entry["label"] == "TRELLIS.2 (NVIDIA)"
    assert "port" not in entry["label"] and "Slowest" not in entry["tradeoff"]
    assert entry["setup_fetches_weights"] is True
    assert "Hugging Face sign-in" in entry["caveat"] and "stops after" not in entry["caveat"]


def test_trellis_on_nvidia_needs_its_weights_to_be_ready():
    entry = bc.BY_ID["trellis"].describe(bc.NVIDIA)
    none_present = [{"bytes_present": 0, "bytes_expected": 10 * bc.GB}]
    assert bc._state(none_present, built=True, setup_fetches=entry["setup_fetches_weights"]) != "ready"


def test_host_overrides_only_name_fields_describe_knows():
    for backend in bc.CATALOG:
        for overrides in backend.overrides_by_host.values():
            assert set(overrides) <= bc.HOST_OVERRIDABLE, backend.id


def test_setup_fetch_flag_is_answered_per_machine():
    trellis = bc.BY_ID["trellis"]
    assert trellis.setup_fetches_on(bc.NVIDIA) is True
    assert trellis.setup_fetches_on(bc.APPLE) is False
    assert bc.BY_ID["pixal3d"].setup_fetches_on(bc.NVIDIA) is True


# --- Setup page tabs: one per machine family, so nobody reads another machine's cards ---
def test_catalog_offers_a_tab_per_machine_family():
    tabs = bc.catalog_status(bc.NVIDIA)["platforms"]
    assert [t["id"] for t in tabs] == [bc.APPLE, bc.NVIDIA, "amd"]
    assert next(t for t in tabs if t["id"] == "amd")["coming"] is True


def test_each_tab_lists_only_what_runs_on_that_machine():
    views = bc.catalog_status(bc.NVIDIA)["views"]
    nvidia = {b["id"] for b in views[bc.NVIDIA]}
    apple = {b["id"] for b in views[bc.APPLE]}
    assert "hunyuan-cuda" in nvidia and "hunyuan_xiong" not in nvidia
    assert "hunyuan_xiong" in apple and "hunyuan-cuda" not in apple
    assert {"pixal3d", "trellis"} <= nvidia & apple


def test_a_tab_describes_its_own_machine():
    views = bc.catalog_status(bc.NVIDIA)["views"]
    trellis_mac = next(b for b in views[bc.APPLE] if b["id"] == "trellis")
    trellis_nv = next(b for b in views[bc.NVIDIA] if b["id"] == "trellis")
    assert trellis_nv["label"] == "TRELLIS.2 (NVIDIA)" and trellis_mac["label"] != trellis_nv["label"]
    pixal_mac = next(b for b in views[bc.APPLE] if b["id"] == "pixal3d")
    pixal_nv = next(b for b in views[bc.NVIDIA] if b["id"] == "pixal3d")
    assert "Xcode" in pixal_mac["tradeoff"] and "NVIDIA" not in pixal_mac["tradeoff"]
    assert "CUDA" in pixal_nv["tradeoff"] and "Xcode" not in pixal_nv["tradeoff"]


def test_a_routes_removable_bytes_leave_out_shared_files():
    # Pixal3D's Remove button counted the shared background remover as its own.
    pixal = bc.BY_ID["pixal3d"]
    lite = next(w for w in pixal.weights if "BiRefNet" in w.source)
    assert bc.is_shared(pixal, lite) is True
    assert bc.is_shared(bc.BY_ID["matte"], bc.BY_ID["matte"].weights[0]) is False
    assert "bytes_removable" in pixal.describe(bc.NVIDIA)
