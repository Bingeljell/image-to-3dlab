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

REPO = Path(__file__).resolve().parents[1]


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
    """The Mac port's bootstrap fetches no weights; the NVIDIA and AMD ones fetch them once
    the build is done, and the card decides which flag the browser sees."""
    def flag_for(host):
        entry = next(b for b in bc.catalog_status(host)["backends"] if b["id"] == "trellis")
        return entry["setup_fetches_weights"]

    assert flag_for(bc.APPLE) is False
    assert flag_for(bc.NVIDIA) is True
    assert flag_for(bc.AMD) is True


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
    assert trellis.runs_on == (bc.APPLE, bc.NVIDIA, bc.AMD)
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
    assert [t["id"] for t in tabs] == [bc.APPLE, bc.NVIDIA, bc.AMD]
    assert all(t["coming"] is False for t in tabs)


def test_each_tab_lists_only_what_runs_on_that_machine():
    views = bc.catalog_status(bc.NVIDIA)["views"]
    nvidia = {b["id"] for b in views[bc.NVIDIA]}
    apple = {b["id"] for b in views[bc.APPLE]}
    assert "hunyuan-cuda" in nvidia and "hunyuan_xiong" not in nvidia
    assert "hunyuan_xiong" in apple and "hunyuan-cuda" not in apple
    assert {"pixal3d", "trellis"} <= nvidia & apple


def test_amd_gets_a_3d_route_and_the_image_one():
    """An AMD box must be able to generate 3D, not only pictures: Pixal3D compiles with
    HIP, Qwen-Image has a prebuilt ROCm build, and the cut-out runs on the CPU."""
    amd = {b["id"] for b in bc.catalog_status(bc.AMD)["views"][bc.AMD]}
    assert {"pixal3d", "qwen-image", "matte"} <= amd


def test_trellis_is_offered_on_amd_but_hunyuan_cuda_is_not():
    """TRELLIS.2 runs on AMD through its ROCm route; Hunyuan3D-2.1's CUDA route does not,
    and offering it there means a download that cannot finish. Pixal3D works either way."""
    assert bc.AMD in bc.BY_ID["trellis"].runs_on
    assert bc.AMD not in bc.BY_ID["hunyuan-cuda"].runs_on
    assert bc.BY_ID["pixal3d"].runs_here(bc.AMD) is True


def test_trellis_uses_a_separate_rocm_checkout_from_the_cuda_one():
    """A two-card machine keeps both installs: they are different builds of the same source,
    so sharing a directory would mean one card's compiled extensions overwriting the other's."""
    trellis = bc.BY_ID["trellis"]
    assert trellis.install_by_host[bc.AMD] == "scripts/bootstrap_trellis_rocm.py"
    assert trellis.build_probes_by_host[bc.AMD] != trellis.build_probes_by_host[bc.NVIDIA]
    assert "trellis-rocm" in str(trellis.build_probes_by_host[bc.AMD][0])


@pytest.mark.parametrize("backend_id", ["hunyuan-cuda"])
def test_an_amd_user_is_told_why_not_and_what_to_use_instead(backend_id):
    """"Needs an NVIDIA GPU" is true of an AMD machine and explains nothing: it reads as a
    card problem rather than a port problem, and sends the user off to check their drivers.
    The note has to say what is missing and name the route that does work here."""
    note = bc.BY_ID[backend_id]._platform_note(bc.AMD)
    assert "AMD" in note
    assert "Pixal3D" in note
    assert "download gigabytes" not in note


# --- a build that lost part of itself -------------------------------------------------------
#
# "Set up" with nothing else is what a vanished venv looks like, and it sends people to the
# install script rather than to the directory that is missing. The card now names the file.


def _backend_with_probes(probes, **kwargs):
    return bc.Backend(id="x", label="Stable Fast 3D", kind="3d", best_for="", tradeoff="",
                      license_name="", license_url="", weights=(), install="viewer",
                      build_probes_by_host={bc.AMD: probes}, **kwargs)


def test_a_complete_build_says_nothing(tmp_path):
    probes = (tmp_path / "marker", tmp_path / "venv" / "bin" / "python")
    probes[0].write_text("{}")
    probes[1].parent.mkdir(parents=True)
    probes[1].write_text("")
    backend = _backend_with_probes(probes)
    assert backend.missing_build_probes(bc.AMD) == []
    assert backend.build_hint(bc.AMD) is None


def test_a_never_installed_route_is_not_dressed_up_as_a_loss(tmp_path):
    """Both look like `action == build`. One deserves a rebuild message and one does not."""
    backend = _backend_with_probes((tmp_path / "marker", tmp_path / "venv"))
    assert "not installed" in backend.build_hint(bc.AMD)
    assert "was installed here" not in backend.build_hint(bc.AMD)


def test_a_lost_build_names_the_file_that_is_missing(tmp_path):
    marker = tmp_path / "marker"
    marker.write_text("{}")
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    backend = _backend_with_probes((marker, python))

    python.unlink()          # the venv is what has gone
    hint = backend.build_hint(bc.AMD)
    assert "was installed here" in hint
    assert "venv/bin/python" in hint
    assert backend.missing_build_probes(bc.AMD) == [python]


def test_the_hint_reaches_the_payload_the_card_reads(tmp_path):
    """A field nothing renders is a field nobody benefits from."""
    backend = _backend_with_probes((tmp_path / "marker", tmp_path / "venv"))
    described = backend.describe(bc.AMD)
    assert described["build_present"] is False
    assert described["build_missing"]
    assert described["build_hint"]


def test_a_build_with_all_its_probes_present_reports_none_missing(tmp_path):
    """Synthetic, not the live catalogue: whether a route happens to be built on the machine
    running the tests is not a property of this code, and a test that reads it passes for the
    wrong reason on a developer who built everything and fails on a fresh clone."""
    marker, python = tmp_path / "marker", tmp_path / "venv" / "bin" / "python"
    marker.write_text("{}")
    python.parent.mkdir(parents=True)
    python.write_text("")
    backend = _backend_with_probes((marker, python))
    assert backend.missing_build_probes(bc.AMD) == []
    assert backend.describe(bc.AMD)["build_missing"] == []
    assert backend.describe(bc.AMD)["build_hint"] is None


def test_every_declared_probe_is_a_path():
    """A probe that is a string would silently always exist, and every route would read as
    built on every machine."""
    for backend in bc.CATALOG:
        for host in (bc.APPLE, bc.NVIDIA, bc.AMD):
            for probe in backend.probes_for(host):
                assert isinstance(probe, Path), f"{backend.id}/{host}: {probe!r}"


def test_the_setup_page_renders_the_hint():
    """Pinned in the page's own source: the payload alone proves nothing about the screen."""
    source = (REPO / "viewer" / "studio" / "setup.js").read_text()
    assert "build_hint" in source
    assert "setup-build-hint" in (REPO / "viewer" / "studio" / "setup.css").read_text()


def test_trellis_shows_no_amd_note_because_it_runs_there():
    """A route that runs on the card has nothing to explain. `_platform_note` is only ever
    shown when `runs_here` is False, so what matters is that the blocked-on-AMD note is not
    the one this produces."""
    trellis = bc.BY_ID["trellis"]
    assert trellis.runs_here(bc.AMD) is True
    assert "Not on AMD" not in trellis._platform_note(bc.AMD)


# --- weights that are not in the Hugging Face cache ---------------------------------------
#
# TRELLIS.2 loads from a local directory when I2L_TRELLIS_MODEL_PATH names one, or when the
# route's own vendor/<route>/models/TRELLIS.2-4B exists. The card only looked in the cache,
# so a machine whose weights are local was told it was missing 15 GB it did not need -- on
# the very machine that had just finished a generation.


def test_a_local_weights_directory_satisfies_a_download(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / "weights.safetensors").write_bytes(b"x" * 4096)
    weight = bc.WeightSet("w", "some/repo", 4096, tmp_path / "empty-cache",
                          alt_paths=(local,))
    present, total, alt = weight.state()
    assert (present, total, alt) == (True, 4096, str(local))


def test_an_empty_cache_directory_does_not_settle_it(tmp_path):
    """A Hugging Face cache folder is created with a refs/ file before anything is
    downloaded, so preferring the first location that merely exists picks the empty one."""
    cache = tmp_path / "cache"
    (cache / "refs").mkdir(parents=True)
    local = tmp_path / "local"
    local.mkdir()
    (local / "weights.safetensors").write_bytes(b"x" * 4096)
    weight = bc.WeightSet("w", "some/repo", 4096, cache, alt_paths=(local,))
    assert weight.state()[1] == 4096
    assert weight.state()[2] == str(local)


def test_one_shared_directory_is_counted_once_not_once_per_weight(tmp_path):
    """TRELLIS.2's main weights, its image-large decoder and its DINOv3 encoder all sit in
    one local tree. Summed naively the card read "53.1 GB of 17.9 GB" while being ready."""
    local = tmp_path / "local"
    local.mkdir()
    (local / "w.safetensors").write_bytes(b"x" * 2048)
    weights = [
        bc.WeightSet(f"w{n}", "some/repo", 100, tmp_path / f"cache{n}", alt_paths=(local,)).describe()
        for n in range(3)
    ]
    assert bc._total_present(weights) == 2048


def test_distinct_directories_are_each_counted(tmp_path):
    weights = []
    for n, size in ((0, 100), (1, 250)):
        directory = tmp_path / f"local{n}"
        directory.mkdir()
        (directory / "w.safetensors").write_bytes(b"x" * size)
        weights.append(bc.WeightSet(f"w{n}", "some/repo", size, tmp_path / f"cache{n}",
                                    alt_paths=(directory,)).describe())
    assert bc._total_present(weights) == 350


def test_the_trellis_weights_look_in_both_local_places(monkeypatch, tmp_path):
    monkeypatch.setenv(bc.TRELLIS_MODEL_PATH_ENV, str(tmp_path / "named"))
    assert tmp_path / "named" in bc._local_trellis()
    assert any("trellis-rocm" in str(p) for p in bc._local_trellis())


def test_the_card_and_the_generator_read_the_same_variable():
    """They each read it in their own module; nothing forces the names to agree."""
    sys.path.insert(0, str(REPO / "scripts"))
    import trellis_route

    assert trellis_route.ROCM.model_repo  # sanity: the module loaded
    source = (REPO / "scripts" / "trellis_route.py").read_text()
    assert bc.TRELLIS_MODEL_PATH_ENV in source, (
        f"the card reads {bc.TRELLIS_MODEL_PATH_ENV} but the generator does not set that name")


def test_the_trellis_weights_all_look_in_the_local_places():
    """Without alt_paths the card measures only the Hugging Face cache. No module reload
    here on purpose: reloading backend_catalog binds a *second* copy under the same name, and
    every monkeypatch in this file would then stop reaching the code under test."""
    trellis = {w.label: w for w in bc.BY_ID["trellis"].weights}
    for label in ("TRELLIS.2-4B", "TRELLIS image-large decoder", "DINOv3 image encoder"):
        assert trellis[label].alt_paths, f"{label} only looks in the cache"
        assert any("trellis-rocm" in str(p) for p in trellis[label].alt_paths), label
    # The other two are not part of the TRELLIS.2 tree: TinyCLIP is a separate advisory
    # model and the remover is a single file this repo fetches itself.
    assert not trellis["TinyCLIP input advisor"].alt_paths
    assert not trellis["BiRefNet-lite background remover"].alt_paths


def test_the_amd_note_does_not_leak_into_other_tabs():
    """What must not leak is the AMD *blockade* wording. The platform list naming AMD is
    correct on any machine, because it is now true on all of them."""
    trellis = bc.BY_ID["trellis"]
    for host in (bc.APPLE, bc.NVIDIA):
        note = trellis._platform_note(host)
        assert "Not on AMD" not in note
        assert "does not run on AMD" not in note


def test_sf3d_is_no_longer_a_blocked_route_on_amd():
    """It was blocked for a reason that had nothing to do with the card: its pins broke the
    lab's *shared* environment, so it was left out rather than given a venv. Giving it one
    removed the reason, and the note it used to show must be gone rather than reworded --
    a route that runs here has nothing to explain."""
    entry = bc.BY_ID["sf3d"]
    assert entry.runs_here(bc.AMD, "linux") is True
    assert entry.amd_blocked_reason is None
    assert entry._platform_note(bc.AMD) is None or "Not on AMD" not in entry._platform_note(bc.AMD)
    assert entry.describe(bc.AMD)["platform_note"] is None


def test_trellis_is_no_longer_a_blocked_route_on_amd():
    """It was blocked for a reason that turned out not to be the card: nvdiffrast, CuMesh and
    AuleAttention all have working ROCm builds, and FlexGEMM needed one flag changed. The
    note it used to show must be gone rather than reworded."""
    trellis = bc.BY_ID["trellis"]
    assert trellis.runs_here(bc.AMD, "linux") is True
    assert "Not on AMD" not in trellis._platform_note(bc.AMD)
    assert trellis.amd_blocked_reason in (None, "")


def test_the_cuda_hunyuan_route_points_an_amd_card_at_the_rocm_one():
    """Tencent's code is not NVIDIA-only; this route is. The AMD card gets the same model
    built with HIP, and the note has to name that rather than claim an impossibility."""
    note = bc.BY_ID["hunyuan-cuda"]._platform_note(bc.AMD)
    assert "NVIDIA-only compiled extensions" not in note
    assert "Hunyuan3D-2.1 (AMD)" in note
    assert bc.BY_ID["hunyuan-rocm"].runs_here(bc.AMD, "linux") is True


def test_the_mac_hunyuan_ports_do_not_claim_tencent_is_nvidia_only():
    """Both MLX ports said "Tencent's own needs NVIDIA-only compiled extensions" after the
    AMD route shipped, which was simply no longer true."""
    for backend_id in ("hunyuan_xiong", "hunyuan-mlx"):
        note = bc.BY_ID[backend_id]._platform_note(bc.AMD)
        assert "NVIDIA-only compiled extensions" not in note, backend_id
        assert "Hunyuan3D-2.1 (AMD)" in note, backend_id


def test_every_route_named_as_an_amd_alternative_really_runs_on_amd():
    """A suggestion that is itself unavailable is the same failure as no suggestion, and
    it is what makes a page read as broken rather than as honest."""
    for backend in bc.CATALOG:
        note = backend._platform_note(bc.AMD)
        if not note.startswith("Not on AMD"):
            continue
        # The route being talked about is the one the note is about, not a suggestion:
        # "TRELLIS.2 (clean port) needs NVIDIA-only extensions" contains its own label and
        # is not recommending it.
        for other in bc.CATALOG:
            if other.label == backend.label or other.label not in note:
                continue
            assert other.runs_here(bc.AMD, "linux"), \
                f"{other.label} is suggested to an AMD user but does not run there"


def test_an_amd_user_is_not_sent_to_an_nvidia_machine_they_do_not_have():
    """The MLX ports point at Tencent's own route as the alternative. That is true and
    useless here: it is NVIDIA-only too, so the note has to say Hunyuan is out, not offer
    a machine the reader is not on."""
    for backend_id in ("hunyuan_xiong", "hunyuan-mlx"):
        note = bc.BY_ID[backend_id]._platform_note(bc.AMD)
        assert "on an NVIDIA machine, use" not in note
        assert "AMD" in note and "Pixal3D" in note


def test_a_tab_describes_its_own_machine():
    views = bc.catalog_status(bc.NVIDIA)["views"]
    trellis_mac = next(b for b in views[bc.APPLE] if b["id"] == "trellis")
    trellis_nv = next(b for b in views[bc.NVIDIA] if b["id"] == "trellis")
    assert trellis_nv["label"] == "TRELLIS.2 (NVIDIA)" and trellis_mac["label"] != trellis_nv["label"]
    pixal_mac = next(b for b in views[bc.APPLE] if b["id"] == "pixal3d")
    pixal_nv = next(b for b in views[bc.NVIDIA] if b["id"] == "pixal3d")
    assert "Xcode" in pixal_mac["tradeoff"] and "NVIDIA" not in pixal_mac["tradeoff"]
    assert "CUDA" in pixal_nv["tradeoff"] and "Xcode" not in pixal_nv["tradeoff"]


def test_the_amd_tab_names_hip_and_not_cuda():
    """The wording has to match what the installer does, or a user reads "CUDA" and waits
    for a download that never comes."""
    pixal_amd = next(b for b in bc.catalog_status(bc.AMD)["views"][bc.AMD]
                     if b["id"] == "pixal3d")
    assert "HIP" in pixal_amd["tradeoff"] and "CUDA" not in pixal_amd["tradeoff"]


def test_a_routes_removable_bytes_leave_out_shared_files():
    # Pixal3D's Remove button counted the shared background remover as its own.
    pixal = bc.BY_ID["pixal3d"]
    lite = next(w for w in pixal.weights if "BiRefNet" in w.source)
    assert bc.is_shared(pixal, lite) is True
    assert bc.is_shared(bc.BY_ID["matte"], bc.BY_ID["matte"].weights[0]) is False
    assert "bytes_removable" in pixal.describe(bc.NVIDIA)
