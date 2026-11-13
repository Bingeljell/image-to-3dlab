"""Flux Light: the same picture model in 4-bit files, suggested on machines with 16 GB or less.

A suggestion, never a block: every engine stays choosable, and nothing downloads without a yes.
"""

from __future__ import annotations

import json

import backend_catalog as bc
import bootstrap_flux_image as boot
import image_api as ia
import pytest

from image_to_3dlab import host

GB = 1024 ** 3


def test_sixteen_gigabytes_or_less_counts_as_low_memory():
    assert host.low_memory(8 * GB) and host.low_memory(16 * GB)
    assert not host.low_memory(18 * GB) and not host.low_memory(64 * GB)


def test_unknown_memory_is_not_treated_as_low(monkeypatch):
    # a suggestion must never be built on a guess: no reading means the standard suggestion
    monkeypatch.setattr(host, "total_memory", lambda: None)
    assert host.low_memory() is False


def test_light_is_suggested_first_only_on_low_memory():
    assert [e.id for e in ia.suggested_order(low=True)][:2] == ["flux-light", "flux"]
    assert [e.id for e in ia.suggested_order(low=False)][:2] == ["flux", "flux-light"]
    assert {e.id for e in ia.suggested_order(low=True)} == set(ia.ENGINES)


def test_any_engine_can_still_be_asked_for_by_name():
    for engine in ia.ENGINES:
        assert ia.pick_engine(engine).id == engine


def test_the_default_follows_the_suggestion_among_what_is_installed(monkeypatch):
    installed = {"flux", "flux-light"}
    def ready(e):
        return e.id in installed

    monkeypatch.setattr(ia._host, "low_memory", lambda *a: True)
    assert ia.pick_engine(ready=ready).id == "flux-light"
    monkeypatch.setattr(ia._host, "low_memory", lambda *a: False)
    assert ia.pick_engine(ready=ready).id == "flux"
    installed = {"flux"}                 # a low-memory machine with only standard Flux uses it
    monkeypatch.setattr(ia._host, "low_memory", lambda *a: True)
    assert ia.pick_engine(ready=ready).id == "flux"


def test_light_uses_the_four_bit_files_and_the_shared_vae():
    files = {k: v[1] for k, v in ia.FLUX_LIGHT.weight_files.items()}
    assert files == {"diffusion_model": "flux-2-klein-4b-Q4_0.gguf", "llm": "Qwen3-4B-Q4_K_M.gguf",
                     "vae": "flux2-vae.safetensors"}
    assert ia.FLUX_LIGHT.licence_key == ia.FLUX.licence_key  # same model, same licence
    assert ia.FLUX_LIGHT.defaults == ia.FLUX.defaults


def _record(folder, diffusion_file):
    folder.mkdir(parents=True, exist_ok=True)
    picture = folder / "p.png"
    picture.write_bytes(b"x")
    picture.with_suffix(".provenance.json").write_text(json.dumps({"model": {
        "id": "flux2-klein-4b", "weights": {"diffusion_model": {"file": diffusion_file}}}}))
    return picture


def test_a_redraw_uses_the_same_flux_files_that_drew_the_picture(tmp_path):
    assert ia.engine_of(_record(tmp_path / "a", "flux-2-klein-4b-Q4_0.gguf")).id == "flux-light"
    assert ia.engine_of(_record(tmp_path / "b", "flux-2-klein-4b-Q8_0.gguf")).id == "flux"


def test_an_older_flux_record_without_files_still_reads_as_flux(tmp_path):
    picture = tmp_path / "old.png"
    picture.write_bytes(b"x")
    picture.with_suffix(".provenance.json").write_text(json.dumps({"model": {"id": "flux2-klein-4b"}}))
    assert ia.engine_of(picture).id == "flux"


def test_create_lists_engines_in_the_suggested_order_and_marks_one(monkeypatch):
    import generate_api

    monkeypatch.setattr(ia._host, "low_memory", lambda *a: True)
    engines = generate_api.picture_engines_payload()
    assert [e["id"] for e in engines][:2] == ["flux-light", "flux"]
    assert [e["suggested"] for e in engines] == [True] + [False] * (len(engines) - 1)


def test_setup_puts_start_here_on_the_card_that_suits_this_machine():
    light, standard = bc.BY_ID["flux-image-light"], bc.BY_ID["flux-image"]
    assert light.suggested_here(low=True) and not standard.suggested_here(low=True)
    assert standard.suggested_here(low=False) and not light.suggested_here(low=False)


def test_each_flux_card_counts_only_its_own_files():
    # both cards share Hugging Face folders; a card must not call itself installed on the
    # strength of the other card's files
    q8, q4 = bc.BY_ID["flux-image"].weights[0], bc.BY_ID["flux-image-light"].weights[0]
    assert q8.pattern.endswith("Q8_0.gguf") and q4.pattern.endswith("Q4_0.gguf")
    for card in ("flux-image", "flux-image-light"):
        big = [w for w in bc.BY_ID[card].weights if "VAE" not in w.label]
        assert all(w.pattern for w in big)


def test_light_install_names_its_files_size_and_licence_before_any_download():
    text = boot.announcement(build=False, light=True)
    assert "Light" in text and "flux-2-klein-4b-Q4_0.gguf" in text and "Qwen3-4B-Q4_K_M.gguf" in text
    assert f"{boot.total_gb(light=True):.1f} GB" in text and "Apache-2.0" in text
    assert "Q8_0" not in text


def test_every_light_file_is_pinned_to_a_tested_revision_and_hash():
    for repo, filename, _ in boot.LIGHT_WEIGHTS:
        revision, sha256 = boot.PINS[(repo, filename)]
        assert len(revision) == 40 and len(sha256) == 64


def test_light_setup_without_yes_and_without_a_terminal_downloads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(boot, "install_weights", lambda *a, **k: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.sdcpp, "ensure_binary", lambda: pytest.fail("fetched the engine"))
    monkeypatch.setattr("sys.stdin", None)
    assert boot.main(["--light"]) == 1
    assert "Refusing" in capsys.readouterr().out


def test_the_setup_button_for_light_runs_the_light_install():
    import download_api

    command = download_api.COMMANDS["flux-image-light"]
    assert command[-2:] == ["--light", "--yes"] and command[1].endswith("bootstrap_flux_image.py")


# --- the two Flux cards share Hugging Face folders: check them against a real (fake) cache ---

import dataclasses

REV = "3b1f5a9dc3abb32238b053aeb3d823c30afdacbd"


def _cards_in(hub, monkeypatch):
    """Both Flux cards, with every weight pointed into `hub` (same layout as the real cache)."""
    def move(card):
        return dataclasses.replace(card, weights=tuple(
            dataclasses.replace(w, path=hub / w.path.name) for w in card.weights))

    cards = {cid: move(bc.BY_ID[cid]) for cid in ("flux-image", "flux-image-light")}
    monkeypatch.setattr(bc, "BY_ID", {**bc.BY_ID, **cards})
    return cards


def _cache_file(weight, size):
    """Put one weight's file into the cache the way Hugging Face does: a blob plus a link."""
    name = weight.pattern.split("/")[-1] if weight.pattern else "flux2-vae.safetensors"
    blob = weight.path / "blobs" / (weight.blob or "vae")
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"\0" * size)
    link = weight.path / "snapshots" / REV / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():           # the VAE is shared: both cards may lay it down
        link.symlink_to(blob)


def _shrunk(card):
    """The same card expecting 1000 bytes a file, so a fake cache can read as ready."""
    return dataclasses.replace(card, weights=tuple(
        dataclasses.replace(w, bytes_expected=1000) for w in card.weights))


@pytest.mark.symlinks
def test_with_only_light_installed_the_standard_card_reads_not_installed(tmp_path, monkeypatch):
    cards = {k: _shrunk(v) for k, v in _cards_in(tmp_path, monkeypatch).items()}
    monkeypatch.setattr(bc, "BY_ID", {**bc.BY_ID, **cards})
    for w in cards["flux-image-light"].weights:
        _cache_file(w, 1000)
    # Name the machine. A plain Linux runner is "other", and both cards
    # answer "unsupported" there before they look at the files.
    light, standard = (cards["flux-image-light"].describe(bc.APPLE),
                       cards["flux-image"].describe(bc.APPLE))
    assert light["state"] in ("ready", "missing")    # "missing" only if the engine build is absent
    assert standard["state"] != "partial", "the shared VAE must not make it look half-downloaded"
    assert standard["action"] != "resume"


@pytest.mark.symlinks
def test_with_only_standard_installed_the_light_card_reads_not_installed(tmp_path, monkeypatch):
    cards = {k: _shrunk(v) for k, v in _cards_in(tmp_path, monkeypatch).items()}
    monkeypatch.setattr(bc, "BY_ID", {**bc.BY_ID, **cards})
    for w in cards["flux-image"].weights:
        _cache_file(w, 1000)
    assert cards["flux-image-light"].describe(bc.APPLE)["state"] != "partial"


@pytest.mark.symlinks
def test_each_card_counts_only_its_own_files_in_a_shared_cache(tmp_path, monkeypatch):
    cards = _cards_in(tmp_path, monkeypatch)
    for w in cards["flux-image-light"].weights:
        _cache_file(w, 1000)
    standard_big = [w for w in cards["flux-image"].weights if w.pattern]
    assert all(w.state()[1] == 0 for w in standard_big)


@pytest.mark.parametrize("suffix", [".incomplete", ".1a2b3c4d.incomplete"])  # 1.x, and 2.x's per-process name
def test_a_file_still_downloading_counts_so_progress_moves(tmp_path, monkeypatch, suffix):
    cards = _cards_in(tmp_path, monkeypatch)
    weight = cards["flux-image-light"].weights[0]
    blobs = weight.path / "blobs"
    blobs.mkdir(parents=True)
    (blobs / f"{weight.blob}{suffix}").write_bytes(b"\0" * 1234)
    present, size, _ = weight.state()
    assert (present, size) == (False, 1234)          # counted, but not "there" yet


@pytest.mark.symlinks
def test_remove_on_one_flux_card_leaves_the_other_and_the_shared_vae(tmp_path, monkeypatch):
    import download_api

    cards = _cards_in(tmp_path, monkeypatch)
    monkeypatch.setattr(download_api, "BY_ID", bc.BY_ID)
    monkeypatch.setattr(download_api, "_inside_known_roots", lambda path: True)
    for card in cards.values():
        for w in card.weights:
            _cache_file(w, 1000)
    download_api.remove("flux-image-light")
    light_big = [w for w in cards["flux-image-light"].weights if w.pattern]
    standard = cards["flux-image"].weights
    assert all(w.state()[1] == 0 for w in light_big)                 # Light's own files went
    assert all(w.state()[1] == 1000 for w in standard if w.pattern)  # standard Flux untouched
    assert any((w.path / "snapshots" / REV / "flux2-vae.safetensors").exists()
               for w in standard if not w.pattern)                   # the shared VAE stayed


def test_the_cards_know_the_same_file_fingerprints_the_installer_pins():
    pins = {f: sha for (_, f), (_, sha) in boot.PINS.items()}
    for card in ("flux-image", "flux-image-light"):
        for w in bc.BY_ID[card].weights:
            if w.pattern:
                assert w.blob == pins[w.pattern.split("/")[-1]]
