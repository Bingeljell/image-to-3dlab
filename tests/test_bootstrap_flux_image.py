"""Tests for scripts/bootstrap_flux_image.py: what it says before a byte moves, and consent.

The engine install itself is bootstrap_qwen_image.ensure_binary, tested there; here it is
always replaced, so no test can reach the real vendor/sdcpp or the network.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import backend_catalog
import bootstrap_flux_image as boot

REAL_INSTALL_WEIGHTS = boot.install_weights


@pytest.fixture(autouse=True)
def nothing_real(monkeypatch):
    monkeypatch.setattr(boot.sdcpp, "ensure_binary", lambda: pytest.fail("reached the real engine install"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("reached the real download"))
    monkeypatch.setattr(boot.sdcpp, "binary_present", lambda: True)
    monkeypatch.setattr(boot.sdcpp, "binary_stale", lambda: False)   # never the real install's record


def test_the_announcement_names_backend_route_size_and_licence():
    text = boot.announcement()
    assert "FLUX.2 [klein] 4B" in text and "stable-diffusion.cpp" in text
    assert "8.9 GB" in text and "Apache-2.0" in text and "commercial" in text
    for _, filename, size in boot.WEIGHTS:
        assert Path(filename).name in text and f"{size:.2f}" in text


def test_the_catalogue_card_quotes_the_same_files_and_size():
    card = backend_catalog.BY_ID["flux-image"]
    assert [w.source for w in card.weights] == [repo for repo, _, _ in boot.WEIGHTS]
    assert sum(w.bytes_expected for w in card.weights) / 1e9 == pytest.approx(boot.total_gb(), abs=0.05)
    # listed before Qwen: the Setup page shows new installs Flux first
    ids = [b.id for b in backend_catalog.CATALOG]
    assert ids.index("flux-image") < ids.index("qwen-image")


def test_nothing_downloads_without_a_yes_and_nobody_to_ask(monkeypatch, capsys):
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    assert boot.main([]) == 1
    assert "Refusing to download" in capsys.readouterr().out


def test_a_typed_no_downloads_nothing(monkeypatch):
    monkeypatch.setattr(boot.sys, "stdin", type("Tty", (io.StringIO,), {"isatty": lambda self: True})(""))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert boot.main([]) == 1


def test_a_yes_installs_the_engine_then_the_weights(monkeypatch):
    done = []
    monkeypatch.setattr(boot.sdcpp, "ensure_binary", lambda: done.append("engine") or True)
    monkeypatch.setattr(boot, "install_weights", lambda: done.append("weights"))
    assert boot.main(["--yes"]) == 0 and done == ["engine", "weights"]


def test_an_engine_that_cannot_reach_the_gpu_stops_before_the_weights(monkeypatch):
    monkeypatch.setattr(boot.sdcpp, "ensure_binary", lambda: False)
    assert boot.main(["--yes"]) == 1


def test_the_studio_setup_button_runs_it_with_yes():
    import download_api
    command = download_api.COMMANDS["flux-image"]   # the Setup page's own dialog is the asking
    assert command[-1] == "--yes" and command[1].endswith("bootstrap_flux_image.py")


def test_an_engine_update_is_named_before_the_yes(monkeypatch):
    """ensure_binary replaces an older sd-cli: the announcement must say so (AGENTS.md: name it)."""
    monkeypatch.setattr(boot.sdcpp, "binary_stale", lambda: True)
    monkeypatch.setattr(boot.sdcpp, "target", lambda: "macos-arm64")
    assert "update sd-cli to" in boot.announcement()


def test_weights_come_from_the_tested_revision_and_are_checked(monkeypatch, tmp_path):
    """Security review 2026-10-31: pin each file and check it, whatever the repo serves later."""
    import huggingface_hub
    asked = []

    def fake_download(repo_id, filename, revision=None):
        asked.append((repo_id, revision))
        path = tmp_path / Path(filename).name
        path.write_bytes(b"swapped")
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(SystemExit, match="not the file we tested"):
        REAL_INSTALL_WEIGHTS()
    assert asked == [(boot.WEIGHTS[0][0], boot.PINS[boot.WEIGHTS[0][0]][0])]   # stopped at the first
    assert set(boot.PINS) == {repo for repo, _, _ in boot.WEIGHTS}
    assert all(len(rev) == 40 and len(sha) == 64 for rev, sha in boot.PINS.values())
