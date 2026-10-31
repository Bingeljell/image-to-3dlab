"""Tests for scripts/bootstrap_front_back.py: consent, prerequisites, and what it fetches."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "viewer")]

import backend_catalog
import bootstrap_front_back as boot


@pytest.fixture
def routes_ready(monkeypatch):
    monkeypatch.setattr(boot, "missing_routes", list)


def test_with_flux_the_announcement_names_only_the_shape_weights():
    text = boot.announcement(vision=False)
    assert "5.9 GB" in text and "pixal3d-q8_0-v1" in text and "DINOv3" in text
    assert "mmproj" not in text and "Qwen" not in text


def test_with_only_qwen_it_also_names_the_vision_projector():
    text = boot.announcement(vision=True)
    assert "0.75 GB" in text and "6.6 GB in all" in text and "mmproj" in text and "Qwen" in text


def test_nothing_downloads_without_a_yes_and_nobody_to_ask(monkeypatch, routes_ready):
    monkeypatch.setattr(boot, "needs_vision", lambda: True)
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: pytest.fail("downloaded"))
    monkeypatch.setattr(boot.qwen, "install_vision", lambda *a: pytest.fail("downloaded"))
    assert boot.main([]) == 1


@pytest.mark.parametrize("vision, expected", [(True, ["mv", "vision"]), (False, ["mv"])])
def test_a_yes_fetches_the_projector_only_when_qwen_must_draw_the_back(monkeypatch, routes_ready, vision, expected):
    fetched = []
    monkeypatch.setattr(boot, "needs_vision", lambda: vision)
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: fetched.append("mv"))
    monkeypatch.setattr(boot.qwen, "install_vision", lambda *a: fetched.append("vision"))
    assert boot.main(["--yes"]) == 0 and fetched == expected


def test_needs_vision_only_when_qwen_alone_is_set_up(monkeypatch):
    api = boot.image_api
    installed = {"flux": False, "qwen": True}
    monkeypatch.setattr(api, "is_installed", lambda engine=api.QWEN: installed[engine.id])
    monkeypatch.setattr(api, "resolve_vision", lambda engine=api.QWEN: None)
    assert boot.needs_vision() is True
    installed["flux"] = True
    assert boot.needs_vision() is False


def test_missing_routes_stop_it_before_anything_is_fetched(monkeypatch, capsys):
    monkeypatch.setattr(boot, "missing_routes", lambda: ["Pixal3D (python scripts/bootstrap_pixal3d.py)"])
    monkeypatch.setattr(boot.pixal3d, "install_multiview", lambda *a: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Pixal3D" in capsys.readouterr().out


def test_the_catalogue_entry_quotes_the_same_sizes():
    entry = backend_catalog.BY_ID["front-back"]
    assert [w.bytes_expected for w in entry.weights] == [boot.MV_BYTES]
    assert entry.weights[0].pattern == "*_mv.gguf"  # measured without the files linked from single-view
    # the projector moved to Qwen's own card: it is Qwen's edit mode, and skins need it too
    vision = next(w for w in backend_catalog.BY_ID["qwen-image"].weights if "mmproj" in (w.pattern or ""))
    assert vision.bytes_expected == boot.VISION_BYTES and vision.pattern.endswith(boot.qwen.VISION[1])
