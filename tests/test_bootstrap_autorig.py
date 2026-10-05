"""The auto-rig installer: says what it fetches, never downloads without a yes, pins upstream."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import backend_catalog
import bootstrap_autorig as ba
import download_api


def test_announcement_names_backend_licence_and_size():
    text = ba.announcement()
    assert "SkinTokens" in text and "MIT" in text
    assert "1.6 GB" in text
    assert ba.COMMIT[:7] in text and ba.HF_REVISION[:7] in text


def test_refuses_without_yes_when_nobody_can_answer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", None)
    fetched = []
    monkeypatch.setattr(ba, "install_weights", lambda *a, **k: fetched.append(1))
    monkeypatch.setattr(ba, "clone_pinned", lambda *a, **k: fetched.append(1))
    assert ba.main([]) == 1
    assert not fetched
    assert "Refusing" in capsys.readouterr().out


def test_clone_is_pinned(tmp_path):
    calls = []
    ba.clone_pinned(tmp_path / "SkinTokens", runner=calls.append)
    assert calls[0][:2] == ["git", "clone"]
    assert calls[-1][-1] == ba.COMMIT


def test_existing_clone_is_moved_to_the_pin_not_recloned(tmp_path):
    root = tmp_path / "SkinTokens"
    (root / ".git").mkdir(parents=True)
    calls = []
    ba.clone_pinned(root, runner=calls.append)
    assert len(calls) == 1 and calls[0][-1] == ba.COMMIT


def test_link_checkpoint_replaces_a_stale_link(tmp_path):
    cache = tmp_path / "cache.ckpt"
    cache.write_bytes(b"x")
    checkpoint = ba.CHECKPOINTS[0]
    stale = tmp_path / "root" / checkpoint.path
    stale.parent.mkdir(parents=True)
    stale.symlink_to(tmp_path / "gone.ckpt")
    linked = ba.link_checkpoint(cache, tmp_path / "root", checkpoint)
    assert linked.resolve() == cache.resolve()


def test_weights_present_needs_both_checkpoints_and_the_llm_config(tmp_path):
    root = tmp_path / "SkinTokens"
    for checkpoint in ba.CHECKPOINTS:
        (root / checkpoint.path).parent.mkdir(parents=True, exist_ok=True)
        (root / checkpoint.path).write_bytes(b"x")
    assert not ba.weights_present(root)
    (root / ba.LLM_DIR).mkdir(parents=True)
    (root / ba.LLM_DIR / "config.json").write_text("{}")
    assert ba.weights_present(root)


def test_venv_on_the_wrong_python_is_rebuilt(tmp_path):
    root = tmp_path / "SkinTokens"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.10.14\n")
    assert ba.venv_needs_rebuild(root)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.11.9\n")
    assert not ba.venv_needs_rebuild(root)


def test_catalogue_and_setup_command_agree_with_the_installer():
    backend = backend_catalog.BY_ID["autorig"]
    assert backend.bytes_expected == ba.WEIGHT_BYTES
    assert set(backend.runs_on) == {backend_catalog.APPLE, backend_catalog.NVIDIA}
    command = download_api.COMMANDS["autorig"]
    assert command[1].endswith("bootstrap_autorig.py") and "--yes" in command
