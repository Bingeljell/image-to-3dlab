"""bootstrap_hunyuan_xiong: build whichever venv is missing, then fetch the default route."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap_hunyuan_xiong as bx


def _repo(tmp_path: Path, *built: str) -> Path:
    for half in bx.HALVES:
        (tmp_path / "hunyuan_mlx" / half).mkdir(parents=True)
    for half in built:
        python = bx.venv_python(tmp_path / "hunyuan_mlx" / half)
        python.parent.mkdir(parents=True)
        python.touch()
    return tmp_path


def test_a_missing_paint_venv_is_built_before_the_weights(tmp_path):
    """The #97 machine: shape venv made by hand, paint never made."""
    steps = bx.plan(_repo(tmp_path, "shape"), "uv")
    assert steps[0] == (["uv", "sync"], tmp_path / "hunyuan_mlx" / "paint")
    assert len(steps) == 2 and steps[1][0][1].endswith("download_weights.py")


def test_a_fresh_clone_builds_both_and_a_set_up_one_only_fetches(tmp_path):
    fresh = bx.plan(_repo(tmp_path / "a"), "uv")
    assert [cwd.name for cmd, cwd in fresh[:2]] == ["shape", "paint"]
    assert len(bx.plan(_repo(tmp_path / "b", "shape", "paint"), "uv")) == 1


def test_only_the_default_route_is_fetched(tmp_path):
    # Without --model the downloader fetches every shape checkpoint: 23 GB, not 5.
    fetch = bx.plan(_repo(tmp_path, "shape", "paint"), "uv")[-1][0]
    assert fetch[fetch.index("--model") + 1] == "2.0"
    assert Path(fetch[0]).parent.name in ("bin", "Scripts")  # this OS's venv layout


def test_the_announcement_names_route_size_and_licence(tmp_path):
    text = bx.announcement(bx.plan(_repo(tmp_path), "uv"))
    assert "default route" in text and "GB" in text and "EU" in text


def test_nobody_to_ask_refuses_before_any_download(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bootstrap_hunyuan_xiong.py"])
    monkeypatch.setattr(bx.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(bx.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(bx.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert bx.main() == 1
    assert "nobody to ask" in capsys.readouterr().out


def test_end_of_input_refuses_before_any_download(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["bootstrap_hunyuan_xiong.py"])
    monkeypatch.setattr(bx.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(bx.sys.stdin, "isatty", lambda: True)

    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    monkeypatch.setattr(bx.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("downloaded")))
    assert bx.main() == 1
    assert "nobody to ask" in capsys.readouterr().out


def test_the_downloader_is_not_asked_twice(tmp_path):
    # The bootstrap's own prompt (or the studio's) is the consent; the downloader's second
    # prompt has nobody to answer it inside the studio and would refuse.
    fetch = bx.plan(_repo(tmp_path, "shape", "paint"), "uv")[-1][0]
    assert "--yes" in fetch


def test_setup_installs_and_announces_the_background_remover(tmp_path):
    # The studio cuts out pictures for this route. Without the remover the first run
    # failed, since nothing fetches one the user did not agree to.
    import backend_catalog

    fetched = []
    bx.install_background_remover(tmp_path / "lite.onnx", download=fetched.append)
    assert fetched == [tmp_path / "lite.onnx"]
    assert "BiRefNet-lite" in bx.announcement(bx.plan(_repo(tmp_path), "uv"))
    labels = {w.label for w in backend_catalog.BY_ID["hunyuan_xiong"].weights}
    assert "BiRefNet-lite background remover" in labels
