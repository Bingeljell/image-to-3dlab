"""bootstrap_hunyuan_xiong: build whichever venv is missing, then fetch the default route."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bootstrap_hunyuan_xiong as bx

PYTHONS = {"shape": Path("/mise/3.12/python"), "paint": Path("/mise/3.11/python")}


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
    steps = bx.plan(_repo(tmp_path, "shape"), "uv", PYTHONS)
    assert steps[0][0][:2] == ["uv", "sync"]
    assert steps[0][0][steps[0][0].index("--python") + 1] == str(PYTHONS["paint"])
    assert len(steps) == 2 and steps[1][0][1].endswith("download_weights.py")


def test_a_fresh_clone_builds_both_and_a_set_up_one_only_fetches(tmp_path):
    fresh = bx.plan(_repo(tmp_path / "a"), "uv", PYTHONS)
    assert [cwd.name for cmd, cwd in fresh[:2]] == ["shape", "paint"]
    assert len(bx.plan(_repo(tmp_path / "b", "shape", "paint"), "uv", PYTHONS)) == 1


def test_only_the_default_route_is_fetched(tmp_path):
    # Without --model the downloader fetches every shape checkpoint: 23 GB, not 5.
    fetch = bx.plan(_repo(tmp_path, "shape", "paint"), "uv", PYTHONS)[-1][0]
    assert fetch[fetch.index("--model") + 1] == "2.0"
    assert Path(fetch[0]).parent.name in ("bin", "Scripts")  # this OS's venv layout


def test_the_announcement_names_route_size_and_licence(tmp_path):
    text = bx.announcement(bx.plan(_repo(tmp_path), "uv", PYTHONS))
    assert "default route" in text and "GB" in text and "EU" in text


def test_a_fresh_plan_uses_the_shape_python(tmp_path):
    steps = bx.plan(_repo(tmp_path), "uv", PYTHONS)
    assert steps[0][0] == ["uv", "sync", "--python", str(PYTHONS["shape"])]
