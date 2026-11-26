"""Exercise the shipped command files as scripts without starting Blender."""

import runpy
import socket
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    ("script", "option"),
    [
        ("bake_normals", "--keep-sign"),
        ("import_character", "--facing"),
        ("inspect", "--object"),
        ("light_orbit", "--frames"),
        ("orbit_camera", "--target"),
        ("stage", "--double-sided"),
        ("stage_bake", "--frame-step"),
        ("turntable", "--keep-frames"),
    ],
)
def test_script_help_exits_without_contacting_blender(
    script, option, monkeypatch, capsys
):
    path = SCRIPTS / f"blender_{script}.py"
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "bpy", None)

    def unexpected_connection(*args, **kwargs):
        pytest.fail("Showing help must not contact Blender")

    monkeypatch.setattr(socket, "create_connection", unexpected_connection)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(path), run_name="__main__")

    assert exc.value.code == 0
    output = capsys.readouterr()
    assert output.out.startswith(f"usage: {path.name}")
    assert option in output.out
    assert not output.err
