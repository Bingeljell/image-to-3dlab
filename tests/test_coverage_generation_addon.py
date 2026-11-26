"""Exercise the add-on command with local registry and checkout fixtures."""

import io
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from image_to_3dlab import addon_install
from scripts import addon
from tests.test_addons import RUN_PYTHON, listing, make_installed_addon, manifest


@pytest.fixture(autouse=True)
def local_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(addon, "ADDONS_DIR", tmp_path / "addons")
    monkeypatch.setattr(addon, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(addon, "REPO", tmp_path)
    monkeypatch.setattr(addon_install, "urlopen", lambda *_args, **_kwargs: pytest.fail("network"))


def test_list_shows_review_status_and_only_installed_filter(tmp_path, capsys):
    make_installed_addon(tmp_path, manifest())
    index = tmp_path / "index.json"
    index.write_text(json.dumps([
        listing(run_check={"status": "partial", "skipped": ["Windows"]}, runs="cloud"),
        listing(id="crate", run_check={"status": "partial"}, rating=1, ratings_count=0, links={}),
        listing(id="fence", rating=None, ratings_count=0, summary="", author={"name": "A"}, links={}),
        listing(id="bench", run_check={"status": "passed"}),
    ]))
    assert addon.main(["list", "--index", str(index)]) == 0
    out = capsys.readouterr().out
    assert "vehicle-kit (installed):" in out
    assert "partly run-checked (not run: Windows) (cloud)" in out
    assert "* 4.5 (12)" in out
    assert "* 1" in out
    assert ", partly run-checked\n" in out
    assert ", not run-checked" in out
    assert "site: https://vehiclekit.example" in out
    assert "author: https://github.com/Ahnyth" in out
    assert "repo: https://github.com/Ahnyth/vehicle-kit" in out

    assert addon.main(["list", "--index", str(index), "--installed"]) == 0
    out = capsys.readouterr().out
    assert "vehicle-kit (installed)" in out
    assert all(f"{identifier}:" not in out for identifier in ("crate", "fence", "bench"))


def test_list_read_error_uses_stderr(tmp_path, capsys):
    index = tmp_path / "index.json"
    assert addon.main(["list", "--index", str(index)]) == 1
    assert "could not load the add-on list" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("interactive", "answer", "outcome"),
    [(False, None, False), (True, "  YES  ", True), (True, "sure", False)],
)
def test_addon_prompt(monkeypatch, interactive, answer, outcome):
    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(sys.stdin, "isatty", lambda: interactive)

    def respond(prompt):
        assert interactive
        assert prompt == "Install? [y/N] "
        return answer

    monkeypatch.setattr("builtins.input", respond)
    assert addon._ask_yes("Install?") is outcome


@pytest.mark.parametrize(
    ("mode", "result"),
    [("decline", 1), ("yes", 0), ("plan-mismatch", 1), ("install-mismatch", 1),
     ("invalid-plan", 2), ("invalid-install", 2), ("setup-fails", 1)],
)
def test_cli_add_plans_then_confirms(mode, result, monkeypatch, capsys, tmp_path):
    index = tmp_path / "index.json"
    index.write_text(json.dumps([listing()]))
    cloned = []
    setup = []
    declined = mode == "decline"
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    def run(argv, cwd, **kwargs):
        if argv[:2] == ["git", "clone"]:
            assert argv[2] == listing()["repo"]
            assert cwd == str(tmp_path)
            dest = Path(argv[3])
            dest.mkdir()
            cloned.append(dest)
            invalid = mode == "invalid-plan" or (mode == "invalid-install" and len(cloned) == 2)
            data = manifest(
                id="other" if invalid else "vehicle-kit",
                models=[] if declined else [{"name": "Local sample", "licence": "MIT", "size": "1 MB"}],
            )
            (dest / "addon.json").write_text(json.dumps(data))
        elif argv[:2] == ["git", "checkout"]:
            assert Path(cwd) == cloned[-1]
            assert argv[2] == "a" * 40
        elif argv[:2] == ["git", "rev-parse"]:
            assert argv[2] == "HEAD"
            assert Path(cwd) == cloned[-1]
            mismatch = mode == "plan-mismatch" or (mode == "install-mismatch" and len(cloned) == 2)
            return subprocess.CompletedProcess(argv, 0, stdout=("b" if mismatch else "a") * 40)
        else:
            assert list(argv) == manifest()["setup"]
            assert cwd == str(addon.ADDONS_DIR / "vehicle-kit")
            setup.append(cwd)
            return subprocess.CompletedProcess(argv, int(mode == "setup-fails"))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(addon_install.subprocess, "run", run)
    flags = [] if declined else ["--yes"]
    assert addon.main(["add", "vehicle-kit", "--index", str(index), *flags]) == result
    captured = capsys.readouterr()
    if mode in {"plan-mismatch", "invalid-plan"}:
        assert not captured.out
    else:
        assert ("Local sample (MIT, 1 MB)" in captured.out) is not declined
        assert "Author: Ahnyth" in captured.out
    if declined:
        assert "Nothing installed." in captured.out
    if mode == "yes":
        assert "Installed vehicle-kit" in captured.out
        record = addon.ADDONS_DIR / "vehicle-kit/.addon-install.json"
        assert json.loads(record.read_text())["commit"] == "a" * 40
    if result and not declined:
        assert "error:" in captured.err
    assert len(setup) == int(mode in {"yes", "setup-fails"})
    assert len(cloned) == (1 if mode in {"decline", "plan-mismatch", "invalid-plan"} else 2)
    assert all(not path.exists() for path in cloned)


def test_remove_cli_reports_missing_and_deletes_installed(tmp_path, capsys):
    assert addon.main(["remove", "vehicle-kit"]) == 2
    assert "vehicle-kit is not installed" in capsys.readouterr().err
    path = make_installed_addon(tmp_path, manifest())
    assert addon.main(["remove", "vehicle-kit"]) == 0
    assert "Removed vehicle-kit" in capsys.readouterr().out
    assert not path.exists()


def test_remove_cli_reports_filesystem_error_without_traceback(tmp_path, monkeypatch, capsys):
    path = make_installed_addon(tmp_path, manifest())

    def remove_tree(target, **kwargs):
        assert target == path
        raise PermissionError("read-only folder")

    monkeypatch.setattr(addon_install.shutil, "rmtree", remove_tree)
    assert addon.main(["remove", "vehicle-kit"]) == 1
    assert capsys.readouterr().err == "error: read-only folder\n"
    assert path.is_dir()


@pytest.mark.parametrize("mode", ["success", "tool-error", "unknown-tool"])
def test_run_cli_writes_provenance_and_reports_failures(mode, tmp_path, monkeypatch, capsys):
    path = make_installed_addon(tmp_path, manifest())
    out = tmp_path / "results" if mode == "success" else addon.OUTPUT_DIR
    calls = []

    def run(argv, cwd, env, check):
        assert argv[:2] == [RUN_PYTHON, "shape_guide.py"]
        assert argv[3:] == ["--preset", "bus_12m"]
        assert Path(cwd) == path
        assert env["HF_HUB_OFFLINE"] == "1"
        assert check is False
        output = Path(argv[2]) / "sample.glb"
        assert output.is_relative_to(out)
        calls.append(output)
        if mode == "success":
            output.write_bytes(b"local model")
        return subprocess.CompletedProcess(argv, int(mode == "tool-error"))

    monkeypatch.setattr(addon_install.subprocess, "run", run)
    tool = "missing" if mode == "unknown-tool" else "build"
    command = [
        "run", "vehicle-kit", tool, "--input", "preset=bus_12m",
        "--run-stem", "sample", "--blender", "unused",
    ]
    if mode == "success":
        assert addon.main([*command, "--output-dir", str(out)]) == 0
        assert str(calls[0]) in capsys.readouterr().out
        record = json.loads(calls[0].with_suffix(".provenance.json").read_text())
        assert record["output"]["source"] == "add-on"
    elif mode == "tool-error":
        assert addon.main(command) == 1
        assert "run failed" in capsys.readouterr().err
        assert not list(out.rglob("*.provenance.json"))
    else:
        assert addon.main(command) == 2
        assert "tool 'missing' not found" in capsys.readouterr().err
        assert not calls


def test_addon_cli_rejects_input_missing_equals():
    with pytest.raises(ValueError, match="key=value"):
        addon.main(["run", "vehicle-kit", "build", "--input", "preset"])


def test_addon_script_entrypoint(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [addon.__file__, "--help"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(addon.__file__, run_name="__main__")
    assert exc.value.code == 0
    assert "List the add-ons" in capsys.readouterr().out
