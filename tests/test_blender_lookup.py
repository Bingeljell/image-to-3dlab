"""Finding Blender on every supported OS (image_to_3dlab/blender.py).

Finish and the rig tools run Blender headless; it used to be looked up at the macOS app
path only, so every Finish run on Linux or Windows failed before it started.
"""

from __future__ import annotations

from pathlib import Path

from image_to_3dlab import blender as bl

original_candidates = bl.candidates


def _exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    return path


def test_the_override_wins_over_everything(tmp_path):
    chosen = _exe(tmp_path / "custom" / "blender")
    found = bl.find_blender(env={bl.ENV_VAR: str(chosen)}, which=lambda _: "/usr/bin/blender",
                            family="linux", home=tmp_path)
    assert found == chosen


def test_a_wrong_override_is_reported_not_skipped(tmp_path):
    """Silently falling back would run a different Blender than the one asked for."""
    found = bl.find_blender(env={bl.ENV_VAR: str(tmp_path / "nope")},
                            which=lambda _: "/usr/bin/blender", family="linux", home=tmp_path)
    assert found is None


def test_the_path_comes_before_install_folders(tmp_path):
    assert bl.find_blender(env={}, which=lambda _: "/somewhere/blender",
                           family="linux", home=tmp_path) == Path("/somewhere/blender")


def test_a_blender_org_tarball_in_the_home_folder_is_found_on_linux(tmp_path, monkeypatch):
    unpacked = _exe(tmp_path / "blender-5.2.0-linux-x64" / "blender")
    # Only home-folder candidates exist in the test: a machine with a distro Blender would
    # otherwise find /usr/bin/blender first and fail for reasons that are not the code's.
    monkeypatch.setattr(bl, "candidates", lambda family, home=None: [
        p for p in original_candidates(family, home) if str(p).startswith(str(tmp_path))])
    assert bl.find_blender(env={}, which=lambda _: None, family="linux",
                           home=tmp_path) == unpacked


def test_the_blender_setup_installed_outranks_the_path(tmp_path):
    """A distro Blender on PATH must not win over the one the user installed from Setup."""
    installed = _exe(tmp_path / "blender-lts" / "blender")
    assert bl.find_blender(env={}, which=lambda _: "/usr/bin/blender", family="linux",
                           home=tmp_path) == installed


def test_the_setup_blender_is_a_linux_idea_only(tmp_path):
    _exe(tmp_path / "blender-lts" / "blender")
    assert bl.ours("macos", home=tmp_path) is None


def test_the_environment_variable_still_beats_the_setup_blender(tmp_path):
    _exe(tmp_path / "blender-lts" / "blender")
    chosen = _exe(tmp_path / "mine" / "blender")
    assert bl.find_blender(env={bl.ENV_VAR: str(chosen)}, family="linux",
                           home=tmp_path) == chosen


def test_linux_looks_in_the_usual_places():
    places = [str(p) for p in bl.candidates("linux", home=Path("/home/x"))]
    assert "/usr/bin/blender" in places and "/snap/bin/blender" in places


def test_windows_prefers_the_newest_version_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "x86"))
    _exe(tmp_path / "Blender Foundation" / "Blender 4.2" / "blender.exe")
    newest = _exe(tmp_path / "Blender Foundation" / "Blender 5.2" / "blender.exe")
    assert bl.candidates("windows")[0] == newest


def test_the_mac_app_is_the_first_mac_candidate():
    assert bl.candidates("macos")[0] == bl.MAC_APP


def test_nothing_found_is_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    assert bl.find_blender(env={}, which=lambda _: None, family="windows", home=tmp_path) is None


def test_versions_are_read_from_blenders_banner():
    assert bl.parse_version("Blender 5.2.0 LTS\n\tbuild date: ...") == (5, 2)
    assert bl.parse_version("not blender") is None


def test_old_blenders_are_flagged_and_new_ones_are_not():
    assert "older than 4.2" in bl.version_problem((3, 6))
    assert bl.version_problem((4, 2)) is None
    assert bl.version_problem(None) is None


def test_the_missing_help_says_what_to_do_on_each_os():
    assert "snap install blender" in bl.missing_help("linux")
    assert "Applications" in bl.missing_help("macos")
    assert "installer" in bl.missing_help("windows")
    for family in ("linux", "macos", "windows"):
        assert bl.ENV_VAR in bl.missing_help(family)
