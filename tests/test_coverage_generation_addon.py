"""Exercise the add-on command with local registry and checkout fixtures."""


import pytest

from image_to_3dlab import addon_install
from scripts import addon
from tests.test_addons import make_installed_addon, manifest


@pytest.fixture(autouse=True)
def local_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(addon, "ADDONS_DIR", tmp_path / "addons")
    monkeypatch.setattr(addon, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(addon, "REPO", tmp_path)
    monkeypatch.setattr(addon_install, "urlopen", lambda *_args, **_kwargs: pytest.fail("network"))


def test_remove_cli_reports_filesystem_error_without_traceback(tmp_path, monkeypatch, capsys):
    path = make_installed_addon(tmp_path, manifest())

    def remove_tree(target, **kwargs):
        assert target == path
        raise PermissionError("read-only folder")

    monkeypatch.setattr(addon_install.shutil, "rmtree", remove_tree)
    assert addon.main(["remove", "vehicle-kit"]) == 1
    assert capsys.readouterr().err == "error: read-only folder\n"
    assert path.is_dir()
