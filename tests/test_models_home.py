"""The chosen models folder: where every download goes, and when a new choice takes effect."""

from __future__ import annotations

from pathlib import Path

import pytest

from image_to_3dlab import models_home as mh


def test_nothing_chosen_keeps_the_old_places(tmp_path):
    env: dict[str, str] = {}
    assert mh.chosen(tmp_path, env) is None
    assert mh.vendor_models("pixal3d-cpp", tmp_path, env) == tmp_path / "vendor" / "pixal3d-cpp" / "models"
    assert mh.apply(tmp_path, env) is None
    assert "HF_HOME" not in env and "U2NET_HOME" not in env


def test_a_saved_folder_moves_every_download_under_it(tmp_path):
    models = tmp_path / "big drive" / "models"
    mh.save(str(models), tmp_path)
    env: dict[str, str] = {}
    assert mh.apply(tmp_path, env) == models
    assert env["HF_HOME"] == str(models / "huggingface")
    assert env["U2NET_HOME"] == str(models / "u2net")
    assert mh.vendor_models("pixal3d-cpp", tmp_path, env) == models / "pixal3d-cpp"


def test_a_persons_own_hf_home_still_wins(tmp_path):
    mh.save(str(tmp_path / "models"), tmp_path)
    env = {"HF_HOME": "/mine"}
    mh.apply(tmp_path, env)
    assert env["HF_HOME"] == "/mine"


def test_a_folder_saved_mid_run_waits_for_the_next_start(tmp_path):
    env: dict[str, str] = {}
    started = mh.apply(tmp_path, env)          # the studio starts with nothing chosen
    mh.save(str(tmp_path / "models"), tmp_path)
    # a job launched now inherits env, so it still downloads where the studio looks
    assert mh.vendor_models("pixal3d-cpp", tmp_path, env) == tmp_path / "vendor" / "pixal3d-cpp" / "models"
    shown = mh.describe(started, tmp_path)
    assert shown["in_use"] is None and shown["saved"] == str(tmp_path / "models") and shown["restart_needed"]
    assert mh.vendor_models("pixal3d-cpp", tmp_path, {}) == tmp_path / "models" / "pixal3d-cpp"   # next start


def test_an_empty_choice_goes_back_to_the_old_places(tmp_path):
    mh.save(str(tmp_path / "models"), tmp_path)
    assert mh.save("  ", tmp_path) is None
    assert mh.saved(tmp_path) is None


@pytest.mark.parametrize("bad", ["models", "relative/path"])
def test_a_relative_path_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="full path"):
        mh.save(bad, tmp_path)


def test_a_file_is_not_a_folder(tmp_path):
    file = tmp_path / "a-file"
    file.write_text("x")
    with pytest.raises(ValueError, match="not a folder"):
        mh.save(str(file), tmp_path)
    assert mh.saved(tmp_path) is None


def test_the_installers_and_the_studio_agree_on_pixal3ds_folder():
    import bootstrap_pixal3d
    import pixal3d_generate

    assert bootstrap_pixal3d.MODELS == pixal3d_generate.MODELS == mh.vendor_models("pixal3d-cpp") / "pixal3d-sv"
    assert bootstrap_pixal3d.MV_MODELS == pixal3d_generate.MV_MODELS


def test_the_installed_hunyuan_route_follows_the_folder_and_another_checkout_keeps_its_own(tmp_path):
    import bootstrap_hunyuan_cuda as boot
    import hunyuan_cuda_generate as gen

    assert gen.configure_environment(boot.VENDOR, {})["HY3DGEN_MODELS"] == str(boot.MODELS)
    assert gen.configure_environment(tmp_path, {})["HY3DGEN_MODELS"] == str(tmp_path / "models")


def test_the_setting_lives_in_output_which_git_ignores():
    assert mh.SETTING.parts[0] == "output"
    assert "output/" in Path(mh.REPO / ".gitignore").read_text()


def test_the_setup_endpoint_saves_a_folder_and_asks_for_a_restart(tmp_path, monkeypatch):
    import http.server
    import json
    import threading
    import urllib.error
    import urllib.request

    import generate_api as api

    monkeypatch.setattr(api, "MODELS_SETTING_REPO", tmp_path)
    monkeypatch.setattr(api, "MODELS_STARTED_WITH", None)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/api/models-folder"

    def post(path):
        request = urllib.request.Request(url, json.dumps({"path": path}).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "X-AssetFurnace-Request": "1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            assert json.loads(response.read())["in_use"] is None
        answer = post(str(tmp_path / "models"))
        assert answer["saved"] == str(tmp_path / "models") and answer["restart_needed"] is True
        assert mh.saved(tmp_path) == tmp_path / "models"
        with pytest.raises(urllib.error.HTTPError) as refused:
            post("not/a/full/path")
        assert refused.value.code == 422
    finally:
        server.shutdown()
        server.server_close()
