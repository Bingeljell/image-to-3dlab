"""Exercise Pixal3D patch adapters without loading model weights."""

import importlib
import sys
from types import SimpleNamespace

import patch_pixal3d_model_subset as subset
import patch_pixal3d_rembg as remover
import pytest
from test_patch_pixal3d_model_subset import ALL_MODELS
from test_patch_pixal3d_model_subset import ORIGINAL as PIPELINE
from test_patch_pixal3d_rembg import ORIGINAL as REMBG

from scripts import pixal3d_compat as compat


@pytest.mark.parametrize(("patcher", "source"), [(subset, PIPELINE), (remover, REMBG)])
def test_cli_refreshes_missing_or_stale_helper_after_patch(monkeypatch, tmp_path, patcher, source):
    target = tmp_path / "vendor.py"
    target.write_text(source)
    monkeypatch.setattr(sys, "argv", ["patch", str(target)])
    assert patcher.main() == 0
    before = target.read_bytes()
    helper = target.with_name("_assetfurnace_pixal3d_compat.py")
    assert helper.read_bytes() == patcher.HELPER_SOURCE.read_bytes()

    for missing in (False, True):
        if missing:
            helper.unlink()
        else:
            helper.write_bytes(b"# wrong helper")
        with monkeypatch.context() as arguments:
            arguments.setattr(sys, "argv", ["patch", str(target), "--check"])
            assert patcher.main() == 1
        assert patcher.main() == 0
        assert helper.read_bytes() == patcher.HELPER_SOURCE.read_bytes()
        assert target.read_bytes() == before
    assert patcher.main() == 0
    monkeypatch.setattr(sys, "argv", ["patch", str(target), "--check"])
    assert patcher.main() == 0


@pytest.mark.parametrize("skip", ["", " tex_slat_decoder , ,", ",".join(ALL_MODELS)])
def test_patched_pipeline_calls_copied_subset_before_model_loading(monkeypatch, tmp_path, capsys, skip):
    package_name = "pixal_model_fixture"
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").touch()
    target = package / "pipeline.py"
    target.write_text(PIPELINE)
    monkeypatch.setattr(sys, "argv", ["patch", str(target)])
    assert subset.main() == 0
    capsys.readouterr()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PIXAL3D_SKIP_MODELS", skip)
    try:
        module = importlib.import_module(f"{package_name}.pipeline")
        cls = module.Pixal3DImageTo3DPipeline
        expected = [name for name in ALL_MODELS if name not in {n.strip() for n in skip.split(",")}]
        assert cls.model_names_to_load == expected
        assert compat.model_names_for_run(ALL_MODELS) == expected
        assert not any(name.startswith("_i2l") for name in vars(cls))
        assert ("not loading:" in capsys.readouterr().out) is bool(skip)
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def test_patched_pixal3d_never_hands_bria_identity_to_loader(monkeypatch, tmp_path, capsys):
    package_name = "pixal_remover_fixture"
    package = tmp_path / package_name
    package.mkdir()
    (package / "__init__.py").touch()
    target = package / "BiRefNet.py"
    target.write_text(REMBG)
    monkeypatch.setattr(sys, "argv", ["patch", str(target)])
    assert remover.main() == 0
    capsys.readouterr()
    monkeypatch.syspath_prepend(str(tmp_path))
    requests = []

    def load(name, **args):
        requests.append((name, args))
        return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForImageSegmentation=SimpleNamespace(from_pretrained=load)
    ))
    try:
        cls = importlib.import_module(f"{package_name}.BiRefNet").BiRefNet
        cls("briaai/RMBG-2.0")
        cls("for-example/approved")
        cls()
        assert compat.guard_rembg_model("briaai/RMBG-2.0") == "ZhengPeng7/BiRefNet"
        assert compat.guard_rembg_model("for-example/approved") == "for-example/approved"
        assert requests == [(model, {"trust_remote_code": True}) for model in (
            "ZhengPeng7/BiRefNet", "for-example/approved", "ZhengPeng7/BiRefNet",
        )]
        assert "refusing briaai/RMBG-2.0" in capsys.readouterr().out
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


@pytest.mark.parametrize(("patcher", "source"), [(subset, PIPELINE), (remover, REMBG)])
def test_pixal3d_patch_refuses_repeated_anchor_and_modified_matching_marker(patcher, source):
    with pytest.raises(SystemExit, match="anchor not found or changed"):
        patcher.apply(source + source)
    modified = source.replace(patcher.ANCHOR, patcher.ANCHOR + f"# {patcher.MARKER} customized\n")
    with pytest.raises(SystemExit, match="anchor not found or changed"):
        patcher.apply(modified)


@pytest.mark.parametrize("module", ["patch_pixal3d_model_subset", "patch_pixal3d_rembg"])
def test_pixal3d_cli_as_package_refuses_missing_checkout(module, tmp_path, monkeypatch):
    import runpy

    monkeypatch.setattr(sys, "argv", [module, str(tmp_path / "missing.py")])
    with pytest.raises(SystemExit, match="not found"):
        runpy.run_module(f"scripts.{module}", run_name="__main__")


def test_pixal3d_payload_upgrade_requires_identical_audited_legacy_bytes():
    import hashlib

    original = "class Demo:\n"
    previous = "    # prior_model_selector\n    old_config()\n"
    replacement = "    new_config()\n"
    settings = (original, replacement, "prior_model_selector", "    old_config()\n",
                hashlib.sha256(previous.encode()).hexdigest(), "anchor has changed")
    assert compat.replace_payload(original + previous + "    after()\n", *settings) == (
        original + replacement + "    after()\n"
    )
    altered = previous.replace("old_config()", "old_config()  # adjusted")
    with pytest.raises(SystemExit, match="anchor has changed"):
        compat.replace_payload(original + altered, *settings)


@pytest.mark.parametrize(("cli_enabled", "env", "enabled"), [
    (False, None, False), (False, " 0 ", False), (False, " false ", False),
    (False, "yes", True), (False, "FALSE", True), (True, "0", True),
])
def test_low_vram_patch_loads_helper_without_host_os_import(
    monkeypatch, tmp_path, cli_enabled, env, enabled,
):
    import runpy

    import patch_pixal3d_low_vram as low
    from test_patch_pixal3d_low_vram import ORIGINAL

    target = tmp_path / "generate_mps.py"
    target.write_text(ORIGINAL.replace("import os\n", ""))
    monkeypatch.setattr(sys, "argv", ["patch", str(target)])
    assert low.main() == 0
    assert (tmp_path / compat.HELPER_NAME).read_bytes() == low.HELPER_SOURCE.read_bytes()
    monkeypatch.syspath_prepend(str(tmp_path))
    if env is None:
        monkeypatch.delenv("PIXAL3D_LOW_VRAM", raising=False)
    else:
        monkeypatch.setenv("PIXAL3D_LOW_VRAM", env)
    calls = []
    monkeypatch.setitem(sys.modules, compat.HELPER_NAME.removesuffix(".py"), None)
    sys.modules.pop(compat.HELPER_NAME.removesuffix(".py"))
    try:
        module = runpy.run_path(str(target), init_globals={"_install_low_vram_hooks": lambda *a: calls.append(a)})
        args = SimpleNamespace(low_vram=cli_enabled)
        assert module["image_to_asset"]("pipeline", None, args, "device") == "asset"
        assert calls == ([("pipeline", "device")] if enabled else [])
        assert compat.low_vram_for_run(args) is enabled
    finally:
        sys.modules.pop(compat.HELPER_NAME.removesuffix(".py"), None)
