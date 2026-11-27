"""Exercise installer decisions without installing runtimes or models."""

import io
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile, ZipInfo

import bootstrap_autorig as rig
import bootstrap_blender as blender
import bootstrap_flux_image as flux
import bootstrap_front_back as front
import bootstrap_gltfpack as gltf
import bootstrap_hunyuan_cuda as hy_cuda
import bootstrap_hunyuan_rocm as hy_rocm
import bootstrap_hunyuan_xiong as xiong
import bootstrap_matte as matte
import bootstrap_pixal3d as pixal
import bootstrap_qwen_image as qwen
import bootstrap_sf3d as sf
import bootstrap_sf3d_rocm as sf_rocm
import bootstrap_trellis_cuda as t_cuda
import bootstrap_trellis_rocm as t_rocm
import bootstrap_upscaler as up
import pytest


@pytest.mark.parametrize("script", sorted(Path(rig.__file__).parent.glob("bootstrap_*.py")))
def test_installer_help_runs_without_setup(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    monkeypatch.setattr(sys, "path", sys.path.copy())
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("module", [blender, gltf])
def test_platform_installers_reject_unknown_hosts(module, monkeypatch):
    monkeypatch.setattr(module.platform, "system", lambda: "UnsupportedOS")
    with pytest.raises(SystemExit, match="Linux|No gltfpack build"):
        module.main(["--yes"])


@pytest.mark.parametrize(("terminal", "answer", "expected"), [(False, "", 1), (True, "no", 1), (True, "yes", 0)])
def test_sf3d_checks_consent_before_install(terminal, answer, expected, monkeypatch):
    monkeypatch.setattr(sf, "target", lambda: "macos-arm64")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: terminal))
    ask = Mock(return_value=answer)
    monkeypatch.setattr("builtins.input", ask)
    code, weights = Mock(), Mock()
    monkeypatch.setattr(sf, "install_code", code)
    monkeypatch.setattr(sf, "install_weights", weights)
    assert sf.main([]) == expected
    assert ask.call_count == terminal
    assert code.call_count == weights.call_count == (expected == 0)


def test_sf3d_install_plans_missing_checkout_and_libomp(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, "VENDOR", tmp_path / "checkout")
    monkeypatch.setattr(sf, "baker_installed", lambda: False)
    monkeypatch.setattr(sf, "pip_install_command", lambda: ["pip", "install"])
    run = Mock()
    monkeypatch.setattr(sf.subprocess, "run", run)
    sf.install_code("linux")
    assert run.call_args_list[0].args[0][:2] == ["git", "init"]
    assert run.call_args.args[0][-1] == "requirements.txt"
    exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if p.name == "libomp" else exists(p))
    run.reset_mock()
    with pytest.raises(SystemExit, match="libomp"):
        sf.install_code("macos-arm64")
    run.assert_not_called()


@pytest.mark.parametrize("module", [sf, sf_rocm, t_rocm])
def test_weight_file_plan_passes_declared_patterns_to_cache_boundary(module, monkeypatch):
    import huggingface_hub

    snapshot = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    remover = Mock()
    monkeypatch.setattr(module, "install_background_remover", remover)
    module.install_weights()
    assert [(c.args[0], c.kwargs["allow_patterns"]) for c in snapshot.call_args_list] == [
        (name, selection) for name, selection, _ in module.WEIGHTS
    ]
    assert remover.call_count == (module is not t_rocm)


@pytest.mark.parametrize("failure", ["403 gated repo", "disk full"])
def test_sf3d_rocm_distinguishes_weight_failures(failure, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", Mock(side_effect=RuntimeError(failure)))
    with pytest.raises(SystemExit if failure.startswith("403") else RuntimeError, match="gated|disk full"):
        sf_rocm.install_weights()


@pytest.mark.parametrize("module", [rig, sf_rocm, t_cuda, t_rocm, xiong])
def test_installers_report_missing_uv_before_any_work(module, monkeypatch, tmp_path):
    monkeypatch.setattr(module.host, "find_uv", lambda: None)
    with pytest.raises(SystemExit, match="uv is required"):
        if module is rig:
            module.install_packages(tmp_path, Mock())
        elif module is xiong:
            module.main(["--yes"])
        elif module is t_rocm:
            module.install_code((7, 0))
        elif module is t_cuda:
            module.install_code("source", None)
        else:
            module.install_code(next(iter(module.TORCH)))


@pytest.mark.parametrize("module", [t_cuda, t_rocm])
def test_clone_refreshes_pinned_commit_and_updates_recursive_checkout(module, tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(module, "CHECKOUT", checkout)
    run = Mock()
    monkeypatch.setattr(module, "run", run)
    commit = "a" * 40
    module.clone("local-origin", checkout, commit, recursive=True)
    assert [call.args[0][1] for call in run.call_args_list] == ["clone", "checkout", "submodule"]
    (checkout / ".git").mkdir(parents=True)
    run.reset_mock()
    module.clone("local-origin", checkout, commit)
    run.assert_not_called()
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "old\n")
    module.fetch_checkout(commit)
    assert [call.args[0][1] for call in run.call_args_list] == ["fetch", "checkout", "submodule"]


def test_sf3d_rocm_fetches_missing_checkout_at_requested_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(sf_rocm, "VENDOR", tmp_path)
    monkeypatch.setattr(sf_rocm, "CHECKOUT", tmp_path / "checkout")
    monkeypatch.setattr(sf_rocm.subprocess, "check_output", lambda *args, **kwargs: "old\n")
    run = Mock()
    monkeypatch.setattr(sf_rocm, "run", run)
    sf_rocm.fetch_checkout("new")
    assert [call.args[0][1] for call in run.call_args_list] == ["clone", "fetch", "checkout"]


@pytest.mark.parametrize(("module", "build_flag"), [(rig, "--build-only"), (sf_rocm, "--code-only"), (t_rocm, "--code-only")])
@pytest.mark.parametrize(("args", "terminal", "answer", "status"), [
    ([], False, "", 1), ([], True, "no", 1), ([], True, "yes", 0),
    (["--yes", "--weights-only"], False, "", 0),
])
def test_bootstrap_code_and_weights_obey_consent(
    module, build_flag, args, terminal, answer, status, monkeypatch,
):
    if module is rig:
        monkeypatch.setattr(module, "nvidia_problem", lambda: None)
        monkeypatch.setattr(module, "this_machines_index", lambda: None)
        monkeypatch.setattr(module, "clone_pinned", Mock())
        monkeypatch.setattr(module.portable, "patch", Mock())
        build_name = "install_packages"
    else:
        monkeypatch.setattr(module.host, "build_target", lambda: "linux-amd")
        monkeypatch.setattr(module.host, "rocm_version", lambda: (7, 0))
        monkeypatch.setattr(module, "detect", lambda: ("source", "fixture"))
        monkeypatch.setattr(module, "install_background_remover", Mock())
        monkeypatch.setattr(module, "build_present", lambda: False)
        if module is t_rocm:
            monkeypatch.setattr(module, "check_access", Mock())
        build_name = "install_code"
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: terminal))
    monkeypatch.setattr("builtins.input", lambda _: answer)
    build, weights = Mock(), Mock()
    monkeypatch.setattr(module, build_name, build)
    monkeypatch.setattr(module, "install_weights", weights)
    assert module.main(args or [build_flag]) == status
    assert build.call_count == (status == 0 and not args)
    assert weights.call_count == (status == 0 and bool(args))


@pytest.mark.parametrize("present", [False, True])
def test_existing_rig_and_upscaler_have_probe_only_entry_points(present, monkeypatch):
    monkeypatch.setattr(rig, "build_present", lambda: True)
    monkeypatch.setattr(rig, "weights_present", lambda: present)
    monkeypatch.setattr(up.upscale, "installed", lambda _: present)
    assert rig.main(["--check"]) == up.main(["--check"]) == (0 if present else 1)


def test_declined_upscaler_and_matte_never_download(monkeypatch):
    fetch = Mock()
    monkeypatch.setattr(up.upscale, "installed", lambda _: False)
    monkeypatch.setattr(matte, "download", fetch)
    assert up.main([], ask=lambda _: "no", fetch=fetch) == 1
    assert matte.main([], ask=lambda _: "no") == 1
    fetch.assert_not_called()


@pytest.mark.parametrize(("terminal", "answer", "status"), [(False, "", 1), (True, "no", 1), (True, "yes", 0)])
def test_front_back_lists_dependencies_then_checks_consent(tmp_path, terminal, answer, status, monkeypatch):
    binary, models = tmp_path / "cli", tmp_path / "models"
    monkeypatch.setattr(front.pixal3d, "cli_path", lambda: binary)
    monkeypatch.setattr(front.pixal3d, "MODELS", models)
    monkeypatch.setattr(front.image_api, "is_installed", lambda _: binary.is_file())
    assert len(front.missing_routes()) == 2
    binary.touch()
    models.mkdir()
    assert front.missing_routes() == []
    monkeypatch.setattr(front, "needs_vision", lambda: False)
    install = Mock()
    monkeypatch.setattr(front.pixal3d, "install_multiview", install)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: terminal))
    monkeypatch.setattr("builtins.input", lambda _: answer)
    assert front.main([]) == status
    assert install.call_count == (status == 0)


@pytest.mark.parametrize(("answer", "status"), [("no", 1), ("yes", 0)])
def test_xiong_announces_builds_and_dispatches_local_plan(tmp_path, answer, status, monkeypatch, capsys):
    monkeypatch.setattr(xiong, "REPO", tmp_path)
    monkeypatch.setattr(xiong.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: answer)
    run, remover = Mock(), Mock()
    monkeypatch.setattr(xiong.subprocess, "run", run)
    monkeypatch.setattr(xiong, "install_background_remover", remover)
    assert xiong.main([]) == status
    assert "uv sync in hunyuan_mlx/shape, hunyuan_mlx/paint" in capsys.readouterr().out
    assert run.call_count == (3 if status == 0 else 0)
    assert remover.call_count == (status == 0)


def test_qwen_handles_release_without_matching_assets(tmp_path, monkeypatch):
    monkeypatch.setattr(qwen, "target", lambda: "macos-arm64")
    monkeypatch.setattr(qwen.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"assets":[]}'))
    with pytest.raises(SystemExit, match="no .* build"):
        qwen.install_binary(tmp_path)
    with pytest.raises(SystemExit, match="contained no sd-cli"):
        qwen.finish_install(tmp_path)


def test_qwen_ignores_absent_and_already_modified_archive_links(tmp_path):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        for name in ("exists", "missing"):
            link = ZipInfo(name)
            link.external_attr = 0xA0000000
            archive.writestr(link, "target")
    (tmp_path / "exists").write_text("changed")
    with ZipFile(buffer) as archive:
        qwen.restore_symlinks(archive, tmp_path)
    assert (tmp_path / "exists").read_text() == "changed"
    assert not (tmp_path / "missing").exists()


def test_rig_installs_venv_when_missing_and_preserves_model_links(tmp_path, monkeypatch):
    monkeypatch.setattr(rig.host, "find_uv", lambda: "uv")
    commands = Mock()
    assert rig.venv_needs_rebuild(tmp_path)
    rig.install_packages(tmp_path, commands, index="https://packages.invalid/cpu")
    assert commands.call_args_list[0].args[0][:2] == ["uv", "venv"]
    assert commands.call_args_list[-1].args[0][-1] == str(tmp_path / "requirements.txt")
    import huggingface_hub

    cached = tmp_path / "cache"
    cached.write_text("tiny placeholder")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: str(cached))
    snapshot = Mock()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    rig.install_weights(tmp_path)
    assert all((tmp_path / checkpoint.path).read_text() == cached.read_text() for checkpoint in rig.CHECKPOINTS)
    assert snapshot.call_args.kwargs["ignore_patterns"] == ["*.bin", "*.safetensors"]


@pytest.mark.parametrize("module", [hy_cuda, t_cuda])
def test_cuda_build_without_toolkit_keeps_original_paths(module):
    env = module.build_env(None, None, {"PATH": "/prior", "CUDA_HOME": "/old"})
    assert env["PATH"] == "/prior"
    assert env["CUDA_HOME"] == "/old"
    assert "TORCH_CUDA_ARCH_LIST" not in env
    assert int(env["MAX_JOBS"]) >= 1


@pytest.mark.parametrize(("module", "key"), [(sf_rocm, "ROCM_HOME"), (t_rocm, "ROCM_PATH"), (hy_rocm, "ROCM_HOME")])
def test_hip_build_handles_compiler_outside_rocm_root(module, key, monkeypatch):
    monkeypatch.setattr(module.host, "rocm_root", lambda _: None)
    kw = {"format_workaround": False} if module is t_rocm else {}
    env = module.build_env([], "/tools/bin/hipcc", {"PATH": "/prior"}, **kw)
    assert key not in env
    assert env["PATH"].split(module.os.pathsep)[0].replace("\\", "/").endswith("/tools/bin")


def test_rocm_plans_reject_unknown_torch_tags(monkeypatch, tmp_path):
    monkeypatch.setattr(hy_rocm.host, "torch_rocm_index", lambda _: "https://packages.invalid/future")
    for module in (hy_rocm, sf_rocm):
        args = ((7, 0), "/tools/hipcc", ["gfx1100"])
        route, reason = module.choose_route(*args)
        assert route is None
        assert "future" in reason
    with pytest.raises(ValueError, match="no install plan"):
        hy_rocm.pip_commands("wrong", "uv", tmp_path / "python", [])
    with pytest.raises(ValueError, match="no install plan"):
        sf_rocm.pip_commands("wrong", "uv", tmp_path / "python")
    with pytest.raises(ValueError, match="without a ROCm version"):
        t_rocm.pip_commands("source", "uv", tmp_path / "python")


def test_cuda_and_rocm_use_only_their_own_pinned_requirements(tmp_path):
    cuda_lock, rocm_lock = [tmp_path / f"packages-{route}.txt" for route in ("cuda", "rocm")]
    cuda_lock.touch()
    rocm_lock.touch()
    assert hy_cuda.requirement_args(cuda_lock) == ["-r", str(cuda_lock)]
    assert t_rocm.basic_args(rocm_lock) == ["-r", str(rocm_lock)]
    assert t_rocm.basic_args(cuda_lock) == t_rocm.BASIC_ROCM


@pytest.mark.parametrize("route", ["source", "pinned"])
def test_trellis_cuda_installer_runs_commands_and_records_selected_source(tmp_path, monkeypatch, route):
    monkeypatch.setattr(t_cuda, "VENDOR", tmp_path)
    marker = tmp_path / "installed.json"
    monkeypatch.setattr(t_cuda, "BUILT_MARKER", marker)
    monkeypatch.setattr(t_cuda.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(t_cuda, "pinned_commit", lambda: "fixture")
    monkeypatch.setattr(t_cuda, "venv_python", lambda: tmp_path / "python")
    monkeypatch.setattr(t_cuda, "fetch_checkout", Mock())
    monkeypatch.setattr(t_cuda.bria_patch, "apply", Mock(return_value=True))
    monkeypatch.setattr(t_cuda, "install_env", lambda _: {"PATH": "/buildtools"})
    monkeypatch.setattr(t_cuda, "pip_commands", lambda *_: [["uv", "pip", "install", "package"]])
    clone, run = Mock(), Mock()
    monkeypatch.setattr(t_cuda, "clone", clone)
    monkeypatch.setattr(t_cuda, "run", run)
    if route == "pinned":
        (tmp_path / "python").touch()
    t_cuda.install_code(route, "cu128")
    assert clone.call_count == (len(t_cuda.SOURCE_EXTENSIONS) if route == "source" else 0)
    assert run.call_count == (2 if route == "source" else 1)
    run.assert_called_with(["uv", "pip", "install", "package"], env={"PATH": "/buildtools"})
    assert route in marker.read_text()


def test_sf3d_rocm_runs_gpu_probe_before_compiling(tmp_path, monkeypatch):
    python = tmp_path / ".venv/bin/python"
    marker = tmp_path / "installed.json"
    monkeypatch.setattr(sf_rocm, "BUILT_MARKER", marker)
    monkeypatch.setattr(sf_rocm, "venv_python", lambda: python)
    monkeypatch.setattr(sf_rocm.host, "find_uv", lambda: "uv")
    monkeypatch.setattr(sf_rocm, "fetch_checkout", Mock())
    monkeypatch.setattr(sf_rocm.sf3d_patch, "patch", Mock())
    monkeypatch.setattr(sf_rocm, "install_env", dict)
    monkeypatch.setattr(sf_rocm, "pip_commands", lambda *_: [["torch"], ["baker"]])
    trace = []
    monkeypatch.setattr(sf_rocm, "run", lambda cmd, **_: trace.append(cmd[0]))
    monkeypatch.setattr(t_rocm, "gpu_probe", lambda _: (trace.append("probe") or True, "ok"))
    sf_rocm.install_code(next(iter(sf_rocm.TORCH)))
    assert trace == ["uv", "torch", "probe", "baker"]
    assert marker.is_file()
    monkeypatch.setattr(t_rocm, "gpu_probe", lambda _: (False, "fixture device failure"))
    with pytest.raises(SystemExit, match="texture baker was not compiled"):
        sf_rocm.prove_torch(python)


def test_trellis_build_reuses_existing_checkout_and_handles_exact_ref(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(t_rocm, "run", run)
    t_rocm.clone("local", tmp_path / "first", "b" * 40)
    assert [call.args[0][1] for call in run.call_args_list] == ["clone", "checkout"]
    monkeypatch.setattr(t_rocm.host, "amd_torch_build", lambda _: ("2", "1"))
    monkeypatch.setattr(t_rocm.host, "torch_rocm_index", lambda _: None)
    assert [candidate.label for candidate in t_rocm.torch_candidates((99, 0))] == ["AMD's own index"]


def test_hunyuan_rocm_consent_runs_code_only(tmp_path, monkeypatch):
    monkeypatch.setattr(hy_rocm.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(hy_rocm, "detect", lambda: (next(iter(hy_rocm.TORCH)), "fixture"))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    build, weights = Mock(), Mock()
    monkeypatch.setattr(hy_rocm, "install_code", build)
    monkeypatch.setattr(hy_rocm, "install_weights", weights)
    assert hy_rocm.main(["--code-only"]) == 0
    build.assert_called_once()
    weights.assert_not_called()


@pytest.mark.parametrize("module", [flux, qwen])
def test_picture_engines_decline_and_list_exact_weight_files(module, monkeypatch):
    import huggingface_hub

    backend = flux.sdcpp if module is flux else module
    inputs, install = Mock(return_value="no"), Mock()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", inputs)
    monkeypatch.setattr(backend, "target", lambda: next(iter(backend.BUILDS)))
    monkeypatch.setattr(backend, "binary_present", lambda: False)
    monkeypatch.setattr(backend, "ensure_binary", install)
    assert module.main([]) == 1
    install.assert_not_called()
    fetch = Mock()
    monkeypatch.setattr(backend, "fetch_pinned", fetch)
    download = Mock(side_effect=AssertionError("the weight helper may only plan downloads in this test"))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    module.install_weights()
    expected = (*qwen.WEIGHTS, qwen.VISION) if module is qwen else flux.WEIGHTS
    assert [call.args[1:4] for call in fetch.call_args_list] == list(expected)
    download.assert_not_called()


def test_upscaler_checks_release_size_before_decoding(tmp_path, monkeypatch):
    target = tmp_path / "check.safetensors"
    target.write_bytes(b"123")
    monkeypatch.setattr(up.upscale, "WEIGHTS_BYTES", 4)
    assert up.verify(target) == "size 3 is not the release's 4"
    monkeypatch.setattr(up.upscale, "WEIGHTS_BYTES", 3)
    monkeypatch.setattr(up.upscale, "network", Mock(side_effect=ValueError("invalid tensor")))
    assert "invalid tensor" in up.verify(target)


@pytest.mark.parametrize(("kind", "hip", "clang", "result"), [
    ("metal-source", None, None, "build finished without"),
    ("hip-source", None, None, "hipcc.*not on PATH"),
    ("hip-source", "/tools/hipcc", None, "Clang was not found"),
])
def test_pixal_checks_toolkit_and_build_output(tmp_path, monkeypatch, kind, hip, clang, result):
    monkeypatch.setattr(pixal.shutil, "which", lambda command: f"/tools/{command}")
    monkeypatch.setattr(pixal, "find_metal", lambda: {"DEVELOPER_DIR": str(tmp_path)})
    monkeypatch.setattr(pixal.os, "environ", pixal.os.environ.copy())
    monkeypatch.setattr(pixal, "find_hipcc", lambda: hip)
    monkeypatch.setattr(pixal, "hip_compiler", lambda _: clang)
    monkeypatch.setattr(pixal, "find_nvcc", lambda: None)
    monkeypatch.setattr(pixal.host, "compute_capability", lambda: None)
    monkeypatch.setattr(pixal, "fetch_source", Mock())
    for patch in ("apply_steps_patch", "apply_dump_patch", "apply_shape_patch"):
        monkeypatch.setattr(pixal, patch, Mock())
    monkeypatch.setattr(pixal, "build_present", lambda: False)
    run = Mock()
    monkeypatch.setattr(pixal.subprocess, "run", run)
    with pytest.raises(SystemExit, match=result):
        pixal.build_from_source(kind)
    assert run.call_count == (2 if kind == "metal-source" else 0)


def test_pixal_source_checkout_keeps_preexisting_weights_and_reuses_git(tmp_path, monkeypatch):
    monkeypatch.setattr(pixal, "VENDOR", tmp_path)
    weights = tmp_path / "weights"
    weights.write_text("fixture")
    run = Mock()
    monkeypatch.setattr(pixal.subprocess, "run", run)
    pixal.fetch_source("pinned")
    assert run.call_args_list[0].args[0] == ["git", "init", "-q", str(tmp_path)]
    assert run.call_count == 5
    (tmp_path / ".git").mkdir()
    run.reset_mock()
    pixal.fetch_source("pinned")
    assert run.call_count == 1
    assert run.call_args.args[0][-3:] == ["update", "--init", "--recursive"]
    assert weights.read_text() == "fixture"


def test_pixal_refuses_missing_tools_and_prebuilt_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(pixal.shutil, "which", lambda _: None)
    with pytest.raises(SystemExit, match="cmake not found"):
        pixal.build_from_source("metal-source")
    monkeypatch.setattr(pixal.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b'{"assets":[]}'))
    with pytest.raises(SystemExit, match="has no"):
        pixal.install_prebuilt(next(iter(pixal.PREBUILTS)))
    monkeypatch.setattr(pixal, "driver_cuda", lambda: (11, 8))
    assert "needs" in pixal.no_route_message("windows-nvidia")
    assert "nvcc" not in pixal.no_route_message("windows-nvidia")


@pytest.mark.parametrize(("terminal", "answer", "status"), [(False, "", 1), (True, "no", 1), (True, "yes", 0)])
def test_pixal_reports_prebuilt_cannot_rebuild_without_replacing_it(monkeypatch, terminal, answer, status, capsys):
    monkeypatch.setattr(pixal, "target", lambda: "linux-nvidia")
    monkeypatch.setattr(pixal, "current_kind", lambda *_: "prebuilt")
    monkeypatch.setattr(pixal, "build_present", lambda: True)
    monkeypatch.setattr(pixal, "build_stale", lambda: False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: terminal))
    monkeypatch.setattr("builtins.input", lambda _: answer)
    install = Mock()
    monkeypatch.setattr(pixal, "install_build", install)
    assert pixal.main(["--rebuild", "--build-only"]) == status
    install.assert_not_called()
    assert ("nothing to rebuild" in capsys.readouterr().out) == (status == 0)


def test_pixal_rebuild_and_shared_weight_reuse(tmp_path, monkeypatch):
    src, dest = tmp_path / "single", tmp_path / "multi"
    dest.mkdir()
    for shared in pixal.MV_SHARED:
        (dest / shared).parent.mkdir(parents=True, exist_ok=True)
        (dest / shared).write_text("preserve")
    assert pixal.link_shared(src, dest) == []
    assert all((dest / shared).read_text() == "preserve" for shared in pixal.MV_SHARED)
    monkeypatch.setattr(pixal, "build_present", lambda: False)
    for patch in ("apply_steps_patch", "apply_dump_patch", "apply_shape_patch"):
        monkeypatch.setattr(pixal, patch, Mock())
    with pytest.raises(SystemExit, match="rebuild finished without"):
        pixal.rebuild_existing(runner=Mock())


def test_qwen_installs_matching_archive_fixture_not_host_tools(tmp_path, monkeypatch):
    archive_name = "fixture-darwin-arm64.zip"
    payload = io.BytesIO()
    binary = qwen.host.executable(tmp_path, "sd-cli")
    with ZipFile(payload, "w") as archive:
        archive.writestr(binary.name, b"fixture")
    monkeypatch.setattr(qwen, "target", lambda: "macos-arm64")
    release = b'{"tag_name":"unit","assets":[{"name":"fixture-darwin-arm64.zip","browser_download_url":"unused"}]}'
    monkeypatch.setattr(qwen.urllib.request, "urlopen", lambda *_, **kwargs: io.BytesIO(release))
    monkeypatch.setattr(qwen.urllib.request, "urlretrieve", lambda _, path: path.write_bytes(payload.getvalue()))
    import hashlib

    monkeypatch.setitem(qwen.ARCHIVE_SHA256, archive_name, hashlib.sha256(payload.getvalue()).hexdigest())
    assert qwen.install_binary(tmp_path) == binary
    assert binary.read_bytes() == b"fixture"
    assert not (tmp_path / "release.zip").exists()


@pytest.mark.parametrize("module", [flux, qwen])
def test_picture_engine_consent_provisions_after_showing_the_plan(module, monkeypatch):
    backend = flux.sdcpp if module is flux else module
    monkeypatch.setattr(backend, "target", lambda: "macos-arm64")
    monkeypatch.setattr(backend, "binary_present", lambda: False)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda _: "YES")
    binary, weights = Mock(return_value=True), Mock()
    monkeypatch.setattr(backend, "ensure_binary", binary)
    monkeypatch.setattr(module, "install_weights", weights)
    assert module.main([]) == 0
    binary.assert_called_once()
    weights.assert_called_once()


@pytest.mark.parametrize("module", [sf, xiong])
def test_background_remover_can_be_dispatched_without_scripts_on_sys_path(module, monkeypatch, tmp_path):
    scripts = str(module.REPO / "scripts")
    monkeypatch.setattr(sys, "path", [path for path in sys.path if path != scripts])
    install = Mock()
    monkeypatch.setattr(matte, "install_if_missing", install)
    downloader = Mock()
    module.install_background_remover(tmp_path / "fixture", downloader)
    assert scripts in sys.path
    install.assert_called_once_with(tmp_path / "fixture", downloader)


def test_xiong_keeps_both_existing_environments(tmp_path, monkeypatch):
    monkeypatch.setattr(xiong, "REPO", tmp_path)
    monkeypatch.setattr(xiong.host, "find_uv", lambda: "uv")
    for stage in xiong.HALVES:
        python = xiong.venv_python(tmp_path / "hunyuan_mlx" / stage)
        python.parent.mkdir(parents=True)
        python.touch()
    steps = xiong.plan(tmp_path, "uv")
    assert len(steps) == 1
    assert "uv sync" not in xiong.announcement(steps)
    run = Mock()
    monkeypatch.setattr(xiong.subprocess, "run", run)
    monkeypatch.setattr(xiong, "install_background_remover", Mock())
    assert xiong.main(["--yes"]) == 0
    run.assert_called_once_with(steps[0][0], cwd=tmp_path, check=True)


def test_hunyuan_falls_back_to_filtered_upstream_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(hy_cuda, "CHECKOUT", tmp_path)
    (tmp_path / "requirements.txt").write_text("--index-url https://unused.invalid\nexample==1.0\n")
    assert hy_cuda.requirement_args(tmp_path / "missing.txt") == ["example==1.0"]


@pytest.mark.parametrize("module", [t_cuda, t_rocm])
def test_pinned_build_at_correct_head_fetches_only_submodules(tmp_path, monkeypatch, module):
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(module, "CHECKOUT", checkout)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "pinned\n")
    run = Mock()
    monkeypatch.setattr(module, "run", run)
    module.fetch_checkout("pinned")
    run.assert_called_once_with(["git", "submodule", "update", "--init", "--recursive"], cwd=checkout)


def test_cmake_hip_flags_keep_default_rocm_discovery(monkeypatch):
    monkeypatch.setattr(pixal.host, "rocm_root", lambda _: None)
    monkeypatch.setattr(pixal, "hip_compiler", lambda _: None)
    flags = pixal.cmake_flags("hip-source", None, None, "/tools/hipcc", ["gfx1100"])
    assert "-DGPU_ARCHS=gfx1100" not in flags
    assert "-DAMDGPU_TARGETS=gfx1100" in flags
    assert not any(flag.startswith("-DROCM_PATH=") for flag in flags)


def test_upscaler_passes_affirmative_consent_to_selected_path(tmp_path, monkeypatch):
    target = tmp_path / "fixture"
    monkeypatch.setattr(up.upscale, "WEIGHTS_FILE", target)
    monkeypatch.setattr(up.upscale, "installed", lambda _: False)
    fetch = Mock()
    assert up.main([], ask=lambda _: "yes", fetch=fetch) == 0
    fetch.assert_called_once_with(target)
