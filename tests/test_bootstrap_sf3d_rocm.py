"""scripts/bootstrap_sf3d_rocm.py: the AMD route's install decisions.

The questions are the ones a wrong answer costs an afternoon on: which PyTorch the card
gets, whether the CUDA wheel sneaks in from PyPI, whether the texture baker is told to
build its GPU kernels at all, and what the announcement says before anyone consents to a
download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "viewer"))

import backend_catalog
import bootstrap_sf3d as apple
import bootstrap_sf3d_rocm as boot

HIPCC = "/opt/rocm/bin/hipcc"
GFX = ["gfx1100"]


def test_the_route_is_chosen_from_the_installed_rocm():
    route, why = boot.choose_route((7, 2), HIPCC, GFX)
    assert route == "rocm7.2"
    assert "gfx1100" in why and "2.14.1" in why


@pytest.mark.parametrize("version,expected", [
    ((7, 2), "rocm7.2"), ((7, 1), "rocm7.1"), ((7, 0), "rocm7.0"),
    ((6, 4), "rocm6.4"), ((6, 3), "rocm6.3"), ((6, 2), "rocm6.2"),
])
def test_every_rocm_build_pytorch_publishes_has_a_torch_here(version, expected):
    assert boot.choose_route(version, HIPCC, GFX)[0] == expected
    assert expected in boot.TORCH


@pytest.mark.parametrize("version,hipcc,gfx,missing", [
    (None, HIPCC, GFX, "ROCm"),
    ((7, 2), None, GFX, "hipcc"),
    ((6, 0), HIPCC, GFX, "ROCm"),
    ((7, 2), HIPCC, [], "gfx"),
])
def test_it_refuses_rather_than_installing_something_that_cannot_run(version, hipcc, gfx,
                                                                     missing):
    route, why = boot.choose_route(version, hipcc, gfx)
    assert route is None
    assert missing in why


def test_torch_comes_from_the_rocm_index_not_from_pypi():
    """PyPI's Linux torch is a CUDA build: it installs cleanly and then reports no device."""
    commands = boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"))
    torch_step = next(c for c in commands if any(a.startswith("torch==") for a in c))
    assert "--index-url" in torch_step
    assert torch_step[torch_step.index("--index-url") + 1] == \
        "https://download.pytorch.org/whl/rocm7.2"
    assert any(a.startswith("torchvision==") for a in torch_step)


def test_the_requirements_file_is_passed_with_dash_r():
    """uv reads a bare path as an archive to install, and rejects a requirements.txt for
    not ending in .whl or .tar.gz. That cost a whole install before it was checked."""
    commands = boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"))
    requirements_step = next(c for c in commands if "requirements.txt" in " ".join(c))
    assert requirements_step[requirements_step.index("--no-build-isolation") + 1] == "-r"
    assert requirements_step[-1].endswith("requirements.txt")


def test_the_extensions_build_against_the_torch_just_installed():
    """--no-build-isolation: texture_baker's setup.py imports torch, and must not compile
    against a fresh one."""
    commands = boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"))
    assert any(a.startswith("torch==") for a in commands[0]), "torch is installed first"
    assert "--no-build-isolation" in commands[-1]


def test_the_build_is_told_to_include_the_gpu_kernels():
    """USE_CUDA=1 is what keeps baker_kernel.cu in the build. Without it the extension
    builds and has no GPU kernels in it, with nothing to say so."""
    env = boot.build_env(GFX, HIPCC, base={"PATH": "/usr/bin"})
    assert env["USE_CUDA"] == "1"
    assert env["USE_METAL"] == "0"
    assert env["PYTORCH_ROCM_ARCH"] == "gfx1100"
    assert env["ROCM_HOME"] == "/opt/rocm"
    assert "/opt/rocm/bin" in env["PATH"]


def test_a_machine_with_no_card_still_gets_a_readable_environment():
    env = boot.build_env([], None, base={"PATH": "/usr/bin"})
    assert "PYTORCH_ROCM_ARCH" not in env
    assert env["PATH"] == "/usr/bin"


def test_it_installs_the_same_upstream_commit_as_the_mac_route():
    """One checkout for both, so the two cannot drift in what they built."""
    assert boot.COMMIT == apple.COMMIT
    assert boot.VENDOR == apple.VENDOR


def test_the_patches_run_before_anything_builds(monkeypatch, tmp_path):
    """An unpatched setup.py compiles a CPU-only baker and says nothing, so this has to
    happen first rather than be noticed afterwards."""
    order: list[str] = []
    monkeypatch.setattr(boot, "fetch_checkout", lambda *a, **k: order.append("clone"))
    monkeypatch.setattr(boot.sf3d_patch, "patch",
                        lambda *a, **k: order.append("patch"))
    monkeypatch.setattr(boot, "venv_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [["uv", "pip", "install", "x"]])
    monkeypatch.setattr(boot, "install_env", dict)
    monkeypatch.setattr(boot, "BUILT_MARKER", tmp_path / ".i2l-build-complete")
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(boot, "run", lambda *a, **k: None)
    monkeypatch.setattr(boot, "prove_torch", lambda *a, **k: None)
    boot.install_code("rocm7.2")
    assert order.index("patch") < len(order)
    assert order[0] == "clone" and order[1] == "patch"


def test_a_build_that_dies_leaves_no_marker(monkeypatch, tmp_path):
    marker = tmp_path / ".i2l-build-complete"
    monkeypatch.setattr(boot, "fetch_checkout", lambda *a, **k: None)
    monkeypatch.setattr(boot.sf3d_patch, "patch", lambda *a, **k: None)
    monkeypatch.setattr(boot, "venv_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [["uv", "pip", "install", "x"]])
    monkeypatch.setattr(boot, "install_env", dict)
    monkeypatch.setattr(boot, "BUILT_MARKER", marker)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(boot, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("build failed")))
    with pytest.raises(RuntimeError, match="build failed"):
        boot.install_code("rocm7.2")
    assert not marker.exists()


def test_the_announcement_names_the_route_the_size_and_the_licence():
    """AGENTS.md: a download path must name the backend, name the route, state the size,
    and the gated licence with its revenue condition."""
    text = boot.announcement("rocm7.2", "PyTorch 2.14.1", code=True, weights=True)
    assert "backend: Stable Fast 3D" in text
    assert "AMD / ROCm" in text and "rocm7.2" in text
    assert f"{boot.total_gb():.1f} GB" in text
    assert "gated" in text.lower()
    assert "$1M" in text


def test_the_announcement_does_not_claim_a_size_when_nothing_is_downloading():
    assert "GB total" not in boot.announcement("rocm7.2", "x", code=True, weights=False)


def test_it_refuses_to_run_on_the_wrong_machine_before_asking_anything(monkeypatch, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda: "macos-arm64")
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "bootstrap_sf3d.py" in out
    assert "Nothing downloaded" in out


def test_it_refuses_without_rocm(monkeypatch, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: (None, "no ROCm"))
    monkeypatch.setattr(boot, "build_present", lambda: False)
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_it_will_not_install_without_a_yes_and_nobody_to_ask(monkeypatch, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: ("rocm7.2", "PyTorch 2.14.1"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out


def test_the_weights_are_the_ones_the_apple_route_fetches():
    """One list for both, so the Setup card and the installer cannot disagree."""
    assert boot.WEIGHTS == apple.WEIGHTS
    assert boot.LICENCE == apple.LICENCE


def test_the_catalogue_offers_this_route_on_amd():
    entry = backend_catalog.BY_ID["sf3d"]
    assert entry.runs_here(backend_catalog.AMD, "linux") is True
    assert "bootstrap_sf3d_rocm.py" in entry.install_for(backend_catalog.AMD)
    # The Apple route keeps its own installer, which installs into the shared interpreter.
    assert "bootstrap_sf3d.py" in entry.install_for(backend_catalog.APPLE)


def test_the_catalogue_readiness_probe_is_this_routes_venv_not_the_apple_checkout():
    """Both routes share one checkout, so a probe of a source file would report the AMD
    route ready on a Mac that has only ever built the Apple one."""
    entry = backend_catalog.BY_ID["sf3d"]
    apple_probes = entry.probes_for(backend_catalog.APPLE)
    amd_probes = entry.probes_for(backend_catalog.AMD)
    # The venv interpreter is what proves this route built; the marker proves it finished.
    assert any(".venv" in str(p) for p in amd_probes), amd_probes
    assert any(p.name.endswith(".i2l-build-complete") for p in amd_probes), amd_probes
    assert apple_probes != amd_probes

def test_the_background_remover_is_announced_and_fetched_with_the_weights(monkeypatch,
                                                                          tmp_path):
    """sf3d_backend.py cuts pictures out with BiRefNet-lite. A setup that skips it leaves the
    first run to fetch u2net nobody agreed to."""
    assert "BiRefNet-lite" in boot.announcement("rocm7.2", "x", code=False, weights=True)
    fetched = []
    boot.install_background_remover(tmp_path / "lite.onnx", download=fetched.append)
    assert fetched == [tmp_path / "lite.onnx"]


def _amd_main(monkeypatch, built: bool) -> list[str]:
    """main() on an AMD box with every slow step replaced by a note of having run."""
    done: list[str] = []
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: ("rocm7.2", "PyTorch 2.14.1"))
    monkeypatch.setattr(boot, "build_present", lambda: built)
    monkeypatch.setattr(boot, "install_code", lambda route: done.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: done.append("weights"))
    return done


def test_a_rerun_after_a_gated_licence_failure_skips_the_build(monkeypatch, capsys):
    """The build finished, the gated weights did not. Again should not mean another
    compile."""
    done = _amd_main(monkeypatch, built=True)
    assert boot.main(["--yes"]) == 0
    assert done == ["weights"]
    out = capsys.readouterr().out
    assert "already built" in out and "--rebuild" in out


def test_rebuild_builds_over_a_finished_build(monkeypatch):
    done = _amd_main(monkeypatch, built=True)
    assert boot.main(["--yes", "--rebuild"]) == 0
    assert done == ["code", "weights"]


def test_a_first_install_builds_then_fetches(monkeypatch):
    done = _amd_main(monkeypatch, built=False)
    assert boot.main(["--yes"]) == 0
    assert done == ["code", "weights"]


def test_code_only_on_a_finished_build_does_nothing_and_asks_nothing(monkeypatch):
    done = _amd_main(monkeypatch, built=True)
    monkeypatch.setattr(sys, "stdin", None)  # a prompt would refuse, not pass
    assert boot.main(["--code-only"]) == 0
    assert done == []


def test_a_torch_that_cannot_run_on_the_card_stops_before_the_baker_compiles(monkeypatch,
                                                                            tmp_path):
    ran: list[list[str]] = []
    marker = tmp_path / ".i2l-build-complete"
    monkeypatch.setattr(boot, "fetch_checkout", lambda *a, **k: None)
    monkeypatch.setattr(boot.sf3d_patch, "patch", lambda *a, **k: None)
    monkeypatch.setattr(boot, "venv_python", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(boot, "install_env", dict)
    monkeypatch.setattr(boot, "BUILT_MARKER", marker)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(boot, "run", lambda command, **k: ran.append(command))
    import bootstrap_trellis_rocm

    monkeypatch.setattr(bootstrap_trellis_rocm, "gpu_probe",
                        lambda python: (False, "hipErrorInvalidKernelFile"))
    with pytest.raises(SystemExit, match="hipErrorInvalidKernelFile"):
        boot.install_code("rocm7.2")
    assert len(ran) == 2 and "torch==" in " ".join(ran[1]), "the baker compiled anyway"
    assert not marker.exists()


def test_install_code_only_ever_deletes_the_venv_it_was_pointed_at(monkeypatch, tmp_path):
    """It used to delete VENDOR/.venv by name, so every test that faked venv_python() wiped
    the real SF3D venv on the machine running the suite."""
    real = tmp_path / "real" / ".venv"
    real.mkdir(parents=True)
    monkeypatch.setattr(boot, "VENDOR", tmp_path / "real")
    monkeypatch.setattr(boot, "fetch_checkout", lambda *a, **k: None)
    monkeypatch.setattr(boot.sf3d_patch, "patch", lambda *a, **k: None)
    monkeypatch.setattr(boot, "venv_python", lambda: tmp_path / "fake" / ".venv" / "bin" / "python")
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [["uv", "pip", "install", "x"]])
    monkeypatch.setattr(boot, "install_env", dict)
    monkeypatch.setattr(boot, "BUILT_MARKER", tmp_path / ".i2l-build-complete")
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    ran: list[list[str]] = []
    monkeypatch.setattr(boot, "run", lambda command, **k: ran.append(command))
    monkeypatch.setattr(boot, "prove_torch", lambda *a, **k: None)
    boot.install_code("rocm7.2")
    assert real.is_dir(), "deleted a venv it was not pointed at"
    assert ran[0][:3] == ["uv", "venv", str(tmp_path / "fake" / ".venv")]


def test_a_marker_without_its_venv_is_not_a_build(monkeypatch, tmp_path):
    marker = tmp_path / ".i2l-build-complete"
    python = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setattr(boot, "BUILT_MARKER", marker)
    monkeypatch.setattr(boot, "venv_python", lambda: python)
    marker.write_text("{}")
    assert boot.build_present() is False
    python.parent.mkdir(parents=True)
    python.write_text("")
    assert boot.build_present() is True
