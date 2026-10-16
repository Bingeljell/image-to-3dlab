"""scripts/bootstrap_trellis_rocm.py: what it would install, and what it refuses to do.

The plan is pure so the order of the installs can be checked without a 40-minute build,
and the route choice is pure because "why can't this machine install it" is a question the
viewer asks on its own, long before anybody runs this.

The rule that matters most here is AGENTS.md's: a download path must name the backend, name
the route, state the size, and require an affirmative answer. A test that fails is a test
that stops a silent 16 GB download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_trellis_rocm as boot

from image_to_3dlab import host

PYTHON = Path("/venv/bin/python")


# --- the route: can this machine install it at all -----------------------------------------

def test_this_machine_has_a_route():
    route, why = boot.detect()
    assert route == "source", why
    assert "gfx" in why


@pytest.mark.parametrize("version, hipcc, gfx, expected", [
    ((7, 2), "/opt/rocm/bin/hipcc", ["gfx1100"], "source"),
    ((6, 4), "/opt/rocm/bin/hipcc", ["gfx1030"], "source"),
    # No card: nothing to compile for, so nothing to install.
    (None, "/opt/rocm/bin/hipcc", [], None),
    ((7, 2), "/opt/rocm/bin/hipcc", [], None),
    # No toolkit: these extensions are compiled, so hipcc is not optional.
    ((7, 2), None, ["gfx1100"], None),
    # Unreadable ROCm version: the PyTorch wheel is published per ROCm release.
    (None, "/opt/rocm/bin/hipcc", ["gfx1100"], None),
])
def test_choose_route(version, hipcc, gfx, expected):
    route, why = boot.choose_route(version, hipcc, gfx)
    assert route == expected
    assert why, "every route decision must explain itself"


def test_an_unknown_rocm_release_is_refused_with_the_supported_ones():
    route, why = boot.choose_route((5, 7), "/opt/rocm/bin/hipcc", ["gfx1100"])
    assert route is None
    assert "5.7" in why and "7.2" in why, "the message should list what would have worked"


def test_every_supported_rocm_release_has_a_torch_pair():
    for version in boot.TORCH:
        torch_version, vision_version = boot.TORCH[version]
        assert torch_version.count(".") == 2, version
        assert vision_version.count(".") == 2, version


# --- the announcement, which is the only thing standing between a typo and 16 GB -----------

def test_the_announcement_names_the_backend_the_route_and_the_size():
    text = boot.announcement("source", "because", code=True, weights=True)
    assert "TRELLIS.2" in text
    assert "AMD / ROCm" in text
    assert f"{boot.total_gb():.1f} GB" in text
    for repo, _, _ in boot.WEIGHTS:
        assert repo in text, f"{repo} is fetched but not announced"


def test_the_announcement_names_the_compiled_extensions_and_which_source_they_come_from():
    text = boot.announcement("source", "because", code=True, weights=False)
    assert "nvdiffrast" in text and "hipcc" in text
    assert "minutes" in text, "a 40 minute compile should not be a surprise"


def test_the_announcement_is_honest_about_which_fork_is_used():
    """Only nvdiffrast comes from a fork; saying otherwise would misstate the trust boundary."""
    text = boot.announcement("source", "because", code=True, weights=False)
    assert "dmonkman" in text
    assert "upstream" in text


def test_the_licence_names_the_gated_model():
    text = boot.announcement("source", "because", code=True, weights=True)
    assert boot.GATED in text
    assert "gated" in text.lower()
    assert "BRIA" in text, "the guardrail must be stated, not just applied"


def test_weights_alone_do_not_mention_the_compile():
    text = boot.announcement("source", "because", code=False, weights=True)
    assert "GB" in text
    assert "hipcc" not in text


# --- the install plan ----------------------------------------------------------------------

def test_the_plan_is_torch_then_the_extensions_then_o_voxel():
    """Order matters: the extensions must compile against the torch just installed, and
    o-voxel goes last so its git dependencies resolve to what we just built."""
    commands = boot.pip_commands("source", "uv", PYTHON, (7, 2))
    joined = [" ".join(c) for c in commands]
    torch_at = next(i for i, c in enumerate(joined) if "torch==" in c)
    first_ext = next(i for i, c in enumerate(joined) if "--no-build-isolation" in c)
    assert torch_at < first_ext
    assert joined[-1].endswith("o-voxel")
    assert "--no-build-isolation" in joined[-1]


def test_every_extension_is_installed_no_build_isolation():
    commands = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (7, 2))]
    for name, *_ in boot.SOURCE_EXTENSIONS:
        matching = [c for c in commands if f"extensions/{name}" in c]
        assert len(matching) == 1, f"{name} is installed {len(matching)} times"
        assert "--no-build-isolation" in matching[0], (
            f"{name}: without it pip builds against a throwaway torch, not the installed one")


def test_o_voxel_is_installed_without_dependencies():
    """o-voxel names CuMesh and FlexGEMM as git dependencies and would compile them again."""
    last = boot.pip_commands("source", "uv", PYTHON, (7, 2))[-1]
    assert "--no-deps" in last


def test_aule_is_installed_and_flash_attn_is_not():
    """AuleAttention is the whole reason attention works here; flash-attn cannot build."""
    joined = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (7, 2))]
    assert any(boot.AULE_ATTENTION in c for c in joined)
    assert not any("flash-attn" in c or "flash_attn" in c for c in joined)


def test_torch_comes_from_the_rocm_index_not_the_default_one():
    commands = boot.pip_commands("source", "uv", PYTHON, (7, 2))
    torch_command = next(c for c in commands if any("torch==" in a for a in c))
    index = torch_command[torch_command.index("--index-url") + 1]
    assert index == host.torch_rocm_index((7, 2))
    assert "rocm" in index, "a CUDA torch on an AMD card is the bug this route exists to fix"


def test_onnxruntime_is_not_installed():
    """Its 1.16.3 wheel asks for an executable stack the kernel here refuses, and nothing
    on the generation path needs it."""
    joined = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (7, 2))]
    assert not any("onnxruntime" in c for c in joined)


def test_an_unsupported_rocm_release_has_no_plan():
    with pytest.raises(ValueError, match="no PyTorch build"):
        boot.pip_commands("source", "uv", PYTHON, (5, 7))
    with pytest.raises(ValueError, match="no install plan"):
        boot.pip_commands("pinned", "uv", PYTHON, (7, 2))


def test_both_published_rocm_pairs_are_used_together():
    for version, (torch_version, vision_version) in boot.TORCH.items():
        commands = boot.pip_commands("source", "uv", PYTHON, version)
        line = next(c for c in commands if any("torch==" in a for a in c))
        assert f"torch=={torch_version}" in line
        assert f"torchvision=={vision_version}" in line


# --- the build environment ------------------------------------------------------------------

def test_archs_are_semicolon_separated():
    """These extensions split GPU_ARCHS on ';'. PYTORCH_ROCM_ARCH's comma would silently
    compile for a card named 'gfx1100,gfx1101', which is not a thing."""
    assert boot.ARCH_SEPARATOR == ";"
    assert boot.arch_list(["gfx1100", "gfx1030"]) == "gfx1100;gfx1030"


def test_the_build_environment_names_the_card_the_toolkit_and_the_jobs():
    env = boot.build_env(["gfx1100"], "/opt/rocm/bin/hipcc", {})
    assert env["GPU_ARCHS"] == "gfx1100"
    assert env["ROCM_PATH"] == "/opt/rocm"
    assert "/opt/rocm/bin" in env["PATH"]
    assert int(env["MAX_JOBS"]) >= 1


def test_the_build_environment_without_a_toolkit_still_names_the_card():
    """So the failure is hipcc's, not a missing GPU_ARCHS."""
    env = boot.build_env(["gfx1100"], None, {})
    assert env["GPU_ARCHS"] == "gfx1100"
    assert "ROCM_PATH" not in env


def test_only_nvdiffrast_comes_from_a_fork():
    """CuMesh, FlexGEMM and o-voxel carry HIP support upstream. Forking them would mean
    running different source on the two cards for no reason."""
    urls = {name: url for name, url, _, _ in boot.SOURCE_EXTENSIONS}
    assert "dmonkman" in urls["nvdiffrast"]
    for name in ("CuMesh", "FlexGEMM"):
        assert "dmonkman" not in urls[name], f"{name} should be upstream"
    # ... and the build patch has to agree.
    patched = {path for path, _ in boot.rocm_patch.CXX20_FILES}
    assert "extensions/CuMesh/setup.py" in patched
    assert "extensions/FlexGEMM/setup.py" in patched
    assert "TRELLIS.2/o-voxel/setup.py" in patched
    assert boot.rocm_patch.DEVICE_LIB_FILES == ("extensions/nvdiffrast/setup.py",)


# --- the command line -----------------------------------------------------------------------

def test_it_refuses_on_a_non_amd_machine(monkeypatch, capsys):
    monkeypatch.setattr(host, "build_target", lambda *a, **k: "linux-nvidia")
    assert boot.main([]) == 1
    out = capsys.readouterr().out
    assert "Nothing downloaded" in out
    assert "bootstrap_trellis_cuda.py" in out, "it should point at the NVIDIA route"


def test_it_refuses_without_confirmation_when_nobody_can_answer(monkeypatch, capsys):
    monkeypatch.setattr(boot, "detect", lambda: ("source", "because"))
    monkeypatch.setattr(host, "build_target", lambda *a, **k: "linux-amd")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out


def test_declining_downloads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(boot, "detect", lambda: ("source", "because"))
    monkeypatch.setattr(host, "build_target", lambda *a, **k: "linux-amd")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    called = []
    monkeypatch.setattr(boot, "install_code", lambda *a, **k: called.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: called.append("weights"))
    boot.main([])
    assert called == [], "declining still started an install"
    assert "Nothing downloaded" in capsys.readouterr().out


def test_an_impossible_machine_downloads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(host, "build_target", lambda *a, **k: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: (None, "no card here"))
    called = []
    monkeypatch.setattr(boot, "install_code", lambda *a, **k: called.append("code"))
    assert boot.main(["--yes"]) == 1
    assert called == []
    assert "no card here" in capsys.readouterr().out


def test_a_gated_model_stops_with_instructions(monkeypatch, capsys):
    monkeypatch.setattr(host, "build_target", lambda *a, **k: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: ("source", "because"))
    monkeypatch.setattr(host, "rocm_version", lambda *a, **k: (7, 2))
    monkeypatch.setattr(host, "rocm_gfx_targets", lambda *a, **k: ["gfx1100"])
    monkeypatch.setattr(boot, "check_access",
                        lambda *a, **k: (_ for _ in ()).throw(boot.GatedAccess(boot.GATED)))
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "gated" in out and "hf auth login" in out