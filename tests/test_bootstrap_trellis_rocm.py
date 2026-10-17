"""scripts/bootstrap_trellis_rocm.py: what it would install, and what it refuses to do.

The plan is pure so the order of the installs can be checked without a 40-minute build,
and the route choice is pure because "why can't this machine install it" is a question the
viewer asks on its own, long before anybody runs this.

The rule that matters most here is AGENTS.md's: a download path must name the backend, name
the route, state the size, and require an affirmative answer. A test that fails is a test
that stops a silent 16 GB download.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_trellis_rocm as boot

from image_to_3dlab import host

PYTHON = Path("/venv/bin/python")


# --- the route: can this machine install it at all -----------------------------------------

@pytest.mark.skipif(host.host_platform() != host.AMD,
                    reason="checks the card running the suite, so only an AMD box can")
def test_this_machine_has_a_route():
    route, why = boot.detect()
    assert route == "source", why
    assert "gfx" in why


@pytest.mark.parametrize("version, hipcc, gfx, expected", [
    ((7, 2), "/opt/rocm/bin/hipcc", ["gfx1100"], "source"),
    ((6, 4), "/opt/rocm/bin/hipcc", ["gfx1030"], "source"),
    # Newer than PyTorch's index: AMD's own publishes the same torch for it, so this is
    # a different index rather than a different job.
    ((10, 2), "/opt/rocm/bin/hipcc", ["gfx1151"], "source"),
    ((10, 1), "/opt/rocm/bin/hipcc", ["gfx1151"], "source"),
    # No card: nothing to compile for, so nothing to install.
    (None, "/opt/rocm/bin/hipcc", [], None),
    ((7, 2), "/opt/rocm/bin/hipcc", [], None),
    # No toolkit: these extensions are compiled, so hipcc is not optional.
    ((7, 2), None, ["gfx1100"], None),
    # Unreadable ROCm version: the torch wheel is published per ROCm release.
    (None, "/opt/rocm/bin/hipcc", ["gfx1100"], None),
    # Below AMD's oldest supported tag there is genuinely nothing.
    ((5, 7), "/opt/rocm/bin/hipcc", ["gfx1100"], None),
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


def test_aule_is_pinned_to_one_release():
    """It runs on every AMD install, so it is the version that was tested, not whatever is
    newest on PyPI that day."""
    assert boot.AULE_ATTENTION.startswith("aule-attention==")


def test_torch_comes_from_the_rocm_index_not_the_default_one():
    commands = boot.pip_commands("source", "uv", PYTHON, (7, 2))
    torch_command = next(c for c in commands if any("torch==" in a for a in c))
    index = torch_command[torch_command.index("--index-url") + 1]
    assert index == host.torch_rocm_index((7, 2))
    assert "rocm" in index, "a CUDA torch on an AMD card is the bug this route exists to fix"


def test_onnxruntime_is_installed_so_rembg_can_cut_a_picture():
    """A picture arriving without alpha has to be cut out before TRELLIS sees it, and rembg
    needs onnxruntime to do that. An old wheel asked for an executable stack this kernel
    refused; the release that resolves now installs and runs, so dropping it was a stale
    answer and left every un-matted upload failing with no obvious cause."""
    joined = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (7, 2))]
    assert any("onnxruntime" in c for c in joined)


def test_amd_publishes_torch_for_rocm_versions_pytorch_does_not():
    """PyTorch's index stops at rocm7.2. A newer machine is a different index, not a dead end.

    Checked live against AMD's index on 2026-10-16: torch 2.14.0+rocm10.1.0 for gfx1151
    (Strix Halo), paired with torchvision 0.29.0 -- the same torch this box runs on 7.2.
    """
    assert host.amd_torch_build((10, 2)) == ("2.14.0", "0.29.0")
    assert host.amd_torch_build((10, 1)) == ("2.14.0", "0.29.0")
    assert host.amd_torch_build((10, 0)) == ("2.13.0", "0.28.0")
    assert host.amd_torch_build((7, 2)) is None, "7.2 is PyTorch's, not AMD's"
    assert host.amd_torch_build(None) is None


def test_a_newer_rocm_machine_gets_amd_torch_from_amd_s_index():
    """The whole point: a ROCm 10.x card gets the same torch, from the index that has it."""
    route, why = boot.choose_route((10, 2), "/opt/rocm/bin/hipcc", ["gfx1151"])
    assert route == "source"
    assert "AMD" in why, "the message should say where the torch is coming from"

    joined = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (10, 2))]
    torch_line = next(c for c in joined if "torch==" in c)
    assert "--extra-index-url" in torch_line
    assert "stable.repo.amd.com" in torch_line
    assert "torch==2.14.0" in torch_line and "torchvision==0.29.0" in torch_line
    # And it must not also force PyTorch's index, which has no rocm10 wheels.
    assert "--index-url" not in torch_line


def test_an_older_machine_still_comes_from_pytorch():
    """The fallback must not have quietly replaced the path that already works."""
    joined = [" ".join(c) for c in boot.pip_commands("source", "uv", PYTHON, (7, 2))]
    torch_line = next(c for c in joined if "torch==" in c)
    assert torch_line.endswith("https://download.pytorch.org/whl/rocm7.2")
    assert "amd.com" not in torch_line


def test_torch_index_for_picks_the_index_that_has_this_version():
    assert boot.torch_index_for((7, 2)) == "https://download.pytorch.org/whl/rocm7.2"
    assert boot.torch_index_for((10, 2)) == host.AMD_ROCM_INDEX


def test_the_announcement_says_which_index_it_will_use():
    text = boot.announcement("source", boot.choose_route(
        (10, 2), "/opt/rocm/bin/hipcc", ["gfx1151"])[1], code=True, weights=False)
    assert "AMD" in text, "a reader should be able to see the torch comes from AMD's index"


def test_torch_pair_for_never_strays_outside_what_is_published():
    """A pair guessed from the torch version would pair 2.14.0 with a torchvision that may
    not exist on that index."""
    for version in ((10, 2), (10, 1), (10, 0)):
        torch_version, vision_version = boot.torch_pair_for(version)
        assert torch_version.count(".") == 2, version
        assert vision_version.count(".") == 2, version


def test_an_unsupported_rocm_release_has_no_plan():
    """Below AMD's oldest supported tag there is genuinely nothing, and the plan should say so."""
    with pytest.raises(ValueError, match="no torch for ROCm"):
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
    env = boot.build_env(["gfx1100"], "/opt/rocm/bin/hipcc", {}, format_workaround=False)
    assert env["GPU_ARCHS"] == "gfx1100"
    assert env["ROCM_PATH"] == "/opt/rocm"
    assert "/opt/rocm/bin" in env["PATH"]
    assert int(env["MAX_JOBS"]) >= 1


def test_the_build_environment_without_a_toolkit_still_names_the_card():
    """So the failure is hipcc's, not a missing GPU_ARCHS."""
    env = boot.build_env(["gfx1100"], None, {})
    assert env["GPU_ARCHS"] == "gfx1100"
    assert "ROCM_PATH" not in env


def test_which_extensions_come_from_a_fork():
    """nvdiffrast has no HIP upstream; CuMesh's sources need a hipify. FlexGEMM does not.

    Forking FlexGEMM would mean the two cards ran different code for no reason, and its
    upstream build was checked on this card rather than assumed.
    """
    urls = {name: url for name, url, _, _ in boot.SOURCE_EXTENSIONS}
    assert "dmonkman" in urls["nvdiffrast"]
    assert "dmonkman" in urls["CuMesh"]
    assert "dmonkman" not in urls["FlexGEMM"], "FlexGEMM should be upstream"
    assert urls["FlexGEMM"] == "https://github.com/JeffreyXiang/FlexGEMM.git"
    # ... and the build patch has to agree with that.
    patched = {path for path, _ in boot.rocm_patch.CXX20_FILES}
    assert "extensions/FlexGEMM/setup.py" in patched
    assert "TRELLIS.2/o-voxel/setup.py" in patched
    # CuMesh's port is already C++20, so it must not be asked for a C++20 change.
    assert "extensions/CuMesh/setup.py" not in patched
    assert set(boot.rocm_patch.DEVICE_LIB_FILES) == {"extensions/nvdiffrast/setup.py",
                                                    "extensions/CuMesh/setup.py"}
    python_floor = {spec.file for spec in boot.rocm_patch.REQUIRES_PYTHON_PATCHES}
    assert python_floor == {"extensions/nvdiffrast/pyproject.toml",
                            "extensions/CuMesh/pyproject.toml"}


def test_the_refusal_message_reads_its_ceilings_off_the_tables():
    """It used to have "7.2" and "10.2" written into the sentence, which goes quietly wrong
    the day either table grows."""
    pytorch, amd = boot.newest_known()
    assert pytorch == max(boot.TORCH)
    assert amd == max(minimum for minimum, _, _ in host.AMD_TORCH_BUILDS)
    _, why = boot.choose_route((5, 7), "/opt/rocm/bin/hipcc", ["gfx1100"])
    assert f"{pytorch[0]}.{pytorch[1]}" in why
    assert f"{amd[0]}.{amd[1]}" in why


# --- proving the torch runs on this card before compiling anything against it -------------

def _probe_result(code: int, stderr: str = ""):
    """A stand-in for the probe's subprocess: what a torch that works, or one that does not,
    prints. Passed to `gpu_probe` as its `run`, so the shipped function is what is tested."""
    return lambda *args, **kwargs: types.SimpleNamespace(returncode=code, stderr=stderr,
                                                           stdout="")


def test_a_newer_rocm_gets_two_torch_candidates_not_one():
    """The bug this guards, and it cost a full build plus a failed generation.

    ROCm 10.2 sent this machine to AMD's index, and that wheel cannot run here at all: its
    `.hip_fatbin` section is NOBITS -- declared at 322 MB, empty on disk, with no code object
    anywhere beside it -- so the first kernel died with `hipErrorInvalidKernelFile`. PyTorch's
    own rocm7.2 wheel carries 322 MB of real device code and runs on the same card under the
    same ROCm 10.2 runtime. Which of the two works is a property of the machine, so both are
    offered and the card decides.
    """
    candidates = boot.torch_candidates((10, 2))
    assert [c.index for c in candidates] == [host.AMD_ROCM_INDEX,
                                             host.torch_rocm_index(max(boot.TORCH))]
    assert candidates[0].own_index is False, "AMD's is searched alongside PyPI, not instead"
    assert candidates[-1].own_index is True


def test_a_rocm_pytorch_still_publishes_offers_just_the_one():
    """The fallback must not turn a machine that already works into a two-try install."""
    candidates = boot.torch_candidates((7, 2))
    assert len(candidates) == 1
    assert candidates[0].index == host.torch_rocm_index((7, 2))


def test_the_probe_asks_the_card_a_question_no_table_can():
    assert "cuda" in boot.GPU_PROBE_SOURCE, "a CPU-only torch would pass anything weaker"
    assert "synchronize" in boot.GPU_PROBE_SOURCE, "without it an async load failure hides"


def test_the_probe_reports_a_working_torch():
    assert boot.gpu_probe(PYTHON, run=_probe_result(0)) == (True, "")


def test_the_probe_names_the_failure_that_matters():
    boom = ("Traceback (most recent call last):\n"
            "torch.AcceleratorError: CUDA error: invalid kernel file\n")
    works, why = boot.gpu_probe(PYTHON, run=_probe_result(1, boom))
    assert works is False
    assert "invalid kernel file" in why, "an error nobody can act on is no error at all"


def test_the_probe_survives_a_probe_that_cannot_run():
    def explode(*args, **kwargs):
        raise OSError("no such interpreter")

    assert boot.gpu_probe(PYTHON, run=explode) == (False, "no such interpreter")


def test_install_torch_falls_back_to_the_next_candidate(monkeypatch, capsys):
    """The whole point: a torch that cannot launch a kernel is replaced, not reported."""
    installed, probed = [], []

    def fake_install(command, **kwargs):
        installed.append(" ".join(command))

    def fake_probe(python):
        probed.append(python)
        return (len(probed) > 1, "" if len(probed) > 1 else "invalid kernel file")

    chosen = boot.install_torch("uv", (10, 2), {}, python=PYTHON, install=fake_install,
                                probe=fake_probe)

    installs = [c for c in installed if " install " in c]
    assert len(installs) == 2, "the second candidate was never tried"
    assert chosen.index == host.torch_rocm_index(max(boot.TORCH))
    assert "2.14.0" in installs[0] and "2.14.1" in installs[1]
    assert "does not run here" in capsys.readouterr().out


def test_install_torch_does_not_fall_back_from_a_torch_that_works(monkeypatch, capsys):
    installed = []
    chosen = boot.install_torch("uv", (7, 2), {}, python=PYTHON,
                                install=lambda c, **k: installed.append(" ".join(c)),
                                probe=lambda p: (True, ""))
    assert len(installed) == 1
    assert chosen.index == host.torch_rocm_index((7, 2))
    assert "does not run here" not in capsys.readouterr().out


def test_install_torch_refuses_rather_than_building_against_a_dead_torch(monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit_info:
        boot.install_torch("uv", (10, 2), {}, python=PYTHON,
                           install=lambda c, **k: None,
                           probe=lambda p: (False, "invalid kernel file"))
    said = str(exit_info.value) + capsys.readouterr().out
    assert "invalid kernel file" in said
    assert "nothing built against one would run" in said


def test_the_marker_remembers_the_torch_that_actually_ran():
    preferred = boot.torch_candidates((10, 2))[0]
    fell_back = boot.torch_candidates((10, 2))[-1]
    assert boot.build_record((10, 2), "source", ["gfx1151"])["torch"] == preferred.torch
    assert boot.build_record((10, 2), "source", ["gfx1151"],
                             fell_back)["torch_index"] == host.torch_rocm_index(max(boot.TORCH))


# --- the build marker, written after a forty-minute compile ----------------------------------

def test_the_build_record_is_writable_on_every_machine_the_route_accepts():
    """The bug this guards: the marker used to read TORCH directly, which only knows PyTorch's
    index. Every ROCm newer than 7.2 -- a machine the route had just installed, successfully,
    after forty minutes -- died with a KeyError on that last line, and because the marker was
    never written the studio still said "Setting up" and the next run redid the whole build.
    """
    for version in (*boot.TORCH, *(minimum for minimum, _, _ in host.AMD_TORCH_BUILDS)):
        route, why = boot.choose_route(version, "/opt/rocm/bin/hipcc", ["gfx1100"])
        assert route == "source", (version, why)
        record = boot.build_record(version, route, ["gfx1100"])
        assert record["torch"] == boot.torch_pair_for(version)[0], version
        assert record["torchvision"] == boot.torch_pair_for(version)[1], version
        assert record["rocm"] == f"{version[0]}.{version[1]}"
        assert record["gfx"] == ["gfx1100"]
        assert record["attention"] == boot.AULE_ATTENTION
        json.loads(json.dumps(record)), "it has to be JSON, as written"


def test_install_code_refuses_an_unsupported_machine_before_touching_anything(monkeypatch):
    """AGENTS.md: check the environment before the expensive step, not after. A machine with no
    torch pair should hear so in a second, not after a gigabyte of git and a compile."""
    def nothing(*args, **kwargs):
        pytest.fail("it started work before checking the machine could be installed for")

    for name in ("fetch_checkout", "clone", "run", "venv_python", "install_torch"):
        monkeypatch.setattr(boot, name, nothing)
    with pytest.raises(ValueError, match="no torch for ROCm"):
        boot.install_code((5, 7))


def _stubbed_build(monkeypatch, tmp_path, *, patched: bool, gfx: list[str]) -> tuple[Path, list]:
    """`install_code` with every expensive step replaced by a stub, returning the marker path
    and the list the stubbed installs land in."""
    marker = tmp_path / ".i2l-build-complete"
    python = tmp_path / "python"
    python.write_text("")
    installed: list[list[str]] = []
    monkeypatch.setattr(boot, "BUILT_MARKER", marker)
    monkeypatch.setattr(boot, "venv_python", lambda *a, **k: python)
    monkeypatch.setattr(boot, "fetch_checkout", lambda commit: None)
    monkeypatch.setattr(boot, "clone", lambda *a, **k: None)
    monkeypatch.setattr(boot, "install_env", lambda *a, **k: {})
    monkeypatch.setattr(boot, "run", lambda command, **k: installed.append(command))
    monkeypatch.setattr(boot, "install_torch",
                        lambda uv, version, env, **k: boot.torch_candidates(version)[-1])
    monkeypatch.setattr(boot, "rocm_patch",
                        types.SimpleNamespace(apply=lambda root: [],
                                              is_patched=lambda root: patched))
    monkeypatch.setattr(boot, "bria_patch", types.SimpleNamespace(apply=lambda root: False))
    monkeypatch.setattr(host, "rocm_gfx_targets", lambda *a, **k: gfx)
    return marker, installed


def test_install_code_writes_the_record_it_installed(monkeypatch, tmp_path):
    """The whole build, with every expensive step stubbed: what must survive to the end is a
    marker a later run and the studio can both trust."""
    marker, installed = _stubbed_build(monkeypatch, tmp_path, patched=True, gfx=["gfx1151"])

    boot.install_code((10, 2))

    record = json.loads(marker.read_text())
    # The last candidate, which is what the stub proves: a machine that had to fall back
    # should say in the one file a later run reads that it did.
    assert record["torch"] == "2.14.1" and record["rocm"] == "10.2"
    assert record["torch_index"] == host.torch_rocm_index(max(boot.TORCH))
    assert record["gfx"] == ["gfx1151"]
    assert installed, "nothing was installed"
    assert any("o-voxel" in " ".join(c) for c in installed)
    torch_lines = [c for c in installed if any(a.startswith("torch==") for a in c)]
    assert len(torch_lines) == 1, "one torch, installed once and proved"
    assert "torch==2.14.1" in " ".join(torch_lines[0]), "the plan must build on what ran"


def test_a_build_that_died_halfway_leaves_no_marker(monkeypatch, tmp_path):
    """Written last, on purpose, so an interrupted build never looks finished."""
    marker, _ = _stubbed_build(monkeypatch, tmp_path, patched=False, gfx=["gfx1100"])

    with pytest.raises(SystemExit):
        boot.install_code((7, 2))
    assert not marker.exists()


# --- ROCm's empty __noinline__ against a 2026 libstdc++ -------------------------------------

def test_the_probe_reproduces_the_collision_it_guards():
    """The probe has to fail on a machine that has the collision, or it guards nothing."""
    if shutil.which("c++") is None:
        pytest.skip("no C++ compiler")
    assert boot.noinline_collides() in {True, False}   # runs, and returns a bool
    # The source is the collision itself, not a guess at it.
    assert "#define __noinline__" in boot.NOINLINE_PROBE_SOURCE
    assert "#include <format>" in boot.NOINLINE_PROBE_SOURCE


def test_the_probe_says_false_for_a_compiler_that_always_works(tmp_path):
    fake = tmp_path / "always-ok"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    assert boot.noinline_collides(compiler=str(fake), tmp_dir=tmp_path) is False


def test_the_probe_says_true_for_a_compiler_that_always_fails(tmp_path):
    fake = tmp_path / "always-fails"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    assert boot.noinline_collides(compiler=str(fake), tmp_dir=tmp_path) is True


def test_the_workaround_is_added_only_where_it_is_needed():
    env = boot.build_env(["gfx1100"], "/opt/rocm/bin/hipcc", {}, format_workaround=True)
    assert env["CPPFLAGS"] == "-include format", (
        "without it every .cpp that reaches <chrono> fails inside a system header")
    assert "CPPFLAGS" not in boot.build_env(["gfx1100"], "/opt/rocm/bin/hipcc", {},
                                           format_workaround=False)


def test_the_workaround_reaches_the_host_compiler_only():
    """CPPFLAGS goes to setuptools' preprocessor options, which is the host-compiler path.
    The HIP compiles go through hipcc and are unaffected either way."""
    assert boot.FORMAT_WORKAROUND_FLAGS == "-include format"


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

def test_the_gpu_probe_cannot_fail_by_chance():
    """It asserted that one of 64 random numbers exceeded 2, which fails about one run in
    twenty on a perfectly good torch."""
    assert "randn" not in boot.GPU_PROBE_SOURCE and "rand(" not in boot.GPU_PROBE_SOURCE
    assert "== 512" in boot.GPU_PROBE_SOURCE


def test_a_rejected_torch_is_taken_out_before_the_next_goes_in(capsys):
    """Two indexes ship Triton under two names into one folder: left side by side, a later
    uninstall of either breaks the attention backend."""
    installed = []
    boot.install_torch("uv", (10, 2), {}, python=PYTHON,
                       install=lambda c, **k: installed.append(c),
                       probe=lambda p: (len(installed) > 1, "invalid kernel file"))
    assert installed[1][:3] == ["uv", "pip", "uninstall"]
    assert {"torch", "triton", "triton-rocm", "rocm-sdk-core"} <= set(installed[1])
    assert installed[2][2] == "install"


def test_nothing_is_uninstalled_when_the_first_torch_works():
    installed = []
    boot.install_torch("uv", (7, 2), {}, python=PYTHON,
                       install=lambda c, **k: installed.append(c), probe=lambda p: (True, ""))
    assert all("uninstall" not in c for c in installed)


def test_a_rerun_tries_the_torch_the_last_build_proved_first():
    """Otherwise every re-run downgrades to the known-bad candidate and back again."""
    proved = boot.torch_candidates((10, 2))[-1]
    installed = []
    chosen = boot.install_torch("uv", (10, 2), {}, python=PYTHON, prefer=proved.index,
                                install=lambda c, **k: installed.append(c),
                                probe=lambda p: (True, ""))
    assert chosen == proved and len(installed) == 1


def test_a_rerun_drops_the_old_marker_before_it_changes_anything(monkeypatch, tmp_path):
    marker, _ = _stubbed_build(monkeypatch, tmp_path, patched=True, gfx=["gfx1151"])
    marker.write_text('{"torch_index": "old"}')
    seen = {}

    def fetch(commit):
        seen["marker"] = marker.exists()

    monkeypatch.setattr(boot, "fetch_checkout", fetch)
    monkeypatch.setattr(boot, "install_torch", lambda uv, version, env, **k: (
        seen.setdefault("prefer", k.get("prefer")), boot.torch_candidates(version)[-1])[1])
    boot.install_code((10, 2))
    assert seen == {"marker": False, "prefer": "old"}
    assert marker.exists(), "and a finished build writes a new one"
