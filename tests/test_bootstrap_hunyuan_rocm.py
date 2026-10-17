"""scripts/bootstrap_hunyuan_rocm.py: the AMD twin of the NVIDIA Hunyuan bootstrap.

The questions here are the ones a wrong answer costs a real afternoon on: which PyTorch the
card gets, whether the CUDA wheel sneaks in from PyPI, what gets compiled and for which
gfx target, and whether the announcement says enough for someone to consent to 19.5 GB.
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
import bootstrap_hunyuan_cuda as cuda
import bootstrap_hunyuan_rocm as boot

HIPCC = "/opt/rocm/bin/hipcc"
ROCM_GFX = ["gfx1100"]


def test_the_route_is_chosen_from_the_installed_rocm():
    route, why = boot.choose_route((7, 2), HIPCC, ROCM_GFX)
    assert route == "rocm7.2"
    assert "gfx1100" in why and "2.14.1" in why


@pytest.mark.parametrize("version,expected", [
    ((7, 2), "rocm7.2"), ((7, 1), "rocm7.1"), ((7, 0), "rocm7.0"),
    ((6, 4), "rocm6.4"), ((6, 3), "rocm6.3"), ((6, 2), "rocm6.2"),
])
def test_every_rocm_build_pytorch_publishes_has_a_torch_here(version, expected):
    """host picks the index tag; this route has to have picked a torch for that tag. A tag
    in one table and not the other is how a card ends up on somebody's CUDA wheel."""
    assert boot.choose_route(version, HIPCC, ROCM_GFX)[0] == expected
    assert expected in boot.TORCH


def test_no_rocm_is_refused_rather_than_installed_and_left_on_the_cpu():
    route, why = boot.choose_route(None, HIPCC, ROCM_GFX)
    assert route is None
    assert "ROCm" in why


def test_no_hipcc_is_refused_because_the_rasterizer_needs_compiling():
    route, why = boot.choose_route((7, 2), None, ROCM_GFX)
    assert route is None
    assert "hipcc" in why


def test_a_rocm_too_old_for_any_pytorch_build_is_refused():
    route, why = boot.choose_route((6, 0), HIPCC, ROCM_GFX)
    assert route is None
    assert "older" in why


def test_no_gfx_target_means_nothing_to_compile_for():
    route, why = boot.choose_route((7, 2), HIPCC, [])
    assert route is None
    assert "gfx" in why


def test_torch_comes_from_the_rocm_index_not_from_pypi():
    """PyPI's Linux torch is a CUDA build: it installs cleanly and then reports no device,
    which is the whole reason this route exists."""
    commands = boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"), ["numpy"])
    torch_step = next(c for c in commands if any(a.startswith("torch==") for a in c))
    assert "--index-url" in torch_step
    assert torch_step[torch_step.index("--index-url") + 1] == \
        "https://download.pytorch.org/whl/rocm7.2"
    # The torchvision pairing is pinned too: a mismatched pair fails at the first kernel.
    assert any(a.startswith("torchvision==") for a in torch_step)


def test_the_rasterizer_is_compiled_against_the_torch_that_was_just_installed():
    """--no-build-isolation, and last, or it compiles against whatever torch is lying about."""
    commands = boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"), ["numpy"])
    assert any(a.startswith("torch==") for a in commands[0]) or \
        any(a.startswith("torch==") for a in commands[1]), "torch is not installed early"
    last = commands[-1]
    assert "custom_rasterizer" in last[-1]
    assert "--no-build-isolation" in last


def test_the_cuda_only_wheels_stay_out():
    steps = [" ".join(c) for c in
             boot.pip_commands("rocm7.2", "uv", Path("/venv/bin/python"),
                               boot.filter_requirements(_UPSTREAM_REQUIREMENTS))]
    for banned in ("cupy-cuda12x", "gradio", "deepspeed", "onnxruntime=="):
        assert all(banned not in step for step in steps), banned


def test_onnxruntime_is_replaced_not_skipped():
    """Upstream's pin cannot load here; skipping the package was the wrong remedy.

    `hy3dshape.rembg` imports it, `image_to_3dlab/matte.py` imports it, and rembg imports
    onnxruntime at module import — so a venv built by the skip would have raised
    ModuleNotFoundError the first time it cut out a background, having traded one ImportError
    for another. The pin is replaced with the floor requirements.txt already asks for, which
    resolves to an ordinary shared object.
    """
    assert "onnxruntime" not in boot.SKIP_REQUIREMENTS, "rembg needs it; do not drop it"
    assert boot.REPLACED["onnxruntime"] == "onnxruntime>=1.18"
    filtered = boot.filter_requirements(_UPSTREAM_REQUIREMENTS)
    assert "onnxruntime>=1.18" in filtered
    assert "onnxruntime==1.16.3" not in filtered, "the wheel that cannot load is still pinned"
    # The CUDA route keeps upstream's pin: it has its own generated lock, resolved elsewhere.
    assert "onnxruntime==1.16.3" in cuda.filter_requirements(_UPSTREAM_REQUIREMENTS)


def test_the_replacement_actually_loads_rather_than_being_believed():
    """A pin nobody has run is a guess. onnxruntime 1.31 is what this resolves to, it imports,
    and rembg imports through it — checked against this repo's own venv, not a comment."""
    import subprocess

    torch_free = ["hunyuan-rocm", "trellis-rocm"]
    for route in torch_free:
        python = REPO / "vendor" / route / ".venv" / "bin" / "python"
        if not python.is_file():
            pytest.skip(f"{route} is not installed on this machine")
        found = subprocess.run(
            [str(python), "-c", "import onnxruntime, rembg; print(onnxruntime.__version__)"],
            capture_output=True, text=True, check=False)
        assert found.returncode == 0, f"{route}: {found.stderr.strip()[-300:]}"


def test_the_skip_list_does_not_grow_by_accident():
    """Every entry has to be one that genuinely is unused on the inference path."""
    assert not boot.SKIP_REQUIREMENTS - cuda.SKIP_REQUIREMENTS, (
        "a new skip needs its own reasoning; replacing a pin is usually the right answer, "
        "because something that is merely unimported today may import it tomorrow")
    for name in boot.SKIP_REQUIREMENTS:
        assert name not in boot.REPLACED, name


def test_the_bpy_pin_survives_the_filter():
    """Upstream pins bpy==4.0, which PyPI no longer has; the route substitutes 4.2."""
    filtered = boot.filter_requirements(_UPSTREAM_REQUIREMENTS)
    assert "bpy==4.2.0" in filtered
    assert not any(line.startswith("bpy==4.0") for line in filtered)


def test_the_compile_targets_this_card_and_not_some_other_one():
    env = boot.build_env(["gfx1100"], HIPCC, base={"PATH": "/usr/bin"})
    assert env["PYTORCH_ROCM_ARCH"] == "gfx1100"
    assert env["ROCM_HOME"] == "/opt/rocm"
    assert "/opt/rocm/bin" in env["PATH"]
    assert env["MAX_JOBS"]


def test_every_gfx_target_is_listed_when_there_is_more_than_one():
    env = boot.build_env(["gfx1100", "gfx1030"], HIPCC, base={"PATH": "/usr/bin"})
    assert env["PYTORCH_ROCM_ARCH"] == "gfx1100,gfx1030"


def test_a_machine_with_no_card_still_gets_a_readable_environment():
    """build_env is called before the checks finish, so it must not raise on empty input."""
    env = boot.build_env([], None, base={"PATH": "/usr/bin"})
    assert "PYTORCH_ROCM_ARCH" not in env
    assert env["PATH"] == "/usr/bin"


def test_it_installs_the_same_upstream_commit_as_the_nvidia_route():
    """Two installs of the same model that differ in what they built are worse than one."""
    assert boot.COMMIT == cuda.COMMIT
    assert boot.UPSTREAM == cuda.UPSTREAM


def test_the_announcement_names_the_route_the_size_and_the_licence_before_asking():
    """AGENTS.md: a download path must name the backend, name the route, state the size,
    and the one licence with real teeth (Hunyuan's territorial limits)."""
    text = boot.announcement("rocm7.2", "PyTorch 2.14.1", code=True, weights=True)
    assert "backend: Hunyuan3D-2.1" in text
    assert "AMD / ROCm" in text and "rocm7.2" in text
    assert f"{boot.total_gb():.1f} GB" in text
    assert "EU, the UK or South Korea" in text


def test_the_announcement_does_not_claim_a_size_when_it_is_not_downloading_any():
    assert "GB total" not in boot.announcement("rocm7.2", "x", code=True, weights=False)


def test_it_refuses_to_run_on_the_wrong_machine_before_asking_anything(monkeypatch, capsys):
    """A user on an NVIDIA card who ran the AMD bootstrap should be pointed at the other
    name. The guard is checked here rather than by running the real guard on whatever card
    the suite happens to be on, which would install things."""
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-nvidia")
    assert boot.main(["--yes"]) == 1
    out = capsys.readouterr().out
    assert "bootstrap_hunyuan_cuda.py" in out
    assert "Nothing downloaded" in out


def test_it_refuses_on_a_machine_without_rocm(monkeypatch, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: (None, "no ROCm"))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_it_will_not_install_without_a_yes_and_nobody_to_ask(monkeypatch, capsys):
    """AGENTS.md: a download path needs an affirmative answer. Defaulting to yes is the
    thing it forbids."""
    monkeypatch.setattr(boot.host, "build_target", lambda: "linux-amd")
    monkeypatch.setattr(boot, "detect", lambda: ("rocm7.2", "PyTorch 2.14.1"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert boot.main([]) == 1
    assert "--yes" in capsys.readouterr().out


def test_weights_match_the_catalogue():
    """Same rule as the NVIDIA route: what the bootstrap fetches and what the Setup card
    promises have to be the same list, or one of them is lying."""
    entry = backend_catalog.BY_ID["hunyuan-rocm"]
    announced = {w["label"]: w["bytes_expected"]
                 for w in entry.describe(backend_catalog.AMD)["weights"]}
    remover = next(label for label in announced if "background remover" in label.lower())
    assert set(announced) - {remover} == {label for label, *_ in boot.WEIGHTS}
    for label, _, _, _, size in boot.WEIGHTS:
        assert announced[label] == pytest.approx(size * backend_catalog.GB, rel=0.05), label


def test_the_catalogue_measures_where_the_weights_land():
    entry = backend_catalog.BY_ID["hunyuan-rocm"]
    announced = {w["label"]: w["path"] for w in entry.describe(backend_catalog.AMD)["weights"]}
    for label, source, _, local, _ in boot.WEIGHTS:
        if local is not None:
            assert announced[label] == str(local), label


def test_the_amd_route_does_not_write_over_the_nvidia_routes_weights():
    """A machine with both cards must be able to keep both installs."""
    for _, _, _, local, _ in boot.WEIGHTS:
        if local is not None:
            assert cuda.VENDOR not in local.parents and cuda.VENDOR != local, local


def _no_real_install(monkeypatch, tmp_path, order=None):
    """Everything install_code does, stubbed out except the sequence of the first three.

    Nothing here touches the network, the vendor folder or a venv: a test that calls the
    real thing on whatever card the suite happens to run on starts a multi-gigabyte
    download, and that is how this file once left a 324 MB clone behind.
    """
    def record(name):
        # Not `order and order.append(...)`: an empty list is falsy, so that never records.
        def stub(*a, **k):
            if order is not None:
                order.append(name)
        return stub

    monkeypatch.setattr(boot, "fetch_checkout", record("clone"))
    monkeypatch.setattr(boot.hunyuan_patch, "patch", record("patch"))
    monkeypatch.setattr(boot, "venv_needs_rebuild", lambda v: False)
    monkeypatch.setattr(boot, "install_env", dict)
    monkeypatch.setattr(boot, "requirement_args", list)
    monkeypatch.setattr(boot.shutil, "which", lambda name: "uv")
    monkeypatch.setattr(boot, "run", lambda *a, **k: None)
    monkeypatch.setattr(boot, "BUILT_MARKER", tmp_path / ".i2l-build-complete")


def test_it_runs_the_patches_before_building(monkeypatch, tmp_path):
    """A build against an unpatched checkout compiles and then crashes in the scheduler."""
    order: list[str] = []
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [])
    monkeypatch.setattr(boot, "compile_mesh_painter",
                        lambda *a, **k: order.append("compile"))
    _no_real_install(monkeypatch, tmp_path, order)
    boot.install_code("rocm7.2")
    assert order.index("patch") < order.index("compile")


def test_the_build_marker_is_written_last(monkeypatch, tmp_path):
    """A build that died halfway must never look installed."""
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [])
    monkeypatch.setattr(boot, "compile_mesh_painter", lambda *a, **k: None)
    _no_real_install(monkeypatch, tmp_path)
    boot.install_code("rocm7.2")
    assert (tmp_path / ".i2l-build-complete").is_file()


def test_a_build_that_dies_leaves_no_marker(monkeypatch, tmp_path):
    """The marker is the only thing Setup reads, so a half-built venv must not write one."""
    monkeypatch.setattr(boot, "pip_commands", lambda *a, **k: [["uv", "pip", "install", "boom"]])
    monkeypatch.setattr(boot, "compile_mesh_painter", lambda *a, **k: None)
    _no_real_install(monkeypatch, tmp_path)
    monkeypatch.setattr(boot, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("build failed")))
    with pytest.raises(RuntimeError, match="build failed"):
        boot.install_code("rocm7.2")
    assert not (tmp_path / ".i2l-build-complete").exists()


_UPSTREAM_REQUIREMENTS = """\
--extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/
gradio==5.33.0
fastapi==0.115.12
uvicorn==0.34.3
pythreejs
deepspeed
cupy-cuda12x==13.4.1
bpy==4.0
torchmetrics==1.6.0
transformers==4.46.0
# a comment
numpy==1.24.4
onnxruntime==1.16.3
"""


def test_the_fixture_is_upstreams_real_list():
    """Every name this route skips must appear in the fixture, or the filter tests below pass
    on a list that never had anything to skip. Upstream's requirements.txt really does pin
    `onnxruntime==1.16.3`, under an "# ONNX Runtime" heading, and that is the line that leaked.
    """
    listed = {line.split("==")[0].strip() for line in _UPSTREAM_REQUIREMENTS.splitlines()
              if line.strip() and not line.startswith(("#", "-"))}
    missing = boot.SKIP_REQUIREMENTS - listed
    assert not missing, f"the fixture never mentions {missing}, so nothing tests skipping it"