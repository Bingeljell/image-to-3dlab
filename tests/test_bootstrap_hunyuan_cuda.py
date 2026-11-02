"""The NVIDIA Hunyuan3D-2.1 installer: picks torch from the card, installs only from PyPI,
announces before it fetches, and never downloads without a yes. No network here."""

from __future__ import annotations

import io

import backend_catalog
import bootstrap_hunyuan_cuda as boot
import pytest

# The shape of upstream's requirements.txt at the pinned commit, abridged.
UPSTREAM_REQUIREMENTS = """\
--extra-index-url https://mirrors.cloud.tencent.com/pypi/simple/
--extra-index-url https://mirrors.aliyun.com/pypi/simple

# Build Tools
ninja==1.11.1.1
transformers==4.46.0
rembg==2.0.65
basicsr==1.4.2
open3d==0.18.0
gradio==5.33.0
fastapi==0.115.12
uvicorn==0.34.3
cupy-cuda12x==13.4.1
bpy==4.0
pythreejs
deepspeed
timm
"""


REMOVER = "BiRefNet-lite background remover"


def test_weights_match_the_catalogue():
    """Every set the installer fetches is one the Setup page announces, at the same size."""
    entry = backend_catalog.BY_ID["hunyuan-cuda"]
    announced = {w.label: w.bytes_expected for w in entry.weights}
    # The background remover is the shared one, installed by bootstrap_matte.
    assert set(announced) - {REMOVER} == {label for label, *_ in boot.WEIGHTS}
    for label, _, _, _, size in boot.WEIGHTS:
        assert announced[label] == pytest.approx(size * backend_catalog.GB, rel=0.05), label


def test_the_catalogue_measures_where_the_weights_land():
    """The Setup page's "present" check must look where the installer writes."""
    entry = backend_catalog.BY_ID["hunyuan-cuda"]
    paths = {w.label: w.path for w in entry.weights}
    for label, source, _, local, _ in boot.WEIGHTS:
        if local is not None:
            assert paths[label] == local, label
        else:
            assert paths[label].name == "models--" + source.replace("/", "--"), label


def test_requirements_come_from_pypi_only():
    kept = boot.filter_requirements(UPSTREAM_REQUIREMENTS)
    assert not any(line.startswith("-") or "mirrors" in line for line in kept)


def test_demo_and_training_packages_are_dropped():
    kept = " ".join(boot.filter_requirements(UPSTREAM_REQUIREMENTS))
    for name in ("gradio", "fastapi", "uvicorn", "deepspeed", "pythreejs", "cupy"):
        assert name not in kept, name


def test_open3d_is_kept_for_the_paint_remesh():
    # Seen on a real pod: paint's remesh calls trimesh's simplify, which imports open3d,
    # so dropping it as "unused" killed every paint run after a finished shape.
    assert "open3d==0.18.0" in boot.filter_requirements(UPSTREAM_REQUIREMENTS)


def test_the_inference_path_is_kept_pinned():
    kept = boot.filter_requirements(UPSTREAM_REQUIREMENTS)
    for line in ("transformers==4.46.0", "rembg==2.0.65", "basicsr==1.4.2",
                 "timm", "ninja==1.11.1.1"):
        assert line in kept, line


def test_bpy_is_one_pypi_still_has():
    # Seen on a real pod: bpy 4.0 is gone from PyPI (oldest left is 4.2.0, Python 3.11
    # only), so upstream's pin made setup unsatisfiable. The paint stage only uses bpy to
    # turn its OBJ into a GLB, with operators 4.2 still has.
    kept = boot.filter_requirements(UPSTREAM_REQUIREMENTS)
    assert "bpy==4.0" not in kept and boot.BPY_PIN in kept
    assert boot.BPY_PIN == "bpy==4.2.0" and boot.PYTHON_VERSION == "3.11"


@pytest.mark.parametrize("driver,cap,nvcc,route", [
    ((12, 8), "89", (12, 8), "tested"),       # 4090, current driver
    ((12, 4), "86", (12, 4), "tested"),       # 3090 on upstream's exact CUDA
    ((13, 0), "86", (12, 8), "tested"),       # newer driver, 12.x toolkit
    ((12, 8), "120", (12, 8), "blackwell"),   # 5090: 2.5.1 has no kernels for it
])
def test_route_follows_the_card(driver, cap, nvcc, route):
    assert boot.choose_route(driver, cap, nvcc)[0] == route


@pytest.mark.parametrize("driver,cap,nvcc,needle", [
    (None, "89", (12, 8), "nvidia-smi"),
    ((12, 2), "89", (12, 2), "12.4 or newer"),
    ((12, 4), "120", (12, 4), "12.8 or newer"),
    ((12, 8), "89", None, "nvcc"),
    ((13, 0), "89", (13, 0), "12.x toolkit"),  # torch 2.5.1 is CUDA 12; nvcc 13 cannot build for it
])
def test_no_route_says_why(driver, cap, nvcc, needle):
    route, why = boot.choose_route(driver, cap, nvcc)
    assert route is None and needle in why


def test_plan_installs_torch_first_and_builds_against_it():
    commands = boot.pip_commands("tested", "uv", boot.venv_python(), ["timm"])
    flat = [" ".join(c) for c in commands]
    assert "torch==2.5.1" in flat[1] and "cu124" in flat[1]
    assert all("--no-build-isolation" in line for line in flat[2:])
    assert flat[-1].endswith("custom_rasterizer")
    assert not any("extra-index-url" in line for line in flat)


def test_blackwell_plan_uses_a_torch_with_its_kernels():
    commands = boot.pip_commands("blackwell", "uv", boot.venv_python(), [])
    assert "torch==2.7.1" in commands[1] and any("cu128" in part for part in commands[1])


def test_unknown_route_has_no_plan():
    with pytest.raises(ValueError):
        boot.pip_commands("pinned", "uv", boot.venv_python(), [])


def test_mesh_painter_builds_for_the_venv_python():
    """upstream's script asks python3-config, which in a venv can name another Python."""
    command = boot.mesh_painter_command("-I/a -I/b", ".cpython-310-x86_64-linux-gnu.so")
    assert command[-1] == "mesh_inpaint_processor.cpython-310-x86_64-linux-gnu.so"
    assert "-I/a" in command and "-I/b" in command
    assert not any("python3-config" in part for part in command)


def test_build_env_sets_arch_jobs_and_toolkit():
    env = boot.build_env("89", "/usr/local/cuda/bin/nvcc", base={"PATH": "/usr/bin"})
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert env["CUDA_HOME"] == "/usr/local/cuda"
    assert env["PATH"].startswith("/usr/local/cuda/bin")
    assert int(env["MAX_JOBS"]) >= 1


def test_weights_never_land_inside_the_checkout():
    """--weights-only may run before the clone; git will not clone into a non-empty folder."""
    for *_, local, _ in boot.WEIGHTS:
        if local is not None:
            assert boot.CHECKOUT not in local.parents, local


def test_announcement_names_backend_route_size_and_territory():
    text = boot.announcement("tested", "PyTorch 2.5.1 + cu124")
    assert "Hunyuan3D-2.1" in text and "NVIDIA" in text
    assert f"{boot.total_gb():.1f} GB" in text
    assert "EU" in text and "UK" in text and "South Korea" in text
    assert boot.COMMIT[:7] in text


@pytest.fixture
def linux_nvidia(monkeypatch):
    monkeypatch.setattr(boot.host, "build_target", lambda *a, **k: "linux-nvidia")
    monkeypatch.setattr(boot, "detect", lambda: ("tested", "test route"))
    calls = []
    monkeypatch.setattr(boot, "install_code", lambda *a: calls.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: calls.append("weights"))
    # the remover download has its own test; here it must never reach the network
    monkeypatch.setattr(boot, "install_background_remover", lambda *a, **k: None)
    return calls


def test_refuses_without_yes_when_nobody_can_answer(monkeypatch, linux_nvidia, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert boot.main([]) == 1
    assert linux_nvidia == []
    assert "Refusing" in capsys.readouterr().out


def test_a_no_downloads_nothing(monkeypatch, linux_nvidia):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "n")
    assert boot.main([]) == 1
    assert linux_nvidia == []


def test_yes_builds_then_fetches(linux_nvidia):
    assert boot.main(["--yes"]) == 0
    assert linux_nvidia == ["code", "weights"]


def test_code_only_and_weights_only(linux_nvidia):
    boot.main(["--yes", "--code-only"])
    boot.main(["--yes", "--weights-only"])
    assert linux_nvidia == ["code", "weights"]


@pytest.mark.parametrize("target", ["macos-arm64", "windows-nvidia", None])
def test_other_machines_are_refused_before_any_download(monkeypatch, target, capsys):
    monkeypatch.setattr(boot.host, "build_target", lambda *a, **k: target)
    monkeypatch.setattr(boot, "install_code", lambda *a: pytest.fail("built"))
    monkeypatch.setattr(boot, "install_weights", lambda: pytest.fail("downloaded"))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_no_route_refuses_before_asking(monkeypatch, linux_nvidia, capsys):
    monkeypatch.setattr(boot, "detect", lambda: (None, "no nvcc"))
    assert boot.main(["--yes"]) == 1
    assert linux_nvidia == []
    assert "no nvcc" in capsys.readouterr().out


def test_build_is_present_only_after_the_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(boot, "BUILT_MARKER", tmp_path / ".done")
    assert boot.build_present() is False
    (tmp_path / ".done").write_text("{}")
    assert boot.build_present() is True


def test_installs_use_patient_downloads(monkeypatch):
    monkeypatch.setattr(boot.host, "compute_capability", lambda: "86")
    monkeypatch.setattr(boot.host, "find_nvcc", lambda: "/usr/local/cuda/bin/nvcc")
    env = boot.install_env({"PATH": "/usr/bin"})
    assert "UV_HTTP_TIMEOUT" in env and env["TORCH_CUDA_ARCH_LIST"] == "8.6"


def test_setup_installs_and_announces_the_background_remover(tmp_path):
    # Without it, the first generation from a plain picture fetched u2net unannounced.
    fetched = []
    boot.install_background_remover(tmp_path / "lite.onnx", download=fetched.append)
    assert fetched == [tmp_path / "lite.onnx"]
    assert "BiRefNet-lite" in boot.announcement("source", "test")
    sources = {w.label for w in backend_catalog.BY_ID["hunyuan-cuda"].weights}
    assert REMOVER in sources


def test_a_venv_on_the_wrong_python_is_rebuilt(tmp_path):
    # Installs whose setup failed on bpy 4.0 were left with a Python 3.10 venv.
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.10.18\n")
    assert boot.venv_needs_rebuild(venv) is True
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\nversion_info = 3.11.13\n")
    assert boot.venv_needs_rebuild(venv) is False
    assert boot.venv_needs_rebuild(tmp_path / "missing") is True


def test_setuptools_keeps_pkg_resources():
    # Seen on a real pod: setuptools 84 has no pkg_resources, and lightning_fabric (from
    # upstream's pytorch-lightning 1.9.5) imports it, so the paint stage died on start.
    for command in boot.pip_commands("tested", "uv", boot.venv_python(), ["timm"]):
        assert "setuptools" not in command, "unpinned setuptools"
    first = boot.pip_commands("tested", "uv", boot.venv_python(), [])[0]
    assert boot.SETUPTOOLS_PIN in first and boot.SETUPTOOLS_PIN == "setuptools<81"
    # The requirements install must not float it back up either.
    reqs = boot.pip_commands("tested", "uv", boot.venv_python(), ["timm"])[2]
    assert boot.SETUPTOOLS_PIN in reqs


def test_installs_everything_but_torch_from_the_lock():
    # Unpinned installs broke on a real pod as upstreams moved (bpy, setuptools, open3d).
    assert boot.requirement_args() == ["-r", str(boot.PACKAGE_LOCK)] and boot.PACKAGE_LOCK.is_file()


def test_the_lock_holds_the_pins_that_broke():
    lock = boot.PACKAGE_LOCK.read_text()
    for pin in ("bpy==4.2.0", "open3d==0.18.0", "numpy==1.24.4", "transformers==4.46.0"):
        assert f"\n{pin}\n" in lock, pin
    setuptools = next(line for line in lock.splitlines() if line.startswith("setuptools=="))
    assert int(setuptools.split("==")[1].split(".")[0]) < 81
    assert not any(line.startswith(("torch==", "torchvision==", "nvidia-"))
                   for line in lock.splitlines())


def test_the_lock_covers_what_we_install_from_upstream():
    lock = boot.PACKAGE_LOCK.read_text().lower().replace("_", "-")
    for line in boot.filter_requirements(UPSTREAM_REQUIREMENTS):
        name = line.split("==")[0].split("<")[0].lower().replace("_", "-")
        assert f"\n{name}==" in lock, name
