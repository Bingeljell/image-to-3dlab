"""The auto-rig installer: says what it fetches, never downloads without a yes, pins upstream."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import backend_catalog
import bootstrap_autorig as ba
import download_api


def test_announcement_names_backend_licence_and_size():
    text = ba.announcement()
    assert "SkinTokens" in text and "MIT" in text
    assert "1.6 GB" in text
    assert ba.COMMIT[:7] in text and ba.HF_REVISION[:7] in text


def test_refuses_without_yes_when_nobody_can_answer(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", None)
    fetched = []
    monkeypatch.setattr(ba, "install_weights", lambda *a, **k: fetched.append(1))
    monkeypatch.setattr(ba, "clone_pinned", lambda *a, **k: fetched.append(1))
    assert ba.main([]) == 1
    assert not fetched
    assert "Refusing" in capsys.readouterr().out


def test_clone_is_pinned(tmp_path):
    calls = []
    ba.clone_pinned(tmp_path / "SkinTokens", runner=calls.append)
    assert calls[0][:2] == ["git", "clone"]
    assert calls[-1][-1] == ba.COMMIT


def test_existing_clone_is_moved_to_the_pin_not_recloned(tmp_path):
    root = tmp_path / "SkinTokens"
    (root / ".git").mkdir(parents=True)
    calls = []
    ba.clone_pinned(root, runner=calls.append)
    assert len(calls) == 1 and calls[0][-1] == ba.COMMIT


def test_link_checkpoint_replaces_a_stale_link(tmp_path):
    cache = tmp_path / "cache.ckpt"
    cache.write_bytes(b"x")
    checkpoint = ba.CHECKPOINTS[0]
    stale = tmp_path / "root" / checkpoint.path
    stale.parent.mkdir(parents=True)
    stale.symlink_to(tmp_path / "gone.ckpt")
    linked = ba.link_checkpoint(cache, tmp_path / "root", checkpoint)
    assert linked.resolve() == cache.resolve()


def _no_symlinks(self, target):
    # What Windows says without Developer Mode or admin rights.
    raise OSError(1314, "A required privilege is not held by the client")


def test_without_symlink_rights_the_checkpoint_is_hard_linked(tmp_path, monkeypatch):
    """Windows without admin: a hard link is still the cached file, so no second 1.6 GB."""
    monkeypatch.setattr(Path, "symlink_to", _no_symlinks)
    cache = tmp_path / "cache.ckpt"
    cache.write_bytes(b"weights")
    linked = ba.link_checkpoint(cache, tmp_path / "root", ba.CHECKPOINTS[0])
    assert linked.samefile(cache)


def test_with_no_links_at_all_the_checkpoint_is_copied(tmp_path, monkeypatch):
    """The cache on another drive: a copy costs disk, but setup still finishes."""
    monkeypatch.setattr(Path, "symlink_to", _no_symlinks)

    def no_hard_links(self, target):
        raise OSError(17, "Cannot create a link across drives")
    monkeypatch.setattr(Path, "hardlink_to", no_hard_links)
    cache = tmp_path / "cache.ckpt"
    cache.write_bytes(b"weights")
    linked = ba.link_checkpoint(cache, tmp_path / "root", ba.CHECKPOINTS[0])
    assert linked.read_bytes() == b"weights" and not linked.is_symlink()


def test_weights_present_needs_both_checkpoints_and_the_llm_config(tmp_path):
    root = tmp_path / "SkinTokens"
    for checkpoint in ba.CHECKPOINTS:
        (root / checkpoint.path).parent.mkdir(parents=True, exist_ok=True)
        (root / checkpoint.path).write_bytes(b"x")
    assert not ba.weights_present(root)
    (root / ba.LLM_DIR).mkdir(parents=True)
    (root / ba.LLM_DIR / "config.json").write_text("{}")
    assert ba.weights_present(root)


def test_venv_on_the_wrong_python_is_rebuilt(tmp_path):
    root = tmp_path / "SkinTokens"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.10.14\n")
    assert ba.venv_needs_rebuild(root)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.11.9\n")
    assert not ba.venv_needs_rebuild(root)


def test_catalogue_and_setup_command_agree_with_the_installer():
    backend = backend_catalog.BY_ID["autorig"]
    assert backend.bytes_expected == ba.WEIGHT_BYTES
    # SkinTokens is pure PyTorch with no compiled extension, so AMD is a real route and
    # not an aspiration: the catalogue must not leave it out or setup would be refused.
    assert set(backend.runs_on) == {backend_catalog.APPLE, backend_catalog.NVIDIA,
                                    backend_catalog.AMD}
    command = download_api.COMMANDS["autorig"]
    assert command[1].endswith("bootstrap_autorig.py") and "--yes" in command


def test_mac_gets_pypi_torch():
    assert ba.torch_index("macos", nvidia=False, driver=None) is None


def test_amd_gets_the_rocm_build_not_pypi_s_cuda_one():
    # PyPI's Linux torch is a CUDA build: on an AMD card it reports no device at all, so
    # without this the run would install happily and then die looking for a GPU.
    assert ba.torch_index("linux", nvidia=False, driver=None,
                          amd=True, rocm=(7, 2)).endswith("/rocm7.2")


def test_a_linux_box_with_no_gpu_at_all_still_gets_pypi_torch():
    assert ba.torch_index("linux", nvidia=False, driver=None) is None


def test_an_amd_card_with_an_unusable_rocm_stops_with_a_reason():
    try:
        ba.torch_index("linux", nvidia=False, driver=None, amd=True, rocm=None)
    except SystemExit as stop:
        assert "ROCm" in str(stop) and "6.2" in str(stop)
    else:
        raise AssertionError("an AMD card with no ROCm must stop setup, not guess")


def test_nvidia_torch_matches_the_driver_not_pypi():
    # The 0.4.0 pod check had a CUDA 13 driver, so PyPI's cu130 torch passed; the common
    # 570-series driver (CUDA 12.8) cannot run it.
    assert ba.torch_index("linux", nvidia=True, driver=(12, 8)).endswith("/cu128")
    assert ba.torch_index("linux", nvidia=True, driver=(13, 0)).endswith("/cu130")
    assert ba.torch_index("windows", nvidia=True, driver=(12, 9)).endswith("/cu128")


def test_driver_too_old_stops_with_a_reason():
    try:
        ba.torch_index("linux", nvidia=True, driver=(12, 4))
    except SystemExit as stop:
        assert "CUDA 12.4" in str(stop) and "12.8" in str(stop)
    else:
        raise AssertionError("an unusable driver must stop setup")


def test_torch_installs_first_from_its_index(tmp_path):
    first, second = ba.pip_commands("uv", tmp_path / "python", tmp_path, "https://x/cu128")
    assert first[-4:] == ["torch", "torchvision", "--index-url", "https://x/cu128"]
    assert "torch" not in second and second[-2:] == ["-r", str(tmp_path / "requirements.txt")]
    mac_torch, _ = ba.pip_commands("uv", tmp_path / "python", tmp_path, None)
    assert "--index-url" not in mac_torch


def test_install_packages_runs_the_planned_commands(tmp_path, monkeypatch):
    root = tmp_path / "SkinTokens"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.11.9\n")
    monkeypatch.setattr(ba.shutil, "which", lambda name: "/bin/uv")
    monkeypatch.setattr(ba.host, "os_family", lambda: "linux")
    monkeypatch.setattr(ba.host, "has_nvidia_gpu", lambda: True)
    monkeypatch.setattr(ba.host, "driver_cuda_version", lambda: (12, 8))
    ran = []
    ba.install_packages(root, runner=ran.append)
    assert ran == ba.pip_commands("/bin/uv", ba.venv_python(root), root,
                                  ba.host.TORCH_INDEX + "cu128")


def test_install_packages_on_amd_installs_the_rocm_torch(tmp_path, monkeypatch):
    """The whole point on AMD: torch must come from the ROCm index, or torch sees no card."""
    root = tmp_path / "SkinTokens"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.11.9\n")
    monkeypatch.setattr(ba.shutil, "which", lambda name: "/bin/uv")
    monkeypatch.setattr(ba.host, "os_family", lambda: "linux")
    monkeypatch.setattr(ba.host, "has_nvidia_gpu", lambda: False)
    monkeypatch.setattr(ba.host, "has_amd_gpu", lambda: True)
    monkeypatch.setattr(ba.host, "rocm_version", lambda: (7, 2))
    ran = []
    ba.install_packages(root, runner=ran.append, probe=lambda python: (True, ""))
    assert ran[0][-1] == ba.host.TORCH_INDEX + "rocm7.2"


def test_the_prompt_names_the_amd_route_and_its_torch_index():
    index = ba.host.TORCH_INDEX + "rocm7.2"
    text = ba.announcement(route=ba.route_label("linux", index))
    assert "AMD / ROCm" in text and index in text


@pytest.mark.parametrize("family,index,expected", [
    ("linux", "https://x/cu128", "NVIDIA / CUDA"),
    ("macos", None, "Apple Silicon"),
    ("linux", None, "no supported GPU"),
])
def test_every_machine_gets_a_route_line(family, index, expected):
    assert expected in ba.route_label(family, index)


def test_main_announces_the_route_before_asking(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr(ba, "this_machines_index", lambda: ba.host.TORCH_INDEX + "rocm7.2")
    monkeypatch.setattr(ba.host, "os_family", lambda: "linux")
    assert ba.main([]) == 1
    assert "AMD / ROCm on Linux, torch from" in capsys.readouterr().out


def _amd_install(tmp_path, monkeypatch):
    root = tmp_path / "SkinTokens"
    (root / ".venv").mkdir(parents=True)
    (root / ".venv" / "pyvenv.cfg").write_text("version_info = 3.11.9\n")
    monkeypatch.setattr(ba.shutil, "which", lambda name: "/bin/uv")
    return root


def test_a_rocm_torch_that_cannot_run_on_the_card_stops_before_the_rest(tmp_path,
                                                                        monkeypatch):
    root = _amd_install(tmp_path, monkeypatch)
    ran = []
    with pytest.raises(SystemExit, match="hipErrorInvalidKernelFile"):
        ba.install_packages(root, runner=ran.append, index=ba.host.TORCH_INDEX + "rocm7.2",
                            probe=lambda python: (False, "hipErrorInvalidKernelFile"))
    assert len(ran) == 1, "the requirements went in on a torch that does not run"


def test_the_card_is_only_probed_on_amd(tmp_path, monkeypatch):
    root = _amd_install(tmp_path, monkeypatch)
    ba.install_packages(root, runner=lambda c: None, index="https://x/cu128",
                        probe=lambda python: pytest.fail("probed an NVIDIA install"))


def test_the_default_probe_is_the_trellis_one(monkeypatch):
    """One probe for every AMD bootstrap, so they agree on what "runs" means."""
    import bootstrap_trellis_rocm

    seen = []
    monkeypatch.setattr(bootstrap_trellis_rocm, "gpu_probe",
                        lambda python: seen.append(python) or (True, ""))
    ba.prove_torch_on_amd(Path("/venv/python"))
    assert seen == [Path("/venv/python")]
