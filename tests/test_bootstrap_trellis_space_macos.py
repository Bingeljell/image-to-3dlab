"""Source and dependency contracts for the clean Mac bootstrap."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bootstrap_trellis_space_macos.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_trellis_space_macos", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


def test_every_bootstrap_source_is_pinned():
    lock = json.loads(bootstrap.LOCK.read_text())["upstreams"]
    assert set(bootstrap.TARGETS).issubset(lock)
    for name in bootstrap.TARGETS:
        assert len(lock[name]["commit"]) == 40
        assert lock[name]["url"].startswith("https://")


# --- the machine check, which is the one this file was missing --------------------------
#
# Everything below `install()` builds Metal operators through xcrun. Without an early check
# the script clones four repositories and builds a dependency before failing, so the failure
# arrives minutes in and reads as a broken checkout rather than a machine that cannot host
# it. Hit for real on an AMD card, which is a machine TRELLIS.2 *can* host, through the
# wrong script.


def test_it_runs_on_apple_silicon():
    assert bootstrap.wrong_machine("macos-arm64") is None


def test_it_refuses_on_amd_and_names_the_route_that_works():
    problem = bootstrap.wrong_machine("linux-amd", card=bootstrap.host.AMD, family="linux")
    assert problem is not None
    assert "xcrun" in problem
    assert "bootstrap_trellis_rocm.py" in problem
    assert "bootstrap_trellis_space_macos.py" not in problem


def test_it_refuses_on_nvidia_and_names_the_route_that_works():
    problem = bootstrap.wrong_machine("linux-nvidia", card=bootstrap.host.NVIDIA,
                                      family="linux")
    assert "bootstrap_trellis_cuda.py" in problem


def test_a_machine_with_no_trellis_route_is_told_so_rather_than_left_guessing(monkeypatch):
    # Faked, or on a Mac the real machine answers and there is no problem to report.
    monkeypatch.setattr(bootstrap.host, "build_target", lambda *a, **k: None)
    problem = bootstrap.wrong_machine(card=bootstrap.host.OTHER, family="linux")
    assert "no route for it" in problem
    assert "Nothing downloaded" in problem


def test_every_card_it_can_redirect_names_a_real_bootstrap():
    for card, script in bootstrap.ELSEWHERE.items():
        assert (ROOT / script).is_file(), f"{card} is pointed at {script}, which is not there"


def test_it_checks_the_machine_before_cloning_anything(tmp_path, monkeypatch):
    """The whole point: nothing is fetched, and nothing is built."""
    monkeypatch.setattr(bootstrap.host, "build_target", lambda *a, **k: "linux-amd")
    monkeypatch.setattr(bootstrap, "bootstrap", lambda *a, **k: pytest.fail(
        "it cloned repositories before checking the machine"))
    assert bootstrap.main() == 1


def test_bootstrap_does_not_clone_or_execute_shiv_tree():
    source = SCRIPT.read_text()
    assert "shiv_trellis_mac" not in bootstrap.TARGETS
    assert "mps_compat.py" not in source
    assert "patch_trellis_space_core.py" in source
    assert "patch_trellis_metal_backends.py" in source


def test_microsoft_space_versions_are_preserved():
    requirements = bootstrap.REQUIREMENTS.read_text()
    for pin in (
        "torch==2.11.0",
        "torchvision==0.26.0",
        "transformers==4.57.3",
        "trimesh==4.10.1",
        "kornia==0.8.2",
    ):
        assert pin in requirements
