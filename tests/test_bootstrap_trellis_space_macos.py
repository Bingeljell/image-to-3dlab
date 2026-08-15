"""Source and dependency contracts for the clean Mac bootstrap."""

import importlib.util
import json
from pathlib import Path


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
