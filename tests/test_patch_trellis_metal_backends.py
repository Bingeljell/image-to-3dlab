"""Contracts for replaying proven fixes over Pedro's Metal repositories."""

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_trellis_metal_backends.py"


def test_patch_keeps_production_fixes_and_excludes_investigation_logging():
    source = SCRIPT.read_text()
    for marker in (
        "MTLBVH_QUERY_CHUNK",
        "FixedStack48",
        "closest_triangle_stackless",
        "@autoreleasepool",
        "idx == 0xFFFFFFFFu",
        "self._entries = {}",
        "alpha_mode = 'OPAQUE'",
        "doubleSided=False",
    ):
        assert marker in source
    for diagnostic in ("I2L_REMESH_DIAG", "I2L_CHECKPOINT_PREFIX", "MTLBVH_TRACE"):
        assert diagnostic not in source


def test_port_does_not_modify_clean_pedro_dependencies_that_need_no_local_fix():
    source = SCRIPT.read_text()
    assert "deps/mtldiffrast" not in source
    assert "deps/mtlgemm" not in source
