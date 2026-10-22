"""Tests for the Windows (MSVC) build fix to Pixal3D's uv_bake.cpp.

What matters: the one constant a lambda reads without capturing becomes static (MSVC's
C3493 stops the Windows NVIDIA build otherwise), nothing else changes, re-running is a no-op,
and a moved anchor is refused rather than guessed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "patch_pixal3d_msvc.py"
VENDOR_SOURCE = REPO / "vendor" / "pixal3d-cpp" / "src" / "uv_bake.cpp"


def _load():
    spec = importlib.util.spec_from_file_location("patch_pixal3d_msvc", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patch = _load()

# The shape of upstream's weld at commit c89126d (raven38/pixal3d.cpp).
ORIGINAL = '''    const float eps2 = step * step;
    constexpr int64_t BIAS = (int64_t)1 << 20;
    auto pack = [](int64_t cx, int64_t cy, int64_t cz) -> uint64_t {
        auto clampc = [](int64_t c) { return std::max<int64_t>(-BIAS + 1, std::min<int64_t>(BIAS - 2, c)); };
        return ((uint64_t)(clampc(cx) + BIAS) << 42);
    };
'''


def test_the_constant_becomes_static_and_nothing_else_changes():
    out = patch.apply(ORIGINAL)
    assert "static constexpr int64_t BIAS = (int64_t)1 << 20;" in out
    assert out.replace("static constexpr", "constexpr", 1) == ORIGINAL


def test_running_twice_changes_nothing():
    once = patch.apply(ORIGINAL)
    assert patch.apply(once) == once
    assert patch.is_patched(once) and not patch.is_patched(ORIGINAL)


def test_a_moved_anchor_is_refused():
    with pytest.raises(SystemExit, match="anchor"):
        patch.apply(ORIGINAL.replace("constexpr int64_t BIAS", "constexpr long BIAS"))


@pytest.mark.skipif(not VENDOR_SOURCE.is_file(), reason="no vendored pixal3d.cpp checkout")
def test_the_vendored_source_has_the_anchor_or_is_patched():
    source = VENDOR_SOURCE.read_text(encoding="utf-8")
    assert patch.is_patched(source) or patch.ANCHOR in source
