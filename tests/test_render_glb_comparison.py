"""Test the headless GLB comparison command's framing and layout helpers."""

from __future__ import annotations

import importlib.util
import math
import ntpath
from pathlib import Path, PureWindowsPath

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_glb_comparison.py"
SPEC = importlib.util.spec_from_file_location("render_glb_comparison", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_bounding_sphere_covers_the_diagonal_not_the_longest_side():
    """An elongated subject turns its diagonal to the camera at some angles.

    Sizing on the longest side alone crops it at those azimuths and not others, which makes
    a comparison image lie about the assets rather than about the camera.
    """
    radius = mod.bounding_sphere_radius((0, 0, 0), (10, 2, 2))
    assert radius == pytest.approx(math.sqrt(25 + 1 + 1))
    assert radius > 10 / 2


def test_bounding_sphere_never_returns_zero():
    # A degenerate (flat or empty) box would otherwise put the camera inside the subject.
    assert mod.bounding_sphere_radius((1, 1, 1), (1, 1, 1)) == 1.0


def test_camera_distance_frames_the_sphere():
    # At exactly radius/sin(half) the sphere touches the frame edge; the margin backs off.
    exact = 5.0 / math.sin(0.5)
    assert mod.camera_distance(5.0, 0.5, margin=1.0) == pytest.approx(exact)
    assert mod.camera_distance(5.0, 0.5, margin=1.1) == pytest.approx(exact * 1.1)


@pytest.mark.parametrize("half_angle", [0.0, -0.3, math.pi / 2, 2.0])
def test_camera_distance_rejects_impossible_fields_of_view(half_angle):
    with pytest.raises(ValueError):
        mod.camera_distance(1.0, half_angle)


def test_shared_crop_box_spans_every_panel():
    box = mod.shared_crop_box([(100, 50, 200, 300), (80, 70, 210, 280)], (400, 400), pad=0)
    assert box == (80, 50, 210, 300)


def test_shared_crop_box_is_clamped_to_the_frame():
    box = mod.shared_crop_box([(2, 2, 398, 398)], (400, 400), pad=50)
    assert box == (0, 0, 400, 400)


def test_shared_crop_box_rejects_an_empty_set():
    with pytest.raises(ValueError):
        mod.shared_crop_box([], (10, 10))


def test_asset_spec_splits_path_and_label():
    path, label = mod.parse_asset("output/a.glb:MLX fp16")
    assert path == Path("output/a.glb")
    assert label == "MLX fp16"


def test_asset_spec_without_a_label_uses_the_file_stem():
    path, label = mod.parse_asset("output/storm_ram.glb")
    assert path == Path("output/storm_ram.glb")
    assert label == "storm_ram"


@pytest.mark.parametrize(("suffix", "label"), [("", "storm_ram"), (":MLX fp16", "MLX fp16")])
def test_asset_spec_with_windows_drive(monkeypatch, suffix, label):
    class MissingWindowsPath(PureWindowsPath):
        def exists(self):
            return False

    monkeypatch.setattr(mod, "Path", MissingWindowsPath)
    monkeypatch.setattr(mod, "splitdrive", ntpath.splitdrive, raising=False)
    path = r"C:\models\storm_ram.glb"
    assert mod.parse_asset(path + suffix) == (PureWindowsPath(path), label)


def test_blender_render_code_returns_the_shipped_worker():
    assert mod.blender_render_code() == mod.BLENDER_WORKER.read_text(encoding="utf-8")
