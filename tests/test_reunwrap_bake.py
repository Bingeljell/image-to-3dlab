"""Tests for the re-unwrap and re-bake step.

Only the bpy-free parts are testable here, which is exactly why they are bpy-free: the
argument parsing decides where a 300 MB bake lands, and finding the base-colour image by
walking the node link (rather than by node name) is what makes the script survive a
Blender version bump.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "blender_reunwrap_bake.py"


def _load():
    """Import the real module. It only touches bpy inside main()."""
    spec = importlib.util.spec_from_file_location("reunwrap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reunwrap = _load()


def test_parses_arguments_after_the_separator():
    source, dest, size, _, _ = reunwrap.parse_args(
        ["blender", "-b", "--python", "x.py", "--", "a.glb", "b.glb"]
    )
    assert (source, dest, size) == ("a.glb", "b.glb", 2048)


def test_accepts_an_explicit_atlas_size():
    assert reunwrap.parse_args(["--", "a.glb", "b.glb", "4096"])[2] == 4096


def test_angle_limit_is_converted_to_radians():
    """Blender's operator takes radians; passing degrees would barely cut any island."""
    import math

    assert reunwrap.parse_args(["--", "a.glb", "b.glb", "2048", "89"])[3] == pytest.approx(
        math.radians(89)
    )


def test_island_margin_defaults_tiny():
    """A fraction of the atlas PER ISLAND: 0.005 with thousands of islands consumed the
    whole atlas and collapsed coverage to 1%, median island 2 texels."""
    assert reunwrap.parse_args(["--", "a.glb", "b.glb"])[4] < 0.001


def test_rejects_an_island_margin_big_enough_to_eat_the_atlas():
    with pytest.raises(SystemExit):
        reunwrap.parse_args(["--", "a.glb", "b.glb", "2048", "89", "0.05"])


def test_rejects_an_out_of_range_angle_limit():
    with pytest.raises(SystemExit):
        reunwrap.parse_args(["--", "a.glb", "b.glb", "2048", "180"])


def test_rejects_an_unsupported_atlas_size():
    """A typo here would silently bake into a wrong-sized atlas."""
    with pytest.raises(SystemExit):
        reunwrap.parse_args(["--", "a.glb", "b.glb", "3000"])


def test_missing_arguments_exit_with_usage():
    with pytest.raises(SystemExit):
        reunwrap.parse_args(["--", "only-one.glb"])


def test_no_separator_means_no_arguments():
    with pytest.raises(SystemExit):
        reunwrap.parse_args(["blender", "-b"])


# --- finding the texture to re-bake --------------------------------------------------


class _Socket:
    def __init__(self, link=None):
        self.is_linked = link is not None
        self.links = [link] if link else []


class _Link:
    def __init__(self, node):
        self.from_node = node


class _Node:
    def __init__(self, type_, image=None, inputs=None):
        self.type = type_
        self.image = image
        self.inputs = inputs or {}


class _Material:
    def __init__(self, nodes, use_nodes=True):
        self.use_nodes = use_nodes
        self.node_tree = type("T", (), {"nodes": nodes})()


def test_finds_the_image_linked_to_base_color():
    image = object()
    tex = _Node("TEX_IMAGE", image=image)
    bsdf = _Node("BSDF_PRINCIPLED", inputs={"Base Color": _Socket(_Link(tex))})
    assert reunwrap.base_colour_image(_Material([bsdf, tex])) is image


def test_falls_back_to_any_image_when_base_color_is_unlinked():
    """Some exporters leave Base Color unlinked; a lone texture is still the albedo."""
    image = object()
    tex = _Node("TEX_IMAGE", image=image)
    bsdf = _Node("BSDF_PRINCIPLED", inputs={"Base Color": _Socket()})
    assert reunwrap.base_colour_image(_Material([bsdf, tex])) is image


def test_returns_none_when_there_is_no_texture():
    bsdf = _Node("BSDF_PRINCIPLED", inputs={"Base Color": _Socket()})
    assert reunwrap.base_colour_image(_Material([bsdf])) is None


def test_returns_none_for_a_material_without_nodes():
    assert reunwrap.base_colour_image(_Material([], use_nodes=False)) is None


def test_returns_none_for_no_material():
    assert reunwrap.base_colour_image(None) is None
