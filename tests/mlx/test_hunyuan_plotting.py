"""Render tiny assets with Matplotlib's off-screen backend."""

import runpy
import sys

import matplotlib
import numpy as np
import pytest
import trimesh
from PIL import Image

from hunyuan_mlx.shape.scripts import render_compare, render_gallery

pytestmark = pytest.mark.mlx_unit


@pytest.fixture
def plot_inputs(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Read the real output as well as the titles and layout of the saved figure.
    figures = []
    original = plt.Figure.savefig

    def save(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", save)
    image = tmp_path / "input.png"
    Image.new("RGB", (12, 12), "#fc4606").save(image)
    mesh = trimesh.creation.box(extents=(1, 2, 3))
    direct = tmp_path / "mesh.ply"
    mesh.export(direct)
    scene = tmp_path / "mesh.glb"
    trimesh.Scene(mesh).export(scene)
    assert isinstance(trimesh.load(direct), trimesh.Trimesh)
    assert isinstance(trimesh.load(scene), trimesh.Scene)
    return figures, image, direct, scene


def assert_render(out, aspect, has_input):
    with Image.open(out) as result:
        assert result.format == "PNG"
        assert result.width / result.height == pytest.approx(aspect, abs=0.15)
        rgb = np.array(result.convert("RGB"), dtype=int)
    blue = ((rgb[..., 2] - rgb[..., 0] > 30) & (rgb[..., 0] < 160)).sum()
    red = ((rgb[..., 0] > 210) & (rgb[..., 1] < 100) & (rgb[..., 2] < 40)).sum()
    assert blue > 100
    assert (red > 100) == has_input


def assert_view(axis, elevation, azimuth):
    assert (axis.elev, axis.azim) == (elevation, azimuth)
    assert axis.get_xlim3d() == (-0.5, 0.5)
    assert axis.get_ylim3d() == (-1, 1)
    assert axis.get_zlim3d() == (-1.5, 1.5)
    np.testing.assert_allclose(axis.get_box_aspect() / axis.get_box_aspect()[0], [1, 2, 3])


def test_compare_titles_layout_and_camera(plot_inputs, tmp_path, capsys, monkeypatch):
    figures, _, mesh, scene = plot_inputs
    out = tmp_path / "compare.png"
    monkeypatch.setattr(sys, "argv", [render_compare.__file__, str(out), f"Box={mesh}", f"Scene={scene}"])
    runpy.run_path(render_compare.__file__, run_name="__main__")
    assert_render(out, 1, False)
    axes = figures[0].axes
    assert [ax.get_title() for ax in axes] == ["Box\n0K verts", "", "Scene\n0K verts", ""]
    assert [ax.get_subplotspec().num1 for ax in axes] == [0, 2, 1, 3]
    for i, ax in enumerate(axes):
        assert_view(ax, *((90, -90) if i % 2 == 0 else (20, -60)))
    assert capsys.readouterr().out == f"wrote {out}\n"


def test_gallery_titles_layout_and_camera(plot_inputs, tmp_path, capsys, monkeypatch):
    figures, image, mesh, scene = plot_inputs
    out = tmp_path / "gallery.png"
    monkeypatch.setattr(sys, "argv", [render_gallery.__file__, str(out), f"{image}={mesh}", f"{image}={scene}"])
    runpy.run_path(render_gallery.__file__, run_name="__main__")
    assert_render(out, 1.5, True)
    axes = figures[0].axes
    assert [ax.get_title() for ax in axes] == ["input", "mesh front  (0K)", "mesh 3/4  (0K)", "", "", ""]
    assert [ax.get_subplotspec().num1 for ax in axes] == list(range(6))
    for i in (0, 3):
        assert len(axes[i].images) == 1
        assert_view(axes[i + 1], 90, -90)
        assert_view(axes[i + 2], 20, -60)
    assert capsys.readouterr().out == f"wrote {out}\n"
