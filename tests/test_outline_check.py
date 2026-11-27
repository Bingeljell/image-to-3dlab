"""Tests for the check that finds parts of a model sticking out past the pictures it was made from.

Small synthetic scenes where the right answer is obvious: a camera at +z looking down -z, a
picture whose cut-out is a known rectangle, and quads that stay inside it or reach past it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from image_to_3dlab import outline_check as oc
from image_to_3dlab import photo_paint as pp

IDENTITY = ((0, 1, 2), (1, 1, 1))
SIZE = 128


def _view(box=(32, 32, 96, 96), name="front", z=3.0, flip=False) -> pp.View:
    """A camera at +z (or -z with `flip`) looking at the origin; its picture's cut-out is
    the pixel box (x0, y0, x1, y1)."""
    image = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    image[..., :3] = 90
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1, 3] = 255
    c2w = np.eye(4)
    c2w[2, 3] = z
    if flip:  # half a turn about the up axis: at -z looking down +z
        c2w[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        c2w[2, 3] = -z
    return pp.View(image, c2w, 0.6, name)


def _pixels_to_world(view: pp.View, px: float, py: float) -> tuple[float, float]:
    """Where pixel (px, py) of `view` lands on the z = 0 plane."""
    depth = view.cam_to_world[2, 3]
    return ((px - SIZE / 2) * depth / view.focal_px, -(py - SIZE / 2) * depth / view.focal_px)


def _quad(view: pp.View, x0, y0, x1, y1, offset=0):
    """A quad on z = 0 covering the pixel box (x0, y0, x1, y1) of `view`, finely split so the
    parts that reach past the cut-out are made of vertices of their own."""
    (wx0, wy1), (wx1, wy0) = _pixels_to_world(view, x0, y0), _pixels_to_world(view, x1, y1)
    steps = 24
    xs, ys = np.linspace(wx0, wx1, steps + 1), np.linspace(wy0, wy1, steps + 1)
    gx, gy = np.meshgrid(xs, ys)
    positions = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    faces = []
    for row in range(steps):
        for col in range(steps):
            a = row * (steps + 1) + col
            b, c, d = a + 1, a + steps + 2, a + steps + 1
            faces += [[a, b, c], [a, c, d]]
    return positions, np.array(faces) + offset


def _join(*meshes):
    positions, faces, offset = [], [], 0
    for p, f in meshes:
        positions.append(p)
        faces.append(f - f.min() + offset)
        offset += len(p)
    return np.concatenate(positions), np.concatenate(faces)


# --- what sticks out --------------------------------------------------------------------------


def test_a_model_inside_its_cut_out_has_nothing_past_the_outline():
    view = _view()
    positions, faces = _quad(view, 40, 40, 88, 88)
    assert oc.parts_past(positions, faces, [view], frame=IDENTITY) == []


def test_a_part_reaching_far_below_the_cut_out_is_found_below_and_low():
    view = _view(box=(32, 20, 96, 80))
    positions, faces = _quad(view, 40, 24, 88, 104)   # the cut-out ends at 80, the model at 104
    parts = oc.parts_past(positions, faces, [view], frame=IDENTITY)
    assert len(parts) == 1
    part = parts[0]
    assert part.view == "front" and part.side == "below"
    # 24 of the model's 80 pixels of height lie past the outline
    assert part.reach == pytest.approx(24 / 80, abs=0.03)
    assert part.low == pytest.approx(0.0, abs=0.01) and part.high < 0.35


def test_a_part_just_past_the_outline_is_within_the_tolerance():
    view = _view(box=(32, 32, 96, 96))
    positions, faces = _quad(view, 40, 40, 88, 97)   # one pixel past: matte edges are this rough
    assert oc.parts_past(positions, faces, [view], frame=IDENTITY) == []


def test_two_separate_pieces_past_the_outline_are_two_parts():
    view = _view(box=(40, 32, 88, 96))
    body = _quad(view, 44, 36, 84, 92)
    left = _quad(view, 20, 50, 46, 60)    # sticks out 20 px to the left
    right = _quad(view, 82, 70, 110, 80)  # sticks out 22 px to the right
    positions, faces = _join(body, left, right)
    parts = oc.parts_past(positions, faces, [view], frame=IDENTITY)
    assert sorted(part.side for part in parts) == ["left", "right"]


def test_a_part_off_the_picture_counts_as_past_it():
    view = _view(box=(32, 32, 96, 128))
    positions, faces = _quad(view, 40, 40, 88, 150)   # runs out of the picture at the bottom
    parts = oc.parts_past(positions, faces, [view], frame=IDENTITY)
    assert len(parts) == 1 and parts[0].side == "below"
    assert parts[0].reach > 0.15


def test_only_the_picture_that_sees_it_past_reports_it():
    front = _view(box=(32, 32, 96, 104), name="front")
    back = _view(box=(32, 32, 96, 80), name="back", flip=True)
    positions, faces = _quad(front, 40, 40, 88, 100)
    parts = oc.parts_past(positions, faces, [front, back], frame=IDENTITY)
    assert [part.view for part in parts] == ["back"]


def test_reach_scales_to_centimetres_with_the_real_height():
    view = _view(box=(32, 20, 96, 80))
    positions, faces = _quad(view, 40, 24, 88, 104)
    part = oc.parts_past(positions, faces, [view], frame=IDENTITY)[0]
    assert oc.describe(part, height_m=1.6).startswith(f"{part.reach * 160:.0f} cm below")


# --- the report -------------------------------------------------------------------------------


def test_a_clean_report_says_so():
    report = oc.report([], views=2, height_m=None)
    assert report["clean"] is True and report["parts"] == []
    assert "Nothing sticks out" in report["message"]


def test_a_report_names_each_part_in_plain_words():
    view = _view(box=(32, 20, 96, 80))
    positions, faces = _quad(view, 40, 24, 88, 104)
    parts = oc.parts_past(positions, faces, [view], frame=IDENTITY)
    report = oc.report(parts, views=1, height_m=None)
    assert report["clean"] is False
    assert "1 part sticks out" in report["message"]
    assert "below" in report["message"] and "front picture" in report["message"]
    assert json.loads(json.dumps(report)) == report   # plain JSON for logs and the studio


def test_a_part_to_the_side_reads_as_plain_english():
    part = oc.Part(view="back", side="right", reach=0.2, low=0.5, high=0.8, vertices=9)
    assert oc.describe(part, height_m=0.6) == "12 cm to the right of the outline in the back picture, at 50%-80% of the model's height"


def test_many_parts_name_the_farthest_and_count_the_rest():
    parts = [oc.Part(view="front", side="below", reach=0.1 - i / 100, low=0.0, high=0.1, vertices=5) for i in range(5)]
    message = oc.report(parts, views=2)["message"]
    assert message.startswith("5 parts stick out")
    assert message.count("below the outline") == 3 and "and 2 smaller ones" in message
    assert len(oc.report(parts, views=2)["parts"]) == 5   # the JSON keeps every part


def test_pixal3d_view_files_get_plain_names():
    assert [oc.view_label(name) for name in ("view_000.png", "view_180.png", "view_090.png", "extra.png")] == [
        "front", "back", "side (90°)", "extra"]


def test_the_marked_picture_paints_what_sticks_out_red():
    view = _view(box=(32, 20, 96, 80))
    positions, faces = _quad(view, 40, 24, 88, 104)
    picture = oc.marked_picture(positions, faces, [view], frame=IDENTITY)
    assert picture.shape == (SIZE, SIZE, 3)
    red = picture[100, 64]
    assert red[0] > 200 and red[1] < 80 and red[2] < 80   # far past the cut-out: red
    inside = picture[50, 64]
    assert not (inside[0] > 200 and inside[1] < 80)       # inside the cut-out: not marked


# --- inputs -----------------------------------------------------------------------------------


def _write_views(directory: Path, view: pp.View) -> None:
    directory.mkdir(parents=True)
    Image.fromarray(view.image).save(directory / "view_000.png")
    meta = {"camera_angle_x": view.fov_x, "mesh_scale": 1,
            "frames": [{"file_path": "view_000.png", "transform_matrix": view.cam_to_world.tolist()}]}
    (directory / "transforms.json").write_text(json.dumps(meta))


def test_a_finish_run_folder_gives_its_mesh_and_its_pictures(tmp_path):
    view = _view()
    run = tmp_path / "wolf__finish__20261117-120000"
    _write_views(run / "input" / "source.views", view)
    positions, faces = _quad(view, 40, 40, 88, 88)
    (run / "steps").mkdir(parents=True)
    trimesh.Trimesh(positions, faces, process=False).export(run / "steps" / "1_orient.glb")
    trimesh.Trimesh(positions * 9, faces, process=False).export(run / "input" / "source.glb")
    model, views_dir = oc.find_inputs(run)
    assert model == run / "steps" / "1_orient.glb"   # the Finish mesh, not the heavy raw one
    assert views_dir == run / "input" / "source.views"


def test_a_pixal3d_model_finds_the_pictures_beside_it(tmp_path):
    view = _view()
    _write_views(tmp_path / "wolf__pixal3d__20261117.svviews", view)
    positions, faces = _quad(view, 40, 40, 88, 88)
    glb = tmp_path / "wolf__pixal3d__20261117.glb"
    trimesh.Trimesh(positions, faces, process=False).export(glb)
    assert oc.find_inputs(glb) == (glb, tmp_path / "wolf__pixal3d__20261117.svviews")


def test_a_model_without_pictures_is_refused_plainly(tmp_path):
    glb = tmp_path / "chair.glb"
    trimesh.creation.box().export(glb)
    with pytest.raises(FileNotFoundError, match="pictures"):
        oc.find_inputs(glb)


def test_check_reads_a_run_end_to_end(tmp_path):
    view = _view(box=(32, 20, 96, 80))
    _write_views(tmp_path / "views", view)
    positions, faces = _quad(view, 40, 24, 88, 104)
    glb = tmp_path / "model.glb"
    trimesh.Trimesh(positions, faces, process=False).export(glb)
    report = oc.check(glb, tmp_path / "views", frame=IDENTITY)
    assert report["clean"] is False and report["parts"][0]["side"] == "below"


def test_the_script_runs_and_writes_its_picture(tmp_path, monkeypatch, capsys):
    view = _view(box=(32, 20, 96, 80))
    _write_views(tmp_path / "views", view)
    positions, faces = _quad(view, 40, 24, 88, 104)
    glb = tmp_path / "model.glb"
    trimesh.Trimesh(positions, faces, process=False).export(glb)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import outline_check as script

    monkeypatch.setattr(oc, "PIXAL3D_FRAME", IDENTITY)
    picture = tmp_path / "marked.png"
    code = script.main([str(glb), "--views", str(tmp_path / "views"), "--picture", str(picture)])
    assert code == 1   # something sticks out
    assert picture.is_file()
    assert "sticks out" in capsys.readouterr().out
