"""Four-side previews: which file, which way is front, the cache, and the sheet."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viewer"))
import preview_api as pa


@pytest.mark.parametrize("facing, expected", [
    ([0, 0, 1], 0.0), ([1, 0, 0], 90.0), ([0, 0, -1], 180.0), ([-1, 0, 0], -90.0), (None, 0.0), ([0, 1, 0], 0.0),
])
def test_front_follows_the_models_facing(facing, expected):
    assert pa.front_azimuth(facing) == pytest.approx(expected)


def test_latest_is_the_furthest_step_and_an_animated_asset_shows_its_rig():
    asset = {"picture": "p.png", "model": "m.glb", "finished": "f.glb", "rigged": "r.glb", "clips": [{"name": "idle"}]}
    assert pa.stage_file(asset) == ("animated", "r.glb")
    assert pa.stage_file({"picture": "p.png", "model": "m.glb"}) == ("model", "m.glb")
    assert pa.stage_file(asset, "finished") == ("finished", "f.glb")
    assert pa.stage_file({"picture": "p.png"}, "model") is None
    with pytest.raises(ValueError):
        pa.stage_file(asset, "sideways")


def test_cache_key_changes_with_the_file_the_camera_and_the_look(tmp_path):
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"one")
    key = pa.cache_key(glb, 0.0, "paint")
    assert key == pa.cache_key(glb, 0.0, "paint")
    assert key != pa.cache_key(glb, 90.0, "paint") != pa.cache_key(glb, 0.0, "clay")
    glb.write_bytes(b"two!")
    assert key != pa.cache_key(glb, 0.0, "paint")


def _render(path: Path, box: tuple[int, int, int, int]) -> Path:
    image = Image.new("RGB", (pa.PANEL_W, pa.PANEL_H), (13, 13, 14))
    image.paste((200, 160, 40), box)
    image.save(path)
    return path


def test_sheet_has_one_panel_per_view_cropped_to_one_shared_box(tmp_path):
    panels = [(name, _render(tmp_path / f"{name}.png", (200 + i * 10, 100, 300, 500))) for i, (name, _) in enumerate(pa.VIEWS)]
    sheet = Image.open(pa.compose_sheet(panels, tmp_path / "out.png", title="knight · finished"))
    width = 100 + 2 * 16                          # widest subject (200..300) plus padding, the same for every panel
    assert sheet.width == 4 * width + 3 * 6
    assert sheet.height > 400 + 2 * pa.LABEL_PX   # room for the title row and the big labels
    assert not (tmp_path / "out.part.png").exists()


def test_labels_use_a_real_font_at_a_readable_size():
    font = pa.label_font()
    assert font.getbbox("Front")[3] >= pa.LABEL_PX * 0.6


def test_a_picture_asset_previews_without_blender(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (300, 600), (255, 0, 0)).save(tmp_path / "images" / "front.png")
    Image.new("RGB", (300, 600), (0, 0, 255)).save(tmp_path / "images" / "back.png")
    asset = {"name": "fox", "picture": "images/front.png", "back": "images/back.png"}
    path, shown = pa.preview(tmp_path, asset, blender=None)
    assert shown == "picture" and Image.open(path).size == (606, 600)
    assert path.parent == tmp_path / pa.PREVIEWS_DIR


def test_a_model_without_blender_says_how_to_get_it(tmp_path):
    (tmp_path / "m.glb").write_bytes(b"glb")
    with pytest.raises(RuntimeError, match="Setup"):
        pa.preview(tmp_path, {"model": "m.glb"}, blender=None)


def test_files_outside_output_are_refused(tmp_path):
    with pytest.raises(ValueError):
        pa.preview(tmp_path, {"model": "../../etc/hosts.glb"}, blender=None)
    with pytest.raises(ValueError, match="look"):
        pa.preview(tmp_path, {"model": "m.glb"}, look="neon")


def test_a_move_is_four_moments_from_one_angle_and_a_model_is_four_sides():
    move = pa.shots_for(True)
    assert [frac for _, _, frac in move] == list(pa.MOMENTS) and {az for _, az, _ in move} == {pa.MOVE_AZIMUTH}
    assert [name for name, _, _ in pa.shots_for(False)] == ["Front", "Three-quarter", "Side", "Back"]
    assert pa.shots_arg([("Front", 0.0, None), ("m0", 35.0, 0.1)]) == "Front=0.0=;m0=35.0=0.1"
    assert pa.moment_label(1.04) == "1.0 s" and pa.moment_label(None) == ""


def test_a_move_is_found_by_name_and_a_missing_one_lists_what_there_is():
    asset = {"name": "dancer", "clips": [{"name": "wave", "file": "animate/d/d_wave.glb"}]}
    assert pa.find_move(asset, "wave") == "animate/d/d_wave.glb"
    with pytest.raises(ValueError, match="it has: wave"):
        pa.find_move(asset, "moonwalk")


def test_moves_and_models_never_share_a_cached_picture(tmp_path):
    glb = tmp_path / "a.glb"
    glb.write_bytes(b"x")
    assert pa.cache_key(glb, 0.0, "paint") != pa.cache_key(glb, 0.0, "paint:move")
