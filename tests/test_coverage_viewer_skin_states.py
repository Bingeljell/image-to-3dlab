"""A skin's approved images and last valid build survive failed steps."""

from pathlib import Path
from unittest.mock import Mock

import pixal3d_generate
import pytest
import skins_api as skins
from PIL import Image

from tests.test_coverage_viewer_workers import Child


@pytest.fixture
def front_skin(tmp_path):
    (tmp_path / "front.png").touch()
    (tmp_path / "model.shape.bin").touch()
    store = skins.SkinStore(tmp_path)
    skin = skins.Skin("Red", "paint red", {"front": "front.png", "shape": "model.shape.bin", "finish": {}})
    store.save(skin)
    return store, skin


def test_preview_does_not_offer_back_or_edits_after_skin_is_done(front_skin):
    store, skin = front_skin
    for method in (store.picture_try, store.approve):
        with pytest.raises(skins.SkinError, match="no back picture"):
            method(skin, "back")
    with pytest.raises(skins.SkinError, match="no front picture to approve"):
        store.approve(skin, "front")
    with pytest.raises(skins.SkinError, match="Keep the change under"):
        store.picture_try(skin, "front", "x" * (skins.MAX_INSTRUCTION + 1))
    skin.base["back"] = "back.png"
    with pytest.raises(skins.SkinError, match="Approve the front"):
        store.approve(skin, "back")
    skin.status = "done"
    with pytest.raises(skins.SkinError, match="finished"):
        store.picture_try(skin, "front")
    assert store.get(skin.id).tries == {"front": [], "back": []}


def test_finished_skin_is_not_rebuilt(front_skin):
    store, skin = front_skin
    skin.approved["front"] = "front.png"
    skin.status = "done"
    with pytest.raises(skins.SkinError, match="already finished"):
        skins.SkinBuilds(store).start(skin, runner=Mock(side_effect=AssertionError))


def test_corrupt_skin_record_does_not_hide_other_skins(front_skin):
    store, skin = front_skin
    broken = store.folder("b" * 12)
    broken.mkdir()
    (broken / skins.RECORD).write_text("{broken", encoding="utf-8")
    assert [item.id for item in store.all()] == [skin.id]


@pytest.mark.parametrize("empty", [False, True])
def test_cut_out_box_thresholds_alpha_and_uses_image_size(tmp_path, monkeypatch, empty):
    picture = tmp_path / "picture.png"
    Image.new("RGB", (8, 10)).save(picture)

    def remove(image):
        assert image.mode == "RGB" and image.size == (8, 10)
        result = Image.new("RGBA", image.size, (0, 0, 0, 127))
        if not empty:
            result.putpixel((2, 6), (255, 0, 0, 200))
        return result, "fixture remover"

    monkeypatch.setattr(pixal3d_generate, "_rembg_remove", remove)
    assert skins.cut_out_box(picture) == (None if empty else (0.25, 0.6, 0.375, 0.7))


def test_repaint_keeps_worker_auto_settings_when_original_has_no_overrides(front_skin):
    store, skin = front_skin
    skin.approved["front"] = "front.png"
    command = skins.repaint_command(store.output, skin, "python")
    assert command[-2:] == ["--reuse-shape", str(store.output / "model.shape.bin")]
    for flag in ("--steps", "--seed", "--back"):
        assert flag not in command


def test_build_stops_if_cancelled_before_first_process(front_skin):
    store, skin = front_skin
    skin.approved["front"] = "front.png"
    book = skins.SkinBuilds(store)
    job = skins.SkinBuild("a" * 32, skin.id)
    book.jobs[job.id] = job
    book.active = job.id
    emit = job.emit

    def ask_cancel(event):
        emit(event)
        if event["phase"] == "check":
            job.cancel_requested = True

    job.emit = ask_cancel
    child = Mock(side_effect=AssertionError("cancelled skin started a process"))
    book.run(job, skin, runner=child, box=lambda _: None)
    child.assert_not_called()
    assert job.status == "cancelled"
    assert job.events[-1]["phase"] == "cancelled"
    assert store.get(skin.id).approved["front"] == "front.png"


def test_skin_reports_if_finish_process_writes_no_model(front_skin, monkeypatch):
    import finish_api

    store, skin = front_skin
    skin.approved["front"] = "front.png"
    monkeypatch.setattr(finish_api.blender_lookup, "find_blender", lambda: Path("fixture-blender"))
    runner = Mock(side_effect=[Child("[1/6] start\n\n"), Child("I2L_STAGE::bake::Bake\n")])
    book = skins.SkinBuilds(store)
    job = book.start(skin, background=False, runner=runner, box=lambda _: (0, 0, 1, 1))
    assert runner.call_count == 2
    assert job.status == "error"
    assert "without writing the skin" in job.error
    assert job.status == store.get(skin.id).status
    assert book.active is None
    assert store.get(skin.id).approved["front"] == "front.png"


def test_cached_all_skins_variant_does_not_rewrite_on_second_read(front_skin, monkeypatch):
    from image_to_3dlab import glb_skins
    from tests.test_glb_skins import _glb

    store, skin = front_skin
    original = store.output / "base.glb"
    paint = store.output / "paint.glb"
    original.write_bytes(_glb(b"BASE", b"BASE-SHINE"))
    paint.write_bytes(_glb(b"RED", b"RED-SHINE"))
    skin.status, skin.result = "done", paint.name
    wrapped = Mock(wraps=glb_skins.with_variants)
    monkeypatch.setattr(glb_skins, "with_variants", wrapped)
    with pytest.raises(skins.SkinError, match="not a model"):
        skins.all_skins(store.output, "missing.glb", [skin])
    made = skins.all_skins(store.output, original.name, [skin])
    assert (store.output / made).is_file()
    assert skins.all_skins(store.output, original.name, [skin]) == made
    wrapped.assert_called_once()
