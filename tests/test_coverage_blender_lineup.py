"""Check the render sweep's camera positions, timing and render setup."""

from itertools import pairwise
from types import SimpleNamespace as NS

import blender_lineup_sweep as lineup
import pytest


@pytest.mark.parametrize("count", [1, 4])
def test_sweep_camera_covers_the_lineup_at_constant_speed(count):
    positions = list(lineup.sweep_positions(count))
    span = (count - 1) * lineup.SPACING / 2
    assert len(positions) == lineup.FRAMES
    assert positions[0] == (1, (-span - 1.8, -7.2, 1.5), (-span, 0, 0.55))
    assert positions[-1][0] == lineup.FRAMES
    assert positions[-1][1] == pytest.approx((span + 1.8, -7.2, 1.5))
    assert positions[-1][2] == pytest.approx((span, 0, 0.55))
    steps = [right[1][0] - left[1][0] for left, right in pairwise(positions)]
    assert steps == pytest.approx([steps[0]] * (lineup.FRAMES - 1))


@pytest.mark.parametrize("engine", ["BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"])
def test_render_configuration_supports_both_eevee_versions(engine, monkeypatch):
    import sys

    scene = NS(render=NS(image_settings=NS()))
    types = NS(
        RenderSettings=NS(
            bl_rna=NS(
                properties={
                    "engine": NS(enum_items=[NS(identifier=engine)]),
                }
            )
        )
    )
    monkeypatch.setitem(sys.modules, "bpy", NS(context=NS(scene=scene), types=types))
    lineup.configure_render("frames/out")
    assert scene.render.engine == engine
    assert (scene.render.resolution_x, scene.render.resolution_y) == (1280, 720)
    assert scene.render.fps == 30
    assert (scene.frame_start, scene.frame_end) == (1, 150)
    assert scene.render.image_settings.file_format == "PNG"
    assert scene.render.filepath == "frames/out"


def test_camera_tracks_target_with_linear_keys(monkeypatch):
    import sys

    class KeyedObject:
        def __init__(self, name, data):
            self.name = name
            self.data = data
            self.keys = []
            self.constraints = NS(new=self.new_constraint)

        def new_constraint(self, kind):
            self.constraint = NS(kind=kind)
            return self.constraint

        def keyframe_insert(self, attribute, *, frame):
            self.keys.append((frame, getattr(self, attribute)))

    linked = []
    app = NS(
        data=NS(
            cameras=NS(new=lambda name: NS(name=name)),
            objects=NS(new=KeyedObject),
        ),
        context=NS(
            scene=NS(),
            collection=NS(objects=NS(link=linked.append)),
            preferences=NS(edit=NS()),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    lineup.build_camera(3)
    camera, target = linked
    assert app.context.scene.camera is camera
    assert app.context.preferences.edit.keyframe_new_interpolation_type == "LINEAR"
    assert camera.data.lens == 40
    assert camera.constraint.target is target
    assert (
        camera.constraint.kind,
        camera.constraint.track_axis,
        camera.constraint.up_axis,
    ) == (
        "TRACK_TO",
        "TRACK_NEGATIVE_Z",
        "UP_Y",
    )
    for key, camera_pos, target_pos in lineup.sweep_positions(3):
        assert camera.keys[key - 1] == (key, camera_pos)
        assert target.keys[key - 1] == (key, target_pos)


def test_clear_scene_drops_only_unused_blocks(monkeypatch):
    import sys

    class Objects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    bpy = NS(data=NS(objects=Objects([object(), object()])))
    for name in ("meshes", "materials", "images", "cameras", "lights", "actions"):
        setattr(bpy.data, name, [NS(users=0), NS(users=1)])
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    lineup.clear_scene()
    assert not bpy.data.objects
    for name in ("meshes", "materials", "images", "cameras", "lights", "actions"):
        assert [item.users for item in getattr(bpy.data, name)] == [1]
