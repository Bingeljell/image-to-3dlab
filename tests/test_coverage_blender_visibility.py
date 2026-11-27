"""Check visibility decisions using small face-ID buffers."""

import numpy as np
import pytest

from scripts import blender_visibility_cull as cull
from scripts.visibility_cull import index_to_color


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["in.glb", "out.glb"], ("in.glb", "out.glb", 200, 512)),
        (["in.glb", "out.glb", "8"], ("in.glb", "out.glb", 8, 512)),
        (["in.glb", "out.glb", "8", "64"], ("in.glb", "out.glb", 8, 64)),
    ],
)
def test_render_arguments(argv, expected):
    assert cull.parse_args(argv) == expected


def test_missing_output_explains_usage():
    with pytest.raises(SystemExit, match="usage:"):
        cull.parse_args(["in.glb"])


def test_each_view_keeps_previous_faces_and_ignores_unknown_pixels():
    seen = np.array([True, False, False])
    pixels = np.ones((4, 4), dtype=np.float32)
    pixels[:3, :3] = index_to_color(np.array([1, 1, 3])) / 255
    pixels[3, :3] = 0
    cull.record_visible_faces(seen, pixels.ravel())
    assert seen.tolist() == [True, True, False]


def test_missing_mesh_stops_before_render(monkeypatch):
    import sys
    from types import SimpleNamespace as NS

    operations = []

    def reset(*, use_empty):
        operations.append(("reset", use_empty))

    def gltf(*, filepath):
        operations.append(("import", filepath))

    bpy = NS(
        data=NS(objects=[NS(type="CAMERA")]),
        ops=NS(
            wm=NS(read_factory_settings=reset),
            import_scene=NS(gltf=gltf),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bmesh", NS())
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=object))
    monkeypatch.setattr(sys, "argv", ["blender", "--", "in.glb", "out.glb"])
    with pytest.raises(SystemExit, match="no mesh in in.glb"):
        cull.main()
    assert operations == [("reset", True), ("import", "in.glb")]


def test_face_id_buffer_repeats_face_colors_for_each_polygon_corner():
    colors = cull.face_id_colors(np.array([3, 4, 3]))
    assert colors.shape == (10, 4)
    assert np.all(colors[:, 3] == 1)
    assert np.array_equal(
        np.rint(colors[:, :3] * 255).astype(np.uint8),
        index_to_color(np.array([0] * 3 + [1] * 4 + [2] * 3)),
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_visibility_render_disables_pixel_transforms(monkeypatch, legacy):
    from types import SimpleNamespace as NS

    class Choice:
        def __init__(self, label, allowed, **attributes):
            self.label, self.allowed = label, allowed
            for name, value in attributes.items():
                setattr(self, name, value)

        def __setattr__(self, name, value):
            if name == getattr(self, "label", None) and value != self.allowed:
                raise TypeError(f"unsupported {name}: {value}")
            super().__setattr__(name, value)

    engine = "BLENDER_EEVEE" if legacy else "BLENDER_EEVEE_NEXT"
    view_transform = "Standard" if legacy else "Raw"
    render = Choice("engine", engine, image_settings=NS())
    view = Choice("view_transform", view_transform)
    scene = NS(render=render, view_settings=view, display_settings=NS())
    if legacy:
        scene.eevee = NS(taa_render_samples=16, use_gtao=True)
    camera = object()
    cull.configure_render(scene, camera, 128)
    assert scene.camera is camera
    assert render.engine == engine and view.view_transform == view_transform
    assert render.resolution_x == render.resolution_y == 128
    assert render.filter_size == render.dither_intensity == 0
    assert render.film_transparent and render.image_settings.color_mode == "RGBA"
    assert view.look == "None" and scene.display_settings.display_device == "sRGB"
    if legacy:
        assert scene.eevee.taa_render_samples == 1
        assert scene.eevee.use_gtao is False
