"""Verify occlusion bake setup with a small imported scene."""

import sys
from types import SimpleNamespace as NS

import blender_bake_ao as ao
import pytest


@pytest.mark.parametrize("world_exists", [False, True])
def test_occlusion_bakes_into_active_texture_at_asset_scale(monkeypatch, world_exists):
    scene = NS(
        world=NS(light_settings=NS()) if world_exists else None,
        cycles=NS(),
        render=NS(bake=NS()),
    )
    nodes = NS(new=lambda kind: NS(kind=kind))
    material = NS(node_tree=NS(nodes=nodes))
    events = []
    first = NS(
        type="MESH",
        name="first",
        select_set=lambda selected: events.append(("first", selected)),
        dimensions=NS(x=0, y=3, z=4),
        data=NS(polygons=[object()], uv_layers=[object()]),
        material_slots=[NS(material=None), NS(material=material)],
    )
    second = NS(
        type="MESH", select_set=lambda selected: events.append(("second", selected))
    )
    active = NS(active=None)
    objects = [first, second]
    texture = NS(save=lambda: events.append("save"))

    def add_image(name, *, width, height):
        assert name == "AO"
        assert (width, height) == (64, 64)
        return texture

    def join():
        assert active.active is first
        assert events[-2:] == [("first", True), ("second", True)]
        events.append("joined")

    def bake(*, type):
        assert type == "AO"
        assert scene.cycles.device == "CPU"
        assert scene.cycles.samples == 6
        assert scene.render.bake.use_clear
        assert scene.render.bake.margin == 8
        assert scene.world.light_settings.distance == pytest.approx(0.5)
        assert nodes.active.image is texture and nodes.active.select
        events.append("bake")

    def world(name):
        assert not world_exists and name == "W"
        events.append("world")
        return NS(light_settings=NS())

    app = NS(
        data=NS(objects=objects, images=NS(new=add_image), worlds=NS(new=world)),
        context=NS(scene=scene, view_layer=NS(objects=active)),
        ops=NS(
            wm=NS(
                read_factory_settings=lambda *, use_empty: events.append(
                    ("reset", use_empty)
                )
            ),
            import_scene=NS(
                gltf=lambda *, filepath: events.append(("import", filepath))
            ),
            object=NS(
                select_all=lambda *, action: events.append(action), join=join, bake=bake
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    ao.main(["asset.glb", "out.png", "6", "64", "0.1"])
    assert events == [
        ("reset", True),
        ("import", "asset.glb"),
        "DESELECT",
        ("first", True),
        ("second", True),
        "joined",
        *([] if world_exists else ["world"]),
        "bake",
        "save",
    ]
    assert texture.filepath_raw == "out.png" and texture.file_format == "PNG"
    assert material.use_nodes


@pytest.mark.parametrize("mesh, message", [(False, "no mesh"), (True, "has no UVs")])
def test_occlusion_rejects_missing_mesh_or_uv_before_baking(monkeypatch, mesh, message):
    first = NS(
        type="MESH",
        name="one",
        data=NS(polygons=[], uv_layers=[]),
        select_set=lambda _: None,
    )
    app = NS(
        data=NS(objects=[first] if mesh else []),
        context=NS(view_layer=NS(objects=NS(active=None))),
        ops=NS(
            wm=NS(read_factory_settings=lambda *, use_empty: None),
            import_scene=NS(gltf=lambda *, filepath: None),
            object=NS(select_all=lambda *, action: None),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", app)
    monkeypatch.setattr(sys, "argv", ["blender", "--", "asset.glb", "out.png"])
    with pytest.raises(SystemExit, match=message):
        ao.main()
