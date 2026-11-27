"""Check shell thickness, selection and export across differently sized meshes."""

import sys
from types import SimpleNamespace as NS

import blender_solidify as solidify
import pytest


@pytest.mark.parametrize("fraction", [None, 0.01])
def test_shell_thickness_scales_and_exports_meshes_only(monkeypatch, fraction):
    class SceneObjects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    def mesh(name, dimensions):
        result = NS(
            type="MESH", name=name, dimensions=dimensions, data=NS(polygons=[object()])
        )
        result.select_set = lambda selected: setattr(result, "selected", selected)

        def modifier(modifier_name, kind):
            assert modifier_name == "Solidify" and kind == "SOLIDIFY"
            result.modifier = NS()
            return result.modifier

        result.modifiers = NS(new=modifier)
        return result

    small, big = mesh("small", (1, 2, 3)), mesh("big", (1, 2, 10))
    light = NS(type="LIGHT")
    light.select_set = lambda selected: setattr(light, "selected", selected)
    objects = SceneObjects([NS(type="LIGHT")])
    active = NS(active=None)
    settings = []

    def import_scene(*, filepath):
        assert filepath == "in.glb"
        assert not objects
        objects.extend([small, big, light])

    def apply(*, modifier):
        assert modifier == "Solidify"
        obj = active.active
        options = obj.modifier
        assert options.use_rim and not options.use_rim_only and options.offset == 0
        settings.append((obj.name, options.thickness))
        obj.data.polygons.extend([object(), object(), object()])

    def export(**kwargs):
        assert kwargs == {
            "filepath": "out.glb",
            "use_selection": True,
            "export_format": "GLB",
            "export_yup": True,
        }
        assert [obj.name for obj in objects if obj.selected] == ["small", "big"]

    bpy = NS(
        data=NS(objects=objects),
        context=NS(view_layer=NS(objects=active)),
        ops=NS(
            import_scene=NS(gltf=import_scene),
            object=NS(modifier_apply=apply),
            export_scene=NS(gltf=export),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    args = ["blender", "--", "in.glb", "out.glb"]
    if fraction is not None:
        args.append(str(fraction))
    monkeypatch.setattr(sys, "argv", args)
    solidify.main()
    multiplier = fraction if fraction is not None else 0.004
    assert settings == [("small", 3 * multiplier), ("big", 10 * multiplier)]


def test_solidify_rejects_an_import_without_a_mesh(monkeypatch):
    def import_scene(*, filepath):
        assert filepath == "in.glb"

    bpy = NS(data=NS(objects=[]), ops=NS(import_scene=NS(gltf=import_scene)))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    with pytest.raises(SystemExit, match="no mesh"):
        solidify.solidify("in.glb", "out.glb", 0.004)
