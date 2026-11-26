"""Check which meshes headless remeshing and projection operate on."""

import sys
from types import SimpleNamespace as NS

import blender_shrinkwrap as wrap
import blender_voxel_remesh as remesh
import pytest


class Mesh:
    type = "MESH"

    def __init__(self, name, faces):
        self.name = name
        self.data = NS(polygons=[object()] * faces)
        self.select_set(False)
        self.modifiers = NS(new=self.new_modifier)

    def select_set(self, value):
        self.selected = value

    def new_modifier(self, *, name, type):
        self.modifier = NS(name=name, type=type)
        return self.modifier


@pytest.fixture
def scene(monkeypatch):
    class Objects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    objects = Objects([Mesh("old scene", 1)])
    active = NS(active=None)
    imported = {}
    audit = NS(joined=[], exports=[], modifiers=[])

    def gltf(*, filepath):
        assert all(obj.name != "old scene" for obj in objects)
        objects.extend(imported.get(filepath, []))

    def select_all(*, action):
        assert action == "DESELECT"
        for obj in objects:
            obj.select_set(False)

    def join():
        obj = active.active
        others = [other for other in objects if other is not obj and other.selected]
        audit.joined.append((obj.name, [other.name for other in others]))
        for other in others:
            obj.data.polygons.extend(other.data.polygons)
            objects.remove(other, do_unlink=True)

    def modifier_apply(*, modifier):
        obj = active.active
        audit.modifiers.append(obj.modifier)
        assert modifier == obj.modifier.name
        if obj.modifier.type == "REMESH":
            obj.data.polygons = [object()] * 7

    def export(**opts):
        audit.exports.append((opts, [obj for obj in objects if obj.selected]))

    bpy = NS(
        data=NS(objects=objects),
        context=NS(view_layer=NS(objects=active)),
        ops=NS(
            import_scene=NS(gltf=gltf),
            object=NS(select_all=select_all, join=join, modifier_apply=modifier_apply),
            export_scene=NS(gltf=export),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    return imported, audit


def assert_single_mesh_export(exports, obj):
    assert exports == [
        (
            {
                "filepath": "out.glb",
                "use_selection": True,
                "export_format": "GLB",
                "export_materials": "NONE",
                "export_normals": True,
            },
            [obj],
        )
    ]


@pytest.mark.parametrize("extra_piece", [False, True])
def test_voxel_remesh_joins_shards_and_exports_the_rebuilt_surface(
    scene, monkeypatch, extra_piece, capsys
):
    imported, audit = scene
    mesh = Mesh("largest", 5)
    imported["in.glb"] = [Mesh("other", 1), mesh] if extra_piece else [mesh]
    monkeypatch.setattr(sys, "argv", ["blender", "--", "in.glb", "out.glb", "0.025"])
    remesh.main()
    assert audit.joined == ([("largest", ["other"])] if extra_piece else [])
    (modifier,) = audit.modifiers
    assert modifier.mode == "VOXEL" and modifier.voxel_size == 0.025
    assert modifier.adaptivity == 0 and modifier.use_smooth_shade is False
    assert "-> 7 faces" in capsys.readouterr().out
    assert_single_mesh_export(audit.exports, mesh)


def test_shrinkwrap_only_exports_the_lower_resolution_mesh(scene, monkeypatch):
    imported, audit = scene
    low, high = Mesh("low", 5), Mesh("high", 500)
    imported["remeshed.glb"] = [low]
    imported["decode.glb"] = [Mesh("secondary high", 3), high]
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "remeshed.glb", "decode.glb", "out.glb", "0.01"]
    )
    wrap.main()
    assert audit.joined == [("high", ["secondary high"])]
    (modifier,) = audit.modifiers
    assert modifier.wrap_method == "NEAREST_SURFACEPOINT"
    assert modifier.target is high and modifier.offset == 0.01
    assert_single_mesh_export(audit.exports, low)


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (remesh, ["absent.glb", "out.glb"]),
        (wrap, ["absent.glb", "high.glb", "out.glb"]),
    ],
)
def test_mesh_tools_reject_an_import_without_meshes_before_export(
    scene, monkeypatch, tool, args
):
    _, audit = scene
    monkeypatch.setattr(sys, "argv", ["blender", "--", *args])
    with pytest.raises(SystemExit, match="no mesh in absent.glb"):
        tool.main()
    assert not audit.modifiers and not audit.exports
