"""Check the head-and-body join's scene selection and export options."""

import sys
from types import SimpleNamespace as NS

import hero_blender as hero
import pytest


class Object:
    def __init__(self, name, kind):
        self.name = name
        self.type = kind
        self.selected = True
        self.data = NS(polygons=["one"], materials=["surface"])

    def select_set(self, selected):
        self.selected = selected


@pytest.mark.parametrize("has_mesh", [True, False])
def test_hero_join_exports_one_unanimated_scene(
    monkeypatch, tmp_path, capsys, has_mesh
):
    source, output = tmp_path / "source.glb", tmp_path / "joined.glb"
    head, body, camera = (
        Object("Head", "MESH"),
        Object("Body", "MESH"),
        Object("Camera", "CAMERA"),
    )
    contents = [head, body, camera] if has_mesh else [camera]
    objects = []
    active = NS(active=None)
    operations = []

    def reset(*, use_empty):
        assert use_empty
        objects.clear()
        operations.append("reset")

    def load(*, filepath):
        assert filepath == str(source)
        objects.extend(contents)
        operations.append("import")

    def transform(**options):
        assert options == {"location": True, "rotation": True, "scale": True}
        assert active.active is head
        assert [obj for obj in objects if obj.selected] == [head, body]
        operations.append("transform")

    def join():
        head.data.polygons.extend(body.data.polygons)
        head.data.materials.extend(body.data.materials)
        objects.remove(body)
        operations.append("join")

    def remove(obj, *, do_unlink):
        assert do_unlink
        objects.remove(obj)
        operations.append("remove")

    def export(**options):
        assert objects == [head]
        assert options == {
            "filepath": str(output),
            "use_active_scene": True,
            "export_format": "GLB",
            "export_animations": False,
        }
        operations.append("export")

    bpy = NS(
        ops=NS(
            wm=NS(read_factory_settings=reset),
            import_scene=NS(gltf=load),
            object=NS(transform_apply=transform, join=join),
            export_scene=NS(gltf=export),
        ),
        context=NS(scene=NS(objects=objects), view_layer=NS(objects=active)),
        data=NS(objects=NS(remove=remove)),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "join", str(source), str(output)]
    )
    assert hero.main() == (0 if has_mesh else 1)
    captured = capsys.readouterr()
    if has_mesh:
        assert operations == [
            "reset",
            "import",
            "transform",
            "join",
            "remove",
            "export",
        ]
        assert head.name == "Character"
        assert "(2 faces, 2 materials)" in captured.out
    else:
        assert operations == ["reset", "import"]
        assert "no mesh" in captured.err
