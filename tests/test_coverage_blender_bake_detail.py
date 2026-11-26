"""Check bake selection and material traversal at Blender's API boundary."""

import sys
from types import SimpleNamespace as NS

import blender_bake_detail as detail
import pytest


def linked(node):
    return NS(links=[NS(from_node=node)])


def test_metallic_roughness_follows_a_linked_channel_split():
    texture = NS(type="TEX_IMAGE", image=object())
    split = NS(type="SEPARATE_COLOR", inputs=[linked(texture)])
    bsdf = NS(
        type="BSDF_PRINCIPLED",
        inputs={
            "Metallic": linked(NS(type="VALUE", inputs=[])),
            "Roughness": linked(split),
        },
    )
    mat = NS(use_nodes=True, node_tree=NS(nodes=[texture, bsdf]))
    assert detail._mr_image(mat) is texture.image
    texture.image = None
    assert detail._mr_image(mat) is None
    split.inputs = [NS(links=[])]
    assert detail._mr_image(mat) is None
    mat.node_tree.nodes = [texture]
    assert detail._mr_image(mat) is None
    mat.use_nodes = False
    assert detail._mr_image(mat) is None
    assert detail._mr_image(None) is None


class SceneObject:
    def __init__(self, name, kind, uv_layers=None):
        self.name = name
        self.type = kind
        self.data = NS(uv_layers=uv_layers)
        self.selected = False

    def select_set(self, selected):
        self.selected = selected


def test_bake_import_returns_only_new_meshes():
    original = SceneObject("original", "MESH")
    imported = SceneObject("new", "MESH")
    camera = SceneObject("camera", "CAMERA")
    objects = [original]

    def gltf(*, filepath):
        assert filepath == "source.glb"
        objects.extend([imported, camera])

    bpy = NS(data=NS(objects=objects), ops=NS(import_scene=NS(gltf=gltf)))
    assert detail._import(bpy, "source.glb") == [imported]


def test_bake_selects_only_sources_and_the_active_target():
    src = SceneObject("source", "MESH")
    target = SceneObject("target", "MESH")
    other = SceneObject("other", "MESH")
    other.selected = True
    active = NS(active=other)
    baked = []

    def select_all(*, action):
        assert action == "DESELECT"
        for obj in (src, target, other):
            obj.select_set(False)

    def bake(*, type):
        baked.append(
            (
                type,
                active.active,
                tuple(o.name for o in (src, target, other) if o.selected),
            )
        )

    bpy = NS(
        ops=NS(object=NS(select_all=select_all, bake=bake)),
        context=NS(view_layer=NS(objects=active)),
    )
    detail._bake(bpy, [src], target, "NORMAL")
    assert baked == [("NORMAL", target, ("source", "target"))]


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        ([], "expected one mesh in target.glb, found 0"),
        ([SceneObject("one", "MESH"), SceneObject("two", "MESH")], "found 2"),
        ([SceneObject("one", "MESH", [])], "has no UVs"),
    ],
)
def test_bake_stops_before_baking_invalid_target(monkeypatch, targets, message):
    objects = []
    imports = []
    source = SceneObject("source", "MESH")

    def reset(*, use_empty):
        assert use_empty
        objects.clear()

    def gltf(*, filepath):
        imports.append(filepath)
        objects.extend([source] if filepath == "source.glb" else targets)

    bpy = NS(
        data=NS(objects=objects),
        ops=NS(wm=NS(read_factory_settings=reset), import_scene=NS(gltf=gltf)),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Matrix=object))
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "source.glb", "target.glb", "out.glb"]
    )
    with pytest.raises(SystemExit, match=message):
        detail.main()
    assert imports == ["source.glb", "target.glb"]


def test_bake_image_replaces_the_active_node_in_each_material():
    class Nodes(list):
        def new(self, kind):
            assert kind == "ShaderNodeTexImage"
            node = NS(select=False)
            self.append(node)
            return node

    materials = [NS(node_tree=NS(nodes=Nodes([NS(select=True)]))) for _ in range(2)]
    target = NS(data=NS(materials=materials))
    image = NS(colorspace_settings=NS())

    def new(name, *, width, height, alpha):
        assert (name, width, height, alpha) == ("normals", 64, 64, False)
        return image

    bpy = NS(data=NS(images=NS(new=new)))
    baked_image, targets = detail._target_image(bpy, target, "normals", 64)
    assert baked_image is image
    assert image.colorspace_settings.name == "Non-Color"
    assert [original for original, node in targets] == materials
    for material, node in targets:
        assert material.node_tree.nodes.active is node
        assert [item.select for item in material.node_tree.nodes] == [False, True]
        assert node.image is image


def test_bake_detail_rejects_mismatched_world_bounds(monkeypatch):
    import numpy as np

    source = SceneObject("source", "MESH")
    target = SceneObject("target", "MESH", [object()])
    source.matrix_world = target.matrix_world = np.eye(3)
    source.bound_box = [(0, 0, 0), (2, 1, 1)]
    target.bound_box = [(10, 20, 30), (10.5, 22, 33)]
    material = NS(use_nodes=False)
    target.data.materials = [material]
    objects = []

    def gltf(*, filepath):
        objects.append(source if filepath == "source.glb" else target)

    bpy = NS(
        data=NS(objects=objects),
        ops=NS(
            wm=NS(read_factory_settings=lambda *, use_empty: objects.clear()),
            import_scene=NS(gltf=gltf),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Matrix=object, Vector=np.array))
    monkeypatch.setattr(
        sys, "argv", ["blender", "--", "source.glb", "target.glb", "out.glb"]
    )
    with pytest.raises(SystemExit, match="do not line up"):
        detail.main()
    assert material.use_nodes
    np.testing.assert_array_equal(source.matrix_world, np.eye(3))
