"""Test eye edit state and export cleanup through a small Blender API substitute."""

import sys
from types import SimpleNamespace as Object

import numpy as np
import pytest

from scripts import paint_eyes


class Coordinates(list):
    def foreach_get(self, prop, out):
        out[:] = np.array([getattr(item, prop) for item in self]).ravel()

    def foreach_set(self, prop, values):
        for item, value in zip(self, np.asarray(values).reshape(len(self), -1), strict=True):
            setattr(item, prop, tuple(value) if len(value) > 1 else value.item())


class Layers(list):
    def new(self, *, name, do_init):
        coordinates = [corner.uv if do_init else (0, 0) for corner in self[0].data]
        layer = Object(name=name, data=Coordinates(Object(uv=uv) for uv in coordinates))
        self.append(layer)
        self.active = layer
        return layer

    def get(self, name):
        return next((layer for layer in self if layer.name == name), None)


class Materials(list):
    def pop(self, *, index):
        return super().pop(index)


class SceneObject:
    def __init__(self, *, data=None, parent=None, selected=False):
        self.data = data
        self.parent = parent
        self.selected = selected

    def select_get(self):
        return self.selected

    def select_set(self, value):
        self.selected = value


@pytest.fixture
def eye_scene(monkeypatch):
    atlas = [[.2, .2], [.3, .2], [.2, .3]]
    patch = [[.6, .6], [.7, .6], [.6, .7]]
    base = Object(name="UVMap", data=Coordinates(Object(uv=uv) for uv in patch))
    layers = Layers([base])
    backup = layers.new(name=paint_eyes.BACKUP_UV_LAYER, do_init=False)
    backup.data.foreach_set("uv", np.asarray(atlas).ravel())
    materials = Materials([Object(name="Body"), Object(name="EyePatch")])
    mesh = Object(
        loops=[None] * 3,
        polygons=Coordinates([Object(material_index=1, loop_indices=[0, 1, 2])]),
        uv_layers=layers,
        materials=materials,
        updates=0,
    )

    def update():
        mesh.updates += 1

    mesh.update = update
    arm = SceneObject()
    head = SceneObject(data=mesh, parent=arm)
    unrelated = SceneObject(selected=True)
    scene = Object(
        data=Object(objects={"Head": head}),
        ops=Object(export_scene=Object()),
        context=Object(
            scene=Object(objects=[arm, head, unrelated]),
            view_layer=Object(objects=Object(active=unrelated)),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", scene)
    return scene, head, arm, unrelated, np.array(atlas), np.array(patch)


def test_revert_restores_atlas_and_removes_eye_material(eye_scene):
    _scene, head, _arm, _unrelated, atlas, _patch = eye_scene
    assert paint_eyes.revert_in_blender("Head") == {"reverted": True, "faces": 1}
    mesh = head.data
    assert mesh.updates == 1
    assert mesh.polygons[0].material_index == 0
    assert [material.name for material in mesh.materials] == ["Body"]
    assert mesh.uv_layers.get(paint_eyes.BACKUP_UV_LAYER) is None
    np.testing.assert_allclose([corner.uv for corner in mesh.uv_layers.active.data], atlas)
    assert not paint_eyes.revert_in_blender("Head")["reverted"]


def test_revert_restores_atlas_after_material_was_removed(eye_scene):
    _scene, head, _arm, _unrelated, atlas, _patch = eye_scene
    head.data.materials.pop(index=1)
    head.data.polygons[0].material_index = 0
    assert paint_eyes.revert_in_blender("Head") == {"reverted": True, "faces": 0}
    np.testing.assert_allclose([corner.uv for corner in head.data.uv_layers.active.data], atlas)


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("with_backup", [False, True])
def test_export_restores_selection_and_backup_even_after_failure(eye_scene, tmp_path, fail, with_backup):
    scene, head, arm, unrelated, atlas, _patch = eye_scene
    path = tmp_path / "head.glb"
    layers = head.data.uv_layers
    if not with_backup:
        layers.remove(layers.get(paint_eyes.BACKUP_UV_LAYER))
        layers.active = layers[0]
        head.parent = None

    def gltf(**kwargs):
        assert kwargs == {
            "filepath": str(path), "export_format": "GLB",
            "use_selection": True, "export_materials": "EXPORT",
        }
        assert head.selected and arm.selected is with_backup and not unrelated.selected
        assert scene.context.view_layer.objects.active is head
        assert layers.get(paint_eyes.BACKUP_UV_LAYER) is None
        if fail:
            raise OSError("fixture export failed")
        path.write_bytes(b"fixture")

    scene.ops.export_scene.gltf = gltf
    if fail:
        with pytest.raises(OSError, match="fixture export failed"):
            paint_eyes.export_in_blender("Head", str(path))
    else:
        assert paint_eyes.export_in_blender("Head", str(path)) == {"path": str(path), "bytes": 7}

    assert not head.selected and not arm.selected and unrelated.selected
    assert scene.context.view_layer.objects.active is unrelated
    backup = layers.get(paint_eyes.BACKUP_UV_LAYER)
    if with_backup:
        np.testing.assert_allclose([corner.uv for corner in backup.data], atlas)
        assert layers.active is layers[0]
    else:
        assert backup is None


class Pixels:
    def __init__(self, values):
        self.values = np.asarray(values)

    def foreach_get(self, out):
        out[:] = self.values

    def foreach_set(self, values):
        self.values = np.array(values, copy=True)


class Images(dict):
    def new(self, name, width, height, *, alpha):
        assert not alpha
        image = Object(name=name, size=(width, height), pixels=Pixels([]), colorspace_settings=Object())
        image.pack = lambda: setattr(image, "packed", True)
        self[name] = image
        return image

    def remove(self, image):
        del self[image.name]


class ShaderSockets(dict):
    def __missing__(self, name):
        socket = Object(name=name)
        self[name] = socket
        return socket


class ShaderNodes(list):
    def __init__(self, links):
        self.links = links

    def clear(self):
        super().clear()
        self.links.clear()

    def new(self, kind):
        node = Object(type=kind, inputs=ShaderSockets(), outputs=ShaderSockets())
        self.append(node)
        return node


class ShaderLinks(list):
    def new(self, source, target):
        self.append((source.name, target.name))


class ShaderMaterials(dict):
    def new(self, name):
        links = ShaderLinks()
        material = Object(name=name, node_tree=Object(nodes=ShaderNodes(links), links=links))
        self[name] = material
        return material


@pytest.mark.parametrize("packed_surface", [False, True])
def test_apply_bakes_and_reapplies_from_original_uvs(monkeypatch, packed_surface):
    left, right = np.array([[-2, 0, 0], [2, 0, 0]])
    relative = np.array([[-.7, 0, -.7], [.7, 0, -.7], [0, 0, .7]])
    positions = [(left + relative), (right + relative), (left + relative[::-1])]
    uvs = [(.1, .1), (.9, .1), (.5, .9)] * 3
    layer = Object(name="UVMap", data=Coordinates(Object(uv=uv) for uv in uvs))
    layers = Layers([layer])
    layers.active = layer
    material_library = ShaderMaterials()
    base = material_library.new("Body")
    base_image = Object(size=(2, 2), pixels=Pixels([.05, .05, .05, 1] * 4))

    def source_link(image, source_type, destination):
        return Object(
            from_node=Object(type=source_type, image=image),
            to_socket=Object(name=destination),
            to_node=Object(name=destination),
        )

    base.node_tree.links.append(source_link(base_image, "VALUE", "Base Color"))
    base.node_tree.links.append(source_link(base_image, "TEX_IMAGE", "Base Color"))
    if packed_surface:
        surface = Object(size=(2, 2), pixels=Pixels([.25, .65, .75, 1] * 4))
        base.node_tree.links.append(source_link(surface, "TEX_IMAGE", "Separate Color"))
    else:
        principled = base.node_tree.nodes.new("BSDF_PRINCIPLED")
        principled.inputs["Roughness"].default_value = .65
        principled.inputs["Metallic"].default_value = .75

    polygons = Coordinates(
        Object(loop_indices=list(range(i, i + 3)), material_index=0) for i in range(0, 9, 3)
    )
    mesh = Object(
        polygons=polygons,
        uv_layers=layers,
        materials=Materials([base]),
        loops=Coordinates(Object(vertex_index=i) for i in range(9)),
        vertices=Coordinates(Object(co=pos) for pos in np.vstack(positions)),
        updated=0,
    )

    def update():
        mesh.updated += 1

    mesh.update = update
    model = Object(data=mesh, matrix_world=np.eye(4))
    bpy = Object(data=Object(objects={"Head": model}, images=Images(), materials=material_library))
    monkeypatch.setitem(sys.modules, "bpy", bpy)

    kwargs = {"extent": 1, "patch": 24, "splay": 0}
    first = paint_eyes.apply_in_blender("Head", [left, right], **kwargs)
    assert first["material_slot"] == 1
    assert first["patch"] == [48, 24]
    assert [poly.material_index for poly in polygons] == [1, 1, 0]
    np.testing.assert_allclose([corner.uv for corner in layer.data][-3:], uvs[-3:])
    backup = layers.get(paint_eyes.BACKUP_UV_LAYER)
    np.testing.assert_allclose([corner.uv for corner in backup.data], uvs)
    albedo = bpy.data.images["EyePatch_albedo"]
    surface = bpy.data.images["EyePatch_orm"]
    assert albedo.packed and albedo.colorspace_settings.name == "sRGB"
    assert surface.packed and surface.colorspace_settings.name == "Non-Color"
    pixels = surface.pixels.values.reshape(24, 48, 4)
    assert pixels[12, 12, 2] == pytest.approx(0)
    assert pixels[0, 0, 2] == pytest.approx(.75)
    assert pixels[0, 0, 0] == pytest.approx(.25 if packed_surface else 0)

    expected_links = [
        ("Color", "Base Color"), ("Color", "Color"), ("Green", "Roughness"),
        ("Blue", "Metallic"), ("BSDF", "Surface"),
    ]
    assert list(mesh.materials[1].node_tree.links) == expected_links
    second = paint_eyes.apply_in_blender("Head", [left, right], **kwargs)
    assert second == first
    assert list(mesh.materials[1].node_tree.links) == expected_links
    assert len(layers) == len(mesh.materials) == mesh.updated == 2
    assert layers.get(paint_eyes.BACKUP_UV_LAYER) is backup
    np.testing.assert_allclose(bpy.data.images["EyePatch_albedo"].pixels.values, albedo.pixels.values)


def test_eye_adapter_rejects_nontriangular_mesh_before_writing_backup(eye_scene):
    _scene, head, _arm, _unrelated, _atlas, _patch = eye_scene
    head.data.polygons[0].loop_indices = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="triangulated mesh"):
        paint_eyes.apply_in_blender("Head", [(-1, 0, 0), (1, 0, 0)])
    assert head.data.updates == 0


def test_eye_material_rejects_missing_base_colour_and_uses_default_surface():
    base = Object(
        name="No atlas",
        node_tree=Object(nodes=[Object(type="OUTPUT_MATERIAL")], links=[]),
    )
    obj = Object(data=Object(materials=[base]))
    with pytest.raises(ValueError, match="no image texture feeds 'Base Color'"):
        paint_eyes._named_image(obj, "Base Color")
    assert paint_eyes._constant_surface(obj) == (.8, 0)


def test_eye_rasterizer_skips_offscreen_degenerate_and_subpixel_triangles():
    triangles = np.array([
        [[2, 2], [3, 2], [2, 3]],  # off-screen
        [[.5, .5], [.6, .6], [.7, .7]],  # line
        [[.45, .45], [.55, .45], [.45, .55]],  # no centres of the four texels
    ])
    atlas, coverage = paint_eyes.rasterise(triangles, np.ones((3, 3, 1)), 2, 2)
    assert not coverage.any()
    assert not atlas.any()
