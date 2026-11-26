"""Drive the bake entry points using a small scene and real pixel buffers."""

import copy
import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import blender_bake_detail as detail
import blender_retopo_bake as retopo
import numpy as np
import pytest

from image_to_3dlab.glb_turn import build_glb, parse_glb
from tests.test_blender_bake_detail import _glb


class Transform(np.ndarray):
    def __new__(cls, rows):
        return np.asarray(rows, dtype=float).view(cls)

    @staticmethod
    def Translation(offset):
        matrix = Transform(np.eye(4))
        matrix[:3, 3] = offset
        return matrix

    @staticmethod
    def Scale(scale, size):
        matrix = Transform(np.eye(size))
        matrix[:3, :3] *= scale
        return matrix

    def __matmul__(self, value):
        if value.shape == (3,):
            return (np.asarray(self) @ np.r_[value, 1])[:3]
        return super().__matmul__(value)


class Socket:
    def __init__(self, name, value=0):
        self.name = name
        self.links = []
        self.default_value = value

    @property
    def is_linked(self):
        return bool(self.links)


NODE_PORTS = {
    "ShaderNodeOutputMaterial": ("OUTPUT_MATERIAL", ["Surface"], []),
    "ShaderNodeBsdfPrincipled": ("BSDF_PRINCIPLED", ["Base Color", "Normal", "Roughness", "Metallic", "IOR"], ["BSDF"]),
    "ShaderNodeTexImage": ("TEX_IMAGE", [], ["Color"]),
    "ShaderNodeEmission": ("EMISSION", ["Color"], ["Emission"]),
    "ShaderNodeNormalMap": ("NORMAL_MAP", ["Color"], ["Normal"]),
    "ShaderNodeSeparateColor": ("SEPARATE_COLOR", ["Color"], ["Red", "Green", "Blue"]),
    "ShaderNodeVertexColor": ("VERTEX_COLOR", [], ["Color"]),
}


class Nodes(list):
    active = None

    def new(self, kind):
        name, inputs, outputs = NODE_PORTS[kind]
        node = NS(type=name, inputs={k: Socket(k) for k in inputs}, outputs={k: Socket(k) for k in outputs},
                  select=False, image=None)
        for socket in node.outputs.values():
            socket.node = node
        self.append(node)
        return node


class Material:
    def __init__(self, name, *, shader=True, ior=True):
        self.name = name
        self.use_nodes = True
        nodes = Nodes()
        nodes.new("ShaderNodeOutputMaterial")
        if shader:
            nodes.new("ShaderNodeBsdfPrincipled")
        if not ior:
            original = nodes.new

            def new(kind):
                node = original(kind)
                node.inputs.pop("IOR", None)
                return node

            nodes.new = new
        self.node_tree = NS(nodes=nodes, links=NS(new=self.link))

    @property
    def bsdf(self):
        return next(node for node in self.node_tree.nodes if node.type == "BSDF_PRINCIPLED")

    @staticmethod
    def link(output, destination):
        destination.links[:] = [NS(from_node=output.node, from_socket=output)]


class Pixels:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32).ravel()

    def foreach_get(self, buffer):
        buffer[:] = self.values

    def foreach_set(self, values):
        self.values[:] = values

    def __len__(self):
        return self.values.size


class MeshObject:
    def __init__(self, name, *, materials=(), dimensions=(1, 1, 1)):
        self.name, self.dimensions = name, dimensions
        self.type = "MESH"
        self.selected = False
        self.matrix_world = Transform(np.eye(4))
        self.bound_box = [(0, 0, 0), dimensions]
        self.data = NS(materials=list(materials), polygons=list(range(1000)), uv_layers=["uv"])
        self.data.copy = lambda: NS(materials=list(self.data.materials), polygons=self.data.polygons.copy())
        self.material_slots = [NS(material=m) for m in materials]

    def select_set(self, selected):
        self.selected = selected

    def copy(self):
        return copy.copy(self)


class Objects(list):
    def remove(self, obj, *, do_unlink=False):
        super().remove(obj)


class BakeScene:
    def __init__(self):
        self.objects = Objects()
        self.imports = {}
        self.images = {}
        self.active = NS(active=None)
        self.render = NS(bake=NS())
        self.scene = NS(render=self.render, cycles=NS(), collection=NS(objects=NS(link=self.objects.append)))
        self.ops = NS(
            object=NS(select_all=self.deselect, bake=self.bake),
            wm=NS(read_factory_settings=self.reset),
            import_scene=NS(gltf=self.import_meshes),
            export_scene=NS(gltf=self.export),
        )
        self.bpy = NS(
            data=NS(objects=self.objects, materials=NS(new=Material),
                    images=NS(new=self.new_image, load=lambda path: self.images[path])),
            context=NS(scene=self.scene, view_layer=NS(objects=self.active, update=lambda: None)),
            ops=self.ops,
        )
        self.bakes = []
        self.exports = []
        self.export_data = build_glb({"asset": {"version": "2.0"}}, b"")

    def reset(self, *, use_empty):
        assert use_empty
        self.objects.clear()

    def import_meshes(self, *, filepath):
        self.objects.extend(self.imports[filepath])

    def new_image(self, name, *, width, height, alpha):
        assert not alpha
        image = NS(name=name, size=(width, height), colorspace_settings=NS(), packed=False,
                   pixels=Pixels(np.zeros((height, width, 4))))
        image.pack = lambda: setattr(image, "packed", True)
        self.images[name] = image
        return image

    def deselect(self, *, action):
        assert action == "DESELECT"
        for obj in self.objects:
            obj.select_set(False)

    def bake(self, *, type):
        target = self.active.active
        targets = [m.node_tree.nodes.active for m in target.data.materials]
        assert target.selected and all(node.select for node in targets)
        if hasattr(target, "active_material_index"):
            assert target.active_material_index == 0
        names = tuple(obj.name for obj in self.objects if obj.selected)
        self.bakes.append((type, names, target.name, [n.image.name for n in targets]))
        for image in {node.image.name: node.image for node in targets}.values():
            pixel = (128 / 255, 128 / 255, 0, 1) if type == "NORMAL" else (0.2, 0.6, 0.4, 1)
            image.pixels.values[:] = np.tile(pixel, image.size[0] * image.size[1])

    def export(self, *, filepath, export_format, use_selection, **opts):
        assert export_format == "GLB" and use_selection
        self.exports.append((Path(filepath), [obj.name for obj in self.objects if obj.selected], opts))
        Path(filepath).write_bytes(self.export_data)


def textured_material(name, texture, channel):
    mat = Material(name)
    image = mat.node_tree.nodes.new("ShaderNodeTexImage")
    image.image = texture
    mat.link(image.outputs["Color"], mat.bsdf.inputs[channel])
    return mat


@pytest.mark.parametrize("maps", ["transfer", "preserve", "no-map"])
def test_detail_bake_aligns_meshes_and_commits_the_requested_maps(tmp_path, monkeypatch, capsys, maps):
    scene = BakeScene()
    metallic = NS(name="source roughness")
    source_mat = textured_material("source", metallic, "Metallic") if maps != "no-map" else Material("source")
    flat_mat = Material("plain")
    flat_mat.bsdf.inputs["Metallic"].default_value = 0.2
    flat_mat.bsdf.inputs["Roughness"].default_value = 0.7
    source = MeshObject("source", materials=[source_mat, flat_mat, None, Material("node-less", shader=False)])
    target_mat = (textured_material("repaint", NS(name="repaint roughness"), "Roughness")
                  if maps == "preserve" else Material("target"))
    target = MeshObject("target", materials=[] if maps == "no-map" else [target_mat, Material("no shader", shader=False)],
                        dimensions=(2, 2, 2))
    target.matrix_world = Transform.Translation((10, -5, 2))
    scene.imports = {"high.glb": [source], "low.glb": [target]}
    scene.export_data = _glb([(0, 0, 1)], [(0, 0, 0, 0)])
    out = tmp_path / "result.glb"
    args = ["blender", "--", "high.glb", "low.glb", str(out), str(detail.MIN_SIZE)]
    if maps == "transfer":
        size = detail.MIN_SIZE
        mask = scene.new_image("fade.png", width=size, height=size, alpha=False)
        mask.pixels.values[: (size * size // 2) * 4] = 1
        args.append("fade.png")
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setitem(sys.modules, "bpy", scene.bpy)
    monkeypatch.setitem(sys.modules, "mathutils", NS(Matrix=Transform, Vector=np.array))
    with pytest.raises(SystemExit) as status:
        runpy.run_path(detail.__file__, run_name="__main__")
    assert status.value.code == 0
    report = json.loads(capsys.readouterr().out.split("BAKE_DETAIL::")[1])
    assert report["scale"] == 2 and report["mismatch"] == 0
    assert report["zero_tangents_repaired"] == 1
    assert report["normal_flipped_fraction"] == 1
    assert report["normal_faded_share"] == (0.5 if maps == "transfer" else 0.0)
    assert report["metallic_roughness"] == {
        "transfer": "transferred", "preserve": "kept target's own", "no-map": "none on source",
    }[maps]
    np.testing.assert_array_equal(source.matrix_world @ np.array([1, 1, 1]), [12, -3, 4])
    assert scene.render.engine == "CYCLES"
    assert scene.scene.cycles.device == "CPU"
    assert scene.render.bake.cage_extrusion == pytest.approx(0.04)
    assert scene.render.bake.max_ray_distance == pytest.approx(0.08)
    assert scene.render.bake.normal_space == "TANGENT"
    assert scene.bakes[0][:3] == ("NORMAL", ("source", "target"), "target")
    assert [event[0] for event in scene.bakes] == (["NORMAL", "EMIT"] if maps == "transfer" else ["NORMAL"])
    normal = scene.images["detail_normal"]
    assert normal.packed and normal.colorspace_settings.name == "Non-Color"
    assert normal.pixels.values[2::4].min() > 0.99
    surface = target.data.materials[0].bsdf
    assert surface.inputs["Normal"].links[0].from_node.space == "TANGENT"
    if maps == "transfer":
        assert scene.images["detail_mr"].packed
        expected = [(0, 0.7, 0.2, 1), (0, 0.5, 0.0, 1), (0, 0.5, 0.0, 1)]
        for slot, values in zip(source.material_slots[1:], expected, strict=True):
            emission = next(node for node in slot.material.node_tree.nodes if node.type == "EMISSION")
            assert emission.inputs["Color"].default_value == values
        assert surface.inputs["Roughness"].links[0].from_socket.name == "Green"
        assert surface.inputs["Metallic"].links[0].from_socket.name == "Blue"
    elif maps == "preserve":
        assert surface.inputs["Roughness"].links[0].from_node.image.name == "repaint roughness"
    assert scene.objects == [target]
    assert scene.exports == [(detail.partial_path(out), ["target"], {"export_tangents": True})]
    assert not detail.partial_path(out).exists()
    _, binary = parse_glb(out.read_bytes())
    assert np.frombuffer(binary, dtype="<f4")[3:] == pytest.approx([0, 1, 0, 1])


@pytest.mark.parametrize("legacy", [False, True])
def test_retopo_bakes_only_the_original_albedo_onto_an_independent_mesh(tmp_path, monkeypatch, capsys, legacy):
    scene = BakeScene()
    source_color = NS(size=(64, 64))
    original_material = textured_material("original", source_color, "Base Color")
    original = MeshObject("original", materials=[original_material], dimensions=(3, 1, 2))
    unrelated = MeshObject("unrelated")
    scene.objects.append(unrelated)
    scene.imports["input.glb"] = [original]
    operations = []
    scene.ops.object.mode_set = lambda *, mode: operations.append(mode)
    scene.ops.mesh = NS(
        select_all=lambda *, action: None,
        remove_doubles=lambda *, threshold: operations.append(("weld", threshold)),
        normals_make_consistent=lambda *, inside: operations.append(("normals", inside)),
    )

    def quadriflow(*, target_faces, use_preserve_boundary, use_mesh_symmetry):
        baked = scene.active.active
        assert baked is not original and baked.data is not original.data
        assert original.data.materials == [original_material] and not baked.data.materials
        operations.append(("reduce", target_faces))
        baked.data.polygons = baked.data.polygons[:target_faces]

    scene.ops.object.quadriflow_remesh = quadriflow
    scene.ops.uv = NS(
        smart_project=lambda **kw: operations.append(("project", kw["angle_limit"])),
        seams_from_islands=lambda: operations.append("seams"),
        unwrap=lambda **kw: operations.append(("unwrap", kw["method"])),
        select_overlap=lambda: operations.append("overlap"),
    )
    scene.scene.tool_settings = NS(use_uv_select_sync=False, mesh_select_mode=(True, False, False))
    if legacy:
        del scene.scene.cycles
        scene.bpy.data.materials.new = lambda name: Material(name, ior=False)
        original_material = Material("unlinked")
        original_material.node_tree.nodes.new("ShaderNodeTexImage").image = source_color
        mixer = original_material.node_tree.nodes.new("ShaderNodeEmission")
        original_material.link(mixer.outputs["Emission"], original_material.bsdf.inputs["Base Color"])
        original.data.materials = [original_material]
    version = (5, 2, 0) if legacy else (5, 2, 1)
    scene.bpy.app = NS(version=version, version_string=".".join(map(str, version)))
    monkeypatch.setitem(sys.modules, "bpy", scene.bpy)
    monkeypatch.setitem(sys.modules, "bmesh", NS(from_edit_mesh=lambda data: NS(faces=[])))
    out = tmp_path / "retopo.glb"
    monkeypatch.setattr(sys, "argv", [
        "blender", "--", "input.glb", str(out), "500", "256", "80", "0", "0.1", "0.7", "1.3",
    ])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(retopo.__file__, run_name="__main__")
    assert status.value.code == 0
    [finished] = scene.objects
    assert finished.name == "RETOPO"
    assert len(finished.data.polygons) == 500
    assert ("weld", 1e-6) in operations and ("reduce", 500) in operations
    assert ("project", math.radians(80)) in operations
    assert ("overlap" in operations) is not legacy
    assert ("checking UV overlap" in capsys.readouterr().out) is legacy
    assert scene.scene.tool_settings.mesh_select_mode == (True, False, False)
    assert not scene.scene.tool_settings.use_uv_select_sync
    assert scene.render.engine == "CYCLES"
    assert scene.render.bake.cage_extrusion == pytest.approx(0.015)
    assert scene.render.bake.max_ray_distance == pytest.approx(0.030)
    emit, = original.data.materials
    tex = next(node for node in emit.node_tree.nodes if node.type == "TEX_IMAGE")
    assert tex.image is source_color and tex.interpolation == "Closest"
    dst, = finished.data.materials
    material_output = next(node for node in dst.node_tree.nodes if node.type == "OUTPUT_MATERIAL")
    shader = material_output.inputs["Surface"].links[0].from_node
    target = shader.inputs["Base Color"].links[0].from_node.image
    assert target is scene.images["retopo_albedo"]
    np.testing.assert_allclose(target.pixels.values[:4], [0.2, 0.6, 0.4, 1.0])
    assert [shader.inputs[k].default_value for k in ("Metallic", "Roughness")] == [0.1, 0.7]
    if legacy:
        assert "IOR" not in shader.inputs
    else:
        assert shader.inputs["IOR"].default_value == 1.3
    assert scene.bakes == [("EMIT", ("original", "RETOPO"), "RETOPO", ["retopo_albedo"])]
    assert scene.exports == [(out, ["RETOPO"], {})]
    assert out.read_bytes() == scene.export_data


@pytest.mark.parametrize("mesh", [None, MeshObject("no-material"), MeshObject("untextured", materials=[Material("empty")])])
def test_retopo_refuses_if_source_albedo_cannot_be_baked(tmp_path, monkeypatch, mesh):
    scene = BakeScene()
    scene.imports["input.glb"] = [] if mesh is None else [mesh]
    monkeypatch.setitem(sys.modules, "bpy", scene.bpy)
    out = tmp_path / "retopo.glb"
    monkeypatch.setattr(sys, "argv", ["blender", "--", "input.glb", str(out)])
    message = "no mesh found" if mesh is None else "base colour texture"
    with pytest.raises(SystemExit, match=message):
        retopo.main()
    assert not scene.bakes and not scene.exports
    assert not out.exists()


def test_tangent_repair_leaves_primitives_without_normal_channels_alone():
    document, binary = parse_glb(_glb([(0, 0, 1)], [(1, 0, 0, 1)]))
    primitive = document["meshes"][0]["primitives"][0]
    del primitive["attributes"]["NORMAL"]
    source = build_glb(document, binary)
    repaired, count = detail.repair_zero_tangents(source)
    assert repaired == source and count == 0


def test_tangent_repair_refuses_invalid_component_type():
    document, binary = parse_glb(_glb([(0, 0, 1)], [(0, 0, 0, 0)]))
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    document["accessors"][attributes["TANGENT"]]["componentType"] = 5123
    with pytest.raises(ValueError, match="expected float normals and tangents"):
        detail.repair_zero_tangents(build_glb(document, binary))


@pytest.mark.parametrize(("argv", "message"), [
    (["blender", "input.glb", "output.glb"], "usage:"),
    (["blender", "--", "input.glb", "output.glb", "500", "256", "90"], "angle limit"),
])
def test_retopo_rejects_missing_separator_and_unsupported_angle(argv, message):
    with pytest.raises(SystemExit, match=message):
        retopo.parse_args(argv)
