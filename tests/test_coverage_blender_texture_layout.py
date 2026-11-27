"""Verify the texture bake reads old UVs and publishes the new UV layout."""

import sys
from types import SimpleNamespace as NS

import blender_reunwrap_bake as reunwrap
import pytest


class Node:
    def __init__(self, kind, inputs=(), outputs=(), image=None):
        self.type = kind
        self.select = True
        self.image = image
        self.inputs = {name: NS(links=[], node=self, name=name) for name in inputs}
        self.outputs = {name: NS(links=[], node=self, name=name) for name in outputs}


class Nodes(list):
    def new(self, kind):
        node = {
            "ShaderNodeUVMap": Node("UVMAP", outputs=["UV"]),
            "ShaderNodeTexImage": Node(
                "TEX_IMAGE", inputs=["Vector"], outputs=["Color"]
            ),
            "ShaderNodeEmission": Node(
                "EMISSION", inputs=["Color"], outputs=["Emission"]
            ),
            "ShaderNodeBsdfPrincipled": Node(
                "BSDF_PRINCIPLED", inputs=["Base Color"], outputs=["BSDF"]
            ),
        }[kind]
        self.append(node)
        return node


class UVLayers(list):
    def new(self, *, name):
        layer = NS(name=name)
        self.append(layer)
        return layer

    def get(self, name):
        return next((layer for layer in self if layer.name == name), None)


def link(source, target):
    target.is_linked = True
    target.links[:] = [NS(from_node=source.node, to_node=target.node)]


@pytest.mark.parametrize("existing_shader", [False, True])
def test_reunwrap_reads_original_material_but_exports_new_texture(
    monkeypatch, existing_shader
):
    original_image = NS(size=(64, 64))
    baked_image = NS(size=(1024, 1024))
    original_tex = Node("TEX_IMAGE", outputs=["Color"], image=original_image)
    output = Node("OUTPUT_MATERIAL", inputs=["Surface"])
    nodes = Nodes([original_tex, output])
    if existing_shader:
        existing = nodes.new("ShaderNodeBsdfPrincipled")
        link(original_tex.outputs["Color"], existing.inputs["Base Color"])
    layers = UVLayers()
    layers.active = layers.new(name="original")
    material = NS(use_nodes=True, node_tree=NS(nodes=nodes, links=NS(new=link)))
    obj = NS(
        type="MESH",
        data=NS(materials=[material], uv_layers=layers),
        select_set=lambda _: None,
    )
    scene = NS(render=NS(bake=NS()))
    if existing_shader:
        scene.cycles = NS()
    class SceneObjects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)
            events.append("clear")

    scene_objects = SceneObjects([NS(type="EMPTY")])
    active = NS(active=None)
    events = []

    def gltf(*, filepath):
        assert filepath == "input.glb"
        scene_objects.append(obj)

    def smart_project(*, angle_limit, island_margin):
        assert active.active is obj
        assert layers.active.name == "reuv"
        assert angle_limit == pytest.approx(1.0471975512)
        assert island_margin == 0.0002
        events.append("project")

    def new_image(name, *, width, height, alpha):
        assert (name, width, height, alpha) == ("reuv_albedo", 1024, 1024, False)
        return baked_image

    def bake(*, type):
        assert type == "EMIT"
        assert nodes.active.image is baked_image and nodes.active.select
        assert all(not node.select for node in nodes if node is not nodes.active)
        assert layers.active.name == "reuv"
        uv = next(node for node in nodes if node.type == "UVMAP")
        assert uv.uv_map == "original"
        source = next(
            node
            for node in nodes
            if node is not original_tex and node.image is original_image
        )
        assert source.inputs["Vector"].links[0].from_node is uv
        assert output.inputs["Surface"].links[0].from_node.type == "EMISSION"
        if existing_shader:
            assert scene.cycles.device == "CPU" and scene.cycles.samples == 1
        events.append("bake")
        if existing_shader:
            layers.remove(layers[0])

    def export(*, filepath, export_format):
        assert (filepath, export_format) == ("out.glb", "GLB")
        shader = output.inputs["Surface"].links[0].from_node
        assert shader.type == "BSDF_PRINCIPLED"
        assert shader.inputs["Base Color"].links[0].from_node.image is baked_image
        assert not any(node.type in {"UVMAP", "EMISSION"} for node in nodes)
        if existing_shader:
            assert shader is existing
        assert [layer.name for layer in layers] == ["reuv"]
        events.append("export")

    bpy = NS(
        data=NS(objects=scene_objects, images=NS(new=new_image)),
        context=NS(scene=scene, view_layer=NS(objects=active)),
        ops=NS(
            import_scene=NS(gltf=gltf),
            object=NS(mode_set=lambda *, mode: events.append(mode), bake=bake),
            mesh=NS(select_all=lambda *, action: events.append(action)),
            uv=NS(smart_project=smart_project),
            export_scene=NS(gltf=export),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", ["blender", "--", "input.glb", "out.glb", "1024", "60"])
    assert reunwrap.main() == 0
    assert events == ["clear", "EDIT", "SELECT", "project", "OBJECT", "bake", "export"]


@pytest.mark.parametrize("mesh_present", [True, False])
def test_reunwrap_refuses_import_without_textured_geometry(monkeypatch, mesh_present):
    import runpy

    mesh = NS(type="MESH", data=NS(materials=[]), select_set=lambda _: None)
    objects = []

    def load(*, filepath):
        assert filepath == "input.glb"
        objects.append(mesh if mesh_present else NS(type="EMPTY"))

    bpy = NS(data=NS(objects=objects), ops=NS(import_scene=NS(gltf=load)),
             context=NS(view_layer=NS(objects=NS(active=None))))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", ["blender", "--", "input.glb", "out.glb"])
    message = "base colour texture" if mesh_present else "no mesh found"
    with pytest.raises(SystemExit, match=message):
        runpy.run_path(reunwrap.__file__, run_name="__main__")


def test_reunwrap_ignores_nonimage_material_links():
    procedural = Node("TEX_NOISE", outputs=["Color"])
    shader = Node("BSDF_PRINCIPLED", inputs=["Base Color"])
    link(procedural.outputs["Color"], shader.inputs["Base Color"])
    fallback = Node("TEX_IMAGE", image=object())
    material = NS(use_nodes=True, node_tree=NS(nodes=[shader, procedural, fallback]))
    assert reunwrap.base_colour_image(material) is fallback.image
