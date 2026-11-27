"""Verify that separate region bakes preserve each other's loose textures."""

import bpy


def check_region_outputs(workspace):
    from blender_split_regions_worker import run

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    data = bpy.data.meshes.new("Region fixture")
    vertices, faces = [], []
    for center in (-.6, 0, .5):
        offset = len(vertices)
        vertices.extend([(-.1, center - .1, 0), (.1, center - .1, 0),
                         (.1, center + .1, 0), (-.1, center + .1, 0)])
        faces.append(tuple(range(offset, offset + 4)))
    data.from_pydata(vertices, [], faces)
    uv = data.uv_layers.new(name="source")
    for face in data.polygons:
        for index, point in zip(face.loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1)), strict=True):
            uv.data[index].uv = point
    obj = bpy.data.objects.new("Region fixture", data)
    scene.collection.objects.link(obj)
    material = bpy.data.materials.new("Source color")
    material.use_nodes = True
    image = bpy.data.images.new("Source color", width=4, height=4)
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    material.node_tree.links.new(texture.outputs["Color"],
                                 material.node_tree.nodes.get("Principled BSDF").inputs["Base Color"])
    data.materials.append(material)
    source = workspace / "region-source.glb"
    previous = {}
    for prefix, color in (("first", [1, 0, 0, 1]), ("second", [0, 0, 1, 1])):
        image.pixels[:] = color * 16
        image.pack()
        bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLB", use_active_scene=True)
        result = run({"asset_path": str(source), "out_dir": str(workspace), "size": 16,
                      "head_cut": .25, "tail_cut": -.1, "output_prefix": prefix})
        assert result["error"] is None
        assert result["faces_per_region"] == {"head": 2, "body": 2, "tail": 2}
        for region, name in result["written"].items():
            path = workspace / f"{prefix}_albedo_{region}.png"
            assert str(path) == name
            assert path.is_file()
            if previous:
                assert path.read_bytes() != previous[workspace / f"first_albedo_{region}.png"]
        for path, content in previous.items():
            assert path.read_bytes() == content, "A later bake overwrote an earlier texture"
        if not previous:
            previous = {workspace / f"{prefix}_albedo_{region}.png":
                        (workspace / f"{prefix}_albedo_{region}.png").read_bytes()
                        for region in result["written"]}
    assert len(list(workspace.glob("*_albedo_*.png"))) == 6
    print("PASS two region exports retained all six standalone textures")
    return {"exports": 2, "textures": 6}
