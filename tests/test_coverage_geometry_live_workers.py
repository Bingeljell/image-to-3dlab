"""Exercise the installed live mesh workers on tiny local mesh fixtures."""

import json
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace as Obj

import numpy as np
import pytest


def test_solidify_keeps_largest_component_if_threshold_would_remove_all():
    from scripts import blender_solidify_bake_worker as worker

    neighbours = [[1], [0, 2], [1], [4], [3], []]
    assert worker.component_filter(len(neighbours), neighbours.__getitem__, 8) == ([3, 4, 5], 1, 2)
    assert worker.component_filter(len(neighbours), neighbours.__getitem__, 2) == ([5], 2, 1)
    assert worker.component_filter(len(neighbours), neighbours.__getitem__, 1) == ([], 3, 0)


def test_region_worker_uses_real_polygon_centroids():
    from types import SimpleNamespace

    import numpy as np

    from scripts import blender_split_regions_worker as worker

    vertices = [SimpleNamespace(co=np.array([0, y, 0.0])) for y in (-.6, -.4, 0, .2, .4)]
    polys = [SimpleNamespace(vertices=indices) for indices in ([0, 1, 2], [1, 2, 3], [2, 3, 4])]
    mesh = SimpleNamespace(polygons=polys, vertices=vertices)
    counts = worker.assign_regions(mesh, .25, -.1, lambda: np.zeros(3))
    assert counts == {"head": 1, "body": 1, "tail": 1}
    assert [face.material_index for face in polys] == [0, 1, 2]


class Vector(np.ndarray):
    def __new__(cls, value=(0, 0, 0)):
        return np.asarray(value, dtype=float).view(cls)

    x = property(lambda self: self[0])
    y = property(lambda self: self[1])
    z = property(lambda self: self[2])


class Items(list):
    def get(self, name):
        return next((item for item in self if item.name == name), None)

    def __getitem__(self, key):
        return self.get(key) if isinstance(key, str) else super().__getitem__(key)


class Sockets(dict):
    def __missing__(self, name):
        self[name] = Obj(name=name)
        return self[name]


class Nodes(Items):
    def new(self, kind):
        types = {
            "ShaderNodeOutputMaterial": "OUTPUT_MATERIAL",
            "ShaderNodeBsdfPrincipled": "BSDF_PRINCIPLED",
            "ShaderNodeTexImage": "TEX_IMAGE",
        }
        node = Obj(name=kind, type=types.get(kind, kind), inputs=Sockets(), outputs=Sockets())
        self.append(node)
        return node


class Links(list):
    def new(self, source, target):
        self.append((source.name, target.name))


class SceneLinks(list):
    link = list.append
    unlink = list.remove


class MeshObject:
    users = 0

    def __init__(self, name, data=None):
        self.name = name
        self.type = "MESH" if data else "EMPTY"
        self.data = data
        self.selected = False

    def select_set(self, selected):
        self.selected = selected

    def copy(self):
        result = MeshObject(self.name, self.data)
        result.modifiers = self.modifiers
        return result


@pytest.fixture
def private_scene():
    class CpuOnly:
        @property
        def device(self):
            return "CPU"

        @device.setter
        def device(self, value):
            raise TypeError("GPU unsupported")

    class SceneList(dict):
        def new(self, name):
            render_scene = new_scene(name)
            self[name] = render_scene
            return render_scene

    def new_scene(name):
        scene = Obj(name=name, world=None, objects=SceneLinks(), cycles=Obj(), render=Obj(bake=Obj()))
        scene.collection = Obj(objects=scene.objects, children=SceneLinks())
        return scene

    def build(name, *, existing=True, fail=False, shader=True):
        window = Obj(scene=Obj(name="User", objects=[MeshObject("keep me")]))
        user_scene = window.scene
        scenes = SceneList()
        if existing:
            scene = scenes.new(name)
            scene.objects.append(MeshObject("stale"))
            scene.collection.children.append(
                Obj(name="stale collection", users=0, objects=SceneLinks(), children=SceneLinks())
            )
            scene.world = Obj(name="existing world")
        if not shader:
            scenes.new(name).cycles = CpuOnly()
        events = []
        view_layer = Obj(objects=Obj(active=None), update=lambda: None)
        context = Obj(window=window, view_layer=view_layer)
        class Context:
            def __getattr__(self, name):
                return window.scene if name == "scene" else getattr(context, name)

        def remove(obj, *, do_unlink):
            if do_unlink:
                window.scene.objects.remove(obj)
            else:
                assert obj not in window.scene.objects
            events.append(("remove", obj.name))

        def material(name):
            nodes = Nodes()
            if shader:
                nodes.new("ShaderNodeBsdfPrincipled").name = "Principled BSDF"
            nodes.new("ShaderNodeOutputMaterial")
            return Obj(name=name, node_tree=Obj(nodes=nodes, links=Links()))

        def image(name, width, height, alpha):
            assert not alpha
            img = Obj(name=name, size=(width, height), packed=False)
            img.pack = lambda: setattr(img, "packed", True)
            img.save = lambda: Path(img.filepath_raw).write_bytes(b"png")
            return img

        def bake(*, type):
            events.append(("bake", type, view_layer.objects.active.name))
            if fail:
                raise RuntimeError("fixture bake failed")

        def export(**kwargs):
            assert kwargs["export_format"] == "GLB" and kwargs["use_selection"]
            Path(kwargs["filepath"]).write_bytes(b"glb")
            events.append(("export", Path(kwargs["filepath"]).name))

        def collection_remove(child, *, do_unlink):
            assert not do_unlink and child not in window.scene.collection.children
            events.append(("collection", child.name))

        def deselect(**kwargs):
            assert kwargs == {"action": "DESELECT"}
            for obj in window.scene.objects:
                obj.select_set(False)

        bpy = Obj(
            context=Context(),
            data=Obj(
                scenes=scenes, worlds=Obj(new=lambda name: Obj(name=name)),
                user_map=lambda subset: {},
                collections=Obj(remove=collection_remove),
                images=Obj(new=image), materials=Obj(new=material),
            ),
            ops=Obj(
                import_scene=Obj(),
                export_scene=Obj(gltf=export),
                mesh=Obj(select_all=lambda **kw: None, select_mode=lambda **kw: None),
                uv=Obj(
                    smart_project=lambda **kw: events.append(("unwrap", kw)),
                    select_all=lambda **kw: None,
                    pack_islands=lambda **kw: events.append(("pack", kw)),
                ),
                object=Obj(
                    mode_set=lambda **kw: None, select_all=deselect,
                    modifier_apply=lambda **kw: events.append(("apply", kw)),
                    bake=bake, join=lambda: events.append(("join",)),
                ),
            ),
        )

        class ObjectLibrary:
            def __iter__(self):
                return iter([*user_scene.objects, *window.scene.objects])

        bpy.data.objects = ObjectLibrary()
        bpy.data.objects.remove = remove
        return bpy, user_scene, events

    return build


class Layers(Items):
    def new(self, *, name):
        layer = Obj(name=name, data=[Obj(uv=None) for _ in self.active.data])
        self.append(layer)
        return layer


@pytest.mark.parametrize(
    "scenario,prefix", [("success", None), ("success", "alternate"), ("bake error", None), ("source missing", None)]
)
def test_split_worker_packs_and_bakes_only_in_private_scene(private_scene, tmp_path, monkeypatch, capsys, scenario, prefix):
    from scripts import blender_split_regions_worker as worker

    bpy, user, events = private_scene("IMG3D regions", fail=scenario == "bake error",
                                      existing=scenario != "source missing")
    source_image = Obj(name="source image")
    source_nodes = [Obj(type="OUTPUT_MATERIAL")]
    if scenario != "source missing":
        source_nodes.append(Obj(type="TEX_IMAGE", image=source_image))
    material = Obj(node_tree=Obj(nodes=source_nodes))
    uv = Obj(name="original UV", data=[Obj(uv=(.1, .2)), Obj(uv=(.3, .4))])
    layers = Layers([uv])
    layers.active = uv
    mesh = Obj(
        uv_layers=layers, materials=[material], loops=[0, 1],
        vertices=[Obj(co=Vector((0, y, 0))) for y in (-.6, -.4, 0, .2, .4)],
        polygons=[Obj(vertices=indices) for indices in ([0, 1, 2], [1, 2, 3], [2, 3, 4])],
    )
    actual = MeshObject("Asset", mesh)
    marker = MeshObject("JOINT_test", Obj())
    armature = MeshObject("Armature")
    armature.type = "ARMATURE"

    def load(*, filepath):
        assert filepath == "input.glb"
        bpy.context.scene.objects.extend([marker, armature, actual])

    bpy.ops.import_scene.gltf = load
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", Obj(Vector=Vector))
    args = {"asset_path": "input.glb", "out_dir": str(tmp_path), "size": 8, "head_cut": .25, "tail_cut": -.1}
    expected_prefix = "moss_fox" if prefix is None else prefix
    args["output_prefix"] = expected_prefix
    if scenario == "source missing":
        with pytest.raises(RuntimeError, match="source material has no image texture"):
            worker.run(args, bpy=bpy, vector=Vector)
    else:
        result = worker.run(args)
        assert json.loads(capsys.readouterr().out) == result
        assert result["faces_per_region"] == {"head": 1, "body": 1, "tail": 1}
        assert result["texture_size"] == 8
        assert [event for event in events if event[0] == "bake"] == [("bake", "EMIT", "Asset")]
        packed = [event[1] for event in events if event[0] == "pack"]
        assert len(packed) == 3
        assert all(details["scale"] and details["margin"] == .0002 for details in packed)
        assert layers.active.name == "region_uv"
        assert [point.uv for point in layers.active.data] == [(.1, .2), (.3, .4)]
        for region, mat in zip(worker.REGION_NAMES, mesh.materials, strict=True):
            uv_node = mat.node_tree.nodes.get("ShaderNodeUVMap")
            if scenario == "success":
                assert mat.name == f"{expected_prefix}_{region}"
                assert uv_node.uv_map == "region_uv"
                assert Path(result["written"][region]) == tmp_path / f"{expected_prefix}_albedo_{region}.png"
                assert Path(result["written"][region]).read_bytes() == b"png"
            else:
                assert uv_node.uv_map == "original UV"
        if scenario == "success":
            assert result["error"] is None
            assert layers.get("original UV") is None
            assert Path(result["glb"]) == tmp_path / f"{expected_prefix}_regions.glb"
            assert Path(result["glb"]).read_bytes() == b"glb"
        else:
            assert result["error"] == "fixture bake failed"
            assert result["glb"] is None and result["written"] == {}
            assert layers.get("original UV") is uv
    assert bpy.context.window.scene is user
    assert [obj.name for obj in user.objects] == ["keep me"]
    assert not user.objects[0].selected


@pytest.mark.parametrize("derive,fail", [(True, False), (False, False), (False, True)])
def test_solidify_worker_remeshes_culls_specks_and_restores_scene(private_scene, tmp_path, monkeypatch, derive, fail):
    from scripts import blender_solidify_bake_worker as worker

    bpy, user, events = private_scene("IMG3D solidify", fail=fail, shader=derive, existing=derive)
    material_data = Obj(materials=["original"], polygons=[0], vertices=[0, 1, 2, 3])
    def copy_data():
        return Obj(materials=["original"], polygons=[0], vertices=[0, 1, 2, 3],
                   update=lambda: events.append(("updated",)))

    material_data.copy = copy_data
    source = MeshObject("imported", material_data)
    source.bound_box = list(product((0, 3), (0, 2), (0, 1)))
    source.matrix_world = np.eye(3)
    modifier = Obj(name="Remesh")
    source.modifiers = Obj(new=lambda name, kind: modifier)

    def load(*, filepath):
        assert filepath == "input.glb"
        bpy.context.scene.objects.append(source)
        if derive:
            bpy.context.scene.objects.append(MeshObject("separate patch", material_data))

    bpy.ops.import_scene.gltf = load
    vertices = Items(Obj(index=index, link_edges=[]) for index in range(4))
    vertices.ensure_lookup_table = lambda: None

    def link(first, second):
        def other(vert):
            return vertices[second if vert is vertices[first] else first]
        edge = Obj(other_vert=other)
        vertices[first].link_edges.append(edge)
        vertices[second].link_edges.append(edge)

    link(0, 1)
    link(1, 2)
    deleted = []
    def remove_vertices(_bm, *, geom, context):
        assert context == "VERTS"
        deleted.extend(vert.index for vert in geom)

    def update_mesh(data):
        data.vertices = [index for index in range(4) if index not in deleted]

    bm = Obj(verts=vertices, from_mesh=lambda data: None, to_mesh=update_mesh,
             free=lambda: events.append(("free",)))
    bmesh = Obj(new=lambda: bm, ops=Obj(delete=remove_vertices))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "mathutils", Obj(Vector=Vector))
    monkeypatch.setitem(sys.modules, "bmesh", bmesh)
    output = tmp_path / "out.glb"
    args = {
        "asset_path": "input.glb", "output_path": str(output),
        "voxel_size": 0 if derive else .01,
        "texture_size": 4, "extrusion": 0 if derive else .05,
        "min_component_verts": 2 if derive else 1,
    }
    if fail:
        with pytest.raises(RuntimeError, match="fixture bake failed"):
            worker.run(args, bpy=bpy, vector=Vector, bmesh=bmesh)
        assert not output.exists()
    else:
        result = worker.run(args)
        assert result["extent"] == 3
        assert result["components_kept"] == (1 if derive else 2)
        assert result["debris_components_removed"] == int(derive)
        assert result["verts"] == (3 if derive else 4)
        assert result["output"] == str(output)
        assert output.read_bytes() == b"glb"
        assert ("remove", "SOURCE") in events
        target = bpy.context.view_layer.objects.active
        assert target.selected and target.name == "TARGET"
        material, = target.data.materials
        assert material.node_tree.nodes.active.image.packed
    scene = bpy.data.scenes["IMG3D solidify"]
    assert scene.render.bake.use_selected_to_active
    assert scene.cycles.device == ("GPU" if derive else "CPU")
    cage = .036 if derive else .05
    assert scene.render.bake.cage_extrusion == pytest.approx(cage)
    assert scene.render.bake.max_ray_distance == pytest.approx(2 * cage)
    assert modifier.voxel_size == (.006 if derive else .01)
    assert deleted == ([3] if derive else [])
    assert ("free",) in events
    assert ("join",) in events if derive else ("join",) not in events
    assert bpy.context.window.scene is user
    assert [obj.name for obj in user.objects] == ["keep me"]
    assert not user.objects[0].selected


@pytest.mark.parametrize("filename", ["blender_solidify_bake.py", "blender_split_regions.py"])
def test_live_launcher_help_runs_without_connecting(filename, monkeypatch, tmp_path, capsys):
    from runpy import run_path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(sys, "argv", [filename, "--help"])
    with pytest.raises(SystemExit) as result:
        run_path(str(scripts / filename), run_name="__main__")
    assert result.value.code == 0
    assert "--host" in capsys.readouterr().out


def test_split_launcher_rejects_inverted_cuts_before_creating_output(monkeypatch, tmp_path):
    from runpy import run_path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    output = tmp_path / "out"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(sys, "argv", [
        "blender_split_regions.py", "mesh.glb", str(output), "--head-cut", "-1",
    ])
    with pytest.raises(SystemExit, match="--head-cut must be greater than --tail-cut"):
        run_path(str(scripts / "blender_split_regions.py"), run_name="__main__")
    assert not output.exists()


@pytest.mark.parametrize("prefix,expected", [(None, "moss_fox"), ("sample model", "sample model")])
def test_split_launcher_bundles_default_and_custom_prefix(monkeypatch, tmp_path, capsys, prefix, expected):
    from scripts import blender_split_regions as launcher

    source = tmp_path / "input.glb"
    output = tmp_path / "out"
    call = {}
    build_worker_code = launcher.build_worker_code

    def bundle(path, arguments):
        call.update(arguments)
        return build_worker_code(path, arguments)

    def send(code, host, port):
        assert code
        assert (host, port) == ("127.0.0.1", 9876)
        return "fixture reply"

    monkeypatch.setattr(launcher, "build_worker_code", bundle)
    monkeypatch.setattr(launcher, "send", send)
    flags = [] if prefix is None else ["--output-prefix", prefix]
    monkeypatch.setattr(sys, "argv", ["regions", str(source), str(output), *flags])
    assert launcher.main() == 0
    assert call["asset_path"] == str(source)
    assert call["out_dir"] == str(output)
    assert call["output_prefix"] == expected
    assert capsys.readouterr().out.strip() == "fixture reply"


@pytest.mark.parametrize("prefix", ["", "../other", "elsewhere\\other"])
def test_split_launcher_rejects_directory_prefixes(monkeypatch, tmp_path, prefix):
    from scripts import blender_split_regions as launcher

    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["regions", "mesh.glb", str(output), "--output-prefix", prefix])
    with pytest.raises(SystemExit) as result:
        launcher.main()
    assert result.value.code == 2
    assert not output.exists()
