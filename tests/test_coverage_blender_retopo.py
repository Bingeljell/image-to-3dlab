"""Exercise material lookup and UV finishing with small operator substitutes."""

import sys
from types import SimpleNamespace as NS

import blender_retopo_bake as retopo
import pytest


class Face:
    def __init__(self, select=False, uv_select=False):
        self.select = select
        self.uv_select = uv_select

    def select_set(self, value):
        self.select = value


def test_overlap_selection_combines_mesh_and_valid_uv_selections():
    mesh_face, uv_face, other = Face(select=True), Face(uv_select=True), Face()
    bm = NS(faces=[mesh_face, uv_face, other], uv_select_sync_valid=False)
    assert retopo.overlap_selection(bm) == (1, 0)
    bm.uv_select_sync_valid = True
    assert retopo.overlap_selection(bm) == (2, 1)
    assert uv_face.select and not other.select
    assert retopo.overlap_selection(bm) == (2, 0)


@pytest.mark.parametrize("linked", [True, False])
def test_base_colour_prefers_connected_image_over_fallback(linked):
    fallback = NS(type="TEX_IMAGE", image=object())
    colour = NS(type="TEX_IMAGE", image=object())
    socket = NS(is_linked=linked, links=[NS(from_node=colour)])
    shader = NS(type="BSDF_PRINCIPLED", inputs={"Base Color": socket})
    mat = NS(use_nodes=True, node_tree=NS(nodes=[fallback, shader, colour]))
    assert retopo.base_colour_image(mat) is (colour if linked else fallback).image
    mat.node_tree.nodes = [
        NS(type="TEX_IMAGE", image=None),
        NS(type="BSDF_PRINCIPLED", inputs={}),
    ]
    assert retopo.base_colour_image(mat) is None
    mat.use_nodes = False
    assert retopo.base_colour_image(mat) is None
    assert retopo.base_colour_image(None) is None


class UVOperators:
    def __init__(self, selections):
        self.selections = iter(selections)
        self.face = Face()
        self.projections = []
        self.seams = 0
        self.unwraps = []
        self.isolated = []
        self.updates = []

    def smart_project(self, *, angle_limit, island_margin):
        self.projections.append((angle_limit, island_margin))

    def seams_from_islands(self):
        self.seams += 1

    def unwrap(self, *, method, margin):
        self.unwraps.append((method, margin))

    def select_overlap(self):
        selected = next(self.selections)
        if isinstance(selected, Exception):
            raise selected
        self.face.uv_select = selected

    def select_all(self, *, action):
        self.face.select_set(action == "SELECT")

    def mark_seam(self, *, clear):
        self.isolated.append((self.face.select, clear))

    def edit_mesh(self, mesh):
        return NS(faces=[self.face], uv_select_sync_valid=True)

    def update_mesh(self, mesh, *, loop_triangles, destructive):
        self.updates.append((mesh, loop_triangles, destructive))


@pytest.fixture
def uv_scene(monkeypatch):
    def build(selections, version):
        uv = UVOperators(selections)
        settings = NS(use_uv_select_sync=False, mesh_select_mode=[True, False, False])
        bpy = NS(
            ops=NS(uv=uv, mesh=uv),
            context=NS(scene=NS(tool_settings=settings)),
            app=NS(version=version),
        )
        monkeypatch.setitem(
            sys.modules,
            "bmesh",
            NS(from_edit_mesh=uv.edit_mesh, update_edit_mesh=uv.update_mesh),
        )
        return bpy, uv, settings

    return build


def test_uv_finishing_reprojects_and_isolates_persistent_overlap(uv_scene):
    bpy, uv, settings = uv_scene([True, True, True, True, False], (5, 2, 1))
    mesh = object()
    assert retopo.unwrap_finish(bpy, mesh, 0.8) == 0
    assert uv.projections == [(0.8, 0.0002)] + [(0.4, 0.0002)] * 3
    assert uv.seams == 4
    assert uv.unwraps == [("CONFORMAL", 0.002)] * 5
    assert uv.isolated == [(True, False)]
    assert uv.updates == [(mesh, False, False)] * 4
    assert uv.face.select
    assert settings.use_uv_select_sync is False
    assert settings.mesh_select_mode == (True, False, False)


def test_uv_finishing_stops_when_no_faces_overlap(uv_scene):
    bpy, uv, _ = uv_scene([False, False, False], (5, 2, 1))
    assert retopo.unwrap_finish(bpy, object(), 0.8) == 0
    assert uv.projections == [(0.8, 0.0002)]
    assert not uv.isolated
    assert not uv.updates


def test_uv_finishing_uses_projection_only_on_crashing_blender(uv_scene):
    bpy, uv, _ = uv_scene([], (5, 2, 0))
    assert retopo.unwrap_finish(bpy, object(), 0.8) is None
    assert uv.projections == [(0.8, 0.0002)]
    assert not uv.unwraps


def test_uv_finishing_restores_selection_settings_on_error(uv_scene):
    bpy, _, settings = uv_scene([RuntimeError("bad UV layer")], (5, 2, 1))
    with pytest.raises(RuntimeError, match="bad UV layer"):
        retopo.unwrap_finish(bpy, object(), 0.8)
    assert settings.use_uv_select_sync is False
    assert settings.mesh_select_mode == (True, False, False)


@pytest.mark.parametrize(
    ("objects", "message"),
    [
        ([NS(type="CAMERA")], "no mesh found"),
        ([NS(type="MESH", data=NS(materials=[]))], "base colour texture"),
    ],
)
def test_retopo_rejects_incomplete_import_before_copying(monkeypatch, objects, message):
    class ObjectList(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    scene_objects = ObjectList([NS(type="MESH")])

    def import_mesh(*, filepath):
        assert filepath == "in.glb"
        assert not scene_objects
        scene_objects.extend(objects)

    bpy = NS(data=NS(objects=scene_objects), ops=NS(import_scene=NS(gltf=import_mesh)))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", ["blender", "--", "in.glb", "out.glb"])
    with pytest.raises(SystemExit, match=message):
        retopo.main()
    assert scene_objects == objects


@pytest.mark.parametrize(
    ("outcome", "voxel_fraction", "wanted"),
    [
        ("quad", 0.01, 4),
        ("decimate", 0.0, 6),
        ("quad_error", 0.01, 6),
        ("no_reduction", 0.0, None),
    ],
)
def test_topology_reduction_falls_back_when_quad_solver_cannot_help(
    outcome, voxel_fraction, wanted
):
    calls = []
    data = NS(polygons=[None] * 10, loop_triangles=[None] * 20)

    def mesh_command(name):
        def run(**kwargs):
            calls.append((name, kwargs))

        return run

    def remesh():
        calls.append(("voxel", data.remesh_voxel_size))
        data.polygons = [None] * 12

    def quad_solver(**kwargs):
        calls.append(("quad", kwargs))
        if outcome == "quad":
            data.polygons = [None] * 4
        elif outcome == "quad_error":
            raise RuntimeError("solver failed")

    def new_modifier(name, kind):
        assert kind == "DECIMATE"
        modifier = NS(name=name)
        assert not modifiers
        modifiers.append(modifier)
        return modifier

    def apply_modifier(*, modifier):
        (created,) = modifiers
        assert modifier == created.name
        assert created.decimate_type == "COLLAPSE"
        assert created.ratio == 0.2
        calls.append(("decimate", modifier))
        if outcome != "no_reduction":
            data.polygons = [None] * 6

    modifiers = []
    data.calc_loop_triangles = mesh_command("triangles")
    obj = NS(data=data, dimensions=(1, 2, 3), modifiers=NS(new=new_modifier))
    bpy = NS(
        ops=NS(
            object=NS(
                mode_set=mesh_command("mode"),
                voxel_remesh=remesh,
                quadriflow_remesh=quad_solver,
                modifier_apply=apply_modifier,
            ),
            mesh=NS(
                select_all=mesh_command("select"),
                remove_doubles=mesh_command("weld"),
                normals_make_consistent=mesh_command("normals"),
            ),
        )
    )
    if wanted is None:
        with pytest.raises(SystemExit, match="nothing reduced the mesh"):
            retopo.reduce_topology(bpy, obj, 4, voxel_fraction)
    else:
        retopo.reduce_topology(bpy, obj, 4, voxel_fraction)
        assert len(data.polygons) == wanted
    phases = [kind for kind, _ in calls if kind not in {"mode", "select"}]
    assert phases == [
        "weld",
        "normals",
        *(["voxel"] if voxel_fraction else []),
        "normals",
        "quad",
        *(["triangles", "decimate"] if outcome != "quad" else []),
    ]
    if voxel_fraction:
        assert data.remesh_voxel_size == pytest.approx(0.03)
