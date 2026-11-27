"""Check voxel proxy cleanup and the weight report's counting rules."""

from types import SimpleNamespace as NS

import blender_rebind_weights as weights
import pytest


class Vertices(list):
    def ensure_lookup_table(self):
        pass


class EditMesh:
    def __init__(self, edges):
        self.edge_indices = edges
        self.freed = False

    def from_mesh(self, data):
        self.verts = Vertices(data.vertices)
        self.edges = [
            NS(verts=(self.verts[a], self.verts[b])) for a, b in self.edge_indices
        ]

    def to_mesh(self, data):
        data.vertices = list(self.verts)

    def free(self):
        self.freed = True


@pytest.mark.parametrize(
    ("edges", "expected"),
    [([(0, 1), (1, 2)], [0, 1, 2]), ([(0, 1), (1, 2), (2, 3)], [0, 1, 2, 3])],
)
def test_proxy_strip_keeps_the_largest_island(edges, expected):
    mesh = EditMesh(edges)
    updates = []
    data = NS(
        vertices=[NS(index=i) for i in range(4)], update=lambda: updates.append(True)
    )
    removed = []

    def delete(edit, *, geom, context):
        assert edit is mesh and context == "VERTS"
        removed.extend(v.index for v in geom)
        for v in geom:
            edit.verts.remove(v)

    bmesh = NS(new=lambda: mesh, ops=NS(delete=delete))
    assert weights.strip_loose_islands(bmesh, data) == {
        "before": 4,
        "after": len(expected),
        "removed": 4 - len(expected),
    }
    assert [v.index for v in data.vertices] == expected
    assert removed == ([3] if len(expected) == 3 else [])
    assert updates == ([True] if removed else [])
    assert mesh.freed


def test_proxy_edit_mesh_is_freed_when_removal_fails():
    mesh = EditMesh([(0, 1)])
    data = NS(vertices=[NS(index=i) for i in range(3)])

    def delete(*args, **kwargs):
        raise RuntimeError("bad mesh")

    bmesh = NS(new=lambda: mesh, ops=NS(delete=delete))
    with pytest.raises(RuntimeError, match="bad mesh"):
        weights.strip_loose_islands(bmesh, data)
    assert mesh.freed
    assert len(data.vertices) == 3


def test_weight_report_treats_threshold_as_inclusive_and_ignores_unknown_groups():
    def vertex(*groups):
        return NS(groups=[NS(group=i, weight=w) for i, w in groups])

    mesh = NS(
        vertex_groups=[NS(index=0, name="body"), NS(index=1, name="head")],
        data=NS(
            vertices=[
                vertex((0, 0.25)),
                vertex((1, 0.24)),
                vertex((2, 1)),
                vertex(),
            ]
        ),
    )
    assert weights._nonempty_groups(mesh, 0.25) == {"body"}
    assert weights._unweighted_vertices(mesh, 0.25) == 2


def test_voxel_size_rejects_zero_extent_and_out_of_range_fraction():
    mesh = NS(name="test mesh", dimensions=(0, 0, 0))
    with pytest.raises(RuntimeError, match="zero extent"):
        weights.derived_voxel_size(mesh)
    mesh.dimensions = (0, 5, 0)
    with pytest.raises(ValueError, match="between"):
        weights.derived_voxel_size(mesh, 0.1)
    assert weights.derived_voxel_size(mesh, 0.05) == 0.25


class Modifiers(list):
    def new(self, name, kind):
        modifier = NS(name=name, type=kind)
        self.append(modifier)
        return modifier

    def named(self, name):
        return next(modifier for modifier in self if modifier.name == name)


class MeshData:
    def __init__(self):
        self.vertices = [NS(index=i, groups=[]) for i in range(3)]
        self.polygons = [object()]

    def copy(self):
        return MeshData()


class MeshObject:
    def __init__(self, name):
        self.name = name
        self.selected = False
        self.parent = None
        self.dimensions = (1, 2, 3)
        self.matrix_world = [1, 2, 3]
        self.data = MeshData()
        self.modifiers = Modifiers()
        self.modifiers.new("old rig", "ARMATURE")
        self.modifiers.new("existing bevel", "BEVEL")
        self.vertex_groups = [NS(index=0, name="old weight")]

    def copy(self):
        return MeshObject(self.name)

    def select_set(self, selected):
        self.selected = selected


@pytest.mark.parametrize(
    "failure", [None, "remesh", "parent", "proxy weights", "transfer weights"]
)
def test_transfer_weights_sequences_proxy_and_skinning_steps(monkeypatch, failure):
    import sys

    class SceneObjects(list):
        def remove(self, obj, *, do_unlink):
            assert do_unlink
            super().remove(obj)

    mesh, rig = MeshObject("mesh"), MeshObject("rig")
    previous_proxy = MeshObject("I2L_WEIGHT_PROXY_mesh")
    objects = SceneObjects([mesh, rig, previous_proxy])
    active = NS(active=None)
    actions = []

    def select_all(*, action):
        assert action == "DESELECT"
        for obj in objects:
            obj.select_set(False)

    def proxy():
        return next(obj for obj in objects if obj.name == "I2L_WEIGHT_PROXY_mesh")

    def modifier_apply(*, modifier):
        obj = active.active
        applied = obj.modifiers.named(modifier)
        actions.append(applied.type)
        if applied.type == "REMESH":
            assert obj is proxy() and obj is not previous_proxy
            assert obj.data is not mesh.data
            assert obj.matrix_world == mesh.matrix_world
            assert obj.vertex_groups == []
            assert applied.mode == "VOXEL"
            assert applied.voxel_size == pytest.approx(0.036)
            if failure == "remesh":
                obj.data.polygons.clear()
        else:
            assert obj is mesh
            assert applied.object is proxy()
            assert applied.vert_mapping == "POLYINTERP_NEAREST"
            assert applied.use_vert_data
            assert applied.data_types_verts == {"VGROUP_WEIGHTS"}
            if failure != "transfer weights":
                for vertex in obj.data.vertices[:2]:
                    vertex.groups = [NS(group=0, weight=1.0)]
        obj.modifiers.remove(applied)

    def parent_set(*, type):
        assert active.active is rig
        actions.append(type)
        if type == "ARMATURE_AUTO":
            assert proxy().selected and rig.selected
            if failure == "parent":
                raise RuntimeError("bone heat did not converge")
            if failure != "proxy weights":
                proxy().vertex_groups.append(NS(index=0, name="body"))
                for vertex in proxy().data.vertices:
                    vertex.groups = [NS(group=0, weight=1.0)]
            proxy().parent = rig
        else:
            assert mesh.selected and rig.selected
            mesh.parent = rig

    def data_layout(*, modifier):
        assert active.active is mesh
        transfer = mesh.modifiers.named(modifier)
        assert transfer.object is proxy()
        assert mesh.vertex_groups == []
        mesh.vertex_groups = list(transfer.object.vertex_groups)
        actions.append("layout")

    bpy = NS(
        data=NS(objects=objects),
        context=NS(
            scene=NS(collection=NS(objects=NS(link=objects.append))),
            view_layer=NS(objects=active),
        ),
        ops=NS(
            object=NS(
                select_all=select_all,
                modifier_apply=modifier_apply,
                parent_set=parent_set,
                datalayout_transfer=data_layout,
            )
        ),
    )
    edit = EditMesh([(0, 1), (1, 2)])
    monkeypatch.setitem(sys.modules, "bmesh", NS(new=lambda: edit))
    if failure:
        message = {
            "remesh": "empty proxy",
            "parent": "bone heat did not converge",
            "proxy weights": "proxy produced no non-empty",
            "transfer weights": "transfer produced no non-empty",
        }[failure]
        with pytest.raises(RuntimeError, match=message):
            weights.transfer_weights(bpy, mesh, rig)
        assert previous_proxy not in objects
        if failure == "transfer weights":
            assert objects == [mesh, rig]
    else:
        report = weights.transfer_weights(bpy, mesh, rig)
        assert report == {
            "mesh": "mesh",
            "voxelSize": pytest.approx(0.036),
            "proxyFaces": 1,
            "proxyLooseVertsRemoved": 0,
            "proxyGroups": 1,
            "meshGroups": 1,
            "unweightedVertices": 1,
            "totalVertices": 3,
        }
        assert actions == [
            "REMESH",
            "ARMATURE_AUTO",
            "layout",
            "DATA_TRANSFER",
            "ARMATURE_NAME",
        ]
        assert objects == [mesh, rig]
        assert mesh.parent is rig
        assert [m.type for m in mesh.modifiers] == ["BEVEL"]
