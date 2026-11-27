"""Run the visibility cull with recorded face-ID buffers and a writable mesh adapter."""

import math
import runpy
import sys
from types import SimpleNamespace as NS

import blender_visibility_cull as cull
import numpy as np
import pytest

from tests.test_coverage_bake_extra_scene import (
    BakeScene,
    MeshObject,
    Pixels,
    Transform,
)
from tests.test_coverage_blender_quadruped import Vector


class CameraVector(Vector):
    def __new__(cls, values):
        return super().__new__(cls, tuple(values))

    def to_track_quat(self, track, up):
        assert (track, up) == ("-Z", "Y")
        return self

    def to_euler(self):
        return tuple(self)


class Faces(list):
    def foreach_get(self, name, buffer):
        buffer[:] = [getattr(face, name) for face in self]

    def foreach_set(self, name, values):
        for face, value in zip(self, values, strict=True):
            setattr(face, name, int(value))

    def ensure_lookup_table(self):
        pass


class Materials(list):
    def pop(self, *, index):
        return super().pop(index)


class CornerColors(dict):
    writes = None

    def new(self, *, name, type, domain):
        assert (type, domain) == ("FLOAT_COLOR", "CORNER")
        attr = NS(name=name, data=NS(foreach_set=self.write))
        self[name] = attr
        return attr

    def write(self, field, values):
        assert field == "color"
        self.writes = values.copy().reshape(-1, 4)

    def remove(self, attr):
        del self[attr.name]


@pytest.mark.parametrize("combine", [False, True])
def test_visibility_cull_restores_materials_and_exports_only_seen_faces(tmp_path, monkeypatch, capsys, combine):
    work = BakeScene()
    source = MeshObject("shell", dimensions=(2, 3, 6))
    source.matrix_world = Transform.Translation((5, -4, 7))
    materials = [object(), object()]
    source.data.materials = Materials(materials)
    source.data.polygons = Faces([
        NS(index=index, material_index=value, loop_total=loops)
        for index, (value, loops) in enumerate([(1, 3), (0, 4), (1, 3)])
    ])
    attr = source.data.color_attributes = CornerColors()
    inner = MeshObject("inner fragment")
    work.imports["input.glb"] = [source, inner] if combine else [source]
    made_materials, captures, reads, discarded, removed_faces = [], [], [], [], []
    create_material = work.bpy.data.materials.new

    def create_index_material(name):
        material = create_material(name)
        made_materials.append(material)
        return material

    work.bpy.data.materials.new = create_index_material
    work.bpy.data.materials.remove = made_materials.remove
    work.bpy.app = NS(tempdir=str(tmp_path))
    work.bpy.data.cameras = NS(new=lambda name: NS())

    class SceneObjects(type(work.objects)):
        def new(self, name, camera_data):
            return NS(name=name, data=camera_data, type="CAMERA", selected=False,
                      select_set=lambda selected: None)

    work.bpy.data.objects = work.objects = SceneObjects([MeshObject("previous scene")])
    work.scene.collection.objects.link = work.objects.append
    work.scene.view_settings, work.scene.display_settings = NS(), NS()
    work.scene.eevee = NS(taa_render_samples=16)
    work.render.image_settings = NS()
    joins = []

    def join():
        assert source.selected and inner.selected and work.active.active is source
        joins.append(inner)
        work.bpy.data.objects.remove(inner)

    work.ops.object.join = join

    def render(*, write_still):
        assert write_still
        assert source.data.polygons[0].material_index == 2
        assert made_materials[-1].use_backface_culling is False
        assert work.scene.view_settings.view_transform == "Raw"
        captures.append(work.scene.camera.location.copy())

    def load(path, *, check_existing):
        assert path == str(tmp_path / "fid.png") and not check_existing
        offset = 0 if not reads else 7
        visible_color = attr.writes[offset]  # face 0 first, then face 2, never face 1
        pixels = Pixels([visible_color, (0, 0, 0, 0)])
        image = NS(colorspace_settings=NS(), pixels=pixels)
        reads.append(image)
        return image

    def release(image):
        assert image.colorspace_settings.name == "Non-Color"
        discarded.append(image)

    def delete(mesh, *, geom, context):
        assert context == "FACES"
        assert source.data.materials == materials and not made_materials
        assert [p.material_index for p in source.data.polygons] == [1, 0, 1]
        for face in geom:
            removed_faces.append(face.index)
            mesh.faces.remove(face)

    def from_mesh(mesh):
        edit.faces = Faces(mesh.polygons.copy())

    def to_mesh(mesh):
        mesh.polygons = Faces(edit.faces.copy())

    cleared = []
    edit = NS(from_mesh=from_mesh, to_mesh=to_mesh, free=lambda: cleared.append(True))
    work.ops.render = NS(render=render)
    work.bpy.data.images = NS(load=load, remove=release)
    monkeypatch.setitem(sys.modules, "bpy", work.bpy)
    monkeypatch.setitem(sys.modules, "bmesh", NS(new=lambda: edit, ops=NS(delete=delete)))
    monkeypatch.setitem(sys.modules, "mathutils", NS(Vector=CameraVector))
    out = tmp_path / "visible.glb"
    argv = ["input.glb", str(out), "25", "32"]
    if combine:
        monkeypatch.setattr(sys, "argv", ["blender", "--", *argv])
        runpy.run_path(cull.__file__, run_name="__main__")
    else:
        cull.main(argv)

    assert len(captures) == len(discarded) == 25
    np.testing.assert_allclose([np.linalg.norm(point - [6, -2.5, 10]) for point in captures], 10.5)
    assert work.scene.camera.data.ortho_scale == pytest.approx(math.sqrt(49) * 1.05)
    assert ("25/25 views, 2 faces seen" in capsys.readouterr().out)
    assert joins == ([inner] if combine else [])
    assert np.load(out.with_suffix(".visible.npy")).tolist() == [True, False, True]
    assert cleared == [True] and removed_faces == [1]
    assert [p.index for p in source.data.polygons] == [0, 2]
    assert source.data.materials == materials
    assert "fid" not in attr
    assert work.exports == [(out, ["shell"], {})]
    assert out.read_bytes() == work.export_data


def test_render_configuration_preserves_defaults_if_compatibility_choices_are_unavailable():
    class RenderSettings(NS):
        @property
        def engine(self):
            return "CYCLES"

        @engine.setter
        def engine(self, value):
            raise TypeError(f"{value} unsupported")

    class ViewSettings(NS):
        @property
        def view_transform(self):
            return "Default"

        @view_transform.setter
        def view_transform(self, value):
            raise TypeError(f"{value} unsupported")

    render = RenderSettings(image_settings=NS())
    scene = NS(render=render, view_settings=ViewSettings(), display_settings=NS())
    camera = object()
    cull.configure_render(scene, camera, 64)
    assert scene.camera is camera
    assert render.engine == "CYCLES" and scene.view_settings.view_transform == "Default"
    assert render.filter_size == render.dither_intensity == 0
    assert render.image_settings.color_mode == "RGBA"
