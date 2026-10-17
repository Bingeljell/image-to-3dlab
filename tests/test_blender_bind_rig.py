from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "scripts" / "blender_bind_rig.py"


def load_module():
    spec = importlib.util.spec_from_file_location("blender_bind_rig", MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Obj:
    def __init__(self, name, type_, scale=(1.0, 1.0, 1.0)):
        self.name = name
        self.type = type_
        self.scale = scale


class Bone:
    def __init__(self, name, use_deform=True):
        self.name = name
        self.use_deform = use_deform


class Armature(Obj):
    def __init__(self, name, bones):
        super().__init__(name, "ARMATURE")
        self.data = type("Data", (), {"bones": bones})()


def test_resolves_the_single_mesh_and_armature():
    module = load_module()
    scene = [Obj("Camera", "CAMERA"), Obj("geometry_0", "MESH"),
             Obj("Light", "LIGHT"), Armature("metarig", [Bone("spine")])]
    mesh, armature = module.resolve_pair(scene)
    assert mesh.name == "geometry_0"
    assert armature.name == "metarig"


def test_a_leftover_proxy_does_not_make_the_scene_ambiguous():
    module = load_module()
    scene = [Obj("geometry_0", "MESH"),
             Obj(f"{module.PROXY_PREFIX}geometry_0", "MESH"),
             Armature("metarig", [Bone("spine")])]
    assert module.resolve_pair(scene)[0].name == "geometry_0"


def test_two_meshes_refuse_rather_than_guess():
    module = load_module()
    scene = [Obj("body", "MESH"), Obj("eyes", "MESH"),
             Armature("metarig", [Bone("spine")])]
    with pytest.raises(RuntimeError, match="name one explicitly"):
        module.resolve_pair(scene)
    assert module.resolve_pair(scene, mesh_name="eyes")[0].name == "eyes"


def test_naming_a_missing_object_reports_what_was_there():
    module = load_module()
    scene = [Obj("body", "MESH"), Armature("metarig", [Bone("spine")])]
    with pytest.raises(RuntimeError, match=r"\['body'\]"):
        module.resolve_pair(scene, mesh_name="geometry_0")


def test_an_empty_scene_names_the_missing_kind():
    module = load_module()
    with pytest.raises(RuntimeError, match="no armature"):
        module.resolve_pair([Obj("body", "MESH")])


def test_unapplied_scale_is_what_triggers_the_apply():
    module = load_module()
    assert module.unapplied_scale(Obj("mesh", "MESH", (1.5, 1.5, 1.5)))
    assert module.unapplied_scale(Obj("rig", "ARMATURE", (0.9, 0.9, 0.9)))
    assert module.unapplied_scale(Obj("flat", "MESH", (1.0, 1.0, 2.0)))
    assert not module.unapplied_scale(Obj("clean", "MESH", (1.0, 1.0, 1.0)))
    assert not module.unapplied_scale(Obj("noise", "MESH", (1.00001, 1.0, 1.0)))


def test_only_deforming_bones_are_counted():
    module = load_module()
    armature = Armature("metarig", [Bone("spine"), Bone("WGT-ik", use_deform=False)])
    assert module.deforming_bones(armature) == ["spine"]


def test_rigify_widget_meshes_are_not_bind_candidates():
    """Generating a Rigify rig fills the scene with WGT- control widget meshes; they are
    scene furniture, not the mesh to bind."""
    module = load_module()
    scene = [Obj("geometry_0", "MESH"), Obj("WGT-rig_root", "MESH"),
             Obj("WGT-rig_thigh_ik.L", "MESH"), Armature("metarig", [Bone("spine")])]
    assert module.resolve_pair(scene)[0].name == "geometry_0"


def test_the_generated_rig_is_preferred_over_the_metarig():
    """After generation the scene holds both armatures; weights belong on the DEF- bones
    of the generated rig, never on the metarig."""
    module = load_module()
    metarig = Armature("metarig", [Bone("spine")])
    generated = Armature("rig", [Bone("DEF-spine")])
    assert module.pick_armature([metarig, generated], generated) is generated
    assert module.pick_armature([metarig], None) is metarig


def test_binding_to_a_named_armature_still_wins():
    module = load_module()
    metarig = Armature("metarig", [Bone("spine")])
    generated = Armature("rig", [Bone("DEF-spine")])
    _mesh, armature = module.resolve_pair(
        [Obj("geometry_0", "MESH"), metarig, generated], armature_name="metarig")
    assert armature is metarig
