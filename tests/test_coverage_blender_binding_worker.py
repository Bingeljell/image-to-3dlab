"""Verify the bind command applies scale and reports the rig actually weighted."""

import json
import sys
from types import SimpleNamespace as NS

import blender_bind_rig as bind
import blender_rebind_weights as weights
import pytest


class Object:
    def __init__(self, name, kind, bones=()):
        self.name = name
        self.type = kind
        self.scale = (1, 1, 1)
        self.data = NS(bones=bones)
        self.vertex_groups = []

    def hide_set(self, value):
        self.hidden = value

    def select_set(self, value):
        self.selected = value


@pytest.mark.parametrize("workflow", ["rigify", "plain", "no deforms", "same output"])
def test_bind_worker_routes_object_changes_and_save(
    monkeypatch, tmp_path, capsys, workflow
):
    source, output = tmp_path / "opened.blend", tmp_path / "new" / "bound.blend"
    source.write_bytes(b"original")
    if workflow == "same output":
        output = source
    mesh = Object("creature", "MESH")
    metarig = Object("metarig", "ARMATURE", [NS(name="spine", use_deform=False)])
    generated = Object("generated", "ARMATURE", [NS(name="DEF-spine", use_deform=True)])
    mesh.scale = (2, 2, 2)
    if workflow == "plain" or workflow == "same output":
        metarig.data.bones[0].use_deform = True
    objects = [mesh, metarig]
    active = NS(active=None)
    events = []

    def apply_scale(*, location, rotation, scale):
        assert not location and not rotation and scale
        assert active.active is mesh and mesh.selected
        mesh.scale = (1, 1, 1)
        events.append("scale")

    def generate():
        assert active.active is metarig
        assert mesh.scale == (1, 1, 1)
        metarig.data.rigify_target_rig = generated
        objects.append(generated)
        events.append("generate")

    def transfer(bpy, obj, rig, *, voxel_fraction, weight_threshold):
        assert obj is mesh
        assert rig is (generated if workflow == "rigify" else metarig)
        assert voxel_fraction == 0.01 and weight_threshold == 0.02
        obj.vertex_groups = [NS(name=rig.data.bones[0].name), NS(name="old")]
        events.append("transfer")
        return {"mesh": obj.name}

    def save(*, filepath):
        assert filepath == str(output) and output.parent.is_dir()
        output.write_bytes(b"bound")
        events.append("save")

    bpy = NS(
        data=NS(objects=objects, filepath=str(source)),
        context=NS(object=NS(mode="POSE"), view_layer=NS(objects=active)),
        ops=NS(
            object=NS(
                mode_set=lambda *, mode: events.append(mode),
                select_all=lambda *, action: None,
                transform_apply=apply_scale,
            ),
            preferences=NS(addon_enable=lambda *, module: None),
            pose=NS(rigify_generate=generate),
            wm=NS(save_as_mainfile=save),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(weights, "transfer_weights", transfer)
    args = [
        "--out",
        str(output),
        "--voxel-fraction",
        "0.01",
        "--weight-threshold",
        "0.02",
    ]
    if workflow == "rigify":
        args.append("--generate-rigify")
    else:
        args.append("--no-apply-scale")
    if workflow == "no deforms":
        with pytest.raises(RuntimeError, match="no deforming bones"):
            bind.main(args)
        assert events == ["OBJECT"]
    elif workflow == "same output":
        with pytest.raises(SystemExit, match="blend Blender opened"):
            bind.main(args)
        assert events == ["OBJECT", "transfer"]
    else:
        assert bind.main(args) == 0
        report = json.loads(capsys.readouterr().out.removeprefix("I2L_BIND_REPORT "))
        is_rigify = workflow == "rigify"
        assert report["generatedRigify"] is is_rigify
        assert report["armature"] == ("generated" if is_rigify else "metarig")
        assert report["boundToMissingBones"] == ["old"]
        assert report["scaleApplied"] == (["creature"] if is_rigify else [])
        assert events == [
            "OBJECT",
            *(["scale", "generate"] if is_rigify else []),
            "transfer",
            "save",
        ]
        if is_rigify:
            assert metarig.hidden
        assert output.read_bytes() == b"bound"
    assert source.read_bytes() == b"original"
