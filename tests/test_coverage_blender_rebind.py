"""Check persistent endpoint binding and export selection."""

from types import SimpleNamespace as NS

import blender_rebind as rebind
import pytest


class SceneObject:
    def __init__(self, name, kind, **custom):
        self.custom = custom
        self.name = name
        self.type = kind
        self.selected = False
        self.hidden = True
        self.hide_select = True
        self.parent = None
        self.modifiers = []

    def get(self, key):
        return self.custom.get(key)

    def select_set(self, value):
        self.selected = value

    def hide_set(self, value):
        self.hidden = value


@pytest.fixture
def skinned_scene():
    metarig = SceneObject(
        "renamed metarig", "ARMATURE", **{rebind.OBJECT_ID: "persistent"}
    )
    bone = SceneObject("renamed bone", "BONE", **{rebind.BONE_ID: "stable-bone"})
    unbound = SceneObject("other bone", "BONE")
    metarig.data = NS(
        bones=[bone, unbound],
        edit_bones={bone.name: NS(head=(0, 0, 0), tail=(0, 1, 0))},
    )
    active = NS(active=None)
    actions = []
    bpy = NS(
        context=NS(object=NS(mode="POSE"), view_layer=NS(objects=active)),
        data=NS(objects=[metarig]),
        ops=NS(
            object=NS(
                mode_set=lambda *, mode: actions.append(mode),
                select_all=lambda *, action: actions.append(action),
            )
        ),
    )
    return bpy, metarig, bone, actions


def test_bone_correction_resolves_ids_after_renames(skinned_scene):
    bpy, rig, bone, actions = skinned_scene
    assert rebind._find_metarig(bpy, "persistent") is rig
    change = NS(
        joint_id="neck",
        target_position=(1, 2, 3),
        targets=[NS(bone_id="stable-bone", endpoint="tail", bone_name="former bone")],
    )
    assert rebind.apply_corrections(bpy, rig, [change]) == [
        {
            "joint": "neck",
            "boneId": "stable-bone",
            "boneName": bone.name,
            "expectedBoneName": "former bone",
            "endpoint": "tail",
            "targetPosition": [1, 2, 3],
        }
    ]
    assert rig.data.edit_bones[bone.name].tail == (1, 2, 3)
    assert rig.data.edit_bones[bone.name].head == (0, 0, 0)
    assert actions == ["OBJECT", "DESELECT", "EDIT", "OBJECT"]
    assert rig.selected and not rig.hidden and not rig.hide_select
    assert bpy.context.view_layer.objects.active is rig


@pytest.mark.parametrize(
    "fault", ["no-target", "unknown-id", "duplicate-id", "no-edit-bone"]
)
def test_correction_validates_ids_and_restores_edit_mode(skinned_scene, fault):
    bpy, rig, bone, actions = skinned_scene
    target = NS(bone_id="stable-bone", endpoint="head", bone_name=bone.name)
    change = NS(joint_id="neck", targets=[target], target_position=(1, 2, 3))
    if fault == "duplicate-id":
        rig.data.bones[1].custom[rebind.BONE_ID] = "stable-bone"
        message = "duplicate persistent bone id"
    elif fault == "no-target":
        change.targets = []
        message = "no Blender binding targets"
    elif fault == "unknown-id":
        target.bone_id = "missing"
        message = "missing from metarig"
    else:
        bpy.context.object = None
        rig.data.edit_bones.clear()
        message = "unavailable in edit mode"
    with pytest.raises(RuntimeError, match=message):
        rebind.apply_corrections(bpy, rig, [change])
    assert actions == (
        ["DESELECT", "EDIT", "OBJECT"] if fault == "no-edit-bone" else []
    )


@pytest.mark.parametrize("copies", [0, 2])
def test_metarig_lookup_rejects_missing_or_ambiguous_ids(skinned_scene, copies):
    bpy, rig, _, _ = skinned_scene
    bpy.data.objects = [rig] * copies
    with pytest.raises(RuntimeError, match=f"found {copies}"):
        rebind._find_metarig(bpy, "persistent")


def test_rebind_exports_parented_and_modified_meshes_only(skinned_scene, tmp_path):
    bpy, rig, _, actions = skinned_scene
    parented = SceneObject("parented", "MESH")
    parented.parent = rig
    modified = SceneObject("modified", "MESH")
    modified.modifiers = [NS(type="ARMATURE", object=rig), NS(type="BEVEL")]
    widget = SceneObject("WGT-editor", "MESH")
    widget.parent = rig
    loose = SceneObject("loose", "MESH")
    bpy.data.objects.extend([widget, parented, modified, loose])
    meshes = rebind._bound_meshes(bpy, rig)
    assert [m.name for m in meshes] == ["parented", "modified"]
    output = tmp_path / "export.glb"

    def export(**options):
        assert options == {
            "filepath": str(output),
            "export_format": "GLB",
            "use_selection": True,
            "export_animations": True,
        }
        assert [o.name for o in bpy.data.objects if o.selected] == [
            rig.name,
            "parented",
            "modified",
        ]
        assert bpy.context.view_layer.objects.active is rig
        output.write_bytes(b"GLB")

    bpy.ops.export_scene = NS(gltf=export)
    rebind.export_glb(bpy, rig, meshes, output)
    assert output.read_bytes() == b"GLB"
    assert actions == ["DESELECT"]
    calls = []
    bpy.ops.export_scene.gltf = lambda **options: calls.append(options["filepath"])
    with pytest.raises(RuntimeError, match="without producing"):
        rebind.export_glb(bpy, rig, meshes, tmp_path / "missing.glb")
    assert calls == [str(tmp_path / "missing.glb")]
    bpy.data.objects = [rig, loose, widget]
    with pytest.raises(RuntimeError, match="no bound mesh"):
        rebind._bound_meshes(bpy, rig)


@pytest.mark.parametrize("target", ["declared", "fallback", "invalid", "missing"])
def test_rigify_clears_uploaded_finalize_script_before_generation(
    skinned_scene, target
):
    bpy, metarig, _, actions = skinned_scene
    rig = SceneObject("generated", "ARMATURE")
    metarig.data.rigify_finalize_script = object()
    bpy.data.objects = {"rig": rig} if target == "fallback" else {}
    if target == "invalid":
        metarig.data.rigify_target_rig = SceneObject("wrong", "CAMERA")

    def enable(*, module):
        assert module == "rigify"
        actions.append("enable")

    def generate():
        assert metarig.data.rigify_finalize_script is None
        assert metarig.selected
        assert bpy.context.view_layer.objects.active is metarig
        if target == "declared":
            metarig.data.rigify_target_rig = rig
        actions.append("generate")

    bpy.ops.preferences = NS(addon_enable=enable)
    bpy.ops.pose = NS(rigify_generate=generate)
    if target in {"missing", "invalid"}:
        with pytest.raises(RuntimeError, match="without a target armature"):
            rebind.regenerate_rigify(bpy, metarig)
    else:
        assert rebind.regenerate_rigify(bpy, metarig) is rig
    assert actions == ["enable", "DESELECT", "generate"]


@pytest.mark.parametrize("args, error", [([], "usage:"), (["--", "wrong.obj", "b.rig.json", "c.glb", "d.blend", "e.rig.json", "f.json"], "must end with .glb")])
def test_rebind_entrypoint_rejects_invalid_paths(monkeypatch, args, error):
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["blender", *args])
    with pytest.raises(SystemExit, match=error):
        runpy.run_path(rebind.__file__, run_name="__main__")


@pytest.mark.parametrize("scenario", ["missing", "empty", "apply"])
def test_rebind_reports_corrected_rig_and_output_fingerprints(
    skinned_scene, tmp_path, monkeypatch, capsys, scenario
):
    import json
    import sys

    from test_rig_sidecar import sidecar_for, with_binding

    from image_to_3dlab.rig_sidecar import asset_fingerprint

    bpy, metarig, bone, _ = skinned_scene
    source, opened, sidecar, glb, blend, refreshed, report = (
        tmp_path / path for path in (
            "asset.glb", "opened.blend", "fit.rig.json", "output/rig.glb",
            "output/rig.blend", "output/rig.rig.json", "output/report.json",
        )
    )
    opened.write_bytes(b"original scene")
    source.write_bytes(b"original asset")
    document = with_binding(sidecar_for(source.read_bytes()), opened.read_bytes())
    document["binding"]["metarigObjectId"] = metarig.custom[rebind.OBJECT_ID]
    document["binding"]["joints"]["shoulder.L"]["targets"][0]["boneId"] = bone.custom[rebind.BONE_ID]
    if scenario == "empty":
        document["corrections"] = {}
    if scenario != "missing":
        sidecar.write_text(json.dumps(document), encoding="utf-8")

    generated = SceneObject("generated rig", "ARMATURE")
    metarig.data.rigify_target_rig = generated
    mesh = SceneObject("body", "MESH")
    mesh.parent = generated
    bpy.data.objects.extend([generated, mesh])
    bpy.data.filepath = str(opened)
    bpy.ops.preferences = NS(addon_enable=lambda *, module: None)
    bpy.ops.pose = NS(rigify_generate=lambda: None)

    def bind(_bpy, target, rig):
        assert target is mesh and rig is generated
        return {"vertices": 17, "transferred": 17}

    def save(*, filepath, check_existing):
        assert not check_existing
        assert filepath == str(blend)
        blend.write_bytes(b"updated scene")

    def export(*, filepath, **options):
        assert options == dict(export_format="GLB", use_selection=True, export_animations=True)
        assert mesh.selected and generated.selected and not metarig.hide_select
        assert filepath == str(glb)
        glb.write_bytes(b"updated model")

    bpy.ops.wm = NS(save_as_mainfile=save)
    bpy.ops.export_scene = NS(gltf=export)
    monkeypatch.setattr(rebind, "transfer_weights", bind)
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(sys, "argv", [
        "blender", "--", *map(str, (source, sidecar, glb, blend, refreshed, report)),
    ])
    if scenario == "missing":
        with pytest.raises(SystemExit, match="required input does not exist"):
            rebind.main()
        assert not glb.parent.exists()
        return
    if scenario == "empty":
        with pytest.raises(RuntimeError, match="no fit-joint corrections"):
            rebind.main()
        assert not glb.exists() and not blend.exists()
        return

    assert rebind.main() == 0
    assert metarig.data.edit_bones[bone.name].head == (0.25, 1.1, 0)
    result = json.loads(report.read_text())
    assert result["weightStrategy"] == "voxel-proxy-transfer"
    assert result["correctionsApplied"] == 1 and result["targetsApplied"][0]["boneName"] == bone.name
    assert result["meshes"] == ["body"] and result["weights"] == [{"vertices": 17, "transferred": 17}]
    new_sidecar = json.loads(refreshed.read_text())
    assert new_sidecar["corrections"] == {}
    assert new_sidecar["assetFingerprint"] == asset_fingerprint(glb)
    assert new_sidecar["binding"]["sceneFingerprint"] == asset_fingerprint(blend)
    assert "I2L_REBIND::" in capsys.readouterr().out
