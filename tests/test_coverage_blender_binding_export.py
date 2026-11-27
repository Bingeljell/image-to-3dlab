"""Verify the rig sidecar describes the saved scene and stable joint IDs."""

import hashlib
import sys
from types import SimpleNamespace as NS

import blender_export_rig_binding as exporter
import pytest

from image_to_3dlab.rig_sidecar import FINGERPRINT_PREFIX, load_sidecar

PROFILE = "rigify.basic-quadruped.blender-5.2.v1"


class Properties(dict):
    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class Objects(list):
    def get(self, name):
        return next((obj for obj in self if obj.name == name), None)


@pytest.fixture
def source_rig(monkeypatch, tmp_path):
    profile = exporter.load_rig_profile(PROFILE)
    names = {
        target["bone"]
        for joint in profile["joints"].values()
        for target in joint["targets"]
    }
    bones = {}
    head, tail = (1, 2, 3), (4, 5, 6)
    for name in names:
        bone = Properties()
        bone.head_local, bone.tail_local = head, tail
        bones[name] = bone
    rig = Properties()
    rig.name, rig.type = profile["blenderMetarigNames"][0], "ARMATURE"
    rig.data = NS(bones=bones)
    source = tmp_path / "source.blend"
    source.write_bytes(b"source scene")
    asset = tmp_path / "source.glb"
    asset.write_bytes(b"source asset")
    prepared = tmp_path / "out" / "prepared.blend"
    sidecar = tmp_path / "reports" / "prepared.rig.json"
    calls = []

    def save(*, filepath, check_existing):
        assert filepath == str(prepared) and not check_existing
        assert rig[exporter.OBJECT_ID]
        assert all(bone[exporter.BONE_ID] for bone in bones.values())
        prepared.write_bytes(b"prepared scene")
        calls.append(filepath)

    bpy = NS(
        data=NS(filepath=str(source), objects=Objects([rig])),
        app=NS(version=(5, 2, 1)),
        ops=NS(wm=NS(save_as_mainfile=save)),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "blender",
            "--",
            str(asset),
            str(prepared),
            str(sidecar),
            PROFILE,
        ],
    )
    return bpy, rig, profile, asset, source, prepared, sidecar, calls


@pytest.mark.parametrize("lookup", ["name", "tag"])
def test_rig_binding_exports_persistent_ids_and_exact_fingerprints(source_rig, lookup):
    _, rig, profile, asset, source, prepared, sidecar, calls = source_rig
    if lookup == "tag":
        rig.name = "renamed metarig"
        rig["i2l_adapter"] = PROFILE
        rig[exporter.OBJECT_ID] = "stable"

    assert exporter.main() == 0
    parsed = load_sidecar(sidecar)
    assert parsed["binding"]["metarigObjectId"] == rig[exporter.OBJECT_ID]
    for artifact, fingerprint in (
        (asset, parsed["assetFingerprint"]),
        (prepared, parsed["binding"]["sceneFingerprint"]),
    ):
        assert (
            fingerprint
            == FINGERPRINT_PREFIX + hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
    for ident, joint in profile["joints"].items():
        endpoint = joint["targets"][0]["endpoint"]
        assert parsed["joints"][ident]["position"] == (
            [1, 2, 3] if endpoint == "head" else [4, 5, 6]
        )
        expected = [
            rig.data.bones[target["bone"]][exporter.BONE_ID]
            for target in joint["targets"]
        ]
        assert [
            target["boneId"] for target in parsed["binding"]["joints"][ident]["targets"]
        ] == expected
    assert source.read_bytes() == b"source scene"
    assert calls == [str(prepared)]
    original_ids = [bone[exporter.BONE_ID] for bone in rig.data.bones.values()]
    assert exporter.main() == 0
    assert [bone[exporter.BONE_ID] for bone in rig.data.bones.values()] == original_ids


@pytest.mark.parametrize("failure", ["asset", "version", "metarig", "bone"])
def test_rig_binding_stops_before_saving_when_inputs_are_wrong(source_rig, failure):
    bpy, rig, _, asset, _, prepared, sidecar, calls = source_rig
    if failure == "asset":
        asset.unlink()
        error, message = SystemExit, "source GLB does not exist"
    elif failure == "version":
        bpy.app.version = (4, 5, 0)
        error, message = RuntimeError, "targets Blender"
    elif failure == "metarig":
        rig.name = "not named or tagged"
        error, message = RuntimeError, "expected one metarig"
    else:
        rig.data.bones.pop(next(iter(rig.data.bones)))
        error, message = RuntimeError, "missing adapter bones"
    with pytest.raises(error, match=message):
        exporter.main()
    assert not calls and not prepared.exists() and not sidecar.exists()
