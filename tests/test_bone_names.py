"""image_to_3dlab.bone_names: engine bone names for a rigged humanoid, and at most four weights a vertex.

The skeletons are the synthetic humanoids `tests/test_kimodo_retarget.py` already builds in SkinTokens' tree
shape (shuffled `bN` names), so the names come from the real body mapper.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import kimodo_retarget as kr
from test_kimodo_retarget import _humanoid, _with_extras

from image_to_3dlab import bone_names as bn

GLB, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


def _names(parents, heads, preset="unity"):
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    return bn.engine_names(parents, mapping, preset)


def test_every_body_bone_gets_its_unity_humanoid_name():
    parents, heads, roles = _humanoid()
    names = _names(parents, heads)
    expect = {
        "hips": "Hips", "s1": "Spine", "s2": "Chest", "chest": "UpperChest", "neck": "Neck", "head": "Head",
        "Lsh": "LeftShoulder", "Larm": "LeftUpperArm", "Lfa": "LeftLowerArm", "Lhand": "LeftHand",
        "Rsh": "RightShoulder", "Rarm": "RightUpperArm", "Rfa": "RightLowerArm", "Rhand": "RightHand",
        "Lleg": "LeftUpperLeg", "Lshin": "LeftLowerLeg", "Lfoot": "LeftFoot", "Ltoe": "LeftToes",
        "Rleg": "RightUpperLeg", "Rshin": "RightLowerLeg", "Rfoot": "RightFoot", "Rtoe": "RightToes",
        "Lthumb0": "LeftThumbProximal", "Lthumb2": "LeftThumbDistal",
        "Lindex0": "LeftIndexProximal", "Rmiddle1": "RightMiddleIntermediate", "Lpinky2": "LeftLittleDistal",
    }
    for role, name in expect.items():
        assert names[roles[role]] == name, role


def test_no_name_is_given_twice():
    parents, heads, _ = _humanoid()
    names = _names(parents, heads)
    assert len(set(names.values())) == len(names)


def test_extra_bones_keep_their_own_names():
    # Hair, a cape, a skirt, a sword: not body parts, so no engine slot of their own.
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    names = _names(parents, heads)
    for extra in ("hair0", "hair1", "cape0", "cape2", "skirtF", "skirtB1", "strap", "sword", "blade"):
        assert extra not in names
    assert names[roles["hips"]] == "Hips" and names[roles["Rfa"]] == "RightLowerArm"


def test_mitten_hands_name_the_arm_and_leave_the_fingers_alone():
    # SkinTokens leaves out fingers it cannot see; four look-alike chains cannot be told apart
    # as index, middle, ring and little, so none of them is named.
    parents, heads, roles = _humanoid(fingers=False)
    for lr, sx in (("L", -1), ("R", 1)):
        for k in range(4):
            parent = roles[f"{lr}hand"]
            for j in range(2):
                bone = f"{lr}mit{k}_{j}"
                parents[bone] = parent
                heads[bone] = np.array((sx * (0.72 + 0.03 * j), 0.03 - 0.02 * k, 1.45))
                parent = bone
    names = _names(parents, heads)
    assert names[roles["Lhand"]] == "LeftHand" and names[roles["Rhand"]] == "RightHand"
    assert not any(b.startswith(("Lmit", "Rmit")) for b in names)


def test_a_name_a_kept_bone_already_has_is_not_given_again():
    # An uploaded rig may already call its sword bone "Head"; two bones of one name would confuse an engine.
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    parents["Head"] = parents.pop("sword")
    parents["blade"] = "Head"
    heads["Head"] = heads.pop("sword")
    names = _names(parents, heads)
    assert roles["head"] not in names and "Head" not in names.values()
    assert names[roles["neck"]] == "Neck"


def test_an_unknown_preset_is_refused():
    parents, heads, _ = _humanoid()
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    with pytest.raises(ValueError, match="unity"):
        bn.engine_names(parents, mapping, "maya")


def _glb(doc: dict, binary: bytes = b"\x01\x02\x03\x04") -> bytes:
    chunk = json.dumps(doc).encode()
    chunk += b" " * (-len(chunk) % 4)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<II", len(chunk), JSON_CHUNK) + chunk + struct.pack("<II", len(binary), BIN_CHUNK) + binary
    return struct.pack("<III", GLB, 2, 12 + len(body)) + body


def _parse(glb: bytes):
    json_len = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20:20 + json_len])
    bin_len = struct.unpack_from("<I", glb, 20 + json_len)[0]
    return doc, glb[28 + json_len:28 + json_len + bin_len]


def test_rename_changes_only_the_joint_names_and_keeps_the_binary():
    doc = {"nodes": [{"name": "Body", "mesh": 0}, {"name": "bone_0", "children": [2]}, {"name": "bone_1"}],
           "skins": [{"joints": [1, 2]}], "animations": [{"channels": [{"target": {"node": 2}}]}]}
    out = bn.rename(_glb(doc), {"bone_0": "Hips", "bone_1": "Spine", "Body": "Hips2"})
    magic, version, length = struct.unpack_from("<III", out, 0)
    assert (magic, version, length) == (GLB, 2, len(out))
    assert struct.unpack_from("<I", out, 12)[0] % 4 == 0
    new, binary = _parse(out)
    assert [n["name"] for n in new["nodes"]] == ["Body", "Hips", "Spine"]  # the mesh node is not a joint
    assert new["nodes"][1]["children"] == [2]       # the tree is untouched: extras stay where they hang
    assert new["animations"] == doc["animations"]  # channels point at node indices, which do not move
    assert binary == b"\x01\x02\x03\x04"


def test_rename_refuses_a_file_that_is_not_a_glb():
    with pytest.raises(ValueError):
        bn.rename(b"not a glb at all", {})


def _skinned(joint_sets, weight_sets, weight_type=5126):
    """A GLB whose one primitive carries the given JOINTS_n / WEIGHTS_n (lists of per-vertex 4-tuples)."""
    blob, views, accessors, attributes = b"", [], [], {}

    def add(array, component, kind, name, normalized=False):
        nonlocal blob
        data = np.asarray(array, dtype={5121: "<u1", 5123: "<u2", 5126: "<f4"}[component]).tobytes()
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob += data + b"\0" * (-len(data) % 4)
        accessor = {"bufferView": len(views) - 1, "componentType": component, "count": len(array), "type": kind}
        if normalized:
            accessor["normalized"] = True
        accessors.append(accessor)
        attributes[name] = len(accessors) - 1

    add([[0, 0, 0]] * len(joint_sets[0]), 5126, "VEC3", "POSITION")
    for n, (joints, weights) in enumerate(zip(joint_sets, weight_sets)):
        add(joints, 5121, "VEC4", f"JOINTS_{n}")
        add(weights, weight_type, "VEC4", f"WEIGHTS_{n}", normalized=weight_type != 5126)
    doc = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
           "accessors": accessors, "meshes": [{"primitives": [{"attributes": attributes}]}]}
    return _glb(doc, blob)


def _read(glb, name):
    doc, binary = _parse(glb)
    accessor = doc["accessors"][doc["meshes"][0]["primitives"][0]["attributes"][name]]
    view = doc["bufferViews"][accessor["bufferView"]]
    dtype = {5121: "<u1", 5123: "<u2", 5126: "<f4"}[accessor["componentType"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    values = np.frombuffer(binary, dtype=dtype, count=accessor["count"] * 4, offset=start).reshape(-1, 4)
    if accessor.get("normalized"):
        values = values / np.iinfo(dtype).max
    return values


def test_eight_weights_become_the_four_largest_summing_to_one():
    glb = _skinned([[[1, 2, 3, 4]], [[5, 6, 7, 8]]],
                   [[[0.05, 0.30, 0.02, 0.10]], [[0.20, 0.03, 0.25, 0.05]]])
    out, capped = bn.limit_influences(glb)
    assert capped == 1
    doc, _ = _parse(out)
    attributes = doc["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_1" not in attributes and "WEIGHTS_1" not in attributes
    joints, weights = _read(out, "JOINTS_0")[0], _read(out, "WEIGHTS_0")[0]
    kept = dict(zip(joints.tolist(), weights.tolist()))
    assert set(kept) == {2, 7, 5, 4}   # 0.30, 0.25, 0.20, 0.10
    assert sum(kept.values()) == pytest.approx(1.0, abs=1e-6)
    assert kept[2] / kept[4] == pytest.approx(3.0, rel=1e-5)


def test_normalised_byte_weights_are_capped_and_still_sum_to_one():
    glb = _skinned([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], [[[100, 20, 10, 5]], [[60, 40, 15, 5]]], weight_type=5121)
    out, capped = bn.limit_influences(glb)
    weights = _read(out, "WEIGHTS_0")[0]
    assert capped == 1 and weights.sum() == pytest.approx(1.0, abs=1 / 255)
    assert dict(zip(_read(out, "JOINTS_0")[0].tolist(), weights.tolist())).keys() == {1, 5, 6, 2}


def test_four_weights_or_fewer_leave_the_file_untouched():
    glb = _skinned([[[1, 2, 0, 0], [3, 0, 0, 0]]], [[[0.5, 0.5, 0, 0], [1, 0, 0, 0]]])
    out, capped = bn.limit_influences(glb)
    assert capped == 0 and out == glb


def _rigged(tmp_path, parents, heads, joint_sets=None, weight_sets=None):
    """A GLB file of the skeleton (Z-up heads turned to glTF's Y-up) plus, optionally, one skinned primitive."""
    order = list(parents)
    index = {bone: i for i, bone in enumerate(order)}
    gltf = {b: np.array((p[0], p[2], -p[1])) for b, p in heads.items()}
    nodes = []
    for bone in order:
        parent = parents[bone]
        offset = gltf[bone] - (gltf[parent] if parent else 0)
        nodes.append({"name": bone, "translation": [float(v) for v in offset]})
    for bone in order:
        if parents[bone]:
            nodes[index[parents[bone]]].setdefault("children", []).append(index[bone])
    base = _parse(_skinned(joint_sets, weight_sets)) if joint_sets else ({}, b"")
    doc = {**base[0], "asset": {"version": "2.0"}, "nodes": nodes,
           "skins": [{"joints": list(range(len(order)))}],
           "scenes": [{"nodes": [index[b] for b in order if parents[b] is None]}], "scene": 0}
    path = tmp_path / "rigged.glb"
    path.write_bytes(_glb(doc, base[1]) if joint_sets else _glb(doc))
    return path


def test_name_bones_writes_the_names_and_the_cap_into_the_file(tmp_path):
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    path = _rigged(tmp_path, parents, heads, [[[1, 2, 3, 4]], [[5, 6, 7, 8]]],
                   [[[0.05, 0.30, 0.02, 0.10]], [[0.20, 0.03, 0.25, 0.05]]])
    report = bn.name_bones(path, "unity")
    doc, _ = _parse(path.read_bytes())
    names = [n["name"] for n in doc["nodes"]]
    assert {"Hips", "Spine", "Head", "LeftHand", "RightToes"} <= set(names)
    assert "hair0" in names and "sword" in names
    assert report["preset"] == "unity" and report["capped"] == 1 and report["missing"] == []
    assert report["named"] + len(report["kept"]) == report["bones"] == len(parents)
    hair = names.index("hair0")
    assert any(hair in n.get("children", []) for n in doc["nodes"])   # still hangs from its body bone


def test_name_bones_says_which_required_bones_it_could_not_name(tmp_path):
    # A kept bone already called "Head" (an uploaded rig's sword): the head goes without, and the report says so.
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    parents["Head"] = parents.pop("sword")
    parents["blade"] = "Head"
    heads["Head"] = heads.pop("sword")
    report = bn.name_bones(_rigged(tmp_path, parents, heads), "unity")
    assert report["missing"] == ["Head"]


def test_a_rig_without_a_neck_still_has_all_unity_needs(tmp_path):
    # SkinTokens sometimes puts the head straight on the chest (a Phantom seed did); Unity's neck is optional.
    parents, heads, roles = _humanoid()
    neck, head = roles["neck"], roles["head"]
    parents[head] = parents.pop(neck)
    del heads[neck]
    path = _rigged(tmp_path, parents, heads)
    report = bn.name_bones(path, "unity")
    doc, _ = _parse(path.read_bytes())
    assert report["missing"] == [] and "Head" in [n["name"] for n in doc["nodes"]]
    assert "Neck" not in [n["name"] for n in doc["nodes"]]


def test_a_skeleton_with_no_humanoid_body_is_refused_and_left_as_it_is(tmp_path):
    # Rigged anyway, say a snake: the job keeps the numbered names rather than inventing a body.
    parents, heads, roles = _humanoid()
    for bone in [roles[r] for r in roles if r[1:] in ("leg", "shin", "foot", "toe")]:
        del parents[bone], heads[bone]
    path = _rigged(tmp_path, parents, heads)
    before = path.read_bytes()
    with pytest.raises(LookupError):
        bn.name_bones(path, "unity")
    assert path.read_bytes() == before


def test_name_bones_refuses_an_unknown_preset_before_touching_the_file(tmp_path):
    parents, heads, _ = _humanoid()
    path = _rigged(tmp_path, parents, heads)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        bn.name_bones(path, "maya")
    assert path.read_bytes() == before


def test_describe_says_it_in_one_line():
    line = bn.describe({"preset": "unity", "named": 52, "bones": 60, "kept": ["hair0"], "capped": 0,
                        "missing": []})
    assert "Unity Humanoid" in line and "52 of 60" in line
    assert "missing" in bn.describe({"preset": "unity", "named": 3, "bones": 9, "kept": [], "capped": 0,
                                     "missing": ["LeftFoot"]})
