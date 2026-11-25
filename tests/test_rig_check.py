"""rig_check: does a rigged GLB's skeleton fit the preset moves (a humanoid), read without Blender."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rig_check as rc


def _glb(path: Path, nodes: list[dict], joints: list[int], roots: list[int]) -> Path:
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": roots}], "nodes": nodes,
           "skins": [{"joints": joints}]}
    blob = json.dumps(doc).encode()
    blob += b" " * (-len(blob) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(blob))
                     + struct.pack("<II", len(blob), 0x4E4F534A) + blob)
    return path


def _skeleton(legs: int):
    """A Y-up figure in glTF's local translations: hips, a spine to a chest that forks into a
    neck and two 4-bone arms, and `legs` 4-bone legs hanging down from the hips."""
    nodes: list[dict] = []

    def add(t, parent=None):
        nodes.append({"name": f"b{len(nodes)}", "translation": t})
        if parent is not None:
            nodes[parent].setdefault("children", []).append(len(nodes) - 1)
        return len(nodes) - 1

    hips = add([0, 1.0, 0])
    spine = add([0, 0.15, 0], hips)
    chest = add([0, 0.2, 0], spine)
    neck = add([0, 0.15, 0], chest)
    add([0, 0.1, 0], neck)
    for side in (1, -1):
        b = add([0.08 * side, 0.1, 0], chest)
        for _ in range(3):
            b = add([0.2 * side, 0, 0], b)
    for i in range(legs):
        x = [0.1, -0.1, 0.1, -0.1][i]
        z = [0, 0, 0.4, 0.4][i] if legs == 4 else 0
        b = add([x, -0.05, z], hips)
        b = add([0, -0.45, 0], b)
        b = add([0, -0.45, 0], b)
        add([0, -0.04, 0.12], b)
    return nodes


def test_a_humanoid_skeleton_fits_the_preset_moves(tmp_path):
    nodes = _skeleton(legs=2)
    out = rc.check(_glb(tmp_path / "h.glb", nodes, list(range(len(nodes))), [0]))
    assert out["humanoid"] is True and out["bones"] == len(nodes)
    assert out["forward"] == [0, 0, 1]  # the toes point along +Z


def test_a_four_legged_skeleton_is_rigged_but_not_for_preset_moves(tmp_path):
    nodes = _skeleton(legs=4)
    out = rc.check(_glb(tmp_path / "q.glb", nodes, list(range(len(nodes))), [0]))
    assert out["humanoid"] is False
    assert "humanoids only" in out["message"] and "Blender" in out["message"]


def test_world_positions_follow_the_parents_and_an_armature_above_them(tmp_path):
    # Blender exports an armature node above the root joint; its offset must be carried down
    nodes = [{"name": "Armature", "translation": [0, 0, 5], "children": [1]}] + [
        {**n, "children": [c + 1 for c in n.get("children", [])]} for n in _skeleton(legs=2)]
    path = _glb(tmp_path / "a.glb", nodes, list(range(1, len(nodes))), [0])
    parents, heads = rc.skeleton(path)
    assert parents["b0"] is None and parents["b1"] == "b0"
    assert heads["b0"].tolist() == [0, 1.0, 5]


def test_no_skin_means_no_rig(tmp_path):
    path = tmp_path / "n.glb"
    _glb(path, [{"name": "mesh"}], [], [0])
    doc = rc._gltf_json(path)
    doc["skins"] = []
    blob = json.dumps(doc).encode()
    blob += b" " * (-len(blob) % 4)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 20 + len(blob)) + struct.pack("<II", len(blob), 0x4E4F534A) + blob)
    out = rc.check(path)
    assert out["humanoid"] is False and "no skeleton" in out["message"]


def test_the_rig_job_asks_the_check_and_a_broken_file_never_fails_the_rig(tmp_path):
    from viewer.autorig_api import rig_fit
    nodes = _skeleton(legs=4)
    assert rig_fit(_glb(tmp_path / "q.glb", nodes, list(range(len(nodes))), [0]))["humanoid"] is False
    (tmp_path / "bad.glb").write_bytes(b"nope")
    assert rig_fit(tmp_path / "bad.glb") == {}


# --- does the model itself stand on two legs (the auto-rig's second try, TODO 194 of the support notes) -----------

def _box(lo, hi, n=600, seed=0):
    """Points spread through a box from corner `lo` to corner `hi` (x, y, z; Y up)."""
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 3))


def _biped(feet_apart=0.12):
    legs = [_box([-feet_apart / 2 - 0.12, 0, -0.08], [-feet_apart / 2, 0.85, 0.14], seed=1),
            _box([feet_apart / 2, 0, -0.08], [feet_apart / 2 + 0.12, 0.85, 0.14], seed=2)]
    body = _box([-0.25, 0.85, -0.12], [0.25, 1.5, 0.12], seed=3)
    arms = _box([-0.85, 1.35, -0.05], [0.85, 1.45, 0.05], seed=4)   # a T-pose: as wide as tall
    head = _box([-0.1, 1.5, -0.1], [0.1, 1.75, 0.1], seed=5)
    return np.vstack(legs + [body, arms, head])


def test_a_figure_on_two_legs_is_seen_from_its_mesh():
    out = rc.stands_on_two_legs(_biped())
    assert out["two_legs"] is True and out["feet"] == 2


def test_knuckles_touching_the_floor_beside_two_feet_still_read_as_two_legs():
    knuckles = [_box([x, 0, -0.03], [x + 0.06, 0.15, 0.03], n=40, seed=20 + i) for i, x in enumerate((-0.6, 0.54))]
    out = rc.stands_on_two_legs(np.vstack([_biped()] + knuckles))   # a golem's long arms reach the ground
    assert out["two_legs"] is True and out["feet"] == 4


def test_a_robe_to_the_floor_a_crate_and_feet_together_are_not_two_legs():
    robe = np.vstack([_box([-0.3, 0, -0.25], [0.3, 1.5, 0.25], n=2000), _box([-0.1, 1.5, -0.1], [0.1, 1.75, 0.1])])
    assert rc.stands_on_two_legs(robe)["two_legs"] is False
    assert rc.stands_on_two_legs(_box([0, 0, 0], [1, 1, 1], n=2000))["two_legs"] is False
    assert rc.stands_on_two_legs(_biped(feet_apart=0.0))["two_legs"] is False   # one block: no retry, as before


def test_a_four_legged_creature_is_not_two_legs():
    legs = [_box([x, 0, z], [x + 0.1, 0.4, z + 0.1], seed=i)
            for i, (x, z) in enumerate([(-0.2, -0.5), (0.1, -0.5), (-0.2, 0.4), (0.1, 0.4)])]
    body = _box([-0.2, 0.4, -0.55], [0.2, 0.7, 0.55], n=1500, seed=9)
    assert rc.stands_on_two_legs(np.vstack(legs + [body]))["feet"] == 4
    # its legs fused side by side along the body: two long parts, still not a biped's feet
    sides = [_box([-0.2, 0, -0.5], [-0.1, 0.4, 0.5], seed=11), _box([0.1, 0, -0.5], [0.2, 0.4, 0.5], seed=12)]
    assert rc.stands_on_two_legs(np.vstack(sides + [body]))["two_legs"] is False


def test_the_points_come_from_every_mesh_in_the_glb_with_its_node_moved(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3], [0, 1, 0]], np.float32)
    blob = pts.tobytes()
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0, "translation": [0, 5, 0]}],
           "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
           "bufferViews": [{"buffer": 0, "byteLength": len(blob)}], "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    path = tmp_path / "m.glb"
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
                     + struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob)
    assert rc.mesh_points(path).tolist() == [[0, 5, 0], [1, 7, 3], [0, 6, 0]]
