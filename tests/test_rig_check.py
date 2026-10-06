"""rig_check: does a rigged GLB's skeleton fit the preset moves (a humanoid), read without Blender."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

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
