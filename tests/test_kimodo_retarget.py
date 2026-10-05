"""Tests for `scripts/kimodo_retarget.py`: tree-shape bone mapping and the pose maths.

The skeleton here is a synthetic humanoid built in the same tree shape SkinTokens uses
(root -> spine -> chest forking into neck + 2 arms, 5 fingers per hand, 2 legs), with
bone names shuffled so nothing can lean on `bone_N` numbering.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kimodo_retarget as kr


def _humanoid(face=+1):
    """Z-up T-pose humanoid facing `face`*Y. Returns (parents, heads, role->name)."""
    roles, P = {}, {}
    rng = np.random.default_rng(0)
    names = iter(f"b{n}" for n in rng.permutation(200))

    def add(role, parent_role, pos):
        roles[role] = next(names)
        P[role] = (parent_role, np.array(pos, float))

    fy = face
    add("hips", None, (0, 0, 1.0))
    add("s1", "hips", (0, 0, 1.1)); add("s2", "s1", (0, 0, 1.2)); add("chest", "s2", (0, 0, 1.35))
    add("neck", "chest", (0, 0, 1.5)); add("head", "neck", (0, 0, 1.6))
    for lr, sx in (("L", -fy), ("R", fy)):  # facing +Y: character's left is -X
        add(f"{lr}sh", "chest", (0.05 * sx, 0, 1.45)); add(f"{lr}arm", f"{lr}sh", (0.15 * sx, 0, 1.45))
        add(f"{lr}fa", f"{lr}arm", (0.4 * sx, 0, 1.45)); add(f"{lr}hand", f"{lr}fa", (0.65 * sx, 0, 1.45))
        add(f"{lr}thumb0", f"{lr}hand", (0.68 * sx, 0.03 * fy, 1.43))
        add(f"{lr}thumb1", f"{lr}thumb0", (0.70 * sx, 0.05 * fy, 1.42))
        add(f"{lr}thumb2", f"{lr}thumb1", (0.72 * sx, 0.06 * fy, 1.41))
        for k, fname in enumerate(("index", "middle", "ring", "pinky")):
            y = (0.03 - 0.02 * k) * fy
            add(f"{lr}{fname}0", f"{lr}hand", (0.72 * sx, y, 1.45))
            add(f"{lr}{fname}1", f"{lr}{fname}0", (0.75 * sx, y, 1.45))
            add(f"{lr}{fname}2", f"{lr}{fname}1", (0.78 * sx, y, 1.45))
        add(f"{lr}leg", "hips", (0.1 * sx, 0, 0.95)); add(f"{lr}shin", f"{lr}leg", (0.1 * sx, 0, 0.5))
        add(f"{lr}foot", f"{lr}shin", (0.1 * sx, 0, 0.08)); add(f"{lr}toe", f"{lr}foot", (0.1 * sx, 0.12 * fy, 0.02))
    parents = {roles[r]: (roles[p] if p else None) for r, (p, _) in P.items()}
    heads = {roles[r]: pos for r, (_, pos) in P.items()}
    return parents, heads, roles


@pytest.mark.parametrize("face", [+1, -1])
def test_mapping_finds_every_role_by_tree_shape(face):
    parents, heads, roles = _humanoid(face)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    assert len(m) == len(parents)
    expect = {
        "hips": "Hips", "chest": "Chest", "head": "Head", "neck": "Neck2",  # one neck bone takes the upper joint: its world turn includes Neck1
        "Lhand": "LeftHand", "Rhand": "RightHand", "Lsh": "LeftShoulder", "Rfa": "RightForeArm",
        "Lthumb0": "LeftHandThumb1", "Lthumb2": "LeftHandThumb3",
        "Lindex0": "LeftHandIndex2", "Lpinky2": "LeftHandPinky4", "Rmiddle1": "RightHandMiddle3",
        "Lleg": "LeftLeg", "Rtoe": "RightToeBase", "Lfoot": "LeftFoot",
    }
    for role, soma in expect.items():
        assert m[roles[role]] == soma, role
    assert np.allclose(frame[:, 2], [0, face, 0])  # forward found from the feet


def test_every_mapped_name_is_a_soma_joint():
    parents, heads, _ = _humanoid()
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    assert set(m.values()) <= set(kr.SOMA77)


def test_soma_list_shape():
    assert len(kr.SOMA77) == 77 and kr.SOMA77[0] == "Hips"


def _rest_mats(parents, heads):
    rest = {}
    for b, h in heads.items():
        m = np.eye(4)
        m[:3, 3] = h
        rest[b] = m
    order = [b for b, p in parents.items() if p is None]
    i = 0
    while i < len(order):
        order += [c for c, p in parents.items() if p == order[i]]
        i += 1
    return rest, order


def test_identity_motion_gives_identity_basis():
    parents, heads, _ = _humanoid()
    rest, order = _rest_mats(parents, heads)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    g = np.tile(np.eye(3), (2, 77, 1, 1))
    hips = np.zeros((2, 3))
    for basis in kr.retarget_frames(rest, parents, order, m, kr.soma_to_rig_axes(frame), g, hips, 1.0):
        for mtx in basis.values():
            assert np.allclose(mtx, np.eye(4), atol=1e-9)


def test_raised_left_arm_points_up_on_the_rig():
    """Rotate SOMA's LeftArm +90 deg about forward (Z): a T-pose left arm (+X) swings to +Y (up)."""
    parents, heads, roles = _humanoid(face=+1)
    rest, order = _rest_mats(parents, heads)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    axes = kr.soma_to_rig_axes(frame)
    rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # +90 about Z: X -> Y
    g = np.tile(np.eye(3), (1, 77, 1, 1))
    for name in kr.SOMA77:  # the whole left arm below the shoulder turns together
        if name.startswith(("LeftArm", "LeftForeArm", "LeftHand")):
            g[0, kr.SOMA_INDEX[name]] = rz
    basis = next(kr.retarget_frames(rest, parents, order, m, axes, g, np.zeros((1, 3)), 1.0))
    # Rebuild world positions with Blender's rule and check the hand is now above the shoulder.
    pose = {}
    for b in order:
        p = parents[b]
        rel = rest[b] if p is None else np.linalg.inv(rest[p]) @ rest[b]
        pose[b] = (np.eye(4) if p is None else pose[p]) @ rel @ basis[b]
    arm, hand = pose[roles["Larm"]][:3, 3], pose[roles["Lhand"]][:3, 3]
    assert hand[2] - arm[2] == pytest.approx(0.5, abs=1e-6)  # arm length 0.5, now vertical
    assert abs(hand[0] - arm[0]) < 1e-6


def test_hips_travel_scales_and_maps_axes():
    parents, heads, roles = _humanoid(face=-1)  # rig faces -Y
    rest, order = _rest_mats(parents, heads)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    g = np.tile(np.eye(3), (2, 77, 1, 1))
    hips = np.array([[0, 0.9, 0], [0, 0.9, 2.0]])  # SOMA walks 2 m forward (+Z)
    frames = list(kr.retarget_frames(rest, parents, order, m, kr.soma_to_rig_axes(frame), g, hips, 0.5))
    moved = frames[1][roles["hips"]][:3, 3]
    assert np.allclose(moved, [0, -1.0, 0], atol=1e-9)  # 2 m * 0.5, along the rig's forward (-Y)


def test_soma_hip_height():
    x = np.zeros((1, 77, 3))
    x[0, kr.SOMA_INDEX["Hips"], 1] = 0.95
    x[0, kr.SOMA_INDEX["LeftToeBase"], 1] = -0.02
    assert kr.soma_hip_height(x) == pytest.approx(0.97)


def test_rotation_between_including_opposite():
    for a, b in [([1, 0, 0], [0, 1, 0]), ([0, 0, 1], [0, 0, -1]), ([1, 2, 3], [-2, 0.5, 1])]:
        a, b = np.array(a, float), np.array(b, float)
        r = kr.rotation_between(a, b)
        assert np.allclose(r @ a / np.linalg.norm(a), b / np.linalg.norm(b), atol=1e-9)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)


def test_soma_tpose_recovered_from_a_posed_frame():
    rng = np.random.default_rng(1)
    n = rng.normal(size=(77, 3)) * 0.1
    n[0] = 0
    g = np.empty((1, 77, 3, 3))
    for j in range(77):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        g[0, j] = q * np.sign(np.linalg.det(q))
    x = np.zeros((1, 77, 3))
    x[0, 0] = [0.3, 0.9, 2.0]
    for j, p in enumerate(kr.SOMA77_PARENTS):
        if p >= 0:
            x[0, j] = x[0, p] + g[0, p] @ (n[j] - n[p])
    assert np.allclose(kr.soma_tpose(g, x), n, atol=1e-9)


def _soma_like_tpose():
    """A T-pose in SOMA axes (X left, Y up, Z forward): straight legs, level arms."""
    n = np.zeros((77, 3))
    for j, p in enumerate(kr.SOMA77_PARENTS):
        if p < 0:
            continue
        name = kr.SOMA77[j]
        side = 1.0 if name.startswith("Left") else -1.0
        if any(k in name for k in ("Leg", "Shin", "Foot")):
            step = [0.1 * side, 0, 0] if name.endswith("Leg") else [0, -0.4, 0]
        elif "Toe" in name:
            step = [0, -0.05, 0.1]
        elif name.startswith(("Left", "Right")):
            step = [0.1 * side, 0, 0]
        else:
            step = [0, 0.1, 0]
        n[j] = n[p] + step
    return n


def test_alignment_straightens_a_splayed_leg():
    parents, heads, roles = _humanoid(face=+1)
    # Splay the left shin outward 20 degrees (character's left is -X when facing +Y).
    knee, ankle = heads[roles["Lshin"]], heads[roles["Lfoot"]]
    d = ankle - knee
    ang = np.radians(20)
    heads[roles["Lfoot"]] = knee + np.array([-np.sin(ang), 0, -np.cos(ang)]) * np.linalg.norm(d)
    heads[roles["Ltoe"]] = heads[roles["Lfoot"]] + np.array([0, 0.12, -0.06])
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    axes = kr.soma_to_rig_axes(frame)
    align = kr.rest_alignment(parents, heads, m, axes, _soma_like_tpose())
    shin = heads[roles["Lfoot"]] - heads[roles["Lshin"]]
    fixed = align[roles["Lshin"]] @ shin
    assert np.allclose(fixed / np.linalg.norm(fixed), [0, 0, -1], atol=1e-9)  # straight down


def test_end_bones_turn_with_their_parent_so_no_joint_is_kinked():
    """A toe that kept its rest while the foot was swung put a permanent bend at the ball
    of the foot: the odd toe deformation seen on the knight (foot swung 22 degrees)."""
    parents, heads, roles = _humanoid(face=+1)
    heads[roles["Ltoe"]] = heads[roles["Lfoot"]] + np.array([-0.05, 0.08, -0.06])  # toe turned out
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    axes = kr.soma_to_rig_axes(frame)
    align = kr.rest_alignment(parents, heads, m, axes, _soma_like_tpose())
    foot, toe = roles["Lfoot"], roles["Ltoe"]
    assert not np.allclose(align[foot], np.eye(3))  # the foot really was swung
    assert np.allclose(align[toe], align[foot])
    assert np.allclose(align[roles["head"]], align[roles["neck"]])
    assert np.allclose(align[roles["Lindex2"]], align[roles["Lindex1"]])


def test_arm_spread_only_while_hanging():
    frame = np.stack([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]], axis=1)  # left -X, up Z, fwd Y
    level = frame[:, 0]  # left arm rest direction in rig space
    hanging = kr.rotation_between(level, -frame[:, 1])  # arm turned to point down
    r = kr.arm_spread("LeftForeArm", hanging, level, frame, 10.0)
    swung = r @ (-frame[:, 1])
    assert swung @ frame[:, 0] > 0.17  # moved toward the character's left (sin 10deg)
    raised = kr.rotation_between(level, frame[:, 1])
    assert np.allclose(kr.arm_spread("LeftForeArm", raised, level, frame, 10.0), np.eye(3))
    assert np.allclose(kr.arm_spread("Head", hanging, level, frame, 10.0), np.eye(3))
