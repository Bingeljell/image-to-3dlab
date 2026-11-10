"""Tests for `scripts/kimodo_retarget.py`: tree-shape bone mapping and the pose maths.

The skeleton here is a synthetic humanoid built in the same tree shape SkinTokens uses
(root -> spine -> chest forking into neck + 2 arms, 5 fingers per hand, 2 legs), with
bone names shuffled so nothing can lean on `bone_N` numbering.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kimodo_retarget as kr


def _humanoid(face=+1, fingers=True):
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
        if fingers:
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


def test_a_hand_turns_with_its_forearm_so_the_wrist_keeps_its_rest_bend():
    """An A-pose forearm is swung up to SOMA's T-pose; the hand forks into fingers, so it
    kept its rest and pointed down at the old angle: every move bent the wrist inward by
    the A-pose droop (shark girl: Kimodo 5-13 degrees, ours 29-40)."""
    parents, heads, roles = _humanoid(face=+1)
    elbow = heads[roles["Lfa"]].copy()
    ang = np.radians(30)
    turn = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    for b in kr._subtree(roles["Lhand"], parents):  # droop the forearm and all below it
        heads[b] = elbow + turn @ (heads[b] - elbow)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    align = kr.rest_alignment(parents, heads, m, kr.soma_to_rig_axes(frame), _soma_like_tpose())
    assert not np.allclose(align[roles["Lfa"]], np.eye(3))  # the forearm really was swung
    assert np.allclose(align[roles["Lhand"]], align[roles["Lfa"]])


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


def test_a_fingerless_rig_maps_without_fingers():
    """SkinTokens leaves fingers out for mitten-like hands (a mech got 22 bones, not 52)."""
    parents, heads, roles = _humanoid(face=+1, fingers=False)
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    assert m[roles["Lhand"]] == "LeftHand" and m[roles["Rhand"]] == "RightHand"
    assert not any("Thumb" in v or "Index" in v for v in m.values())


def test_a_mitten_follows_the_middle_finger():
    parents, heads, roles = _humanoid(face=+1, fingers=False)
    hand = roles["Lhand"]
    for i, name in enumerate(("mit0", "mit1")):
        parents[name] = hand if i == 0 else "mit0"
        heads[name] = heads[hand] + np.array([-0.05 * (i + 1), 0, 0])
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    assert m["mit0"].startswith("LeftHandMiddle") and m["mit1"].startswith("LeftHandMiddle")


def test_stiff_fingers_follow_their_hand_and_leave_the_rest():
    parents, heads, roles = _humanoid(face=+1)
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    stiff = kr.stiffen_fingers(m)
    assert stiff[roles["Lindex2"]] == "LeftHand" and stiff[roles["Rthumb0"]] == "RightHand"
    assert stiff[roles["Lhand"]] == "LeftHand" and stiff[roles["Lfa"]] == "LeftForeArm"
    assert stiff[roles["head"]] == m[roles["head"]]


def _rest_and_tails(parents, heads):
    rest = {}
    for b, h in heads.items():
        m = np.eye(4)
        m[:3, 3] = h
        rest[b] = m
    kids = kr._children(parents)
    tails = {b: (heads[kids[b][0]] if kids[b] else heads[b] + np.array([0, 0, 0.05])) for b in heads}
    return rest, tails


def _order(parents):
    order = [b for b, p in parents.items() if p is None]
    kids = kr._children(parents)
    i = 0
    while i < len(order):
        order += kids[order[i]]
        i += 1
    return order


def test_clear_arms_swings_an_arm_out_of_the_chest_and_leaves_a_clear_one_alone():
    import arm_clearance as ac

    parents, heads, _ = _humanoid(face=+1)
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    rest, tails = _rest_and_tails(parents, heads)
    order = _order(parents)
    by_role = {v: k for k, v in m.items()}
    proxy = ac.Proxy(body={by_role["Chest"]: 0.12, by_role["Spine2"]: 0.12},
                     arms={"Left": {"upper": by_role["LeftArm"], "fore": by_role["LeftForeArm"],
                                    "hand": by_role["LeftHand"],
                                    "radius": {by_role["LeftArm"]: 0.03, by_role["LeftForeArm"]: 0.03,
                                               by_role["LeftHand"]: 0.02}}},
                     side_shoulder={"Left": None, "Right": None}, margin=0.01)
    turn = np.array([[-1.0, 0, 0], [0, -1, 0], [0, 0, 1]])  # 180 deg about up: arm through chest
    left_arm = kr._subtree(by_role["LeftArm"], parents)
    bent = {b: (turn if b in left_arm else np.eye(3)) for b in order}
    straight = {b: np.eye(3) for b in order}
    frames = [(dict(bent), heads[order[0]]) for _ in range(5)] + [(dict(straight), heads[order[0]])]
    kr.clear_arms(frames, rest, parents, order, tails, proxy)

    def gap(world_rot):
        pose = kr.pose_matrices(rest, parents, order, world_rot, heads[order[0]])
        seg = kr.posed_segments(rest, pose, tails, order)
        return min(ac.point_segment(p, *seg[b])[0] - (r + ra + proxy.margin)
                   for p, ra in ac._test_points(seg, proxy.arms["Left"])
                   for b, r in proxy.body.items())

    assert gap(bent) < 0
    assert gap(frames[2][0]) > -1e-3


def test_retarget_with_clearance_matches_without_when_nothing_collides():
    import arm_clearance as ac

    parents, heads, _ = _humanoid(face=+1)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    rest, tails = _rest_and_tails(parents, heads)
    order = _order(parents)
    axes = kr.soma_to_rig_axes(frame)
    g = np.tile(np.eye(3), (3, 77, 1, 1))
    hips = np.zeros((3, 3))
    by_role = {v: k for k, v in m.items()}
    proxy = ac.Proxy(body={by_role["Chest"]: 0.1}, arms={"Left": {
        "upper": by_role["LeftArm"], "fore": by_role["LeftForeArm"], "hand": by_role["LeftHand"],
        "radius": {by_role["LeftArm"]: 0.02, by_role["LeftForeArm"]: 0.02, by_role["LeftHand"]: 0.02}}},
        side_shoulder={"Left": None, "Right": None}, margin=0.01)
    plain = list(kr.retarget_frames(rest, parents, order, m, axes, g, hips, 1.0))
    cleared = list(kr.retarget_frames(rest, parents, order, m, axes, g, hips, 1.0,
                                      tails=tails, clearance=proxy))
    for a, b in zip(plain, cleared):
        assert all(np.allclose(a[k], b[k]) for k in a)


def test_key_frames_compress_with_speed_and_keep_the_first_frame():
    assert kr.key_frame(1, 1.25) == 1
    assert kr.key_frame(101, 1.25) == pytest.approx(81)
    assert kr.key_frame(31, 1.0) == 31
    with pytest.raises(ValueError):
        kr.key_frame(5, 0)


def test_posture_is_kept_only_limbs_are_aligned():
    """A cartoon dwarf's neck juts 45 degrees forward; pulling it upright to match the
    mocap actor tipped his whole head back to the sky. Spine, neck and head keep the
    character's own posture; arms, legs and fingers are still straightened."""
    parents, heads, roles = _humanoid(face=+1)
    heads[roles["head"]] = heads[roles["neck"]] + np.array([0, 0.07, 0.07])  # forward neck
    knee = heads[roles["Lshin"]]
    heads[roles["Lfoot"]] = knee + np.array([-0.15, 0, -0.4])  # splayed shin
    heads[roles["Ltoe"]] = heads[roles["Lfoot"]] + np.array([0, 0.12, -0.06])
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    align = kr.rest_alignment(parents, heads, m, kr.soma_to_rig_axes(frame), _soma_like_tpose())
    for role in ("hips", "s1", "s2", "chest", "neck", "head"):
        assert np.allclose(align[roles[role]], np.eye(3)), role
    assert not np.allclose(align[roles["Lshin"]], np.eye(3))


def _with_extras(parents, heads, roles):
    """The test humanoid dressed up: hair, a cape, a skirt, a sword and chest straps."""
    parents, heads = dict(parents), dict(heads)

    def add(name, parent_role_or_name, pos):
        parents[name] = roles.get(parent_role_or_name, parent_role_or_name)
        heads[name] = np.array(pos, float)

    add("hair0", "head", (0, -0.08, 1.58)); add("hair1", "hair0", (0, -0.1, 1.4))  # a ponytail
    add("cape0", "s2", (0, -0.1, 1.3)); add("cape1", "cape0", (0, -0.15, 1.0)); add("cape2", "cape1", (0, -0.15, 0.7))
    add("skirtF", "hips", (0, 0.1, 0.9)); add("skirtF1", "skirtF", (0, 0.12, 0.6))
    add("skirtB", "hips", (0, -0.1, 0.9)); add("skirtB1", "skirtB", (0, -0.12, 0.6))
    add("strap", "chest", (0.02, 0.05, 1.3))
    add("sword", "Rfa", (-0.5, 0.05, 1.4)); add("blade", "sword", (-0.5, 0.4, 1.4))
    return parents, heads


def test_extra_bones_do_not_hide_the_humanoid():
    """Hair, capes, skirts and swords are on most generated characters (reported by a user:
    'most humanoids are not human'). The body is still found, and extras ride along."""
    parents, heads, roles = _humanoid()
    parents, heads = _with_extras(parents, heads, roles)
    m, frame = kr.map_skintokens_to_soma(parents, heads)
    assert len(m) == len(parents)
    for role, soma in {"hips": "Hips", "chest": "Chest", "head": "Head", "Lhand": "LeftHand",
                       "Rhand": "RightHand", "Rfa": "RightForeArm", "Lleg": "LeftLeg",
                       "Rtoe": "RightToeBase"}.items():
        assert m[roles[role]] == soma, role
    assert m["hair1"] == "Head" and m["cape2"] == m[roles["s2"]] and m["skirtB1"] == "Hips"
    assert m["blade"] == "RightForeArm" and m["strap"] == "Chest"
    assert np.allclose(frame[:, 2], [0, 1, 0])


def test_a_four_legged_creature_is_still_turned_away():
    """Front legs on the chest reach the floor: those are legs, not arms."""
    P = {"hips": None, "s1": "hips", "chest": "s1", "neck": "chest", "head": "neck"}
    H = {"hips": (0, 0, 0.8), "s1": (0, 0.4, 0.85), "chest": (0, 0.8, 0.9), "neck": (0, 1.0, 1.1), "head": (0, 1.1, 1.3)}
    for base, at, y in (("f", "chest", 0.8), ("b", "hips", 0.0)):
        for lr, x in (("L", -0.25), ("R", 0.25)):
            P[f"{base}{lr}0"], H[f"{base}{lr}0"] = at, (x, y, 0.7)
            P[f"{base}{lr}1"], H[f"{base}{lr}1"] = f"{base}{lr}0", (x, y, 0.35)
            P[f"{base}{lr}2"], H[f"{base}{lr}2"] = f"{base}{lr}1", (x, y, 0.05)
            P[f"{base}{lr}3"], H[f"{base}{lr}3"] = f"{base}{lr}2", (x, y + 0.08, 0.0)
    # ears and a jaw fork the head three ways, like a chest: they must not pass for arms
    for name, pos in (("earL", (-0.15, 1.1, 1.5)), ("earR", (0.15, 1.1, 1.5)), ("jaw", (0, 1.3, 1.2))):
        P[name], H[name] = "head", pos
    with pytest.raises(ValueError, match="four-legged"):
        kr.map_skintokens_to_soma(P, {k: np.array(v, float) for k, v in H.items()})


def test_arms_hanging_at_the_sides_are_still_arms():
    """Not every upload is T-posed: arms straight down sit close to the body."""
    parents, heads, roles = _humanoid()
    for lr, sx in (("L", -1), ("R", 1)):  # facing +Y: left is -X
        heads[roles[f"{lr}sh"]] = np.array([0.06 * sx, 0, 1.45])
        heads[roles[f"{lr}arm"]] = np.array([0.15 * sx, 0, 1.42])
        heads[roles[f"{lr}fa"]] = np.array([0.17 * sx, 0, 1.15])
        heads[roles[f"{lr}hand"]] = np.array([0.18 * sx, 0, 0.9])
        for role in [r for r in roles if r.startswith(lr) and any(f in r for f in ("thumb", "index", "middle", "ring", "pinky"))]:
            heads[roles[role]] = heads[roles[f"{lr}hand"]] + np.array([0, 0, -0.03 - 0.01 * int(role[-1])])
    m, _ = kr.map_skintokens_to_soma(parents, heads)
    assert m[roles["Lhand"]] == "LeftHand" and m[roles["Rarm"]] == "RightArm" and m[roles["head"]] == "Head"


# The blocky knight's real rig (Y-up, glTF space, read from its rigged GLB on 2026-10-13).
# Its legs have three bones, thigh, knee and ankle, with no toe. The mesh faces +Z; the
# old code took the last leg bone for a toe, read the shin's backward lean as the foot
# pointing backward, and so turned the character away and played every clip in reverse.
_KNIGHT_PARENTS = {
    "bone_0": None, "bone_1": "bone_0", "bone_2": "bone_1", "bone_3": "bone_2", "bone_4": "bone_1",
    "bone_5": "bone_4", "bone_6": "bone_5", "bone_7": "bone_1", "bone_8": "bone_7", "bone_9": "bone_8",
    "bone_10": "bone_0", "bone_11": "bone_10", "bone_12": "bone_11", "bone_13": "bone_0",
    "bone_14": "bone_13", "bone_15": "bone_14",
}
_KNIGHT_HEADS = {
    "bone_0": [-0.0008, -0.1504, -0.0306], "bone_1": [-0.0008, -0.0085, -0.0015],
    "bone_2": [-0.0008, 0.1042, 0.0203], "bone_3": [-0.0008, 0.137, 0.0313],
    "bone_4": [0.0938, 0.0679, 0.024], "bone_5": [0.1774, -0.0049, 0.0167], "bone_6": [0.232, -0.0813, 0.0349],
    "bone_7": [-0.0917, 0.0679, 0.024], "bone_8": [-0.1754, -0.0049, 0.0167], "bone_9": [-0.2336, -0.0813, 0.0349],
    "bone_10": [0.0683, -0.1286, -0.016], "bone_11": [0.0756, -0.2486, -0.0088], "bone_12": [0.0792, -0.3977, -0.0779],
    "bone_13": [-0.0735, -0.1286, -0.016], "bone_14": [-0.0772, -0.2486, -0.0088], "bone_15": [-0.0845, -0.3977, -0.0779],
}
_Y_UP = np.array([0.0, 1.0, 0.0])


def test_the_knights_three_bone_legs_face_forward_not_backward():
    heads = {k: np.array(v, float) for k, v in _KNIGHT_HEADS.items()}
    _, frame = kr.map_skintokens_to_soma(_KNIGHT_PARENTS, heads, up=_Y_UP)
    assert frame[2, 2] > 0.9  # forward is +Z, the way the mesh looks


def _three_bone_legs(face):
    """The `_humanoid` with its toes removed and knees bent forward: thigh, knee, ankle."""
    parents, heads, roles = _humanoid(face)
    for lr in "LR":
        toe = roles[f"{lr}toe"]
        parents.pop(toe), heads.pop(toe)
        heads[roles[f"{lr}shin"]] = heads[roles[f"{lr}shin"]] + np.array([0, 0.05 * face, 0])
        heads[roles[f"{lr}foot"]] = heads[roles[f"{lr}foot"]] + np.array([0, -0.03 * face, 0])
    return parents, heads


@pytest.mark.parametrize("face", [+1, -1])
def test_a_leg_without_a_toe_is_read_by_which_way_the_knee_bends(face):
    parents, heads = _three_bone_legs(face)
    _, frame = kr.map_skintokens_to_soma(parents, heads)
    assert np.allclose(frame[:, 2], [0, face, 0], atol=1e-6)


def test_a_toe_still_decides_when_the_leg_has_one():
    """Four bones to the leg end in a real toe. A backward-bent knee (a dog-legged hero)
    must not outvote it."""
    parents, heads, roles = _humanoid(+1)
    for lr in "LR":
        heads[roles[f"{lr}shin"]] = heads[roles[f"{lr}shin"]] + np.array([0, -0.06, 0])
    _, frame = kr.map_skintokens_to_soma(parents, heads)
    assert np.allclose(frame[:, 2], [0, +1, 0], atol=1e-6)


def test_straight_toeless_legs_fall_back_to_the_old_cue_without_crashing():
    parents, heads = _three_bone_legs(+1)
    for b in heads:  # knees exactly on the hip-ankle line, the whole leg leaning forward a little
        if abs(heads[b][2] - 0.5) < 1e-9:
            heads[b] = np.array([heads[b][0], 0.01, 0.5])
        if abs(heads[b][2] - 0.08) < 1e-9:
            heads[b] = np.array([heads[b][0], 0.02, 0.08])
    _, frame = kr.map_skintokens_to_soma(parents, heads)
    assert np.allclose(frame[:, 2], [0, +1, 0], atol=1e-6)


# ------------------------------------------------------------------ floor and loop metrics


def test_penetration_is_the_deepest_foot_below_the_floor():
    track = np.array([[[0, 0, -0.05], [0, 0, 0.10]], [[0, 0, 0.02], [0, 0, 0.03]]])
    assert kr.penetration(track, 0.0, 1.0) == pytest.approx(0.05)
    assert kr.penetration(track + np.array([0, 0, 0.2]), 0.0, 1.0) == 0
    with pytest.raises(ValueError):
        kr.penetration(track, 0.0, 0)


def _identity_frames(parents, heads, roots):
    rest, order = _rest_mats(parents, heads)
    return [({b: np.eye(3) for b in order}, np.asarray(root, float)) for root in roots], rest, order


def test_loop_gap_ignores_forward_travel_but_not_height():
    parents, heads, _ = _humanoid()
    frames, rest, order = _identity_frames(parents, heads, [[0, 0, 1], [0, 2, 1]])
    height = 1.6
    # 2 m of forward travel is not a pop when games own the movement.
    assert kr.loop_gap(frames, rest, parents, order, height) == pytest.approx(0, abs=1e-12)
    frames, rest, order = _identity_frames(parents, heads, [[0, 0, 1], [0, 0, 1.1]])
    # A 0.1 m jump in height is a pop.
    assert kr.loop_gap(frames, rest, parents, order, height) == pytest.approx(0.1 / height)


def _cycling_frames(parents, heads, roles, frames=40, period=10):
    """One arm swings on a period, while the root walks forward 5 cm a frame."""
    rest, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    out = []
    for t in range(frames):
        angle = 0.5 * np.sin(2 * np.pi * t / period)
        c, s = np.cos(angle), np.sin(angle)
        world = {b: np.eye(3) for b in order}
        world[roles["Larm"]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        out.append((world, root + np.array([0, 0.05 * t, 0])))
    return out, rest, order


def test_best_loop_finds_the_repeated_cycle_not_the_nearest_frame():
    parents, heads, roles = _humanoid()
    frames, rest, order = _cycling_frames(parents, heads, roles)
    start, end = kr.best_loop(frames, rest, parents, order)
    # Any full gait cycle of this exact periodic clip is a perfect match; what matters is
    # that it is one cycle long and starts on the same phase, not which one won the tie.
    assert end - start == 10
    assert start % 10 == 0


def test_best_loop_rejects_a_mirrored_half_step():
    """A walk's half-step swaps left and right; it looks like the other foot, not the same
    pose. The full cycle must win even when the half-step is allowed by the length floor."""
    parents, heads, roles = _humanoid()
    rest, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    frames = []
    for t in range(40):
        phase = 2 * np.pi * t / 20
        world = {b: np.eye(3) for b in order}
        for role, offset in (("Larm", 0.0), ("Rarm", np.pi / 2)):
            angle = 0.5 * np.sin(phase + offset)
            c, s = np.cos(angle), np.sin(angle)
            world[roles[role]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        frames.append((world, root + np.array([0, 0.02 * t, 0])))
    candidates = kr.loop_candidates(frames, rest, parents, order, min_frames=5, min_share=0.0)
    best = candidates[0]
    assert best[1] - best[0] == 20  # one full left+right cycle
    by_length = {end - start: distance for start, end, distance in candidates}
    assert by_length[10] > by_length[20]  # the mirrored half-step is visibly worse


def test_blend_loop_drops_the_duplicate_first_pose_and_keeps_forward_travel():
    parents, heads, roles = _humanoid()
    frames, rest, order = _cycling_frames(parents, heads, roles)
    start, end = kr.best_loop(frames, rest, parents, order)
    height = max(heads[b][2] for b in heads) - min(
        float(np.asarray(heads[bone])[2]) for bone in kr._foot_bones(
            kr.map_skintokens_to_soma(parents, heads)[0]))
    before = kr.loop_gap(frames, rest, parents, order, height)
    assert before > 0
    blended = kr.blend_loop(frames, start, end)
    # The frame that matched the first is dropped; keeping it would show the pose twice.
    assert len(blended) == end - start - 1
    seam = kr.loop_gap(blended, rest, parents, order, height)
    assert 0 < seam < before
    assert seam < 0.05 * height  # a normal frame step at the wrap, not the old jump
    first_quat = kr.matrices_to_quaternions(np.asarray(blended[0][0][roles["Larm"]]))
    last_quat = kr.matrices_to_quaternions(np.asarray(blended[-1][0][roles["Larm"]]))
    assert abs(float(first_quat @ last_quat)) < 1 - 1e-6  # not a duplicate
    # The seam keeps the walk moving forward; only height is pulled back to the start.
    assert blended[-1][1][1] > blended[0][1][1]
    assert blended[-1][1][2] == pytest.approx(blended[0][1][2])


def test_ground_frames_lifts_every_penetrating_frame_onto_the_floor():
    parents, heads, _ = _humanoid()
    rest, order = _rest_mats(parents, heads)
    _, tails = _rest_and_tails(parents, heads)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    base = heads[order[0]]
    frames = [({b: np.eye(3) for b in order}, base + np.array([0, 0, dz]))
              for dz in (-0.1, -0.2, -0.1)]
    floor = kr._rest_floor(rest, tails, mapping)
    height = max(heads[b][2] for b in heads) - floor
    before = kr.penetration(kr._foot_track(frames, rest, parents, order, tails, mapping),
                            floor, height)
    assert before > 0.1
    grounded = kr.ground_frames(frames, rest, parents, order, tails, mapping, floor)
    after = kr.penetration(kr._foot_track(grounded, rest, parents, order, tails, mapping),
                           floor, height)
    assert after == pytest.approx(0, abs=1e-12)


def test_grounding_never_pulls_an_airborne_frame_down():
    """A jump, flying kick or a run's flight phase has both feet above the floor. Grounding
    may only lift, so those frames keep their height instead of being flattened. The take-off
    frame stands on the floor, as in a real jump."""
    parents, heads, _ = _humanoid()
    rest, order = _rest_mats(parents, heads)
    _, tails = _rest_and_tails(parents, heads)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    base = heads[order[0]]
    floor = kr._rest_floor(rest, tails, mapping)
    frames = [({b: np.eye(3) for b in order}, base + np.array([0, 0, dz]))
              for dz in (0.0, 0.4, 0.6, 0.5)]
    grounded = kr.ground_frames(frames, rest, parents, order, tails, mapping, floor)
    for before, after in zip(frames, grounded):
        assert after[1][2] == pytest.approx(before[1][2])


def test_a_move_that_never_touches_the_floor_is_lowered_onto_it():
    """The crouch sneak hovered 5.6 cm above the floor for its whole clip on the space marine
    (2026-11-10). A move whose feet never reach the floor comes down until the lowest one does,
    and keeps its own ups and downs."""
    parents, heads, _ = _humanoid()
    rest, order = _rest_mats(parents, heads)
    _, tails = _rest_and_tails(parents, heads)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    base = heads[order[0]]
    floor = kr._rest_floor(rest, tails, mapping)
    frames = [({b: np.eye(3) for b in order}, base + np.array([0, 0, dz]))
              for dz in (0.06, 0.08, 0.07)]
    grounded = kr.ground_frames(frames, rest, parents, order, tails, mapping, floor)
    low = (kr._foot_track(grounded, rest, parents, order, tails, mapping) @ kr.Z_UP).min(axis=1)
    assert low.min() == pytest.approx(floor, abs=1e-9)
    heights = [after[1][2] - before[1][2] for before, after in zip(frames, grounded)]
    assert heights == pytest.approx([-0.06] * 3)


def test_lift_above_floor_raises_only_the_frames_that_dip():
    parents, heads, _ = _humanoid()
    rest, order = _rest_mats(parents, heads)
    _, tails = _rest_and_tails(parents, heads)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    base = heads[order[0]]
    floor = kr._rest_floor(rest, tails, mapping)
    frames = [({b: np.eye(3) for b in order}, base + np.array([0, 0, dz]))
              for dz in (-0.1, 0.1)]
    lifted = kr.lift_above_floor(frames, rest, parents, order, tails, mapping, floor)
    low = (kr._foot_track(lifted, rest, parents, order, tails, mapping) @ kr.Z_UP).min(axis=1)
    assert low[0] == pytest.approx(floor, abs=1e-9)
    assert lifted[1][1][2] == pytest.approx(frames[1][1][2])  # already above: untouched
    assert low[1] > floor


def test_in_place_frames_stay_on_the_spot_and_keep_the_bob():
    parents, heads, _ = _humanoid()
    _, order = _rest_mats(parents, heads)
    base = heads[order[0]]
    roots = [base + np.array([0.0, 0.0, 0.0]),
             base + np.array([0.4, 1.0, 0.05]),
             base + np.array([-0.2, 2.0, -0.03])]
    world = {b: np.eye(3) for b in order}
    frames = [(world, root) for root in roots]
    moved = kr.in_place_frames(frames)
    assert all(np.allclose(root[:2], roots[0][:2]) for _, root in moved)  # up is Z
    assert [root[2] for _, root in moved] == pytest.approx([root[2] for root in roots])
    assert all(np.array_equal(moved[i][0][b], world[b]) for i in range(3) for b in order)
    assert kr.horizontal_travel(frames) > 1
    assert kr.horizontal_travel(moved) == 0


def test_skin_track_moves_a_sole_vertex_with_its_bone():
    parents, heads, roles = _humanoid()
    rest, order = _rest_mats(parents, heads)
    point = heads[roles["Lfoot"]] + np.array([0, 0, -0.01])
    weights = [{roles["Lfoot"]: 1.0}]
    base = heads[order[0]]
    still = [({b: np.eye(3) for b in order}, base)]
    assert kr.skin_track(still, rest, parents, order, [point], weights)[0, 0] == pytest.approx(point)
    moved = [({b: np.eye(3) for b in order}, base + np.array([0, 1.0, 0]))]
    assert kr.skin_track(moved, rest, parents, order, [point], weights)[0, 0] == pytest.approx(
        point + np.array([0, 1.0, 0]))


def test_moving_average_calms_a_spike_without_shifting_the_ends():
    values = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    smooth = kr._moving_average(values, 3)
    assert smooth.max() < values.max()
    assert smooth[0] == pytest.approx(0) and smooth[-1] == pytest.approx(0)
    assert len(smooth) == len(values)


def test_grounding_keeps_the_hips_own_bob_over_the_feet():
    """Grounding moves the root, not the pose: a frame whose legs are more bent keeps
    its hips lower above the floor than a straight-legged frame."""
    parents, heads, roles = _humanoid()
    rest, order = _rest_mats(parents, heads)
    _, tails = _rest_and_tails(parents, heads)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    base = heads[order[0]]
    straight = {b: np.eye(3) for b in order}
    bent = dict(straight)
    for lr in "LR":  # bend both knees a little: the hips sit lower over the same feet
        bent[roles[f"{lr}shin"]] = kr.rotation_between(
            heads[roles[f"{lr}foot"]] - heads[roles[f"{lr}shin"]], np.array([0, 0.2, -1.0]))
    frames = [(straight, base), (bent, base - np.array([0, 0, 0.1]))]
    floor = kr._rest_floor(rest, tails, mapping)
    grounded = kr.ground_frames(frames, rest, parents, order, tails, mapping, floor)
    hips = [float((kr.pose_matrices(rest, parents, order, world, root)[order[0]][:3, 3])[2])
            for world, root in grounded]
    assert hips[1] < hips[0]



def test_moving_frames_skips_the_standing_start_of_a_travelling_clip():
    """Kimodo clips open standing; two still frames match perfectly, so the picker used to
    choose a sprint's wind-up as its cycle."""
    parents, heads, roles = _humanoid()
    _, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    frames = []
    for t in range(30):
        world = {b: np.eye(3) for b in order}
        if t >= 6:  # the cycle only starts moving at frame 6
            angle = 0.5 * np.sin(2 * np.pi * (t - 6) / 10)
            c, s = np.cos(angle), np.sin(angle)
            world[roles["Larm"]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        frames.append((world, root + np.array([0, 0.0 if t < 6 else 0.04 * (t - 6), 0])))
    usable = kr.moving_frames(frames, height=1.6)
    assert usable is not None
    assert not usable[:6].any() and usable[8:].all()
    # An idle sways but does not travel: every frame is a fair place to start a loop.
    idle = [(world, root + np.array([0, 0, 0.002 * t])) for t, (world, _) in enumerate(frames)]
    assert kr.moving_frames(idle, height=1.6) is None


def test_a_cycle_is_picked_over_a_perfect_match_between_two_still_frames():
    parents, heads, roles = _humanoid()
    rest, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    frames = []
    for t in range(36):
        world = {b: np.eye(3) for b in order}
        moving = t >= 6
        if moving:
            angle = 0.5 * np.sin(2 * np.pi * (t - 6) / 10)
            c, s = np.cos(angle), np.sin(angle)
            world[roles["Larm"]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        frames.append((world, root + np.array([0, 0.0 if not moving else 0.04 * (t - 6), 0])))
    start, _ = kr.best_loop(frames, rest, parents, order, height=1.6)
    assert start >= 6, "a still frame must not start the cycle"


def test_keep_whole_keeps_every_frame_and_only_blends_the_seam():
    parents, heads, roles = _humanoid()
    frames, rest, order = _cycling_frames(parents, heads, roles, frames=40, period=10)
    start, end, candidates = kr.loop_trim(frames, rest, parents, order, height=1.6,
                                          keep_whole=True)
    assert (start, end, candidates) == (0, 40, [])
    trimmed = kr.loop_trim(frames, rest, parents, order, height=1.6)
    assert trimmed[1] - trimmed[0] == 10 and trimmed[2]


def test_a_wider_seam_blends_more_frames():
    assert kr.seam_blend_frames(0.02, 1.6) == 5          # a walk: the shipped five is enough
    wide = kr.seam_blend_frames(0.09, 1.6)               # a sprint's 9% seam
    assert 5 < wide <= kr.LOOP_BLEND_MAX
    assert kr.seam_blend_frames(0.5, 1.6) == kr.LOOP_BLEND_MAX   # never longer than the cap


def _booted_rig(toe_up_deg=30.0, frames=6):
    """A humanoid in boots: the foot bone drops almost straight down into the boot, the toe bends
    forward at the ball, and the toe ends at a tip bone. The left toe is bent up, as the retarget
    does to a planted foot."""
    parents, heads, roles = _humanoid()
    for side, sx in (("L", -1.0), ("R", 1.0)):
        heads[roles[f"{side}foot"]] = np.array([0.1 * sx, 0.0, 0.12])
        heads[roles[f"{side}toe"]] = np.array([0.1 * sx, 0.03, 0.02])
        tip = f"{side}toetip"
        parents[tip] = roles[f"{side}toe"]
        heads[tip] = np.array([0.1 * sx, 0.10, 0.015])
        roles[tip] = tip
    rest, tails = _rest_and_tails(parents, heads)
    order = _order(parents)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    up = np.array([0.0, 0.0, 1.0])
    angle = np.radians(toe_up_deg)
    c, s = np.cos(angle), np.sin(angle)
    rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    world = {b: np.eye(3) for b in order}
    world[roles["Ltoe"]] = rx
    floor = kr._rest_floor(rest, tails, mapping, up)
    pose = kr.pose_matrices(rest, parents, order, world, heads[order[0]])
    feet = [b for b, role in mapping.items() if role in ("LeftFoot", "RightFoot")]
    segs = kr.posed_segments(rest, pose, tails, feet)
    low = min(float(pt @ up) for head, tail in segs.values() for pt in (head, tail))
    root = heads[order[0]] - (low - floor) * up
    return parents, heads, roles, rest, tails, order, mapping, up, [(dict(world), root)] * frames, floor


def _toe_tip_height(rest, tails, parents, order, roles, up, frame):
    pose = kr.pose_matrices(rest, parents, order, *frame)
    tip = kr.posed_segments(rest, pose, tails, [roles["Ltoetip"]])[roles["Ltoetip"]][1]
    return float(tip @ up)


def test_a_planted_toe_sits_at_its_rest_height():
    """The marine's toe tip sat up to 2.2 cm above its rest height when planted. On these rigs
    the foot bone points down into the boot, so flatness is judged at the toe tip."""
    parents, _, roles, rest, tails, order, mapping, up, posed, floor = _booted_rig()
    height = 1.6
    rest_tip = float(tails[roles["Ltoetip"]] @ up)
    assert _toe_tip_height(rest, tails, parents, order, roles, up, posed[0]) > rest_tip + 0.02

    out = kr.plant_feet(posed, rest, parents, order, tails, mapping, floor, height, up)
    assert _toe_tip_height(rest, tails, parents, order, roles, up, out[0]) == pytest.approx(
        rest_tip, abs=0.01)
    # The foot that was not bent is left alone.
    assert np.allclose(out[0][0][roles["Rfoot"]], posed[0][0][roles["Rfoot"]])


def test_a_foot_in_the_air_keeps_its_pointed_toe():
    """A jump's pointed toe is the move, not a bug: only a foot on the floor is levelled."""
    parents, _, roles, rest, tails, order, mapping, up, posed, floor = _booted_rig()
    airborne = [(world, root + np.array([0, 0, 0.6])) for world, root in posed]
    out = kr.plant_feet(airborne, rest, parents, order, tails, mapping, floor, 1.6, up)
    for (world, root), (new_world, new_root) in zip(airborne, out):
        assert np.allclose(np.asarray(world[roles["Ltoe"]]), np.asarray(new_world[roles["Ltoe"]]))


def test_planted_tilt_says_how_far_a_planted_toe_is_from_its_rest_height():
    parents, _, _, rest, tails, order, mapping, up, posed, floor = _booted_rig()
    before = kr.planted_tilt(posed, rest, parents, order, tails, mapping, floor, 1.6, up)
    out = kr.plant_feet(posed, rest, parents, order, tails, mapping, floor, 1.6, up)
    after = kr.planted_tilt(out, rest, parents, order, tails, mapping, floor, 1.6, up)
    assert before > 0.01 and after < 0.005      # both as a share of height


def _clean_loop_frames(parents, heads, roles, frames=21, period=10):
    """A clip that already ends with a copy of its first frame: frame N == frame 0."""
    rest, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    out = []
    for t in range(frames):
        angle = 0.5 * np.sin(2 * np.pi * (t % period) / period)
        c, s = np.cos(angle), np.sin(angle)
        world = {b: np.eye(3) for b in order}
        world[roles["Larm"]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        out.append((world, root + np.array([0, 0.03 * (t % period), 0])))
    return out, rest, order


def test_a_clean_loop_does_not_end_on_a_copy_of_its_first_pose():
    """The sprint clip already is a clean loop: its last frame is its first. Blending the seam
    pulled the frame before it onto the same pose, so the exported clip showed that pose twice
    and the run stuttered at the wrap."""
    parents, heads, roles = _humanoid()
    frames, rest, order = _clean_loop_frames(parents, heads, roles)
    height = 1.6
    blended = kr.blend_loop(frames, 0, len(frames), kr.LOOP_BLEND_FRAMES)
    trimmed = kr.trim_duplicate_tail(blended, rest, parents, order, height)
    gaps = [kr.loop_gap([a, b], rest, parents, order, height)
            for a, b in itertools.pairwise(trimmed)]
    step = float(np.median(gaps))
    wrap = kr.loop_gap([trimmed[-1], trimmed[0]], rest, parents, order, height)
    assert wrap >= 0.5 * step, f"wrap {wrap:.4f} is a duplicate of {step:.4f}"
    assert wrap < 2.5 * step, "the tail lost a real frame"
    assert len(trimmed) < len(frames)


def test_a_three_bone_leg_is_not_planted_by_its_knee():
    """Hip, knee, ankle spread over Leg, Foot, ToeBase make the knee the "Foot". Planting that
    pulled a swinging lower leg down to the floor (review, 2026-11-10), so such legs are skipped."""
    parents, heads = _three_bone_legs(+1)
    rest, tails = _rest_and_tails(parents, heads)
    order = _order(parents)
    mapping, _ = kr.map_skintokens_to_soma(parents, heads)
    assert "LeftShin" not in mapping.values()
    assert kr._foot_sides(mapping, rest) == []
    knee = next(b for b, role in mapping.items() if role == "LeftFoot")
    world = {b: np.eye(3) for b in order}
    c, s = np.cos(1.2), np.sin(1.2)
    world[knee] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])   # heel kicked back
    frames = [(dict(world), heads[order[0]])] * 4
    floor = kr._rest_floor(rest, tails, mapping)
    out = kr.plant_feet(frames, rest, parents, order, tails, mapping, floor, 1.6)
    for (before, _), (after, _) in zip(frames, out):
        assert all(np.allclose(np.asarray(before[b]), np.asarray(after[b])) for b in before)


def test_a_clip_pulled_in_place_still_skips_its_standing_start():
    """In place runs before the loop picker and leaves no travel to measure, so the picker
    takes which frames move from the clip before that step."""
    parents, heads, roles = _humanoid()
    rest, order = _rest_mats(parents, heads)
    root = heads[order[0]]
    frames = []
    for t in range(36):
        world = {b: np.eye(3) for b in order}
        if t >= 6:
            angle = 0.5 * np.sin(2 * np.pi * (t - 6) / 10)
            c, s = np.cos(angle), np.sin(angle)
            world[roles["Larm"]] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        frames.append((world, root + np.array([0, 0.0 if t < 6 else 0.04 * (t - 6), 0])))
    placed = kr.in_place_frames(frames)
    assert kr.moving_frames(placed, height=1.6) is None   # nothing left to measure
    start, _, _ = kr.loop_trim(placed, rest, parents, order, height=1.6,
                               moving=kr.moving_frames(frames, height=1.6))
    assert start >= 6, "a still frame must not start the cycle"
