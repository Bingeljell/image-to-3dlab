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
    knee, ankle = heads[roles["Lshin"]], heads[roles["Lfoot"]]
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
