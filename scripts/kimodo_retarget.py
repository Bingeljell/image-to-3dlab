#!/usr/bin/env python3
"""Play a Kimodo text-to-motion clip on a SkinTokens-rigged humanoid and export an animated GLB.

Kimodo (NVIDIA) writes motion for its own 77-joint SOMA skeleton; SkinTokens (VAST)
invents a skeleton per mesh with bones named `bone_0..N`. Two facts make the hand-off
cheap, and both were checked rather than assumed:

* Kimodo's `global_rot_mats` are rotations *away from SOMA's standard T-pose*: rebuilding
  `posed_joints` from the T-pose plus those rotations matches to 1e-7 m.
* SkinTokens fits a humanoid to one of a few bone *trees* (52 bones on two very
  different characters, 22 on a mech with mitten hands) and only moves the joints. So bones
  are found by tree shape --
  spine, a chest that forks into neck and two arms, hands with five fingers or none --
  never by `bone_N` number, and never by any one character's measurements.

Both rests are T-poses, so each bone gets its SOMA joint's world-space turn applied to its
own rest orientation. Joints SOMA has and the rig lacks (Neck2, finger metacarpals) are
absorbed automatically, because the turn is world-space, not parent-relative. The hips'
travel is scaled by the ratio of hip heights so a short character does not moonwalk.

The maths below is pure numpy; only `main()` needs `bpy` (Blender, or the `bpy` wheel).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# SOMA 77-joint order as Kimodo writes it (kimodo.skeleton.definitions.SOMASkeleton77).
SOMA77 = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye", "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
SOMA_INDEX = {name: i for i, name in enumerate(SOMA77)}
SOMA77_PARENTS = [
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 3, 11, 12, 13, 14, 15, 16, 17, 14, 19, 20, 21, 22, 14, 24, 25, 26,
    27, 14, 29, 30, 31, 32, 14, 34, 35, 36, 37, 3, 39, 40, 41, 42, 43, 44, 45, 42, 47, 48, 49, 50, 42,
    52, 53, 54, 55, 42, 57, 58, 59, 60, 42, 62, 63, 64, 65, 0, 67, 68, 69, 70, 0, 72, 73, 74, 75,
]
FINGERS = ("Index", "Middle", "Ring", "Pinky")
Z_UP = np.array([0.0, 0.0, 1.0])


def _children(parents: dict[str, str | None]) -> dict[str, list[str]]:
    kids: dict[str, list[str]] = {b: [] for b in parents}
    for b, p in parents.items():
        if p is not None:
            kids[p].append(b)
    return kids


def _chain(start: str, kids: dict[str, list[str]]) -> list[str]:
    """Follow single children from `start` until a fork or a leaf (inclusive)."""
    out = [start]
    while len(kids[out[-1]]) == 1:
        out.append(kids[out[-1]][0])
    return out


def _subtree_low(b: str, kids, heads, up) -> float:
    vals = [float(heads[b] @ up)]
    for c in kids[b]:
        vals.append(_subtree_low(c, kids, heads, up))
    return min(vals)


def _spread(chain: list[str], names: list[str]) -> dict[str, str]:
    """Map rig bones in `chain` onto SOMA `names` (same order), stretching or squeezing."""
    if not chain:
        return {}
    if len(chain) == 1:
        return {chain[0]: names[-1]}
    idx = np.round(np.linspace(0, len(names) - 1, len(chain))).astype(int)
    return {b: names[i] for b, i in zip(chain, idx)}


def character_frame(heads: dict[str, np.ndarray], toes: list[str], feet: list[str],
                    up=Z_UP) -> np.ndarray:
    """3x3 matrix whose columns are the character's (left, up, forward) in world space.

    Forward is the mean foot->toe direction flattened onto the ground, the one cue every
    standing humanoid carries regardless of how it was exported.
    """
    f = np.mean([heads[t] - heads[k] for t, k in zip(toes, feet)], axis=0)
    f = f - (f @ up) * up
    f /= np.linalg.norm(f)
    left = np.cross(up, f)
    return np.stack([left, up, f], axis=1)


def _subtree_high(b: str, kids, heads, up) -> float:
    return max([float(heads[b] @ up)] + [_subtree_high(c, kids, heads, up) for c in kids[b]])


def _limb(start: str, kids, score, length: int) -> list[str]:
    """Like `_chain`, but a fork before `length` bones (a sword, a sleeve, a knee pad) is
    passed by following the child that scores highest, instead of ending the limb there."""
    out = [start]
    while kids[out[-1]]:
        options = kids[out[-1]]
        if len(options) == 1:
            out.append(options[0])
        elif len(out) < length:
            out.append(max(options, key=score))
        else:
            break
    return out


def _subtree_bones(b: str, kids) -> list[str]:
    out = [b]
    for c in kids[b]:
        out += _subtree_bones(c, kids)
    return out


def _find_chest(start: str, kids, heads, up, left_axis, reach: float, knee: float):
    """Walk up the spine to the first bone that forks into a neck and two arms.

    Arms are the two children reaching furthest out to each side; anything else on the
    way (a cape, a scabbard, hair, a tail) is passed. Arms that reach the floor are front
    legs, not arms, so a four-legged creature is still turned away.
    """
    b = start
    path = [b]
    while kids[b]:
        if len(kids[b]) >= 3:
            def side_reach(c, sign):
                return max(sign * float((heads[x] - heads[b]) @ left_axis) for x in _subtree_bones(c, kids))
            left = max(kids[b], key=lambda c: side_reach(c, 1))
            right = max(kids[b], key=lambda c: side_reach(c, -1))
            rest = [c for c in kids[b] if c not in (left, right)]
            wide = left != right and side_reach(left, 1) > reach and side_reach(right, -1) > reach
            if wide and max(_subtree_low(a, kids, heads, up) for a in (left, right)) <= knee:
                raise ValueError(f"{b} stands on front legs {left}, {right}: a four-legged body")
            # an arm is shoulder, upper arm, forearm at least; ears and horns are shorter
            arms_ok = wide and min(len(_subtree_bones(a, kids)) for a in (left, right)) >= 3
            if arms_ok and rest:
                neck = max(rest, key=lambda c: _subtree_high(c, kids, heads, up))
                if _subtree_high(neck, kids, heads, up) > float(heads[b] @ up):
                    return path, neck, [left, right]
        b = max(kids[b], key=lambda c: _subtree_high(c, kids, heads, up))
        path.append(b)
    raise ValueError(f"no bone above the hips forks into a neck and two arms (walked {path})")


def map_skintokens_to_soma(parents: dict[str, str | None], heads: dict[str, np.ndarray],
                           up=Z_UP) -> tuple[dict[str, str], np.ndarray]:
    """Return ({rig bone: SOMA joint}, character frame) for a SkinTokens humanoid.

    `parents` maps bone -> parent bone (None for the root); `heads` maps bone -> rest
    joint position in one shared space (armature or world) with `up` as the up axis.

    Extra bones (hair, a cape, a skirt, a tail, a weapon) do not stop it: the body is
    found by shape, and each extra bone follows the nearest body bone above it.
    """
    kids = _children(parents)
    root = next(b for b, p in parents.items() if p is None)
    root_h = float(heads[root] @ up)
    # The spine is the root child reaching highest (the head); the legs are the two of the
    # rest reaching lowest, and both must go below the hips.
    high = {c: _subtree_high(c, kids, heads, up) for c in kids[root]}
    low = {c: _subtree_low(c, kids, heads, up) for c in kids[root]}
    spine_start = max(kids[root], key=lambda c: high[c], default=None)
    legs = sorted((c for c in kids[root] if c != spine_start and low[c] < root_h - 1e-6),
                  key=lambda c: low[c])[:2]
    if spine_start is None or len(legs) != 2 or high[spine_start] <= root_h:
        raise ValueError(f"expected 2 legs + 1 spine under root, got legs={legs} spine={spine_start}")
    leg_chains = [_limb(c, kids, lambda x: -_subtree_low(x, kids, heads, up), 4) for c in legs]
    if min(len(ch) for ch in leg_chains) < 2:
        raise ValueError(f"legs too short to find the feet: {leg_chains}")
    frame = character_frame(heads, [ch[-1] for ch in leg_chains], [ch[-2] for ch in leg_chains], up)
    left_axis = frame[:, 0]

    floor = min(low[c] for c in legs)
    height = max(high.values()) - floor
    # A third chain standing on the floor is a third leg (a four-legged body), not a skirt.
    walkers = [c for c in kids[root] if c != spine_start and low[c] <= floor + 0.03 * height
               and len(_subtree_bones(c, kids)) >= 3]
    if len(walkers) > 2:
        raise ValueError(f"expected 2 legs + 1 spine under root, got legs={walkers} spine={spine_start}")
    knee = floor + 0.25 * (root_h - floor)
    path, neck, arms = _find_chest(spine_start, kids, heads, up, left_axis, 0.12 * height, knee)
    spine = [root] + path
    chest = spine[-1]
    side = {b: float((heads[b] - heads[chest]) @ left_axis) for b in arms}

    m: dict[str, str] = {}
    m.update(_spread(spine, ["Hips", "Spine1", "Spine2", "Chest"]))
    # Up the neck to the head; a single bone hanging down from the head (hair) is not it.
    neck_chain = [neck]
    while len(kids[neck_chain[-1]]) == 1 and heads[kids[neck_chain[-1]][0]] @ up > heads[neck_chain[-1]] @ up:
        neck_chain.append(kids[neck_chain[-1]][0])
    m.update(_spread(neck_chain[:-1], ["Neck1", "Neck2"]))
    m[neck_chain[-1]] = "Head"

    for arm in arms:
        lr = "Left" if side[arm] > 0 else "Right"
        ch = _limb(arm, kids, lambda x: abs(float((heads[x] - heads[chest]) @ left_axis)), 4)
        # A hand with one finger chain does not fork, so the walk runs on into it: the
        # arm is the first four bones, anything past the hand is that one finger.
        ch, past_hand = ch[:4], ch[4:]
        m.update(_spread(ch, [f"{lr}Shoulder", f"{lr}Arm", f"{lr}ForeArm", f"{lr}Hand"]))
        hand = ch[-1]
        fingers = [past_hand] if past_hand else [_chain(c, kids) for c in kids[hand]]
        # SkinTokens leaves fingers out for mitten-like hands (a mech: 22 bones, not 52).
        # Anything short of a full hand moves stiffly with SOMA's middle finger.
        if len(fingers) != 5:
            for fch in fingers:
                m.update(_spread(fch, [f"{lr}HandMiddle{i}" for i in (2, 3, 4)]))
            continue
        # Thumb: the chain rooted nearest the wrist. The rest run front to back.
        thumb = min(fingers, key=lambda f: np.linalg.norm(heads[f[0]] - heads[hand]))
        others = sorted((f for f in fingers if f is not thumb), key=lambda f: -(heads[f[0]] @ frame[:, 2]))
        m.update(_spread(thumb, [f"{lr}HandThumb{i}" for i in (1, 2, 3)]))
        for name, fch in zip(FINGERS, others):
            # SOMA fingers start at the metacarpal (1); SkinTokens starts at the knuckle.
            m.update(_spread(fch, [f"{lr}Hand{name}{i}" for i in (2, 3, 4)]))

    for leg, ch in zip(legs, leg_chains):
        lr = "Left" if float((heads[leg] - heads[root]) @ left_axis) > 0 else "Right"
        m.update(_spread(ch, [f"{lr}Leg", f"{lr}Shin", f"{lr}Foot", f"{lr}ToeBase"]))
    # Extra bones follow the nearest mapped bone above them, rigidly. Parents first, so
    # every extra finds its parent already settled.
    queue = [root]
    while queue:
        b = queue.pop(0)
        if b not in m:
            m[b] = m[parents[b]]
        queue += kids[b]
    return m, frame


def stiffen_fingers(mapping: dict[str, str]) -> dict[str, str]:
    """Fingers and thumbs follow their hand rigidly instead of copying SOMA's curls.

    For hands whose skin weights bleed between fingers, a curled fist pulls skin from
    one finger towards another and stretches it into spikes; a rigid hand cannot.
    """
    return {b: (s.split("Hand")[0] + "Hand" if "Hand" in s and not s.endswith("Hand") else s)
            for b, s in mapping.items()}


def soma_tpose(global_rots: np.ndarray, posed_joints: np.ndarray, t: int = 0) -> np.ndarray:
    """SOMA's standard T-pose joint positions (hips at origin), recovered from any clip frame.

    Kimodo poses joints as  x_j = x_parent + G_parent @ (N_j - N_parent), so the rest
    offsets fall straight out of one frame and no extra Kimodo file is needed.
    """
    n = np.zeros((77, 3))
    for j, p in enumerate(SOMA77_PARENTS):
        if p >= 0:
            n[j] = n[p] + global_rots[t, p].T @ (posed_joints[t, j] - posed_joints[t, p])
    return n


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation (3x3) turning direction `a` onto direction `b`."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: turn 180 degrees about any perpendicular axis
        perp = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        perp /= np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rest_alignment(parents, heads, mapping, axes, tpose) -> dict[str, np.ndarray]:
    """Per-bone rotation that makes the rig's rest pose point like SOMA's T-pose.

    Generated characters rarely stand in a clean T-pose: legs splay, feet turn out, arms
    droop. Copying Kimodo's turns on top of that keeps the error in every frame (a wide,
    duck-footed walk; hands sunk into the hips). Bones with one child are swung so the
    bone points where SOMA's matching bone points; forks (hips, chest, hands) and the
    spine, neck and head keep their rest. Leaves (toes, head, fingertips) have no direction of their own to fix, so they
    turn with their parent: left at rest, a toe under a swung foot kept a permanent bend
    at the ball of the foot, which showed as a deformed toe.
    """
    kids = _children(parents)
    out = {}
    for b in parents:
        out[b] = np.eye(3)
        # Only limbs. A spine's curve and a neck's angle are the character's design: a
        # cartoon dwarf's neck juts forward, and pulling it upright tipped his head back.
        if len(kids[b]) != 1 or not mapping[b].startswith(("Left", "Right")):
            continue
        c = kids[b][0]
        j, k = SOMA_INDEX[mapping[b]], SOMA_INDEX[mapping[c]]
        want = axes @ (tpose[k] - tpose[j])
        have = heads[c] - heads[b]
        if np.linalg.norm(want) > 1e-6 and np.linalg.norm(have) > 1e-6:
            out[b] = rotation_between(have, want)
    for b, p in parents.items():  # a leaf's parent is never a leaf, so it is settled above
        if not kids[b] and p is not None:
            out[b] = out[p]
    return out


def arm_spread(name: str, world_rot_upper_arm: np.ndarray, rest_dir: np.ndarray, frame: np.ndarray,
               degrees: float) -> np.ndarray:  # Arms out, like Fuusuke riding the wind.
    """Outward tilt for one arm-chain bone, faded in only while the upper arm hangs down.

    Bulky characters (armour, big shoulders) swallow their own hands when they copy a
    slim mocap actor's arms-at-sides; tilting the whole arm out by a few degrees clears
    the body. Scaled by how far the upper arm points down so a raised arm (a wave) is
    left alone.
    """
    if degrees == 0 or not any(name.startswith(f"{s}{p}") for s in ("Left", "Right")
                               for p in ("Arm", "ForeArm", "Hand")):
        return np.eye(3)
    up, fwd = frame[:, 1], frame[:, 2]
    hang = max(0.0, -float((world_rot_upper_arm @ rest_dir) @ up))
    sign = 1.0 if name.startswith("Left") else -1.0
    th = np.radians(degrees) * hang * sign
    # Rotation about forward; positive swings a hanging left arm towards the character's left.
    k = np.array([[0, -fwd[2], fwd[1]], [fwd[2], 0, -fwd[0]], [-fwd[1], fwd[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * k @ k


# SOMA's own frame: X = character's left, Y = up, Z = forward.
SOMA_FRAME = np.eye(3)


def soma_to_rig_axes(rig_frame: np.ndarray) -> np.ndarray:
    """Rotation taking SOMA world axes to the rig's world axes (both as left/up/forward)."""
    return rig_frame @ SOMA_FRAME.T


def pose_basis(rest: dict[str, np.ndarray], parents: dict[str, str | None], order: list[str],
               world_rot: dict[str, np.ndarray], root_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Blender `matrix_basis` (4x4) per bone for one frame.

    `rest` holds each bone's rest matrix in armature space (Blender `bone.matrix_local`),
    `world_rot` the wanted armature-space rotation (3x3) for each bone, `root_pos` the
    wanted armature-space head position of the root. `order` is parent-first. Uses
    Blender's rule  pose_b = pose_parent @ inv(rest_parent) @ rest_b @ basis_b.
    """
    return _posed(rest, parents, order, world_rot, root_pos)[1]


def pose_matrices(rest, parents, order, world_rot, root_pos) -> dict[str, np.ndarray]:
    """Each bone's posed armature-space matrix (4x4) for one frame: where it ends up."""
    return _posed(rest, parents, order, world_rot, root_pos)[0]


def _posed(rest, parents, order, world_rot, root_pos):
    pose: dict[str, np.ndarray] = {}
    basis: dict[str, np.ndarray] = {}
    for b in order:
        p = parents[b]
        rel = rest[b] if p is None else np.linalg.inv(rest[p]) @ rest[b]
        parent_pose = np.eye(4) if p is None else pose[p]
        unposed = parent_pose @ rel  # where the bone sits if it adds no motion of its own
        target = np.eye(4)
        target[:3, :3] = world_rot[b]
        target[:3, 3] = root_pos if p is None else unposed[:3, 3]
        pose[b] = target
        basis[b] = np.linalg.inv(unposed) @ target
    return pose, basis


def posed_segments(rest, pose, tails, bones) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(head, tail) of each bone after posing; `tails` holds rest tails in armature space."""
    out = {}
    for b in bones:
        moved = pose[b] @ np.linalg.inv(rest[b])
        out[b] = (pose[b][:3, 3], (moved @ np.append(tails[b], 1.0))[:3])
    return out


def _subtree(root: str, parents: dict[str, str | None]) -> list[str]:
    kids = _children(parents)
    out, stack = [], [root]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(kids[b])
    return out


def clear_arms(frames, rest, parents, order, tails, proxy) -> None:
    """Swing each arm out of the body wherever it went in (see `arm_clearance`), in place.

    `frames` is a list of (world_rot, root_pos); the swing is applied to the whole arm,
    upper arm down to the fingertips, so the hand keeps its pose relative to the forearm.
    """
    import arm_clearance

    sides = list(proxy.arms)
    swings = {side: [] for side in sides}
    for world_rot, root_pos in frames:
        pose = pose_matrices(rest, parents, order, world_rot, root_pos)
        seg = posed_segments(rest, pose, tails, order)
        for side in sides:
            swings[side].append(arm_clearance.arm_swing(seg, proxy, side))
    for side in sides:
        degrees = [arm_clearance.swing_degrees(r) for r in swings[side]]
        if degrees:
            print(f"arm clearance {side}: largest {max(degrees):.0f} deg, typical {float(np.median(degrees)):.0f} deg")
        arm = _subtree(proxy.arms[side]["upper"], parents)
        for (world_rot, _), swing in zip(frames, arm_clearance.smooth_swings(swings[side])):
            for b in arm:
                world_rot[b] = swing @ world_rot[b]


def retarget_frames(rest, parents, order, mapping, axes, global_rots, hips_pos,
                    hip_scale: float, align=None, frame=None, spread_degrees: float = 0.0,
                    tails=None, clearance=None):
    """Yield `pose_basis` dicts for every frame of a Kimodo clip.

    `axes` is `soma_to_rig_axes(...)` expressed in armature space, `global_rots` Kimodo's
    [T, 77, 3, 3] `global_rot_mats`, `hips_pos` its [T, 3] hips track. `align` comes from
    `rest_alignment`; `frame` and `spread_degrees` drive `arm_spread`. With `clearance`
    (an `arm_clearance.Proxy`) and the rest `tails`, arms are kept out of the body.
    """
    root = order[0]
    rest_rot = {b: rest[b][:3, :3] for b in order}
    align = align or {b: np.eye(3) for b in order}
    upper = {lr: next((b for b in order if mapping[b] == f"{lr}Arm"), None) for lr in ("Left", "Right")}
    soma_arm_dir = {lr: axes @ np.array([1.0 if lr == "Left" else -1.0, 0, 0]) for lr in ("Left", "Right")}
    frames = []
    for t in range(len(global_rots)):
        world_rot = {}
        delta = {b: axes @ global_rots[t, SOMA_INDEX[mapping[b]]] @ axes.T for b in order}
        for b in order:
            world_rot[b] = delta[b] @ align[b] @ rest_rot[b]
            if spread_degrees and frame is not None:
                lr = "Left" if mapping[b].startswith("Left") else "Right"
                if upper[lr] is not None:
                    world_rot[b] = arm_spread(mapping[b], delta[upper[lr]], soma_arm_dir[lr], frame,
                                              spread_degrees) @ world_rot[b]
        travel = axes @ (hips_pos[t] - hips_pos[0]) * hip_scale
        frames.append((world_rot, rest[root][:3, 3] + travel))
    if clearance is not None and tails is not None:
        clear_arms(frames, rest, parents, order, tails, clearance)
    for world_rot, root_pos in frames:
        yield pose_basis(rest, parents, order, world_rot, root_pos)


def key_frame(t: int, speed: float) -> float:
    """Timeline frame for clip frame `t` (1-based) played at `speed` (1.25 = 25% faster).

    Kimodo's kicks and punches are slow next to a real strike; playing them faster gives
    them snap without touching the poses. Fractional frames are fine: the glTF exporter
    samples the curve at whole frames.
    """
    if speed <= 0:
        raise ValueError("speed must be positive")
    return 1 + (t - 1) / speed


def matrices_to_quaternions(m: np.ndarray) -> np.ndarray:
    """[..., 3, 3] rotation matrices -> [..., 4] unit quaternions (w, x, y, z), w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    w = np.sqrt(np.clip(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 0, None)) / 2
    x = np.sqrt(np.clip(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 0, None)) / 2
    y = np.sqrt(np.clip(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 0, None)) / 2
    z = np.sqrt(np.clip(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], 0, None)) / 2
    x = np.copysign(x, m[..., 2, 1] - m[..., 1, 2])
    y = np.copysign(y, m[..., 0, 2] - m[..., 2, 0])
    z = np.copysign(z, m[..., 1, 0] - m[..., 0, 1])
    q = np.stack([w, x, y, z], axis=-1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quaternions_to_matrices(q: np.ndarray) -> np.ndarray:
    """[..., 4] quaternions (w, x, y, z), not necessarily unit -> [..., 3, 3]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(world rotations [T,77,3,3], frame-0 joints [77,3], hips track [T,3]) from either
    Kimodo's own `.npz` or one packed by `pack_kimodo_clip.py`."""
    d = np.load(path)
    if "quats" in d.files:
        return quaternions_to_matrices(d["quats"]), d["joints0"].astype(np.float64), \
            d["hips"].astype(np.float64)
    g, x = d["global_rot_mats"], d["posed_joints"]
    if g.ndim == 5:  # [samples, T, J, 3, 3] -> first sample
        g, x = g[0], x[0]
    return g, x[0], x[:, SOMA_INDEX["Hips"]]


def soma_hip_height(posed_joints: np.ndarray) -> float:
    """Hips above the lowest foot joint on frame 0 (SOMA is Y-up)."""
    f0 = posed_joints[0]
    feet = [SOMA_INDEX[n] for n in ("LeftToeBase", "RightToeBase", "LeftFoot", "RightFoot")]
    return float(f0[SOMA_INDEX["Hips"], 1] - f0[feet, 1].min())


def body_proxy(arm, mapping, heads, tails, up):  # pragma: no cover - needs bpy
    """`arm_clearance.Proxy` from the rig's skinned meshes, in armature space."""
    import arm_clearance

    inv = np.array(arm.matrix_world.inverted())
    points, owners = [], []
    import bpy

    for mesh in (o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm):
        to_arm = inv @ np.array(mesh.matrix_world)
        names = {g.index: g.name for g in mesh.vertex_groups}
        for v in mesh.data.vertices:
            best = max(v.groups, key=lambda g: g.weight, default=None)
            if best is None or names.get(best.group) not in heads:
                continue
            points.append((to_arm @ np.append(np.array(v.co), 1.0))[:3])
            owners.append(names[best.group])
    if not points:
        return None
    height = max(h @ up for h in heads.values()) - min(h @ up for h in heads.values())
    segments = {b: (heads[b], tails[b]) for b in heads}
    return arm_clearance.build_proxy(mapping, np.array(points), owners, segments, height)


def main() -> None:  # pragma: no cover - needs bpy
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rig", required=True, type=Path, help="SkinTokens-rigged GLB")
    ap.add_argument("--motion", required=True, type=Path, help="Kimodo .npz (kimodo_gen output)")
    ap.add_argument("--out", required=True, type=Path, help="animated GLB to write")
    ap.add_argument("--blend", type=Path, help="also save a .blend for inspection")
    ap.add_argument("--no-align", action="store_true", help="skip rest-pose alignment (debug)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="play the clip faster (>1) or slower (<1); 1.25 gives strikes snap")
    ap.add_argument("--no-clearance", action="store_true",
                    help="let arms pass through the body (debug; clearance is on by default)")
    ap.add_argument("--stiff-fingers", action="store_true",
                    help="fingers move rigidly with the hand (for gloves, mittens or bad hand weights)")
    ap.add_argument("--arm-spread", type=float, default=0.0,
                    help="degrees to tilt hanging arms outward, for bulky characters (try 8-15)")
    args = ap.parse_args()

    import bpy
    from mathutils import Matrix

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.rig))
    for o in list(bpy.data.objects):  # SkinTokens' `glTF_not_exported` helper
        if o.type == "MESH" and o.name.startswith("Icosphere"):
            bpy.data.objects.remove(o, do_unlink=True)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    bones = arm.data.bones
    parents = {b.name: (b.parent.name if b.parent else None) for b in bones}
    order = [b.name for b in bones if b.parent is None]
    i = 0
    while i < len(order):
        order += [c.name for c in bones[order[i]].children]
        i += 1
    rest = {b.name: np.array(b.matrix_local) for b in bones}
    heads = {n: m[:3, 3] for n, m in rest.items()}
    tails = {b.name: np.array(b.tail_local) for b in bones}

    # Work in armature space; find which armature axis is world up.
    arm_rot = np.array(arm.matrix_world.to_3x3().normalized())
    up = arm_rot.T @ np.array([0.0, 0.0, 1.0])
    mapping, frame = map_skintokens_to_soma(parents, heads, up)
    if args.stiff_fingers:
        mapping = stiffen_fingers(mapping)
    axes = soma_to_rig_axes(frame)

    g, joints0, hips = load_motion(args.motion)
    lowest = min(heads[b] @ up for b in heads)
    rig_hip = float(heads[order[0]] @ up - lowest)
    scale = rig_hip / soma_hip_height(joints0[None])

    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, int(np.ceil(key_frame(len(g), args.speed)))
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    clearance = None if args.no_clearance else body_proxy(arm, mapping, heads, tails, up)
    tpose = soma_tpose(g[:1], joints0[None])
    align = None if args.no_align else rest_alignment(parents, heads, mapping, axes, tpose)
    for t, basis in enumerate(retarget_frames(rest, parents, order, mapping, axes, g, hips,
                                              scale, align, frame, args.arm_spread,
                                              tails, clearance), start=1):
        for name, mtx in basis.items():
            pb = arm.pose.bones[name]
            pb.matrix_basis = Matrix(mtx.tolist())
            pb.keyframe_insert("rotation_quaternion", frame=key_frame(t, args.speed))
            if parents[name] is None:
                pb.keyframe_insert("location", frame=key_frame(t, args.speed))
    print(f"retargeted {len(g)} frames onto {len(order)} bones, hip scale {scale:.3f}")
    print("side check:", {k: v for k, v in mapping.items() if v in ("LeftHand", "RightHand", "Head")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB", export_animations=True)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend))


if __name__ == "__main__":
    main()
