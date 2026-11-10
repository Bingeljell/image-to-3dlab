#!/usr/bin/env python3
"""Play a Kimodo text-to-motion clip on a SkinTokens-rigged humanoid and export an animated GLB,
with feet grounded, repeating clips loopable and looping moves in place.

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
Grounding and looping are separate pure passes over the generated frames:
`ground_frames` puts the lowest sole on the rest floor, and `best_loop` / `blend_loop`
trim a repeating clip to its best cycle and blend the seam.
"""

from __future__ import annotations

import argparse
import itertools
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

# Grounding: every frame's lowest foot is put on the rest floor. The correction is
# smoothed over this many frames, then the exact lift is kept wherever smoothing would
# still let a foot through (see `ground_frames`).
GROUND_SMOOTH_FRAMES = 5
#: A foot whose lowest joint is this close to the floor (a share of body height) is planted.
PLANT_TOLERANCE = 0.02
#: The plant correction is smoothed over this many frames so a foot that lifts does not snap.
PLANT_SMOOTH_FRAMES = 3
# Loop trimming: the shortest cycle worth keeping, and the share of the clip it must span
# (a quarter of a four-second walk is one step, the smallest useful cycle).
LOOP_MIN_FRAMES = 8
LOOP_MIN_SHARE = 0.25
LOOP_BLEND_FRAMES = 5
#: A clip has to travel this much of the character's height before frame speed can tell a
#: still stretch from a moving one (an idle sways too, and every frame of it is fair game).
LOOP_TRAVEL_SHARE = 0.10
#: Frames slower than this share of the clip's own top root speed never end a cycle: Kimodo
#: clips open standing and often settle at the end, and two still frames match perfectly.
LOOP_SPEED_SHARE = 0.35
#: The seam blend grows with how far the two ends differ. Five frames is enough for a walk
#: (about 2% of height); a sprint's halves differ 6-10%, and five frames still showed a hitch.
LOOP_BLEND_FROM_PERCENT = 2.0   # a seam this wide or narrower blends in the base frames
LOOP_BLEND_PER_PERCENT = 0.8
LOOP_BLEND_MAX = 14
#: After blending, the wrap has to be at least this share of a normal frame step: any closer and
#: the last pose is the first one twice, which shows as a stutter.
DUPLICATE_SHARE = 0.5
# SOMA's foot joints, per side. A rig may lack the toe (three-bone legs), so whichever of
# these its mapping has is used.
FOOT_JOINTS = ("Foot", "ToeBase")


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


def _flat(v: np.ndarray, up) -> np.ndarray:
    return v - (v @ up) * up


# A knee that bends less than this share of the leg's length is too straight to tell
# which way is forward, so the older foot cue is used instead.
KNEE_BEND_MIN = 0.01


def leg_forward(heads: dict[str, np.ndarray], chains: list[list[str]], up=Z_UP) -> np.ndarray:
    """The way the legs say is forward, flattened onto the ground (not yet unit length).

    A leg of four bones or more ends in a real foot and toe, and foot -> toe points forward.
    A leg of three has no toe: its last bone is the ankle, and foot -> "toe" is only the
    shin's lean, which can point backward. The blocky knight's did, and it was turned away
    and walked in reverse. Those legs are read by the knee, which bends forward of the
    hip-ankle line. A knee too straight to tell falls back to the foot cue.
    """
    votes = []
    for ch in chains:
        if len(ch) == 3:
            hip, knee, ankle = (heads[b] for b in ch)
            bend = _flat(knee - (hip + ankle) / 2, up)
            if np.linalg.norm(bend) >= KNEE_BEND_MIN * np.linalg.norm(hip - ankle):
                votes.append(bend)
                continue
        votes.append(heads[ch[-1]] - heads[ch[-2]])
    return _flat(np.mean(votes, axis=0), up)


def character_frame(heads: dict[str, np.ndarray], leg_chains: list[list[str]],
                    up=Z_UP) -> np.ndarray:
    """3x3 matrix whose columns are the character's (left, up, forward) in world space.

    Forward comes from the legs (`leg_forward`): the one cue every standing humanoid
    carries regardless of how it was exported.
    """
    f = leg_forward(heads, leg_chains, up)
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
                return max(sign * float((heads[x] - heads[b]) @ left_axis) for x in _subtree_bones(c, kids))  # noqa: B023 - called only in this iteration
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
    frame = character_frame(heads, leg_chains, up)
    left_axis = frame[:, 0]

    floor = min(low[c] for c in legs)
    height = max(high.values()) - floor
    # A third chain standing on the floor is a third leg (a four-legged body), not a skirt.
    walkers = [c for c in kids[root] if c != spine_start and low[c] <= floor + 0.03 * height
               and len(_subtree_bones(c, kids)) >= 3]
    if len(walkers) > 2:
        raise ValueError(f"expected 2 legs + 1 spine under root, got legs={walkers} spine={spine_start}")
    knee = floor + 0.25 * (root_h - floor)
    # Arms hanging straight down still reach ~10% of height out to the side; 5% leaves room.
    path, neck, arms = _find_chest(spine_start, kids, heads, up, left_axis, 0.05 * height, knee)
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
    bone points where SOMA's matching bone points; forks (hips, chest) and the spine,
    neck and head keep their rest, and a hand turns with its forearm. Leaves (toes, head, fingertips) have no direction of their own to fix, so they
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
    # A hand forks into fingers, so it has no single direction to swing; left at rest it
    # kept the A-pose angle while its forearm was swung up, bending every wrist inward.
    for b, p in parents.items():
        if len(kids[b]) > 1 and p is not None and mapping[b].endswith("Hand"):
            out[b] = out[p]
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


def _foot_bones(mapping: dict[str, str]) -> list[str]:
    """Rig bones that carry a foot or toe joint, in a stable order."""
    by_soma = {soma: bone for bone, soma in mapping.items()}
    return [by_soma[f"{side}{joint}"] for side in ("Left", "Right")
            for joint in FOOT_JOINTS if f"{side}{joint}" in by_soma]


def _pose_heads(frames, rest, parents, order) -> np.ndarray:
    """[T, B, 3] armature-space head of every bone, one row per clip frame."""
    out = []
    for world_rot, root_pos in frames:
        pose = pose_matrices(rest, parents, order, world_rot, root_pos)
        out.append([pose[b][:3, 3] for b in order])
    return np.asarray(out)


def _foot_track(frames, rest, parents, order, tails, mapping, up=Z_UP) -> np.ndarray:
    """[T, P, 3] sole sample points: each foot/toe head and tail, one frame per clip frame."""
    bones = _foot_bones(mapping)
    if not bones:
        raise ValueError("the rig maps no foot joints, so the floor cannot be found")
    out = []
    for world_rot, root_pos in frames:
        pose = pose_matrices(rest, parents, order, world_rot, root_pos)
        segments = posed_segments(rest, pose, tails, bones)
        out.append([point for bone in bones for point in segments[bone]])
    return np.asarray(out)


def _rest_floor(rest, tails, mapping, up=Z_UP) -> float:
    """Lowest foot point in the rest pose: the floor the character stands on."""
    points = [float(np.asarray(rest[bone])[:3, 3] @ up) for bone in _foot_bones(mapping)]
    points += [float(np.asarray(tails[bone]) @ up) for bone in _foot_bones(mapping) if bone in tails]
    if not points:
        raise ValueError("the rig maps no foot joints, so the floor cannot be found")
    return min(points)


def penetration(foot_track: np.ndarray, floor: float, height: float, up=Z_UP) -> float:
    """Deepest foot below `floor` over the clip, as a fraction of body height.

    `foot_track` is [T, P, 3] posed sole samples (see `_foot_track`), `floor` the rest
    floor, and `height` the standing height. Zero means no foot ever goes through.
    """
    if height <= 0:
        raise ValueError("height must be positive")
    low = np.asarray(foot_track) @ up
    return float(max(0.0, floor - float(low.min())) / height)


def _horizontal(v: np.ndarray, up=Z_UP) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v - np.expand_dims(v @ up, -1) * up


def loop_gap(frames, rest, parents, order, height: float, up=Z_UP) -> float:
    """Distance between the last and first pose, as a fraction of body height.

    Horizontal root travel is removed first: a walk that has moved forward is not a pop,
    and games move the character themselves. Vertical root height is kept, because a
    loop that jumps in height is a pop too.
    """
    if height <= 0:
        raise ValueError("height must be positive")
    heads = _pose_heads(frames, rest, parents, order)
    root_index = order.index(order[0])
    relative = heads - _horizontal(heads[:, root_index], up)[:, None, :]
    return float(np.linalg.norm(relative[-1] - relative[0], axis=-1).max() / height)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with edge padding, so the clip's ends do not tip."""
    if window <= 1 or len(values) < 2:
        return np.asarray(values, dtype=float)
    window = min(int(window), len(values))
    pad = window // 2
    padded = np.pad(np.asarray(values, dtype=float), pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def ground_frames(frames, rest, parents, order, tails, mapping, floor: float,
                  up=Z_UP, smooth_frames: int = GROUND_SMOOTH_FRAMES, track=None):
    """Shift each frame's root along `up` so the lowest foot sits on `floor`, in place.

    The required lift is smoothed over `smooth_frames` so the body does not jitter, then
    the exact lift is kept wherever smoothing alone would still let a foot through. The
    hips' own bob above the feet is untouched: this moves the whole body, not the pose.
    Pass `track` ([T, P, 3], e.g. `skin_track`) to ground the real sole vertices; left out,
    the foot/toe bone points are used.
    """
    frames = list(frames)
    if track is None:
        track = _foot_track(frames, rest, parents, order, tails, mapping, up)
    low = (np.asarray(track) @ up).min(axis=1)
    # Grounding only ever lifts. Lowering a frame whose feet are already clear would
    # flatten a jump, a flying kick, or the airborne part of a run.
    required = np.maximum(floor - low, 0.0)
    smoothed = _moving_average(required, smooth_frames)
    lift = np.maximum(smoothed, required)
    return [(world_rot, root_pos + float(offset) * up)
            for (world_rot, root_pos), offset in zip(frames, lift)]


def skin_track(frames, rest, parents, order, points, weights) -> np.ndarray:
    """[T, P, 3] posed positions of skin vertices, by linear blend skinning.

    `points` is [P, 3] rest positions in armature space and `weights` is one
    ``{bone: weight}`` dict per point. This is what Blender's armature modifier does,
    and it lets the floor be measured on the actual sole instead of the ankle.
    """
    points = np.asarray(points, dtype=float)
    out = []
    for world_rot, root_pos in frames:
        pose = pose_matrices(rest, parents, order, world_rot, root_pos)
        moved = {b: pose[b] @ np.linalg.inv(rest[b]) for b in order}
        frame = []
        for point, pair in zip(points, weights):
            total = np.zeros(3)
            for bone, weight in pair.items():
                total += float(weight) * (moved[bone] @ np.append(point, 1.0))[:3]
            frame.append(total)
        out.append(frame)
    return np.asarray(out)


def lift_above_floor(frames, rest, parents, order, tails, mapping, floor: float,
                     up=Z_UP, tolerance: float = 1e-9, track=None):
    """Raise only the frames whose lowest foot is below `floor`; leave the rest alone.

    Used after loop blending, which can dip the last few frames even though grounding
    ran first. Frames already on or above the floor keep their smoothed height.
    """
    frames = list(frames)
    if track is None:
        track = _foot_track(frames, rest, parents, order, tails, mapping, up)
    low = (np.asarray(track) @ up).min(axis=1)
    out = []
    for (world_rot, root_pos), height in zip(frames, low):
        lift = float(floor - height)
        out.append((world_rot, root_pos + (lift if lift > tolerance else 0.0) * up))
    return out


def foot_pitch(head, tail, up=Z_UP) -> float:
    """How far a foot points up from level, in radians (negative: toe down)."""
    v = np.asarray(tail, dtype=float) - np.asarray(head, dtype=float)
    run = float(np.linalg.norm(_horizontal(v, up)))
    rise = float(v @ up)
    if run < 1e-9 and abs(rise) < 1e-9:
        return 0.0
    return float(np.arctan2(rise, run))


def _rotate_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit AXIS."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def tilt_to_pitch(head, tail, angle: float, up=Z_UP) -> np.ndarray:
    """Rotation that adds ANGLE to a foot's pitch (positive lifts the toe), about its ankle.

    A foot points along head->tail; the turn is about the horizontal axis across it, so the
    foot keeps facing the way it did and only its angle to the floor changes.
    """
    v = np.asarray(tail, dtype=float) - np.asarray(head, dtype=float)
    level = _horizontal(v, up)
    if np.linalg.norm(level) < 1e-9:
        return np.eye(3)
    axis = np.cross(level / np.linalg.norm(level), up)
    return _rotate_about(axis, -angle)


def _foot_sides(mapping: dict[str, str], rest) -> list[tuple[str, str | None]]:
    """(foot bone, toe bone or None) for each side of the rig."""
    sides = []
    for side in ("Left", "Right"):
        foot = next((b for b, role in mapping.items()
                     if role == f"{side}Foot" and b in rest), None)
        toe = next((b for b, role in mapping.items()
                    if role == f"{side}ToeBase" and b in rest), None)
        if foot is not None:
            sides.append((foot, toe))
    return sides


def _tip(frames_pose, rest, tails, bone):
    return posed_segments(rest, frames_pose, tails, [bone])[bone][1]


def plant_feet(frames, rest, parents, order, tails, mapping, floor: float, height: float,
               up=Z_UP, tolerance: float = PLANT_TOLERANCE,
               smooth_frames: int = PLANT_SMOOTH_FRAMES):
    """Put a planted foot back on the sole angle the character was built with.

    The retarget copies the preset skeleton's foot angle onto a rig whose feet rest at another
    one, so a planted toe ends up pointing up: on the marine it sat 2.2 cm above the heel,
    where at rest it sat level. Judging by the foot bone alone is not enough there, because on
    these rigs the foot bone points almost straight down into the boot (a rest pitch near -76
    degrees) and says little about the sole; the bend at the ball is a separate bone.

    So, while a foot is on the floor, put the toe back at its rest bend and then turn the foot
    and its toes about the ankle until the toe tip sits at the height it has at rest. A foot in
    the air is left alone: a jump's pointed toe is the move, not a bug.
    """
    frames = list(frames)
    sides = _foot_sides(mapping, rest)
    if not frames or not sides or height <= 0:
        return frames
    moved = [(dict(world), np.array(root, dtype=float)) for world, root in frames]
    for foot, toe in sides:
        chain = _subtree(foot, parents)
        tip_bone = toe or foot
        rest_ankle = float(rest[foot][:3, 3] @ up)
        rest_lift = rest_ankle - floor                      # the ankle rides this high at rest
        rest_tip_gap = float(tails[tip_bone] @ up) - rest_ankle   # the tip above the ankle
        bend = (rest[foot][:3, :3].T @ rest[toe][:3, :3]) if toe is not None else None
        turns, planted = [], []
        for world, root in frames:
            pose = pose_matrices(rest, parents, order, world, root)
            ankle = pose[foot][:3, 3]
            on = float(ankle @ up) - floor <= rest_lift + tolerance * height
            planted.append(on)
            if not on:
                turns.append(0.0)
                continue
            probe = dict(world)
            if toe is not None:
                probe[toe] = probe[foot] @ bend             # the rest bend at the ball
            tip = _tip(pose_matrices(rest, parents, order, probe, root), rest, tails, tip_bone)
            arm = np.asarray(tip) - ankle
            span = float(np.linalg.norm(arm))
            if span < 1e-9:
                turns.append(0.0)
                continue
            have = float(np.arcsin(np.clip(float(arm @ up) / span, -1.0, 1.0)))
            want = float(np.arcsin(np.clip(rest_tip_gap / span, -1.0, 1.0)))
            turns.append(have - want)                       # tilt_to_pitch takes off this much
        if not any(planted):
            continue
        smoothed = _moving_average(np.array(turns), smooth_frames)
        # Keep the exact turn on a planted foot; let smoothing carry a foot that lifts.
        angle = np.where(planted & (np.abs(turns) > np.abs(smoothed)), turns, smoothed)
        for index, (world, root) in enumerate(moved):
            if not planted[index]:
                continue
            if toe is not None:
                world[toe] = world[foot] @ bend
            if abs(float(angle[index])) < 1e-9:
                continue
            pose = pose_matrices(rest, parents, order, world, root)
            ankle = pose[foot][:3, 3]
            tip = _tip(pose, rest, tails, tip_bone)
            spin = tilt_to_pitch(ankle, tip, float(angle[index]), up)
            for b in chain:
                world[b] = spin @ np.asarray(world[b])
    return moved


def planted_tilt(frames, rest, parents, order, tails, mapping, floor: float, height: float,
                 up=Z_UP, tolerance: float = PLANT_TOLERANCE) -> float:
    """The worst planted toe tip's height above its rest height, as a share of body height."""
    worst = 0.0
    for foot, toe in _foot_sides(mapping, rest):
        tip_bone = toe or foot
        rest_tip = float(tails[tip_bone] @ up)
        rest_lift = float(rest[foot][:3, 3] @ up) - floor
        for world, root in frames:
            pose = pose_matrices(rest, parents, order, world, root)
            if float(pose[foot][:3, 3] @ up) - floor > rest_lift + tolerance * height:
                continue
            tip = _tip(pose, rest, tails, tip_bone)
            worst = max(worst, abs(float(tip @ up) - rest_tip))
    return float(worst / height)


def horizontal_travel(frames, up=Z_UP) -> float:
    """Net horizontal distance between the first and last root position."""
    frames = list(frames)
    if not frames:
        return 0.0
    return float(np.linalg.norm(_horizontal(frames[-1][1] - frames[0][1], up)))


def in_place_frames(frames, up=Z_UP):
    """Remove horizontal root travel from every frame, keeping height and rotations.

    A walk, run, sneak or strafe then stays on the spot: the game moves the character.
    The vertical bob is untouched, and all bone rotations are untouched.
    """
    frames = list(frames)
    if not frames:
        return frames
    origin = frames[0][1]
    return [(world_rot, root_pos - _horizontal(root_pos - origin, up))
            for world_rot, root_pos in frames]


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path spherical interpolation between two (w, x, y, z) quaternions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - t) * theta) * a + np.sin(t * theta) * b) / np.sin(theta)


def moving_frames(frames, height: float | None, up=Z_UP,
                  share: float = LOOP_SPEED_SHARE) -> np.ndarray | None:
    """Which frames move at near the clip's own top speed, or None when speed says nothing.

    Kimodo clips open standing and often settle at the end. Two still frames match each
    other perfectly, so the picker used to choose a sprint's standing wind-up as its cycle.
    Only travelling clips get this filter: an idle sways on the spot, and every frame of it
    is a fair place to start a loop.
    """
    frames = list(frames)
    n = len(frames)
    if height is None or height <= 0 or n < 3:
        return None
    if horizontal_travel(frames, up) < LOOP_TRAVEL_SHARE * height:
        return None
    roots = np.array([root for _, root in frames])
    steps = np.linalg.norm(_horizontal(np.diff(roots, axis=0), up), axis=1)
    speed = np.empty(n)
    speed[0], speed[-1] = steps[0], steps[-1]
    speed[1:-1] = (steps[:-1] + steps[1:]) / 2
    top = float(speed.max())
    if top <= 0:
        return None
    return speed >= share * top


def loop_candidates(frames, rest, parents, order, up=Z_UP,
                    min_frames: int = LOOP_MIN_FRAMES,
                    min_share: float = LOOP_MIN_SHARE,
                    height: float | None = None) -> list[tuple[int, int, float]]:
    """The best frame pair for each distinct cycle length, best match first.

    Each entry is ``(start, end, distance)``; `end` is excluded and `distance` is the
    largest joint move between the two poses (horizontal root travel removed, vertical
    height kept). Comparing per bone, not as an unordered cloud of points, is what stops
    a mirrored half-step (left leg forward vs right leg forward) counting as a match.

    With HEIGHT (a travelling clip), a cycle's ends must be frames that are actually
    moving: see `moving_frames`.
    """
    frames = list(frames)
    n = len(frames)
    if n < 2:
        return []
    shortest = max(int(min_frames), int(round(n * min_share)))
    if shortest >= n:
        return []
    heads = _pose_heads(frames, rest, parents, order)
    root_index = order.index(order[0])
    relative = heads - _horizontal(heads[:, root_index], up)[:, None, :]
    usable = moving_frames(frames, height, up)
    spots = range(n) if usable is None else [i for i in range(n) if usable[i]]
    by_length: dict[int, tuple[int, int, float]] = {}
    for start in spots:
        for end in spots:
            if end - start < shortest:
                continue
            distance = float(np.linalg.norm(relative[end] - relative[start], axis=-1).max())
            length = end - start
            if length not in by_length or distance < by_length[length][2]:
                by_length[length] = (start, end, distance)
    return sorted(by_length.values(), key=lambda item: item[2])


def best_loop(frames, rest, parents, order, up=Z_UP,
              min_frames: int = LOOP_MIN_FRAMES, min_share: float = LOOP_MIN_SHARE,
              height: float | None = None) -> tuple[int, int]:
    """The pair of frames whose poses match best: `(start, end)`, with `end` excluded.

    See `loop_candidates` for how poses are compared. Falls back to the whole clip when
    nothing is long enough to call a cycle.
    """
    candidates = loop_candidates(frames, rest, parents, order, up, min_frames, min_share, height)
    return (candidates[0][0], candidates[0][1]) if candidates else (0, len(frames))


def loop_trim(frames, rest, parents, order, up=Z_UP, height: float | None = None,
              keep_whole: bool = False):
    """(start, end, candidates) for --loop: the best cycle, or every frame when kept whole.

    Moves built from distinct beats (an idle that looks left, then right) have no repeating
    stretch to cut to; trimming one to its longest matching pair threw away 5 of its 7
    seconds. Kept whole, every frame stays and only the seam is blended.
    """
    if keep_whole:
        return 0, len(frames), []
    candidates = loop_candidates(frames, rest, parents, order, up, height=height)
    if not candidates:
        return 0, len(frames), []
    start, end, _ = candidates[0]
    return start, end, candidates


def seam_blend_frames(gap: float, height: float, base: int = LOOP_BLEND_FRAMES,
                      cap: int = LOOP_BLEND_MAX) -> int:
    """How many frames to blend at the seam, from how far apart its two ends are.

    `gap` is `loop_gap`'s answer (a fraction of `height`). A walk's ends are about 2% of
    height apart and five frames blend it cleanly; a sprint's are 6-10%, where five frames
    left a visible hitch, so the blend grows with the gap.
    """
    if height <= 0:
        return base
    over = max(0.0, gap / height * 100 - LOOP_BLEND_FROM_PERCENT)
    extra = int(round(over * LOOP_BLEND_PER_PERCENT))
    return int(min(cap, max(base, base + extra)))


def _smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def blend_loop(frames, start: int, end: int, blend_frames: int = LOOP_BLEND_FRAMES,
               up=Z_UP):
    """Trim to the cycle `[start, end)` and blend its tail into its first pose.

    Rotations blend as quaternions (shortest path) and the root's height blends too, so
    the seam is smooth. The duplicated last frame is then dropped: keeping it would show
    the first pose twice at the wrap, a one-frame hitch. Horizontal travel is left alone:
    games move the character, and rewinding it would slide them backwards.
    """
    clipped = list(frames)[start:end]
    n = len(clipped)
    if n < 2 or blend_frames < 1:
        return clipped
    _, first_root = clipped[0]
    first_quat = {b: matrices_to_quaternions(np.asarray(m)) for b, m in clipped[0][0].items()}
    tail = min(int(blend_frames), n - 1)
    for step, index in enumerate(range(n - tail, n), start=1):
        world, root = clipped[index]
        t = _smoothstep(step / tail)
        world = {b: quaternions_to_matrices(_slerp(matrices_to_quaternions(np.asarray(m)),
                                                   first_quat[b], t))
                 for b, m in world.items()}
        height = (first_root - root) @ up
        clipped[index] = (world, root + float(height) * up * t)
    return clipped[:-1]


def _step_gaps(frames, rest, parents, order, height: float, up=Z_UP) -> list[float]:
    """The largest joint move between each frame and the next, in shared units."""
    return [loop_gap([a, b], rest, parents, order, height, up)
            for a, b in itertools.pairwise(frames)]


def trim_duplicate_tail(frames, rest, parents, order, height: float, up=Z_UP,
                        share: float = DUPLICATE_SHARE):
    """Drop trailing frames that sit on top of the first pose.

    A clip that is already a clean loop ends with a copy of its first frame. `blend_loop` drops
    that copy, but blending pulls the frame before it almost onto the first pose, so the wrap
    can still be a hair apart and the move stutters. Keep dropping the tail until the wrap is
    at least SHARE of a normal frame step away.
    """
    frames = list(frames)
    if len(frames) < 3 or height <= 0:
        return frames
    gaps = _step_gaps(frames, rest, parents, order, height, up)
    step = float(np.median(gaps)) if gaps else 0.0
    if step <= 0:
        return frames
    while len(frames) > 2 and loop_gap([frames[-1], frames[0]], rest, parents, order,
                                       height, up) < share * step:
        frames = frames[:-1]
    return frames


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


def retarget_pose_frames(rest, parents, order, mapping, axes, global_rots, hips_pos,
                         hip_scale: float, align=None, frame=None, spread_degrees: float = 0.0):
    """Yield ``(world_rot, root_pos)`` for every frame of a Kimodo clip.

    `axes` is `soma_to_rig_axes(...)` expressed in armature space, `global_rots` Kimodo's
    [T, 77, 3, 3] `global_rot_mats`, `hips_pos` its [T, 3] hips track. `align` comes from
    `rest_alignment`; `frame` and `spread_degrees` drive `arm_spread`.

    This is the pose layer: grounding, loop trimming and arm clearance all operate on the
    list it returns, so each one is a plain function a test can call without bpy.
    """
    root = order[0]
    rest_rot = {b: rest[b][:3, :3] for b in order}
    align = align or {b: np.eye(3) for b in order}
    upper = {lr: next((b for b in order if mapping[b] == f"{lr}Arm"), None) for lr in ("Left", "Right")}
    soma_arm_dir = {lr: axes @ np.array([1.0 if lr == "Left" else -1.0, 0, 0]) for lr in ("Left", "Right")}
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
        yield world_rot, rest[root][:3, 3] + travel


def retarget_frames(rest, parents, order, mapping, axes, global_rots, hips_pos,
                    hip_scale: float, align=None, frame=None, spread_degrees: float = 0.0,
                    tails=None, clearance=None):
    """Yield `pose_basis` dicts for every frame, after optional arm clearance.

    See `retarget_pose_frames` for the arguments. With `clearance` (an
    `arm_clearance.Proxy`) and the rest `tails`, arms are kept out of the body.
    """
    frames = list(retarget_pose_frames(rest, parents, order, mapping, axes, global_rots,
                                       hips_pos, hip_scale, align, frame, spread_degrees))
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


def _skinned_meshes(arm):  # pragma: no cover - needs bpy
    import bpy

    return [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == arm]


def mesh_bounds(arm, up):  # pragma: no cover - needs bpy
    """(floor, top) of the skinned meshes' rest pose, in armature space along `up`."""
    inv = np.array(arm.matrix_world.inverted())
    low, high = np.inf, -np.inf
    for mesh in _skinned_meshes(arm):
        to_arm = inv @ np.array(mesh.matrix_world)
        for v in mesh.data.vertices:
            height = float((to_arm @ np.append(np.array(v.co), 1.0))[:3] @ up)
            low, high = min(low, height), max(high, height)
    return low, high


def foot_vertices(arm, mapping):  # pragma: no cover - needs bpy
    """Rest positions and skin weights of the vertices that make up the soles.

    Grounding against the actual sole, not the ankle joint, is what stops a boot from
    clipping the floor when the foot rolls. Vertices are used when a foot or toe bone is
    their strongest influence; the strongest four influences are normalised.
    """
    inv = np.array(arm.matrix_world.inverted())
    foot = set(_foot_bones(mapping))
    points, weights = [], []
    for mesh in _skinned_meshes(arm):
        to_arm = inv @ np.array(mesh.matrix_world)
        names = {g.index: g.name for g in mesh.vertex_groups}
        for v in mesh.data.vertices:
            pairs = [(names[g.group], float(g.weight)) for g in v.groups
                     if names.get(g.group) in foot and g.weight > 0]
            if not pairs or max(weight for _, weight in pairs) < 0.2:
                continue
            pairs.sort(key=lambda pair: -pair[1])
            pairs = pairs[:4]
            total = sum(weight for _, weight in pairs)
            points.append((to_arm @ np.append(np.array(v.co), 1.0))[:3])
            weights.append({bone: weight / total for bone, weight in pairs})
    return np.asarray(points), weights


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
    ap.add_argument("--no-ground", action="store_true",
                    help="let feet pass through the rest floor (debug; grounding is on by default)")
    ap.add_argument("--loop", action="store_true",
                    help="trim to the best cycle and blend the seam; for clips that repeat")
    ap.add_argument("--in-place", action="store_true",
                    help="remove horizontal root travel, keeping the vertical bob; for walking clips")
    ap.add_argument("--keep-whole", action="store_true",
                    help="with --loop, keep every frame and only blend the seam; for moves built "
                         "from distinct beats, which have no cycle to trim to")
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

    tpose = soma_tpose(g[:1], joints0[None])
    align = None if args.no_align else rest_alignment(parents, heads, mapping, axes, tpose)
    frames = list(retarget_pose_frames(rest, parents, order, mapping, axes, g, hips,
                                       scale, align, frame, args.arm_spread))
    clearance = None if args.no_clearance else body_proxy(arm, mapping, heads, tails, up)
    if clearance is not None:
        clear_arms(frames, rest, parents, order, tails, clearance)

    before_travel = horizontal_travel(frames, up)
    if args.in_place:
        frames = in_place_frames(frames, up)

    floor = _rest_floor(rest, tails, mapping, up)
    height = max(float(heads[b] @ up) for b in heads) - floor

    sole_points, sole_weights = foot_vertices(arm, mapping)
    mesh_low, mesh_high = mesh_bounds(arm, up)
    if len(sole_points) and np.isfinite(mesh_low) and mesh_high > mesh_low:
        floor, height = mesh_low, mesh_high - mesh_low
        def track(fs):
            return skin_track(fs, rest, parents, order, sole_points, sole_weights)
    else:  # a rig with no skinned sole vertices: fall back to the ankle joints
        def track(fs):
            return _foot_track(fs, rest, parents, order, tails, mapping, up)

    before_pen = penetration(track(frames), floor, height, up)
    before_gap = loop_gap(frames, rest, parents, order, height, up)

    trim = None
    if args.loop:
        start, end, candidates = loop_trim(frames, rest, parents, order, up, height,
                                           keep_whole=args.keep_whole)
        trim = (start, end, candidates)
        frames = frames[start:end]
    if not args.no_ground:
        frames = ground_frames(frames, rest, parents, order, tails, mapping, floor, up,
                               track=track(frames))
    blend = LOOP_BLEND_FRAMES
    if args.loop:
        seam_gap = loop_gap(frames, rest, parents, order, height, up)
        blend = seam_blend_frames(seam_gap, height)
        frames = blend_loop(frames, 0, len(frames), blend, up=up)
        frames = trim_duplicate_tail(frames, rest, parents, order, height, up)
    if not args.no_ground:
        frames = plant_feet(frames, rest, parents, order, tails, mapping, floor, height, up)
        frames = lift_above_floor(frames, rest, parents, order, tails, mapping, floor, up,
                                  track=track(frames))

    after_tilt = planted_tilt(frames, rest, parents, order, tails, mapping, floor, height, up)
    after_pen = penetration(track(frames), floor, height, up)
    after_gap = loop_gap(frames, rest, parents, order, height, up)
    after_travel = horizontal_travel(frames, up)
    detail = ""
    if args.loop:
        start, end, candidates = trim
        if args.keep_whole:
            detail = f"; kept all {len(frames) + 1} frames, blend {blend}"
        else:
            runner = ""
            if len(candidates) > 1:
                length = candidates[1][1] - candidates[1][0]
                runner = f", runner-up {length} frames at {100 * candidates[1][2] / height:.2f}%"
            detail = (f"; chose frames {start}..{end} ({end - start} frames, "
                      f"distance {100 * candidates[0][2] / height:.2f}%{runner}, blend {blend})")
    print(f"grounding: penetration {100 * before_pen:.2f}% -> {100 * after_pen:.2f}% of height "
          f"({'off' if args.no_ground else 'on'})")
    print(f"feet: planted toe tip {100 * after_tilt:.2f}% of height off its rest height "
          f"({'off' if args.no_ground else 'on'})")
    print(f"loop: gap {100 * before_gap:.2f}% -> {100 * after_gap:.2f}% of height "
          f"({'trimmed' if args.loop else 'off'}, {len(frames)} frames){detail}")
    print(f"in place: root travel {100 * before_travel / height:.1f}% -> "
          f"{100 * after_travel / height:.1f}% of height "
          f"({'on' if args.in_place else 'off'})")

    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start = 1
    scene.frame_end = int(np.ceil(key_frame(len(frames), args.speed)))
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    for t, (world_rot, root_pos) in enumerate(frames, start=1):
        basis = pose_basis(rest, parents, order, world_rot, root_pos)
        for name, mtx in basis.items():
            pb = arm.pose.bones[name]
            pb.matrix_basis = Matrix(mtx.tolist())
            pb.keyframe_insert("rotation_quaternion", frame=key_frame(t, args.speed))
            if parents[name] is None:
                pb.keyframe_insert("location", frame=key_frame(t, args.speed))
    print(f"retargeted {len(frames)} frames onto {len(order)} bones, hip scale {scale:.3f}")
    print("side check:", {k: v for k, v in mapping.items() if v in ("LeftHand", "RightHand", "Head")})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(args.out), export_format="GLB", export_animations=True)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend))


if __name__ == "__main__":
    main()
