"""Keep a retargeted character's arms out of its own body.

Motion made for a slim actor drives a bulky character's forearms into its chest plate (a
punch on an armoured knight) or its hands into its face (a guard on a big-headed stylised
hero): the angles transfer, the bodies do not. This module gives the body a rough outline
of capsules, one per spine, neck, head, shoulder and thigh bone, each as thick as the
character's own skin around that bone. On every frame, when an elbow, forearm, wrist or
hand comes inside one, the whole arm swings out about the shoulder just far enough to
clear it. The swings are then smoothed over neighbouring frames so the arm never pops.

Pure numpy; `kimodo_retarget.py` feeds it posed bone segments from Blender.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDES = ("Left", "Right")
# SOMA roles that make up the body the arms must stay out of: the torso, neck and head,
# where a forearm through a chest plate or a hand through a face shows. Hips and thighs are
# left out on purpose: hanging hands rest beside them, and their outlines (wide trousers,
# bulky legs) pushed every hanging arm 20-50 degrees outward on walks and idles (2026-10-06,
# the bald sensei). A hand brushing a thigh reads far better than arms held out like wings.
BODY_ROLES = ("Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
              "LeftShoulder", "RightShoulder")
ARM_ROLES = ("Arm", "ForeArm", "Hand")
# How much of a bone's skin counts as its thickness. Below the maximum, so a stray vertex
# (a spike of hair, a pauldron's rim) does not inflate the whole capsule.
BODY_PERCENTILE = 85
# A skull is never much wider than the torso; anything past that is hair, a helmet or a hat,
# which a fist in a guard may overlap. Spiky hair made one head 1.5x a plain head and flung
# guard fists out beside it (2026-10-07), so the head counts as at most this x the torso.
HEAD_TO_TORSO = 1.1
TORSO_ROLES = ("Spine1", "Spine2", "Chest")
ARM_PERCENTILE = 60


def point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """(distance, closest point) from `p` to the segment a-b."""
    axis = b - a
    length2 = float(axis @ axis)
    t = 0.0 if length2 == 0 else min(1.0, max(0.0, float((p - a) @ axis) / length2))
    closest = a + t * axis
    return float(np.linalg.norm(p - closest)), closest


def skin_radii(points: np.ndarray, owners: list[str], segments: dict[str, tuple],
               bones: list[str], percentile: float) -> dict[str, float]:
    """For each bone, how far its own vertices (those it mainly moves) sit from it."""
    owners_arr = np.asarray(owners)
    radii = {}
    for bone in bones:
        mine = points[owners_arr == bone]
        if len(mine) == 0 or bone not in segments:
            continue
        a, b = segments[bone]
        radii[bone] = float(np.percentile([point_segment(p, a, b)[0] for p in mine], percentile))
    return radii


def limit_head(radii: dict[str, float], head_bones: set[str], torso_bones: set[str]) -> dict[str, float]:
    """The radii with each head bone no wider than HEAD_TO_TORSO x the widest torso bone."""
    torso = max((r for b, r in radii.items() if b in torso_bones), default=None)
    if torso is None:
        return dict(radii)
    return {b: min(r, torso * HEAD_TO_TORSO) if b in head_bones else r for b, r in radii.items()}


@dataclass
class Proxy:
    body: dict[str, float]       # rig bone -> capsule radius
    arms: dict[str, dict]        # side -> {"upper", "fore", "hand", "radius": {bone: r}}
    side_shoulder: dict[str, str | None]
    margin: float


def build_proxy(mapping: dict[str, str], points: np.ndarray, owners: list[str],
                segments: dict[str, tuple], height: float) -> Proxy | None:
    """The body outline for one rig, from its rest pose. None if it has no arms to fix."""
    by_role = {role: bone for bone, role in mapping.items()}
    body_bones = [by_role[r] for r in BODY_ROLES if r in by_role]
    arms = {}
    for side in SIDES:
        bones = [by_role.get(f"{side}{r}") for r in ARM_ROLES]
        if None in bones:
            continue
        radius = skin_radii(points, owners, segments, bones, ARM_PERCENTILE)
        arms[side] = {"upper": bones[0], "fore": bones[1], "hand": bones[2],
                      "radius": {b: radius.get(b, 0.0) for b in bones}}
    if not arms:
        return None
    body = limit_head(skin_radii(points, owners, segments, body_bones, BODY_PERCENTILE),
                      head_bones={by_role["Head"]} if "Head" in by_role else set(),
                      torso_bones={by_role[r] for r in TORSO_ROLES if r in by_role})
    return Proxy(body=body,
                 arms=arms,
                 side_shoulder={s: by_role.get(f"{s}Shoulder") for s in SIDES},
                 margin=0.01 * height)


def _test_points(seg: dict[str, tuple], arm: dict) -> list[tuple[np.ndarray, float]]:
    fore_a, fore_b = seg[arm["fore"]]
    hand_a, hand_b = seg[arm["hand"]]
    r = arm["radius"]
    return [(fore_a, r[arm["fore"]]), ((fore_a + fore_b) / 2, r[arm["fore"]]),
            (hand_a, r[arm["hand"]]), (hand_b, r[arm["hand"]])]


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _sideways(direction: np.ndarray, along: np.ndarray) -> np.ndarray:
    """`direction` with its component along `along` removed, or zeros if nothing is left."""
    u = along / np.linalg.norm(along)
    side = direction - (direction @ u) * u
    n = float(np.linalg.norm(side))
    return side / n if n > 1e-6 else np.zeros(3)


def arm_swing(seg: dict[str, tuple], proxy: Proxy, side: str, iterations: int = 10) -> np.ndarray:
    """Rotation (3x3, about the shoulder) that takes one arm out of the body on this frame.

    `seg` maps rig bones to their posed (head, tail) in one shared space. Identity when
    the arm is already clear.
    """
    arm = proxy.arms[side]
    pivot = seg[arm["upper"]][0]
    capsules = [(seg[b][0], seg[b][1], r) for b, r in proxy.body.items()
                if b != proxy.side_shoulder.get(side) and b in seg]
    points = _test_points(seg, arm)
    total = np.eye(3)
    for _ in range(iterations):
        worst, push = 0.0, None
        for p, ra in points:
            for a, b, rb in capsules:
                d, closest = point_segment(p, a, b)
                depth = rb + ra + proxy.margin - d
                if depth > worst and d > 1e-9:
                    # A shoulder swing moves a point only sideways to the arm; a push
                    # along the arm itself (an arm straight through the body) would do
                    # nothing, so fall back to the way the shoulder sits off the body.
                    away = _sideways(p - closest, p - pivot)
                    if not away.any():
                        away = _sideways(pivot - point_segment(pivot, a, b)[1], p - pivot)
                    if not away.any():  # shoulder on that line too: turn across the body axis
                        away = _sideways(np.cross(b - a, p - pivot), p - pivot)
                    if away.any():
                        worst, push = depth, (p, away)
        if push is None:
            break
        p, normal = push
        step = _rotation_between(p - pivot, p + normal * worst - pivot)
        total = step @ total
        points = [(pivot + step @ (q - pivot), r) for q, r in points]
    return total


def _log(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis * angle)."""
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def _exp(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def swing_degrees(rotation: np.ndarray) -> float:
    """How far a swing turns the arm, in degrees."""
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0))))


def smooth_swings(swings: list[np.ndarray], radius: int = 4) -> list[np.ndarray]:
    """Spread each frame's swing over its neighbours, so a correction eases in and out
    instead of snapping. Uses the larger of the raw and smoothed swing per frame, so
    smoothing never lets an arm back into the body."""
    vectors = np.array([_log(s) for s in swings])
    weights = np.exp(-0.5 * (np.arange(-radius, radius + 1) / max(radius / 2, 1)) ** 2)
    out = []
    for t in range(len(swings)):
        lo, hi = max(0, t - radius), min(len(swings), t + radius + 1)
        w = weights[lo - t + radius:hi - t + radius]
        blended = (vectors[lo:hi] * w[:, None]).sum(axis=0) / w.sum()
        out.append(_exp(blended if np.linalg.norm(blended) >= np.linalg.norm(vectors[t])
                        else vectors[t]))
    return out
