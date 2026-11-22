"""Hero hands: finding the wrist, cutting it straight, putting a built hand on and stitching it, counting fingers.

The shapes here are small synthetic stand-ins for what the Revolverheld's rig, mittens and built hands showed
(2026-11-22): an arm whose narrowest section lies between a cuff and the palm, a built hand that is two skins
joined only at its cuff's open top, a drawing whose fingers have to be counted. The graft's claim is tested as a
mesh property: after the stitch, every edge is shared by exactly two faces, so there is no gap and no overlap.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from image_to_3dlab import hero_hands as hh

# --- shapes -------------------------------------------------------------------------------------------------


def turned(profile, ring=24, squash=(1.0, 1.0)):
    """A closed surface of revolution round the y axis: PROFILE is (y, radius) from the top down, the last
    radius 0 (a tip). Squash scales x and z (a flat hand). Faces face out."""
    rings = []
    for y, r in profile:
        a = np.linspace(0, 2 * np.pi, ring, endpoint=False)
        rings.append(np.stack([r * np.cos(a) * squash[0], np.full(ring, y), r * np.sin(a) * squash[1]], 1))
    vertices = [np.array([[0.0, profile[0][0], 0.0]])]          # top cap's centre
    faces = []
    for k, points in enumerate(rings[:-1]):
        vertices.append(points)
    tip = len(rings) - 1
    base = 1
    for j in range(ring):                                        # the top cap
        faces.append([0, base + (j + 1) % ring, base + j])
    for k in range(tip - 1):
        a0, b0 = base + k * ring, base + (k + 1) * ring
        for j in range(ring):
            j1 = (j + 1) % ring
            faces += [[a0 + j, a0 + j1, b0 + j], [a0 + j1, b0 + j1, b0 + j]]
    vertices.append(np.array([[0.0, profile[-1][0], 0.0]]))
    last, end = base + (tip - 1) * ring, sum(len(v) for v in vertices) - 1
    for j in range(ring):
        faces.append([last + j, last + (j + 1) % ring, end])
    return np.concatenate(vertices), np.array(faces)


def closed_edges(vertices, faces):
    """(edges used by other than two faces, directed edges used twice) of a mesh welded by position."""
    group = hh.weld_groups(vertices)
    f = group[np.asarray(faces)]
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    directed = directed[directed[:, 0] != directed[:, 1]]
    _, counts = np.unique(np.sort(directed, 1), axis=0, return_counts=True)
    _, same = np.unique(directed, axis=0, return_counts=True)
    return int((counts != 2).sum()), int((same > 1).sum())


# Top down: forearm, a wider cuff, the wrist (narrowest), the palm, the fingers' taper to a tip.
ARM = [(0.30, 0.030), (0.20, 0.030), (0.15, 0.030), (0.149, 0.040), (0.10, 0.040), (0.099, 0.022),
       (0.08, 0.022), (0.06, 0.030), (0.02, 0.028), (-0.02, 0.012), (-0.05, 0.0)]
# A built hand, ten times the size, fingers down: cuff, wrist, palm, tip.
HAND = [(0.50, 0.20), (0.30, 0.20), (0.299, 0.12), (0.25, 0.12), (0.10, 0.16), (-0.10, 0.12), (-0.40, 0.0)]


def figure_arm(x=-0.2):
    """A right arm hanging straight down at X, its hand flat (palm to the body), as the figure's part with one-hot
    weights: joint 1 the forearm (above the rig's wrist joint at y 0.12), joint 2 the hand."""
    v, f = turned(ARM)
    hand = v[:, 1] < 0.12
    v[hand, 0] *= 0.6                                         # the old mitten is flat: its thin side faces x
    v[:, 0] += x
    dense = np.zeros((len(v), 3))
    dense[np.arange(len(v)), np.where(hand, 2, 1)] = 1.0
    uvs = np.stack([v[:, 0], v[:, 1]], 1)
    return {"positions": v, "faces": f, "uvs": uvs, "dense": dense}


def built_hand(double=True):
    """A built hand in Pixal3D's frame (fingers down, back of the hand to +Z, flat in Z). DOUBLE adds the second
    skin Pixal3D makes, 2 % inside, joined to the outer one only at the cuff's open top."""
    v, f = turned(HAND, squash=(1.0, 0.5))
    if not double:
        return {"positions": v, "faces": f, "uvs": v[:, :2].copy()}
    inner_v, inner_f = turned([(y, r) for y, r in HAND[:-1]] + [(-0.39, 0.0)], squash=(0.98, 0.49))
    ring = 24
    outer_top, inner_top = 1, len(v) + 1
    keep = np.flatnonzero((f != 0).all(axis=1))               # open both tops: drop their caps ...
    inner_keep = np.flatnonzero((inner_f != 0).all(axis=1))
    faces = [f[keep], inner_f[inner_keep][:, ::-1] + len(v)]  # ... the inner skin faces inward ...
    for j in range(ring):                                     # ... and join them at the rim
        a, b = outer_top + j, outer_top + (j + 1) % ring
        c, d = inner_top + j, inner_top + (j + 1) % ring
        faces.append(np.array([[a, c, b], [b, c, d]]))
    v = np.concatenate([v, inner_v])
    return {"positions": v, "faces": np.concatenate(faces), "uvs": v[:, :2].copy()}


# --- mesh pieces --------------------------------------------------------------------------------------------

def test_weld_groups_joins_vertices_at_one_place():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 1e-9, 0]], float)
    g = hh.weld_groups(v)
    assert g[0] == g[2] and g[1] == g[3] and g[0] != g[1]


def test_chain_loops_walks_closed_loops_and_keeps_open_chains():
    loops = hh.chain_loops([[0, 1], [1, 2], [2, 0], [5, 6], [6, 7], [7, 5], [10, 11], [11, 12]])
    as_sets = sorted(sorted(loop.tolist()) for loop in loops)
    assert as_sets == [[0, 1, 2], [5, 6, 7], [10, 11, 12]]
    open_chain = next(loop for loop in loops if 10 in loop)
    assert open_chain.tolist() == [10, 11, 12]                # walked from its free end


def test_section_of_a_turned_tube_is_one_circle():
    v, f = turned(ARM, ring=48)
    points, faces, _ = hh.weld(v, f)
    loops = hh.section_loops(points, faces, [0, 0.25, 0], [0, -1, 0])
    assert len(loops) == 1
    assert hh.loop_area(loops[0], [0, 1, 0]) == pytest.approx(np.pi * 0.030 ** 2, rel=0.01)


def test_clip_splits_triangles_on_the_plane_and_mixes_the_uvs():
    v, f = turned(ARM)
    uvs = np.stack([v[:, 1], v[:, 0]], 1)                     # u = height: linear, so a cut keeps it exact
    out_v, out_f, (out_uv,), rim = hh.clip(v, f, [0, 0.0, 0], [0, -1, 0], attributes=(uvs,))
    used = np.unique(out_f)
    assert (out_v[used, 1] >= -1e-12).all()                   # nothing below the plane kept
    on_plane = np.unique(rim)
    assert np.allclose(out_v[on_plane, 1], 0.0) and np.allclose(out_uv[on_plane, 0], 0.0)
    loops = hh.chain_loops(rim)
    assert len(loops) == 1 and len(loops[0]) == 48            # one rim, a point per crossed edge (sides, diagonals)


def test_clip_keeps_faces_it_may_not_cut():
    v, f = turned(ARM)
    middle = v[f].mean(axis=1)
    cuttable = middle[:, 0] < 0                               # only the arm's -x half may be cut
    out_v, out_f, _, _ = hh.clip(v, f, [0, 0.0, 0], [0, -1, 0], cuttable)
    kept_below = out_v[out_f].mean(axis=1)
    assert (kept_below[:, 1] < 0).any() and (kept_below[kept_below[:, 1] < -1e-9, 0] > 0).all()


def test_a_uv_seam_across_the_cut_still_gives_one_rim():
    v, f = turned(ARM)
    # split every vertex at x > 0 into its own copy for the faces on the +z side: a seam along the ring
    seam = (v[:, 0] > 0) & (np.abs(v[:, 2]) < 1e-9)
    copies = np.flatnonzero(seam)
    v2 = np.concatenate([v, v[copies]])
    lookup = np.arange(len(v))
    lookup[copies] = len(v) + np.arange(len(copies))
    f2 = f.copy()
    upper = v[f].mean(axis=1)[:, 2] > 0
    f2[upper] = lookup[f[upper]]
    out_v, _, _, rim = hh.clip(v2, f2, [0, 0.0, 0], [0, -1, 0])
    ids, loop = hh.rim_loop(out_v, rim, [0, 1, 0], [0, 0, 0])
    assert len(ids) == 48 and hh.loop_area(loop, [0, 1, 0]) > 0      # as without the seam: one welded rim


def test_outer_wall_drops_the_second_skin_once_cut_below_its_rim():
    hand = built_hand()
    axis = np.array([0.0, -1.0, 0.0])
    part, rim = hh.cut_part(hand, [0, 0.2, 0], -axis)        # keep the fingers' side
    keep = hh.outer_wall(part["positions"], part["faces"], rim, axis)
    single, _ = hh.cut_part(built_hand(double=False), [0, 0.2, 0], -axis)
    assert keep.sum() == len(single["faces"])                # exactly the outer skin's faces ...
    kept = part["positions"][np.unique(part["faces"][keep])]
    assert np.isclose(np.abs(kept[:, 0]).max(), np.abs(single["positions"][:, 0]).max())   # ... not the inner


# --- the wrist ----------------------------------------------------------------------------------------------

def test_seam_index_skips_a_gap_between_fingers():
    # arm -> cuff -> wrist -> palm -> fingers (several pieces) with a deep gap two sections wide -> tip
    areas = [3.0, 3.0, 5.0, 5.0, 2.0, 2.0, 2.2, 3.0, 3.2, 1.0, 1.0, 0.7, 0.7, 1.0, 1.0, 0.4, 0.1]
    counts = [1] * 9 + [4] * 8
    assert hh.seam_index(areas, counts) in (4, 5)
    assert hh.seam_index(areas) in (11, 12)                # without the pieces, the gap would pass for a wrist


def test_seam_index_takes_the_wrist_below_the_cuff_not_the_sleeve_above_it():
    # a drawn hand with a long sleeve (the Revolverheld's seed-4 drawing): sleeve narrowing to a valley, a cuff,
    # the wrist, the palm. By area the sleeve's valley is the deeper one; the wrist is the one above the hand.
    areas = [0.136, 0.131, 0.124, 0.117, 0.111, 0.1085, 0.1085, 0.110, 0.122, 0.140, 0.140, 0.137,
             0.102, 0.082, 0.065, 0.070, 0.078, 0.085, 0.090, 0.087, 0.078, 0.06, 0.04, 0.02]
    counts = [1] * 21 + [5, 5, 4]
    assert hh.seam_index(areas, counts) in (14, 15)


def test_pieces_counts_outlines_side_by_side_not_one_inside_another_nor_specks():
    def ring(cx, r, n=16):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(a), np.zeros(n), r * np.sin(a)], 1)

    axis = [0, 1, 0]
    assert hh.pieces([ring(0, 1.0), ring(0, 0.98)], axis) == 1          # a double wall: one piece
    assert hh.pieces([ring(-3, 0.5), ring(-1, 0.5), ring(1, 0.5), ring(3, 0.5)], axis) == 4   # four fingers
    assert hh.pieces([ring(0, 1.0), ring(2.5, 0.01)], axis) == 1        # a speck beside the wrist


def test_seam_index_keeps_a_wrist_one_section_wide():
    # the Revolverheld's left wrist: narrow for one section below the cuff, a wide median hid it
    areas = [3.6, 3.6, 4.3, 5.1, 3.9, 1.96, 1.70, 1.15, 1.24, 1.33, 1.41, 1.48, 1.54, 1.50, 1.3, 0.9, 0.4]
    assert hh.seam_index(areas) in (7, 8)


def test_seam_index_says_none_without_a_wrist():
    assert hh.seam_index([3.0, 2.8, 2.6, 2.4, 2.0, 1.0, 0.2]) is None


def test_figure_wrist_cuts_below_the_cuff_and_turns_the_back_out():
    arm = figure_arm()
    plan = hh.figure_wrist(arm["positions"], arm["faces"], arm["dense"].argmax(1), 1, [2],
                           np.array([-0.2, 0.30, 0]), np.array([-0.2, 0.12, 0]), "right")
    assert plan["found"]
    assert 0.06 < plan["cut"][1] < 0.099                      # below the cuff's rim at 0.099, above the palm
    assert np.allclose(plan["fingers"], [0, -1, 0], atol=0.05)
    assert plan["back"][0] < -0.95                            # a right hand at -x: its back faces out, -x
    assert plan["turn_check_deg"] < 10                        # the old mitten's palm faces the body too
    assert plan["length"] == pytest.approx(plan["cut"][1] + 0.05, rel=0.1)


def test_figure_back_puts_the_thumb_forward_on_both_hands():
    fingers = hh.unit([-0.5, -0.85, 0.0])
    right = hh.figure_back(fingers, "right")
    assert np.allclose(np.cross(right, fingers), [0, 0, 1], atol=1e-9)       # thumb = back x fingers, forward
    left = hh.figure_back(hh.unit([0.5, -0.85, 0.0]), "left")
    assert left[0] > 0 and right[0] < 0                                       # both backs face out


def test_placement_puts_seam_fingers_and_back_where_asked():
    tf = hh.placement(np.array([1.0, 2.0, 3.0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0]), 10.0,
                      np.array([-0.2, 0.1, 0.0]), hh.unit([-0.3, -0.9, 0.3]), hh.unit([-0.95, 0.31, 0.0]), 0.5)
    assert np.allclose(hh.moved(tf, [[1.0, 2.0, 3.0]])[0], [-0.2, 0.1, 0.0])
    turn = tf[:3, :3] / 0.05
    assert np.linalg.det(turn) == pytest.approx(1.0)
    assert np.allclose(turn @ [0, -1.0, 0], hh.unit([-0.3, -0.9, 0.3]))


def test_mirror_x_keeps_a_closed_hand_facing_out():
    v, f = turned(HAND)
    mv, mf, _ = hh.mirror_x(v, f)

    def volume(vertices, faces):
        a, b, c = (vertices[faces[:, k]] for k in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)

    assert volume(mv, mf) == pytest.approx(volume(v, f)) and volume(v, f) > 0


def test_blend_wrist_scales_the_rim_and_fades_out():
    ring = np.stack([np.cos(np.linspace(0, 6, 12)), np.zeros(12), np.sin(np.linspace(0, 6, 12))], 1)
    points = np.concatenate([ring, ring + [0, -1.0, 0], ring + [0, -3.0, 0]])
    eased = hh.blend_wrist(points, np.zeros(3), [0, -1, 0], np.zeros(3), np.array([0.5, 0, 0]), 1.2, 2.0)
    assert np.allclose(np.hypot(eased[:12, 0] - 0.5, eased[:12, 2]), 1.2)   # the rim: wider, centred anew
    assert np.allclose(eased[24:], points[24:])                             # past the span: untouched


# --- the stitch ---------------------------------------------------------------------------------------------

def test_bridge_closes_two_rims_into_one_tube():
    a = np.stack([np.cos(np.linspace(0, 2 * np.pi, 20, endpoint=False)), np.zeros(20),
                  np.sin(np.linspace(0, 2 * np.pi, 20, endpoint=False))], 1)
    b = 0.8 * np.stack([np.cos(np.linspace(0, 2 * np.pi, 13, endpoint=False)), np.full(13, -0.1 / 0.8),
                        np.sin(np.linspace(0, 2 * np.pi, 13, endpoint=False))], 1)[::-1]   # turns the other way
    triangles = hh.bridge(a, b, [0, 1, 0])
    assert len(triangles) == 20 + 13
    edges = {}
    for tri in triangles:
        for k in range(3):
            edge = (tuple(tri[k]), tuple(tri[(k + 1) % 3]))
            edges[edge] = edges.get(edge, 0) + 1
    # every rim edge is walked once, against its own rim's direction
    assert all(edges.get(((0, (i + 1) % 20), (0, i))) == 1 for i in range(20))
    assert all(edges.get(((1, (j + 1) % 13), (1, j))) == 1 for j in range(13))


def test_graft_model_leaves_one_closed_surface_with_no_gap_and_no_overlap():
    arm = figure_arm()
    bone = arm["dense"].argmax(1)
    plan = hh.figure_wrist(arm["positions"], arm["faces"], bone, 1, [2], np.array([-0.2, 0.30, 0]),
                           np.array([-0.2, 0.12, 0]), "right")
    hand = built_hand(double=True)
    own = hh.hand_wrist(hand["positions"], hand["faces"])
    assert own["found"] and 0.15 < own["cut"][1] < 0.299 and own["back"][2] > 0.9   # below the cuff
    tf = hh.placement(own["centre"], own["fingers"], own["back"], own["length"],
                      plan["centre"], plan["fingers"], plan["back"], plan["length"])
    body, hands, report = hh.graft_model(arm, bone, {"right": plan}, {"right": {"part": hand, "tf": tf}})
    # the old hand is gone: no body vertex past the cut
    assert ((body["positions"] - plan["cut"]) @ plan["axis"]).max() < 1e-9
    v = np.concatenate([body["positions"], hands["positions"]])
    f = np.concatenate([body["faces"], hands["faces"] + len(body["positions"])])
    assert closed_edges(v, f) == (0, 0)                     # one closed surface, faces turned alike
    side = report["right"]
    assert side["rim_pieces"] == 1 and not side["rims_turn_alike"]
    assert 0.7 < side["wrist_ratio"] < 1.4
    # each stitch triangle is painted from one point of the hand: never blended across the texture's islands
    band = hands["faces"][-side["stitch_faces"]:]
    uv = hands["uvs"][band]
    assert np.allclose(uv[:, 0], uv[:, 1]) and np.allclose(uv[:, 0], uv[:, 2])
    assert np.isin(uv[:, 0], hands["uvs"][:-3 * side["stitch_faces"]]).all()
    # the new hand reaches as far as the old one did, and moves with the hand bone
    tip = ((hands["positions"] - plan["cut"]) @ plan["axis"]).max()
    assert tip == pytest.approx(((arm["positions"] - plan["cut"]) @ plan["axis"]).max(), rel=0.1)
    assert hands["dense"][hands["positions"][:, 1] < 0.0].argmax(1).tolist() == [2] * int(
        (hands["positions"][:, 1] < 0.0).sum())


def test_a_mirrored_hand_on_the_left_arm_also_closes_and_faces_out():
    arm = figure_arm(x=0.2)                                   # the figure's left arm, at +x
    bone = arm["dense"].argmax(1)
    plan = hh.figure_wrist(arm["positions"], arm["faces"], bone, 1, [2], np.array([0.2, 0.30, 0]),
                           np.array([0.2, 0.12, 0]), "left")
    assert plan["back"][0] > 0.95                             # a left hand's back faces out, +x
    hand = built_hand(double=True)
    hand["positions"], hand["faces"], _ = hh.mirror_x(hand["positions"], hand["faces"])
    own = hh.hand_wrist(hand["positions"], hand["faces"])
    tf = hh.placement(own["centre"], own["fingers"], own["back"], own["length"],
                      plan["centre"], plan["fingers"], plan["back"], plan["length"])
    assert np.linalg.det(tf[:3, :3]) > 0                      # a turn, never a mirror, onto the figure
    body, hands, report = hh.graft_model(arm, bone, {"left": plan}, {"left": {"part": hand, "tf": tf}})
    v = np.concatenate([body["positions"], hands["positions"]])
    f = np.concatenate([body["faces"], hands["faces"] + len(body["positions"])])
    assert closed_edges(v, f) == (0, 0) and not report["left"]["rims_turn_alike"]


def test_graft_model_cuts_only_the_arm_it_was_told():
    arm = figure_arm()
    coat = figure_arm(x=0.2)                                  # another limb beside it, same heights, bone 0
    v = np.concatenate([arm["positions"], coat["positions"]])
    f = np.concatenate([arm["faces"], coat["faces"] + len(arm["positions"])])
    dense = np.concatenate([arm["dense"], np.eye(3)[np.zeros(len(coat["positions"]), int)]])
    body = {"positions": v, "faces": f, "dense": dense}
    bone = dense.argmax(1)
    plan = hh.figure_wrist(arm["positions"], arm["faces"], arm["dense"].argmax(1), 1, [2],
                           np.array([-0.2, 0.30, 0]), np.array([-0.2, 0.12, 0]), "right")
    hand = built_hand(double=False)
    own = hh.hand_wrist(hand["positions"], hand["faces"])
    tf = hh.placement(own["centre"], own["fingers"], own["back"], own["length"],
                      plan["centre"], plan["fingers"], plan["back"], plan["length"])
    out, _, _ = hh.graft_model(body, bone, {"right": plan}, {"right": {"part": hand, "tf": tf}})
    beside = out["positions"][out["positions"][:, 0] > 0.1]
    assert beside[:, 1].min() == pytest.approx(-0.05)       # the other limb keeps its tip


def test_arm_stub_takes_the_cuff_above_the_cut_and_not_the_limb_beside_it():
    arm = figure_arm()
    coat = figure_arm(x=-0.1)                                 # another limb 10 cm beside it
    body = {"positions": np.concatenate([arm["positions"], coat["positions"]]),
            "faces": np.concatenate([arm["faces"], coat["faces"] + len(arm["positions"])])}
    plan = hh.figure_wrist(arm["positions"], arm["faces"], arm["dense"].argmax(1), 1, [2],
                           np.array([-0.2, 0.30, 0]), np.array([-0.2, 0.12, 0]), "right")
    stub = hh.arm_stub(body, plan)
    above = (stub["positions"] - plan["cut"]) @ plan["axis"]
    assert len(stub["faces"]) and above.max() <= 1e-9
    assert above.min() > -hh.FOREARM_SHARE * plan["length"] - 0.02
    assert (np.abs(stub["positions"][:, 0] + 0.2) < 0.05).all()   # all on this arm


# --- skin weights -------------------------------------------------------------------------------------------

def test_dense_and_top_weights_round_trip_and_cap_at_four():
    joints = np.array([[0, 3, 5, 0], [1, 2, 3, 4]])
    weights = np.array([[0.5, 0.25, 0.25, 0.0], [0.4, 0.3, 0.2, 0.1]])
    dense = hh.dense_weights(joints, weights, 6)
    assert dense[0, [0, 3, 5]].tolist() == [0.5, 0.25, 0.25]
    dense[1, 5] = 0.5                                         # five influences now
    j, w = hh.top_weights(dense)
    assert np.allclose(w.sum(1), 1.0) and 5 in j[1] and 4 not in j[1]


def test_borrowed_weights_come_from_the_nearest_old_point():
    old = np.array([[0, 0, 0], [0, -1.0, 0]])
    dense = np.array([[1.0, 0.0], [0.0, 1.0]])
    got = hh.borrowed_weights(np.array([[0, -0.9, 0], [0.1, 0.1, 0]]), old, dense)
    assert got.tolist() == [[0.0, 1.0], [1.0, 0.0]]


# --- the rig ------------------------------------------------------------------------------------------------

def stick_figure():
    """A glTF (Y-up) humanoid facing +Z, in SkinTokens' tree shape with five fingers a hand; names shuffled."""
    order = []
    parents, heads = {}, {}
    names = iter(f"bone_{n}" for n in np.random.default_rng(1).permutation(100))
    role = {}

    def add(key, parent, pos):
        role[key] = next(names)
        parents[role[key]] = role.get(parent)
        heads[role[key]] = np.array(pos, float)
        order.append(key)

    add("hips", None, (0, 1.0, 0)); add("s1", "hips", (0, 1.1, 0)); add("s2", "s1", (0, 1.2, 0))
    add("chest", "s2", (0, 1.35, 0)); add("neck", "chest", (0, 1.5, 0)); add("head", "neck", (0, 1.6, 0))
    for side, sx in (("L", 1.0), ("R", -1.0)):        # facing +Z the figure's left is +X
        add(f"{side}sh", "chest", (0.05 * sx, 1.45, 0)); add(f"{side}arm", f"{side}sh", (0.15 * sx, 1.45, 0))
        add(f"{side}fa", f"{side}arm", (0.4 * sx, 1.2, 0)); add(f"{side}hand", f"{side}fa", (0.55 * sx, 0.95, 0))
        for k in range(5):
            z = 0.03 - 0.015 * k
            add(f"{side}f{k}a", f"{side}hand", (0.58 * sx, 0.92 - 0.02 * (k == 0), z))
            add(f"{side}f{k}b", f"{side}f{k}a", (0.60 * sx, 0.89, z))
            add(f"{side}f{k}c", f"{side}f{k}b", (0.62 * sx, 0.86, z))
        add(f"{side}leg", "hips", (0.1 * sx, 0.95, 0)); add(f"{side}shin", f"{side}leg", (0.1 * sx, 0.5, 0))
        add(f"{side}foot", f"{side}shin", (0.1 * sx, 0.08, 0))
        add(f"{side}toe", f"{side}foot", (0.1 * sx, 0.02, 0.12))
    return parents, heads, role


def test_arm_bones_finds_each_hand_by_the_tree_and_the_figures_own_side():
    parents, heads, role = stick_figure()
    arms, frame = hh.arm_bones(parents, heads)
    assert arms["right"]["hand"] == role["Rhand"] and arms["right"]["forearm"] == role["Rfa"]
    assert arms["left"]["hand"] == role["Lhand"]
    assert len(arms["right"]["below"]) == 15 and role["Rf2c"] in arms["right"]["below"]
    assert np.allclose(frame[:, 2], [0, 0, 1], atol=1e-6)     # it faces +Z


def test_hand_points_hold_the_hand_and_the_cuff_above_it():
    arm = figure_arm()
    bone = arm["dense"].argmax(1)
    plan = hh.figure_wrist(arm["positions"], arm["faces"], bone, 1, [2], np.array([-0.2, 0.30, 0]),
                           np.array([-0.2, 0.12, 0]), "right")
    points = hh.hand_points(arm["positions"], bone, plan)
    assert points[:, 1].min() == pytest.approx(-0.05)
    assert plan["cut"][1] < points[:, 1].max() < plan["cut"][1] + hh.FOREARM_SHARE * plan["length"] + 1e-9


# --- the picture side ---------------------------------------------------------------------------------------

def hand_mask(digits=5, thumb=True, gap=14):
    """A flat hand seen from its back, fingers down: a sleeve band on top, a palm, fingers, a shorter thumb."""
    import cv2

    mask = np.zeros((400, 400), np.uint8)
    cv2.rectangle(mask, (120, 0), (280, 60), 1, -1)                           # sleeve
    cv2.rectangle(mask, (130, 60), (270, 200), 1, -1)                         # palm
    fingers = digits - (1 if thumb else 0)
    width = (140 - gap * (fingers - 1)) // fingers
    for k in range(fingers):
        x0 = 130 + k * (width + gap)
        length = 120 if k in (1, 2) else 100
        cv2.rectangle(mask, (x0, 200), (x0 + width, 200 + length), 1, -1)
    if thumb:
        pts = np.array([[270, 110], [300, 160], [320, 215], [300, 225], [262, 170]], np.int32)
        cv2.fillConvexPoly(mask, pts, 1)
    return mask


def test_digit_count_finds_five_digits_and_one_thumb():
    result = hh.digit_count(hand_mask())
    assert (result["digits"], result["thumbs"], result["ok"]) == (5, 1, True)


def test_digit_count_rejects_six_digits_and_a_mitten():
    assert hh.digit_count(hand_mask(digits=6))["digits"] == 6
    assert not hh.digit_count(hand_mask(digits=6))["ok"]
    mitten = hand_mask(digits=5, gap=0)
    assert not hh.digit_count(mitten)["ok"]


def flat_hand(thumb="right", thumb_length=105, splay=False, fingers=4):
    """A flat hand from its back, fingers down, with control over the thumb: its side ('right', 'left', 'both' or
    None), its length (as long as the fingers is allowed) and a little finger splayed outward at the other end."""
    import cv2

    mask = np.zeros((440, 520), np.uint8)
    width, gap, left = 26, 12, 170
    right = left + fingers * (width + gap) - gap                              # the palm is as wide as the fingers
    cv2.rectangle(mask, (left - 10, 0), (right + 10, 60), 1, -1)              # sleeve
    cv2.rectangle(mask, (left, 60), (right, 200), 1, -1)                      # palm, knuckles at y = 200

    def digit(base, tip):                                                     # a digit with a round tip
        cv2.line(mask, base, tip, 1, width)
        cv2.circle(mask, tip, width // 2, 1, -1)

    for k in range(fingers):
        x = left + k * (width + gap) + width // 2
        if splay and k == 0:                                                  # the little finger leans out
            digit((x, 196), (x - 40, 290))
        else:
            digit((x, 196), (x, 200 + (110 if 0 < k < fingers - 1 else 90)))
    for side in (("right", "left") if thumb == "both" else (thumb,) if thumb else ()):
        # a thumb grows from the palm's side well above the knuckles, so its cleft runs deep toward the wrist
        base, tip = (right + 5, 135), (right + 25 + int(0.35 * thumb_length), 135 + thumb_length)
        if side == "left":
            base, tip = (left + right - base[0], base[1]), (left + right - tip[0], tip[1])
        digit(base, tip)
    return mask


def test_a_thumb_as_long_as_the_fingers_is_found_by_its_deep_cleft():
    # the length rule took this for a sixth finger's row without a thumb; its cleft says thumb
    result = hh.digit_count(flat_hand(thumb_length=190))
    assert (result["digits"], result["thumbs"], result["thumb_at"], result["ok"]) == (5, 1, "right", True)


def test_a_splayed_little_finger_is_not_a_second_thumb():
    # the length rule read the short, outward little finger of good drawings as a second thumb
    result = hh.digit_count(flat_hand(splay=True))
    assert (result["digits"], result["thumbs"], result["ok"]) == (5, 1, True)


def test_two_deep_clefts_are_two_thumbs():
    result = hh.digit_count(flat_hand(thumb="both", fingers=3))
    assert result["thumbs"] == 2 and not result["ok"]


def test_a_thumb_on_the_side_the_editor_was_not_told_fails():
    assert hh.digit_count(flat_hand(), thumb_side="right")["ok"]
    left = hh.digit_count(flat_hand(thumb="left"), thumb_side="right")
    assert left["thumb_at"] == "left" and not left["ok"]


def test_five_fingers_in_a_row_have_no_thumb():
    result = hh.digit_count(flat_hand(thumb=None, fingers=5))
    assert result["thumbs"] == 0 and not result["ok"]


def test_views_count_takes_the_views_that_see_the_most_digits():
    assert hh.views_count([(3, 0), (5, 1), (5, 1), (4, 1), (5, 0)]) == {
        "digits": 5, "thumbs": 1, "ok": True, "per_view": [3, 5, 5, 4, 5]}
    assert not hh.views_count([(6, 1), (6, 1), (5, 1)])["ok"]


def test_silhouette_fills_whole_triangles_large_and_tiny():
    big = hh.silhouette(np.array([[10, 10], [90, 10], [10, 90]], float), np.array([[0, 1, 2]]), 100)
    assert big.sum() == pytest.approx(0.5 * 80 * 80, rel=0.05)
    grid = np.stack(np.meshgrid(np.arange(20, 60, 0.7), np.arange(20, 60, 0.7)), -1).reshape(-1, 2)
    n = int(np.sqrt(len(grid)))
    tris = [[i * n + j, i * n + j + 1, (i + 1) * n + j] for i in range(n - 1) for j in range(n - 1)]
    tris += [[i * n + j + 1, (i + 1) * n + j + 1, (i + 1) * n + j] for i in range(n - 1) for j in range(n - 1)]
    tiny = hh.silhouette(grid, np.array(tris), 100)
    assert tiny[25:55, 25:55].all()                         # sub-pixel triangles leave no pinholes


def test_count_built_counts_a_built_hand_from_all_round():
    import trimesh

    parts = [trimesh.creation.box((0.30, 0.20, 0.08), trimesh.transformations.translation_matrix((0, 0.1, 0))),
             trimesh.creation.box((0.34, 0.12, 0.06), trimesh.transformations.translation_matrix((0, 0.26, 0)))]
    for k, x in enumerate((-0.12, -0.04, 0.04, 0.12)):
        length = 0.30 if k in (1, 2) else 0.24
        parts.append(trimesh.creation.box((0.05, length, 0.05),
                                          trimesh.transformations.translation_matrix((x, -length / 2, 0))))
    thumb = trimesh.creation.box((0.05, 0.16, 0.05))
    thumb.apply_transform(trimesh.transformations.rotation_matrix(0.7, (0, 0, 1)))
    thumb.apply_translation((0.21, 0.02, 0))
    parts.append(thumb)
    hand = trimesh.util.concatenate(parts)
    result = hh.count_built(hand.vertices, hand.faces, [0, -1, 0])
    assert result["ok"], result


def test_hand_prompt_names_the_thumbs_side_and_the_look():
    right = hh.hand_prompt("right", "black fingerless gloves.")
    assert "thumb on the right side" in right and "far left" in right and "black fingerless gloves." in right
    left = hh.hand_prompt("left")
    assert "thumb on the left side" in left and hh.DEFAULT_LOOK in left


def test_crop_box_is_square_padded_and_inside():
    x0, y0, x1, y1 = hh.crop_box(np.array([5.0, 100.0]), np.array([600.0, 700.0]), 768, 1376)
    assert x1 - x0 == y1 - y0 == round(100 * hh.HAND_CROP_PAD)
    assert x0 == 0 and y1 <= 1376


# --- GLB in and out -----------------------------------------------------------------------------------------

def tiny_glb(skinned=True, with_image=True):
    """A one-triangle GLB, optionally skinned to two joints and painted with a 1x1 PNG."""
    import io

    from PIL import Image

    from image_to_3dlab.glb_turn import build_glb

    blob = bytearray()
    views, accessors = [], []

    def add(array, component, kind, extra=None):
        data = np.ascontiguousarray(array).tobytes()
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        accessors.append({"bufferView": len(views) - 1, "componentType": component, "type": kind,
                          "count": int(len(array)), **(extra or {})})
        return len(accessors) - 1

    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    attributes = {"POSITION": add(pos, 5126, "VEC3", {"min": [0, 0, 0], "max": [1, 1, 0]}),
                  "TEXCOORD_0": add(np.array([[0, 0], [1, 0], [0, 1]], np.float32), 5126, "VEC2")}
    doc = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0}], "materials": [{"name": "paint"}]}
    if skinned:
        attributes["JOINTS_0"] = add(np.array([[0, 1, 0, 0]] * 3, np.uint8), 5121, "VEC4")
        attributes["WEIGHTS_0"] = add(np.array([[0.75, 0.25, 0, 0]] * 3, np.float32), 5126, "VEC4")
        bind = add(np.tile(np.eye(4, dtype=np.float32).reshape(1, 16), (2, 1)), 5126, "MAT4")
        doc["nodes"] = [{"mesh": 0, "skin": 0}, {"name": "root", "children": [2]}, {"name": "tip"}]
        doc["scenes"] = [{"nodes": [0, 1]}]
        doc["skins"] = [{"joints": [1, 2], "inverseBindMatrices": bind}]
    if with_image:
        png = io.BytesIO()
        Image.new("RGB", (1, 1), (200, 30, 30)).save(png, "PNG")
        data = png.getvalue()
        blob.extend(b"\0" * (-len(blob) % 4))
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
        blob.extend(data)
        doc["images"] = [{"bufferView": len(views) - 1, "mimeType": "image/png"}]
        doc["samplers"] = [{}]
        doc["textures"] = [{"source": 0, "sampler": 0}]
        doc["materials"] = [{"name": "paint", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["meshes"] = [{"primitives": [{"attributes": attributes, "indices": add(np.array([0, 1, 2], np.uint16),
                                                                                5123, "SCALAR"),
                                      "material": 0}]}]
    doc["bufferViews"], doc["accessors"] = views, accessors
    doc["buffers"] = [{"byteLength": len(blob)}]
    return build_glb(doc, bytes(blob))


def test_read_part_reads_a_skinned_model_as_dense_weights():
    doc, _, part = hh.read_part(tiny_glb())
    assert part["faces"].tolist() == [[0, 1, 2]] and part["positions"].shape == (3, 3)
    assert np.allclose(part["dense"], [[0.75, 0.25]] * 3)
    assert part["material"][2] == 0 and len(doc["skins"][0]["joints"]) == 2


def test_read_part_refuses_a_model_moved_by_its_node():
    from image_to_3dlab.glb_turn import build_glb, parse_glb

    doc, binary = parse_glb(tiny_glb(skinned=False))
    doc["nodes"][0]["translation"] = [1, 0, 0]
    with pytest.raises(ValueError, match="frame"):
        hh.read_part(build_glb(doc, binary))


def test_regraft_keeps_the_skeleton_and_carries_each_parts_paint():
    from image_to_3dlab.glb_turn import parse_glb

    template = tiny_glb()
    _, _, body = hh.read_part(template)
    hand_doc, hand_bin, _ = hh.read_part(tiny_glb(skinned=False))
    hand = {"positions": np.array([[0, 0, 1.0], [1, 0, 1], [0, 1, 1]]), "faces": np.array([[0, 1, 2]]),
            "uvs": np.zeros((3, 2)), "dense": np.array([[0.0, 1.0]] * 3), "material": (hand_doc, hand_bin, 0)}
    out = hh.regraft_glb(template, [hh.finish_part(body), hh.finish_part(hand, smooth=True)])
    doc, binary = parse_glb(out)
    assert len(doc["meshes"][0]["primitives"]) == 2 and doc["skins"][0]["joints"] == [1, 2]
    bind = doc["accessors"][doc["skins"][0]["inverseBindMatrices"]]
    view = doc["bufferViews"][bind["bufferView"]]
    assert np.allclose(np.frombuffer(binary, np.float32, 32, view.get("byteOffset", 0)),
                       np.tile(np.eye(4).reshape(16), 2))
    assert len(doc["materials"]) == 2 and len(doc["images"]) == 2
    second = doc["meshes"][0]["primitives"][1]
    from image_to_3dlab.hero import _accessor

    assert _accessor(doc, binary, second["attributes"]["JOINTS_0"])[0, 0] == 1
    assert json.loads(json.dumps(doc))                       # plain JSON all through
    assert struct.unpack_from("<I", out, 0)[0] == 0x46546C67
