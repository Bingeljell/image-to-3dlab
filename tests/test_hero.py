"""The hero route's maths: finding the head, cropping it, putting it back.

Each of these broke once while the route was being worked out by hand (2026-10-21): a fixed
grey background made the whole picture "figure", a guessed scale shrank the head 12%, an
outline made of corner points dropped triangles as holes, a fixed margin trimmed the hair.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from image_to_3dlab import hero


def test_picture_box_prefers_the_cut_out_over_grey():
    # A darker graded background (the elf's ran 202-222) is not the studio's light grey.
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[..., :3] = 210
    rgba[20:80, 30:60] = (90, 120, 90, 255)
    assert hero.picture_box(Image.fromarray(rgba, "RGBA")) == (30, 59, 20, 79)
    # Without a cut-out the same picture reads as all figure: the failure the alpha path fixes.
    assert hero.picture_box(Image.fromarray(rgba[..., :3], "RGB")) == (0, 99, 0, 99)


def test_picture_box_falls_back_to_light_grey():
    rgb = np.full((50, 50, 3), 236, np.uint8)
    rgb[10:40, 5:15] = 40
    assert hero.picture_box(Image.fromarray(rgb, "RGB")) == (5, 14, 10, 39)


def test_project_maps_the_mesh_onto_its_picture_box():
    verts = np.array([[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0]])
    fr = hero.frame(verts, (100, 300, 50, 450))      # 400 px tall for 4 units
    px, py = hero.project(verts, fr)
    assert np.allclose(px, [100, 300]) and np.allclose(py, [450, 50])   # picture y runs down


def test_head_bones_take_the_top_region_and_a_neck_inside_it():
    heights = np.concatenate([np.linspace(0.9, 1.0, 50),    # head
                              np.linspace(0.88, 0.95, 20),  # neck, middle inside the head's span
                              np.linspace(0.6, 0.9, 50)])   # chest
    bone = np.array([5] * 50 + [4] * 20 + [3] * 50)
    assert hero.head_bones(bone, heights) == [4, 5]


def test_head_bones_leave_a_neck_below_the_head():
    heights = np.concatenate([np.linspace(0.9, 1.0, 50), np.linspace(0.8, 0.86, 20)])
    bone = np.array([5] * 50 + [4] * 20)
    assert hero.head_bones(bone, heights) == [5]


def test_crop_box_is_square_padded_and_inside_the_picture():
    px, py = np.array([460.0, 558.0]), np.array([27.0, 168.0])   # the warrior's head pixels
    x0, y0, x1, y1 = hero.crop_box(px, py, 1024, 1024)
    assert x1 - x0 == y1 - y0
    assert x0 < 460 and x1 > 558 and y1 > 168
    assert y0 >= 0                       # a head at the top edge is not cropped off the picture
    assert (x1 - x0) == pytest.approx(141 * hero.CROP_PAD, abs=1)


def test_crop_to_full_puts_a_crop_pixel_back_where_it_came_from():
    full = np.array([[-1.0, -2.0, -0.5], [1.0, 2.0, 0.5]])
    full_frame = hero.frame(full, (100, 300, 50, 450))
    head = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    head_frame = hero.frame(head, (100, 900, 100, 900))          # head picture 1024 px
    crop = (150, 50, 250, 150)                                   # 100 px of the full picture
    tf = hero.crop_to_full(full_frame, head_frame, crop, 1024)
    moved = (np.c_[head, np.ones(2)] @ tf.T)[:, :3]
    px, py = hero.project(moved, full_frame)
    # head picture box (100..900) of 1024 -> full pixels 150 + p * 100/1024
    assert np.allclose(px, [150 + 100 * 100 / 1024, 150 + 900 * 100 / 1024])
    assert np.allclose(py, [50 + 900 * 100 / 1024, 50 + 100 * 100 / 1024])


def test_place_on_matches_centre_and_height_only():
    real = np.array([[0.0, 1.0, 0.0], [0.2, 1.4, 0.1]])
    new = np.array([[5.0, 5.0, 5.0], [5.4, 5.8, 5.2]])
    tf = hero.place_on(real, new)
    moved = (np.c_[new, np.ones(2)] @ tf.T)[:, :3]
    assert np.ptp(moved[:, 1]) == pytest.approx(0.4)
    assert np.allclose(moved.mean(axis=0), real.mean(axis=0))


def test_silhouette_fills_whole_triangles_not_just_corners():
    # One big triangle: its corners are three pixels, its inside must be filled too.
    px, py = np.array([10.0, 90.0, 50.0]), np.array([90.0, 90.0, 10.0])
    mask = hero.silhouette(np.array([[0, 1, 2]]), px, py, (100, 100))
    assert mask[60, 50] and mask.sum() > 2500
    assert not mask[5, 5]


def test_liner_keeps_the_old_head_s_back_half_shrunk_and_drops_its_face():
    verts = np.array([[0, 2, 1.0], [0, 2, -1.0], [0, 0, 0.5]])   # face (+Z), back of head, chest
    real = np.array([True, True, False])
    moved, gone = hero.liner_split(verts, real, 0.1)
    assert gone.tolist() == [True, False, False]      # old face goes; back of head stays as liner
    assert np.allclose(moved[1], [0, 2, -0.9])        # shrunk toward the head's centre (0, 2, 0)
    assert np.allclose(moved[2], verts[2])            # the rest of the body is untouched


def test_liner_off_drops_the_whole_old_head():
    verts = np.array([[0, 2, 1.0], [0, 2, -1.0], [0, 0, 0.5]])
    real = np.array([True, True, False])
    moved, gone = hero.liner_split(verts, real, 0.0)
    assert gone.tolist() == [True, True, False] and np.allclose(moved, verts)


def _ring(height, width, n=60):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(a) * width / 2, np.full(n, height), np.sin(a) * width / 2], axis=1)


def _profile(rows):
    """A head region stacked from rings: (height, width) per row."""
    return np.concatenate([_ring(h, w) for h, w in rows])


def test_neck_height_cuts_between_the_collar_and_the_skull():
    # The warrior's profile (2026-10-22): stray bits, a collar, the narrower jaw, the head. The
    # neck hides inside the collar, so the narrow band above it is the chin: cutting in the middle
    # of it left the old body's mouth showing under the new face. Cut at the collar's top.
    rows = ([(0.862 + 0.002 * i, 0.022) for i in range(3)]
            + [(0.882 + 0.002 * i, 0.090) for i in range(6)]      # collar, top at 0.892
            + [(0.898 + 0.002 * i, 0.064) for i in range(5)]      # chin and jaw
            + [(0.910 + 0.003 * i, 0.078) for i in range(25)])    # head
    pts = _profile(rows)
    cut = hero.neck_height(pts[:, 1], pts[:, 0])
    assert 0.890 <= cut <= 0.899            # at the collar's top, below the chin


def test_neck_height_is_none_without_a_collar():
    # The region ends at a bare neck (the elf): nothing below is wider, so cut nothing extra.
    rows = ([(0.88 + 0.002 * i, 0.050) for i in range(8)]
            + [(0.896 + 0.003 * i, 0.075) for i in range(25)])
    pts = _profile(rows)
    assert hero.neck_height(pts[:, 1], pts[:, 0]) is None


def test_neck_height_ignores_slices_with_too_few_points():
    rows = ([(0.882 + 0.002 * i, 0.090) for i in range(6)]
            + [(0.898 + 0.002 * i, 0.064) for i in range(5)]
            + [(0.910 + 0.003 * i, 0.078) for i in range(25)])
    pts = _profile(rows)
    stray = np.array([[0.0, 0.87, 0.0], [0.001, 0.871, 0.0]])  # a few points below, very narrow
    cut = hero.neck_height(np.r_[pts[:, 1], stray[:, 1]], np.r_[pts[:, 0], stray[:, 0]])
    assert 0.890 <= cut <= 0.899


def test_neck_ring_is_the_skull_s_radius_round_its_centre():
    pts = np.concatenate([_ring(0.95 + 0.005 * i, 0.094) for i in range(6)])     # skull, radius 0.047
    pts[:, 0] += 0.01                                                             # off the origin
    real = np.ones(len(pts), bool)
    axis, radius = hero.neck_ring(pts, real, cut=0.9)
    assert np.allclose(axis, [0.01, 0.0], atol=1e-3) and abs(radius - 0.047) < 0.002


def test_head_cut_splits_the_collar_band_by_the_ring_not_by_height():
    # The warrior (2026-10-22): the new head sits a little lower, so in the collar band the body's
    # old chin (inside the ring) and its collar (outside) share heights with the new mouth.
    # x = distance from the neck's centre line; ring 0.047; collar top 0.90.
    body = np.array([[0.0, 0.95, 0.04],    # old face above the collar: goes
                     [0.0, 0.87, 0.03],    # old chin inside the ring: goes (it showed through)
                     [0.0, 0.87, 0.07],    # its own collar: stays
                     [0.0, 0.80, 0.03]])   # chest, not head region: stays
    real = np.array([True, True, True, False])
    head = np.array([[0.0, 0.95, 0.04],    # new face: stays
                     [0.0, 0.88, 0.03],    # new mouth and chin inside the ring: stay
                     [0.0, 0.87, 0.07]])   # its scuffed collar: goes
    swap, keep = hero.head_cut(body, real, head, cut=0.90, axis=np.zeros(2), ring=0.047)
    assert swap.tolist() == [True, True, False, False]
    assert keep.tolist() == [True, True, False]


def test_head_cut_off_swaps_the_whole_region():
    body = np.array([[0, 0.95, 0.04], [0, 0.87, 0.07], [0, 0.80, 0.03]])
    real = np.array([True, True, False])
    swap, keep = hero.head_cut(body, real, np.array([[0, 0.95, 0.0], [0, 0.85, 0.08]]),
                               cut=None, axis=np.zeros(2), ring=0.047)
    assert swap.tolist() == [True, True, False] and keep.tolist() == [True, True]


def test_paired_points_pair_only_spots_within_reach():
    head = np.array([[0.0, 0, 0], [1.0, 0, 0], [5.0, 5, 5]])
    body = np.array([[0.01, 0, 0], [1.02, 0, 0]])
    head_idx, body_idx = hero.paired_points(head, body, reach=0.05)
    assert head_idx.tolist() == [0, 1] and body_idx.tolist() == [0, 1]   # the far point is left out


def test_match_to_body_lands_an_all_metal_head_on_the_body_s_level():
    # The long-haired woman: Pixal3D called her whole head metal; her body said cloth and skin.
    mr = Image.fromarray(np.dstack([np.zeros((4, 4)), np.full((4, 4), 254), np.full((4, 4), 255)]).astype(np.uint8))
    head_seen = np.tile([0.0, 254 / 255, 1.0], (10, 1))
    body_seen = np.tile([0.0, 0.98, 0.0], (10, 1))
    fixed, report = hero.match_to_body(mr, head_seen, body_seen, channels=(1, 2))
    a = np.asarray(fixed).astype(float) / 255
    assert a[..., 2].max() == 0                         # no metal left
    assert a[..., 1].mean() == pytest.approx(0.98, abs=0.005)
    assert report["big_change"]


def test_match_to_body_moves_level_and_contrast_but_keeps_the_pattern():
    head = np.array([[0.2, 0.4], [0.6, 0.8]])           # a darker, low-contrast head
    image = Image.fromarray(np.round(np.dstack([head] * 3) * 255).astype(np.uint8))
    head_seen = np.repeat(head.reshape(-1, 1), 3, axis=1)
    body_seen = head_seen * 1.2 + 0.05                   # the body: brighter, more contrast
    fixed, report = hero.match_to_body(image, head_seen, body_seen)
    a = np.asarray(fixed).astype(float)[..., 0] / 255
    assert np.allclose(a.mean(), body_seen[:, 0].mean(), atol=0.01)
    assert np.all(np.diff(a.reshape(-1)) > 0)            # the order of light and dark is kept
    assert report["r"]["stretch"] == pytest.approx(1.2, abs=0.01) and not report["big_change"]


def test_sample_texture_reads_glTF_uvs_with_v_up():
    image = Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8))
    got = hero.sample_texture(image, np.array([[0.0, 1.0], [0.0, 0.0]]))   # top-left, bottom-left
    assert np.allclose(got[0], [1, 0, 0]) and np.allclose(got[1], [0, 0, 1])


def _skinned_glb(positions: np.ndarray, joints: np.ndarray, weights: np.ndarray) -> bytes:
    blobs = [positions.astype(np.float32).tobytes(), joints.astype(np.uint8).tobytes(),
             weights.astype(np.float32).tobytes()]
    views, offset = [], 0
    for blob in blobs:
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob)})
        offset += len(blob)
    n = len(positions)
    document = {
        "asset": {"version": "2.0"}, "buffers": [{"byteLength": offset}], "bufferViews": views,
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": n, "type": "VEC3"},
                      {"bufferView": 1, "componentType": 5121, "count": n, "type": "VEC4"},
                      {"bufferView": 2, "componentType": 5126, "count": n, "type": "VEC4"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
    }
    text = json.dumps(document).encode()
    text += b" " * (-len(text) % 4)
    binary = b"".join(blobs)
    binary += b"\0" * (-len(binary) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def test_skinned_vertices_take_each_vertex_s_strongest_joint():
    positions = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], float)
    joints = np.array([[1, 2, 0, 0], [3, 1, 0, 0], [0, 4, 0, 0]])
    weights = np.array([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0], [0.4, 0.6, 0, 0]])
    got_positions, got_joints = hero.skinned_vertices(_skinned_glb(positions, joints, weights))
    assert np.allclose(got_positions, positions)
    assert got_joints.tolist() == [1, 1, 4]


def test_regions_survive_a_rescaled_re_export():
    # The rigger re-exports the mesh at another scale and place; boxing both to one height lines them up.
    verts = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0]], float)
    rig = verts * 3 + 10
    assert hero.regions(verts, rig, np.array([7, 8, 9])).tolist() == [7, 8, 9]
