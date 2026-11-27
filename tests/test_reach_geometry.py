"""Chain length and ankle offset are measured in one space, on the facing axis."""

from __future__ import annotations

import blender_quadruped_walk as walk
import pytest


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __getitem__(self, index):
        return (self.x, self.y, self.z)[index]

    @property
    def length(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


class Scale:
    def __init__(self, scale):
        self.scale = scale

    def __matmul__(self, vector):
        return Vec(vector.x * self.scale, vector.y * self.scale, vector.z * self.scale)


class Bone:
    def __init__(self, head, tail=None):
        self.head = head
        self.head_local = head
        self.tail_local = head if tail is None else tail


class Bag:
    def __init__(self, bones):
        self.bones = bones


class Rig:
    def __init__(self, bones, scale, head, hips):
        self.data = Bag(bones)
        self.matrix_world = Scale(scale)
        self.pose = Bag({"head": Bone(Vec(*head)), "hips": Bone(Vec(*hips))})


def _rig(scale):
    bones = {
        "ORG-front_thigh.L": Bone(Vec(0, 0, 1), Vec(0, 0, 0)),
        "ORG-front_shin.L": Bone(Vec(0, 0, 0), Vec(1, 0, 0)),
    }
    # Head is further from the hips on X than on Y, so forward is +X.
    return Rig(bones, scale, head=(5, 0.2, 0), hips=(0, 0, 0))


def test_ankle_offset_is_measured_along_the_facing_axis():
    got = walk.reach_geometry(_rig(2), None, "front_thigh.L", "front_shin.L")
    # The ankle sits 2 on X and 0 on Y. Forward is X, so the stance is 2, not 0.
    assert got["horizontal_at_rest"] == pytest.approx(2.0)


def test_chain_length_is_in_world_units():
    got = walk.reach_geometry(_rig(2), None, "front_thigh.L", "front_shin.L")
    # Each rest bone is length 1. Scale 2 makes the world chain 4, not 2.
    assert got["chain_length"] == pytest.approx(4.0)
    assert got["vertical_drop"] == pytest.approx(2.0)


def test_a_metarig_does_not_replace_the_rig_measurement():
    metarig_bones = {
        "front_thigh.L": Bone(Vec(0, 0, 0), Vec(10, 0, 0)),
        "front_shin.L": Bone(Vec(0, 0, 0), Vec(0, 10, 0)),
    }
    metarig = Rig(metarig_bones, 1, head=(0, 1, 0), hips=(0, 0, 0))
    got = walk.reach_geometry(_rig(2), metarig, "front_thigh.L", "front_shin.L")
    assert got["chain_length"] == pytest.approx(4.0)
    assert got["horizontal_at_rest"] == pytest.approx(2.0)
