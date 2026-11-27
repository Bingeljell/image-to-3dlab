"""The walk cycle's auto-plant must work in world space.

Paw markers (`JOINT_*`) are world positions. A mesh that is moved, parented or
rotated has different local coordinates, so probing the paws and measuring the
floor with `v.co` plants the feet at the wrong height. The Blender program is a
string, so these tests run the two relevant slices of it against stand-ins.
"""

from __future__ import annotations

import math
import textwrap
import types

import blender_walk_cycle as cycle


class Vec:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def copy(self):
        return Vec(self.x, self.y, self.z)

    def __sub__(self, other):
        return Vec(self.x - other.x, self.y - other.y, self.z - other.z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    @property
    def length_squared(self):
        return self.x**2 + self.y**2 + self.z**2


class Lift:
    """Pure translation up by `dz`, applied with `@` like a Blender matrix."""

    def __init__(self, dz):
        self.dz = dz

    def __matmul__(self, v):
        return Vec(v.x, v.y, v.z + self.dz)


def _code():
    return cycle.TEMPLATE.format(
        frames=20, swing_front=24.0, swing_back=20.0, bend_front=34.0, bend_back=38.0,
        wrist=40.0, ankle=22.0, gait="trot", base_bend=8.0, bob=0.016, sway=2.2,
        head_yaw=0.0, head_pitch=0.0, head_level=0.0, chest_drop=0.0,
        shoulder_tuck=0.0, hind_lead_l=0.0, hind_lead_r=0.0, body_drop=0.0,
        front_crouch=0.0, crouch=0.0, front_sign=-1.0, toe=20.0, auto_plant=True,
    )  # fmt: skip


def _slice(start, end):
    code = _code()
    begin = code.index(start)
    return textwrap.dedent(code[begin : code.index(end, begin)])


def _mesh(dz, local_zs):
    return types.SimpleNamespace(
        matrix_world=Lift(dz),
        data=types.SimpleNamespace(
            vertices=[types.SimpleNamespace(co=Vec(0, 0, z)) for z in local_zs]
        ),
    )


def test_rest_floor_is_the_paw_height_in_world_space():
    # Mesh sits 10 up. Local vertices: paw tip at 0 (world 10), a knee at 5 (world 15).
    # The paw marker is a world position, (0, 0, 10). Matched against local
    # coordinates it picks the knee (local 5 is nearest 10), and the floor reads 5.
    mesh = _mesh(10.0, [0.0, 5.0])
    marker = types.SimpleNamespace(
        name="JOINT_frontL_paw",
        matrix_world=types.SimpleNamespace(translation=Vec(0, 0, 10)),
    )
    scope = {"bpy": types.SimpleNamespace(data=types.SimpleNamespace(objects=[marker])), "mesh": mesh,
             "mk": {"frontL_paw": marker.matrix_world.translation}}  # fmt: skip
    exec(_slice("paw_probe = []", "for frame in range"), scope)
    assert scope["paw_probe"] == [0]
    assert scope["rest_floor"] == 10.0


def test_plant_shift_compares_world_heights():
    # Rest floor is 10 (world). Posed, the paw vertex is at local 0.5 on a mesh
    # sitting 10 up, so world 10.5: the body must drop 0.5. Reading local Z gives
    # 10 - 0.5 = 9.5, which would launch the body.
    evaluated = _mesh(10.0, [0.5])
    mesh = types.SimpleNamespace(evaluated_get=lambda dg: evaluated)
    view_layer = types.SimpleNamespace(update=lambda: None)
    context = types.SimpleNamespace(
        view_layer=view_layer, evaluated_depsgraph_get=lambda: None
    )
    scope = {
        "AUTO_PLANT": True, "paw_probe": [0], "rest_floor": 10.0, "mesh": mesh,
        "base_z": 0.0, "arm": types.SimpleNamespace(location=Vec(0, 0, 0)),
        "bpy": types.SimpleNamespace(context=context), "math": math,
    }  # fmt: skip
    exec(
        _slice(
            "    plant = 0.0\n    if AUTO_PLANT", "    arm.location.z = base_z + plant"
        ),
        scope,
    )
    assert scope["plant"] == -0.5
